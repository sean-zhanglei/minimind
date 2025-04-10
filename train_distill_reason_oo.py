import os
import argparse
import time
import math
import warnings

import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch.utils.tensorboard import SummaryWriter

from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import SFTDataset

warnings.filterwarnings('ignore')

class Logger:
    """处理分布式训练环境下的日志输出"""
    def __init__(self, ddp=False, rank=0):
        self.ddp = ddp
        self.rank = rank
    
    def log(self, content):
        """只在主进程输出日志"""
        if not self.ddp or self.rank == 0:
            print(content)

class ReasoningDistillationConfig:
    """处理推理蒸馏训练的所有配置参数"""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="MiniMind Reasoning Distillation Training")
        self._setup_args()
    
    def _setup_args(self):
        """设置所有命令行参数"""
        # 训练参数
        self.parser.add_argument("--out_dir", type=str, default="out", help="输出目录")
        self.parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
        self.parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
        self.parser.add_argument("--learning_rate", type=float, default=1e-6, help="学习率")
        self.parser.add_argument("--device", type=str, 
                               default="cuda:0" if torch.cuda.is_available() else "cpu",
                               help="训练设备")
        self.parser.add_argument("--dtype", type=str, default="bfloat16", 
                               help="训练精度(bfloat16/float16/float32)")
        self.parser.add_argument("--num_workers", type=int, default=1, 
                               help="数据加载工作线程数")
        self.parser.add_argument("--ddp", action="store_true", 
                               help="是否使用分布式训练")
        self.parser.add_argument("--accumulation_steps", type=int, default=1,
                               help="梯度累积步数")
        self.parser.add_argument("--grad_clip", type=float, default=1.0,
                               help="梯度裁剪阈值")
        self.parser.add_argument("--log_interval", type=int, default=1,
                               help="日志记录间隔")
        self.parser.add_argument("--save_interval", type=int, default=50,
                               help="模型保存间隔")
        self.parser.add_argument('--local_rank', type=int, default=-1,
                               help="分布式训练本地rank")
        
        # 模型参数
        self.parser.add_argument('--dim', default=512, type=int,
                               help="模型维度")
        self.parser.add_argument('--n_layers', default=8, type=int,
                               help="模型层数")
        self.parser.add_argument('--max_seq_len', default=1024, type=int,
                               help="最大序列长度")
        self.parser.add_argument('--use_moe', default=False, type=bool,
                               help="是否使用MoE结构")
        
        # 数据参数
        self.parser.add_argument("--data_path", type=str, default="./dataset/r1_mix_1024.jsonl",
                               help="训练数据路径")
    
    def parse_args(self):
        """解析参数并返回配置对象"""
        args = self.parser.parse_args()
        args.lm_config = LMConfig(
            dim=args.dim,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            use_moe=args.use_moe
        )
        args.save_dir = os.path.join(args.out_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.out_dir, exist_ok=True)
        args.device_type = "cuda" if "cuda" in args.device else "cpu"
        return args

class ReasoningDistillationTrainer:
    """推理蒸馏训练器主类"""
    def __init__(self, args):
        self.args = args
        self.logger = Logger(args.ddp, dist.get_rank() if args.ddp else 0)
        self.writer = None
        self._init_tensorboard()
        
        # 初始化模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
        self.model = self._init_model()
        
        # 初始化数据加载器
        self.train_loader = self._init_data()
        self.iter_per_epoch = len(self.train_loader)
        
        # 初始化优化器和相关组件
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.args.dtype in ['float16', 'bfloat16']))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        self.ctx = nullcontext() if self.args.device_type == "cpu" else torch.cuda.amp.autocast()
        
        # 特殊token处理
        self._init_special_tokens()
        
        # 分布式训练处理
        if self.args.ddp:
            self.model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
            self.model = DistributedDataParallel(
                self.model, 
                device_ids=[dist.get_rank()]
            )
    
    def _init_tensorboard(self):
        """初始化TensorBoard日志记录器"""
        if not self.args.ddp or dist.get_rank() == 0:
            log_dir = os.path.join(self.args.out_dir, 'runs')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
    
    def _init_model(self):
        """初始化模型"""
        model = MiniMindLM(self.args.lm_config)
        moe_path = '_moe' if self.args.lm_config.use_moe else ''
        ckp = f'./out/rlhf_{self.args.lm_config.dim}{moe_path}.pth'
        state_dict = torch.load(ckp, map_location=self.args.device)
        model.load_state_dict(state_dict, strict=False)
        
        self.logger.log(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
        model = model.to(self.args.device)
        return model
    
    def _init_data(self):
        """初始化数据加载器"""
        train_ds = SFTDataset(self.args.data_path, self.tokenizer, 
                            max_length=self.args.lm_config.max_seq_len)
        train_sampler = DistributedSampler(train_ds) if self.args.ddp else None
        return DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=self.args.num_workers,
            sampler=train_sampler
        )
    
    def _init_special_tokens(self):
        """初始化特殊token"""
        self.start_of_think_ids = self.tokenizer('<think>').input_ids
        self.end_of_think_ids = self.tokenizer('</think>').input_ids
        self.start_of_answer_ids = self.tokenizer('<answer>').input_ids
        self.end_of_answer_ids = self.tokenizer('</answer>').input_ids
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    def _calculate_loss(self, logits, labels, loss_mask):
        """计算带有特殊token惩罚的损失"""
        # 计算基础交叉熵损失
        loss = self.loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        ).view(labels.size())
        
        # 识别特殊token位置
        sp_ids = torch.isin(labels.view(-1),
            torch.tensor(self.start_of_think_ids + self.end_of_think_ids
                        + self.start_of_answer_ids + self.end_of_answer_ids)
            .to(self.args.device))
        
        # 在特殊token位置增加惩罚
        loss_mask = loss_mask.view(-1)
        loss_mask_sum = loss_mask.sum()
        loss_mask[sp_ids] = 10  # 10倍惩罚
        loss_mask = loss_mask.view(labels.size())
        
        return (loss * loss_mask).sum() / loss_mask_sum
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        start_time = time.time()
        self.model.train()
        
        for step, (X, Y, loss_mask) in enumerate(self.train_loader):
            X = X.to(self.args.device)
            Y = Y.to(self.args.device)
            loss_mask = loss_mask.to(self.args.device)
            
            # 学习率调整
            lr = self._get_lr(epoch * self.iter_per_epoch + step, 
                            self.args.epochs * self.iter_per_epoch, 
                            self.args.learning_rate)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # 前向传播
            with self.ctx:
                res = self.model(X)
                loss = self._calculate_loss(res.logits, Y, loss_mask)
                
                # MoE额外损失
                if self.args.lm_config.use_moe:
                    loss += res.aux_loss
                
                loss = loss / self.args.accumulation_steps

            # 反向传播和优化
            self.scaler.scale(loss).backward()

            if (step + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            # 日志记录
            if step % self.args.log_interval == 0:
                spend_time = time.time() - start_time
                self.logger.log(
                    f'Epoch:[{epoch}/{self.args.epochs - 1}]({step}/{self.iter_per_epoch}) '
                    f'loss:{loss.item():.4f} lr:{self.optimizer.param_groups[-1]["lr"]:.12f} '
                    f'epoch_Time:{spend_time / (step + 1) * self.iter_per_epoch // 60 - spend_time // 60}min'
                )

                if self.writer is not None:
                    self.writer.add_scalar('Loss/train', loss.item(), 
                                         epoch * self.iter_per_epoch + step)
                    self.writer.add_scalar('LearningRate', 
                                         self.optimizer.param_groups[-1]['lr'],
                                         epoch * self.iter_per_epoch + step)

            # 模型保存
            if (step + 1) % self.args.save_interval == 0 and (not self.args.ddp or dist.get_rank() == 0):
                self._save_model()
    
    def _get_lr(self, current_step, total_steps, lr):
        """计算当前学习率"""
        return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))
    
    def _save_model(self):
        """保存模型"""
        self.model.eval()
        moe_path = '_moe' if self.args.lm_config.use_moe else ''
        ckp = f'{self.args.save_dir}/reason_{self.args.lm_config.dim}{moe_path}.pth'

        if isinstance(self.model, DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        torch.save(state_dict, ckp)
        self.model.train()
    
    def train(self):
        """执行完整训练流程"""
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)

def init_distributed_mode(args):
    """初始化分布式训练"""
    if not args.ddp: 
        return args
    
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    args.device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(args.device)
    return args

if __name__ == "__main__":
    # 初始化配置
    config = ReasoningDistillationConfig()
    args = config.parse_args()
    
    # 分布式训练初始化
    if args.ddp:
        args = init_distributed_mode(args)
    
    # 设置随机种子
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)
    if args.ddp:
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)
    
    # 创建并运行训练器
    trainer = ReasoningDistillationTrainer(args)
    trainer.train()
