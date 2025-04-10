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
from model.dataset import DPODataset

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

class DPOConfig:
    """处理DPO训练的所有配置参数"""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="MiniMind DPO Training")
        self._setup_args()
    
    def _setup_args(self):
        """设置所有命令行参数"""
        # 训练参数
        self.parser.add_argument("--out_dir", type=str, default="out", help="输出目录")
        self.parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
        # self.parser.add_argument("--batch_size", type=int, default=8 , help="批次大小")
        self.parser.add_argument("--batch_size", type=int, default=8 * 2 , help="批次大小")
        self.parser.add_argument("--learning_rate", type=float, default=1e-8, 
                               help="学习率(建议1e-8以下避免遗忘)")
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
        self.parser.add_argument("--log_interval", type=int, default=100,
                               help="日志记录间隔")
        self.parser.add_argument("--save_interval", type=int, default=100,
                               help="模型保存间隔")
        self.parser.add_argument('--local_rank', type=int, default=-1,
                               help="分布式训练本地rank")
        
        # 模型参数
        self.parser.add_argument('--dim', default=512, type=int, help="模型维度")
        self.parser.add_argument('--n_layers', default=8, type=int, help="层数")
        self.parser.add_argument('--max_seq_len', default=1024, type=int, 
                               help="最大序列长度")
        self.parser.add_argument('--use_moe', default=False, type=bool,
                               help="是否使用MoE结构")
        
        # 数据参数
        self.parser.add_argument("--data_path", type=str, default="./dataset/dpo.jsonl",
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

class DPOTrainer:
    """DPO训练器主类"""
    def __init__(self, args):
        self.args = args
        self.logger = Logger(args.ddp, dist.get_rank() if args.ddp else 0)
        self.writer = None
        self._init_tensorboard()
        self._init_models()
        self._init_data()
        self._init_optimizer()
    
    def _init_tensorboard(self):
        """初始化TensorBoard日志记录器"""
        if not self.args.ddp or dist.get_rank() == 0:
            log_dir = os.path.join(self.args.out_dir, 'runs')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
    
    def _init_models(self):
        """初始化模型和参考模型"""
        self.tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
        self.model = MiniMindLM(self.args.lm_config)
        moe_path = '_moe' if self.args.lm_config.use_moe else ''
        ckp = f'./out/full_sft_{self.args.lm_config.dim}{moe_path}.pth'
        state_dict = torch.load(ckp, map_location=self.args.device)
        self.model.load_state_dict(state_dict, strict=False)
        
        # 初始化参考模型
        self.ref_model = MiniMindLM(self.args.lm_config)
        self.ref_model.load_state_dict(state_dict, strict=False)
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        
        self.logger.log(f'LLM总参数量：{sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
        self.model = self.model.to(self.args.device)
        self.ref_model = self.ref_model.to(self.args.device)
    
    def _init_data(self):
        """初始化数据加载器"""
        train_ds = DPODataset(self.args.data_path, self.tokenizer, max_length=self.args.lm_config.max_seq_len)
        train_sampler = DistributedSampler(train_ds) if self.args.ddp else None
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=self.args.num_workers,
            sampler=train_sampler
        )
        self.iter_per_epoch = len(self.train_loader)
    
    def _init_optimizer(self):
        """初始化优化器和相关组件"""
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.args.dtype in ['float16', 'bfloat16']))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        self.ctx = nullcontext() if self.args.device_type == "cpu" else torch.cuda.amp.autocast()
        
        if self.args.ddp:
            self.model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
            self.model = DistributedDataParallel(self.model, device_ids=[dist.get_rank()])
    
    def logits_to_probs(self, logits, labels):
        """将模型输出转换为概率"""
        log_probs = F.log_softmax(logits, dim=2)
        probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
        return probs
    
    def dpo_loss(self, ref_probs, probs, mask, beta=0.1):
        """计算DPO损失"""
        seq_lengths = mask.sum(dim=1, keepdim=True)
        ref_probs = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze()
        probs = (probs * mask).sum(dim=1) / seq_lengths.squeeze()

        batch_size = ref_probs.shape[0]
        chosen_ref_probs = ref_probs[:batch_size // 2]
        reject_ref_probs = ref_probs[batch_size // 2:]
        chosen_probs = probs[:batch_size // 2]
        reject_probs = probs[batch_size // 2:]

        pi_logratios = chosen_probs - reject_probs
        ref_logratios = chosen_ref_probs - reject_ref_probs
        logits = pi_logratios - ref_logratios
        return -F.logsigmoid(beta * logits).mean()
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        start_time = time.time()
        self.model.train()
        
        for step, batch in enumerate(self.train_loader):
            # 准备数据
            x_chosen = batch['x_chosen'].to(self.args.device)
            x_rejected = batch['x_rejected'].to(self.args.device)
            y_chosen = batch['y_chosen'].to(self.args.device)
            y_rejected = batch['y_rejected'].to(self.args.device)
            mask_chosen = batch['mask_chosen'].to(self.args.device)
            mask_rejected = batch['mask_rejected'].to(self.args.device)
            
            x = torch.cat([x_chosen, x_rejected], dim=0)
            y = torch.cat([y_chosen, y_rejected], dim=0)
            mask = torch.cat([mask_chosen, mask_rejected], dim=0)
            
            self.logger.log(
                    f"Forward pass - Batch size: {x.size(0)}, "
                    f"Sequence length: {x.size(1)}, "
                    f"Total elements: {x.numel() + y.numel() + mask.numel()}"
                )

            # 学习率调整
            lr = self._get_lr(epoch * self.iter_per_epoch + step, 
                             self.args.epochs * self.iter_per_epoch, 
                             self.args.learning_rate)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # 前向传播和损失计算
            with self.ctx:
                
                with torch.no_grad():
                    ref_outputs = self.ref_model(x)
                    ref_logits = ref_outputs.logits
                ref_probs = self.logits_to_probs(ref_logits, y)
                
                outputs = self.model(x)
                logits = outputs.logits
                probs = self.logits_to_probs(logits, y)
                
                loss = self.dpo_loss(ref_probs, probs, mask)
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
                    f'Epoch:[{epoch + 1}/{self.args.epochs}]({step + 1}/{self.iter_per_epoch}) '
                    f'loss:{loss.item():.3f} lr:{self.optimizer.param_groups[-1]["lr"]:.12f} '
                    f'step_time:{spend_time / (step + 1):.2f}s '
                    f'epoch_Time:{spend_time / (step + 1) * self.iter_per_epoch // 60 - spend_time // 60}min '
                    f'remaining:{((spend_time / (step + 1)) * (self.iter_per_epoch - step - 1)) // 60}min'
                )

                if self.writer is not None:
                    self.writer.add_scalar('Loss/train', loss.item(), 
                                         epoch * self.iter_per_epoch + step)
                    self.writer.add_scalar('LearningRate', 
                                         self.optimizer.param_groups[-1]['lr'],
                                         epoch * self.iter_per_epoch + step)

            # 模型保存
            if (step + 1) % self.args.save_interval == 0 and (not self.args.ddp or dist.get_rank() == 0):
                self._save_model(epoch)
    
    def _get_lr(self, current_step, total_steps, lr):
        """计算当前学习率"""
        return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))
    
    def _save_model(self, epoch):
        """保存模型"""
        self.model.eval()
        moe_path = '_moe' if self.args.lm_config.use_moe else ''
        ckp = f'{self.args.save_dir}/rlhf_{self.args.lm_config.dim}{moe_path}.pth'

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        torch.save(state_dict, ckp)
        self.model.train()
    
    def train(self):
        """执行完整训练流程"""
        self.logger.log("train starting Main training loop...")
        try:
            for epoch in range(self.args.epochs):
                self.train_epoch(epoch)
            self.logger.log("Training completed successfully!", "SUCCESS")
        except Exception as e:
            self.logger.log(f"Training failed: {str(e)}", "ERROR")
            if hasattr(self, 'writer'):
                self.writer.close()
            raise

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
    config = DPOConfig()
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
    trainer = DPOTrainer(args)
    trainer.train()
