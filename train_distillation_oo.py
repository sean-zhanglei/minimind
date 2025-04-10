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

class DistillationConfig:
    """处理知识蒸馏训练的所有配置参数"""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation Training")
        self._setup_args()
    
    def _setup_args(self):
        """设置所有命令行参数"""
        # 训练参数
        self.parser.add_argument("--out_dir", type=str, default="out", help="输出目录")
        self.parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
        self.parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
        self.parser.add_argument("--learning_rate", type=float, default=5e-6, help="学习率")
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
        
        # 数据参数
        self.parser.add_argument("--data_path", type=str, default="./dataset/sft_data.jsonl",
                               help="训练数据路径")
        
        # 蒸馏参数
        self.parser.add_argument("--alpha", type=float, default=0.5,
                               help="CE损失和蒸馏损失的权重平衡")
        self.parser.add_argument("--temperature", type=float, default=1.0,
                               help="蒸馏温度参数")
    
    def parse_args(self):
        """解析参数并返回配置对象"""
        args = self.parser.parse_args()
        
        # 学生模型和教师模型配置
        args.lm_config_student = LMConfig(dim=512, n_layers=8, max_seq_len=512)
        args.lm_config_teacher = LMConfig(dim=768, n_layers=16, max_seq_len=512)
        
        args.save_dir = os.path.join(args.out_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.out_dir, exist_ok=True)
        args.device_type = "cuda" if "cuda" in args.device else "cpu"
        return args

class ModelManager:
    """管理学生模型和教师模型"""
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.student_model = None
        self.teacher_model = None
    
    def init_models(self):
        """初始化学生模型和教师模型"""
        self.tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
        
        # 初始化学生模型
        self.student_model = MiniMindLM(self.args.lm_config_student)
        moe_path = '_moe' if self.args.lm_config_student.use_moe else ''
        ckp = f'./out/full_sft_{self.args.lm_config_student.dim}{moe_path}.pth'
        state_dict = torch.load(ckp, map_location=self.args.device)
        self.student_model.load_state_dict(state_dict, strict=False)
        
        # 初始化教师模型
        self.teacher_model = MiniMindLM(self.args.lm_config_teacher)
        moe_path = '_moe' if self.args.lm_config_teacher.use_moe else ''
        ckp = f'./out/full_sft_{self.args.lm_config_teacher.dim}{moe_path}.pth'
        state_dict = torch.load(ckp, map_location=self.args.device)
        self.teacher_model.load_state_dict(state_dict, strict=False)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        
        # 打印模型参数信息
        student_params = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters() if p.requires_grad)
        print(f'学生模型参数量: {student_params / 1e6:.3f} 百万')
        print(f'教师模型参数量: {teacher_params / 1e6:.3f} 百万')
        
        # 移动到设备
        self.student_model = self.student_model.to(self.args.device)
        self.teacher_model = self.teacher_model.to(self.args.device)
        return self.student_model, self.teacher_model, self.tokenizer

class DistillationTrainer:
    """知识蒸馏训练器主类"""
    def __init__(self, args):
        self.args = args
        self.logger = Logger(args.ddp, dist.get_rank() if args.ddp else 0)
        self.writer = None
        self._init_tensorboard()
        
        # 初始化模型和数据
        self.model_manager = ModelManager(args)
        self.student_model, self.teacher_model, self.tokenizer = self.model_manager.init_models()
        self._init_data()
        self._init_optimizer()
    
    def _init_tensorboard(self):
        """初始化TensorBoard日志记录器"""
        if not self.args.ddp or dist.get_rank() == 0:
            log_dir = os.path.join(self.args.out_dir, 'runs')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
    
    def _init_data(self):
        """初始化数据加载器"""
        train_ds = SFTDataset(self.args.data_path, self.tokenizer, 
                             max_length=self.args.lm_config_student.max_seq_len)
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
        self.optimizer = optim.AdamW(self.student_model.parameters(), lr=self.args.learning_rate)
        self.ctx = nullcontext() if self.args.device_type == "cpu" else torch.cuda.amp.autocast()
        
        if self.args.ddp:
            self.student_model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
            self.student_model = DistributedDataParallel(
                self.student_model, 
                device_ids=[dist.get_rank()]
            )
    
    def _distillation_loss(self, student_logits, teacher_logits, mask):
        """计算蒸馏损失"""
        # 只在有效token位置做蒸馏
        mask_flat = mask.view(-1) == 1
        student_logits = student_logits.view(-1, student_logits.size(-1))[mask_flat]
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))[mask_flat]
        
        teacher_probs = F.softmax(teacher_logits / self.args.temperature, dim=-1).detach()
        student_log_probs = F.log_softmax(student_logits / self.args.temperature, dim=-1)
        
        kl = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        )
        return (self.args.temperature ** 2) * kl
    
    def _ce_loss(self, logits, labels, mask):
        """计算交叉熵损失"""
        mask_flat = mask.view(-1)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=0,
            reduction='none'
        )
        return torch.sum(loss * mask_flat) / mask_flat.sum()
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        start_time = time.time()
        self.student_model.train()
        
        for step, (X, Y, mask) in enumerate(self.train_loader):
            X = X.to(self.args.device)
            Y = Y.to(self.args.device)
            mask = mask.to(self.args.device)
            
            # 记录输入数据形状和大小
            self.logger.log(
                f"Forward pass - Batch size: {X.size(0)}, "
                f"Sequence length: {X.size(1)}, "
                f"Total elements: {X.numel() + Y.numel() + loss_mask.numel()}"
            )
            
            # 学习率调整
            lr = self._get_lr(epoch * self.iter_per_epoch + step, 
                             self.args.epochs * self.iter_per_epoch, 
                             self.args.learning_rate)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # 前向传播
            with self.ctx:
                # 学生模型前向
                student_output = self.student_model(X)
                student_logits = student_output.logits
                
                # 教师模型前向
                with torch.no_grad():
                    teacher_output = self.teacher_model(X)
                    teacher_logits = teacher_output.logits[..., :student_logits.size(-1)]
                
                # 计算损失
                ce_loss = self._ce_loss(student_logits, Y, mask)
                distill_loss = self._distillation_loss(student_logits, teacher_logits, mask)
                
                # 总损失
                loss = self.args.alpha * ce_loss + (1 - self.args.alpha) * distill_loss
                
                # MoE额外损失
                if self.args.lm_config_student.use_moe:
                    loss += student_output.aux_loss

            # 反向传播和优化
            self.scaler.scale(loss).backward()

            if (step + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            # 日志记录
            if step % self.args.log_interval == 0:
                spend_time = time.time() - start_time
                step_time = time.time() - start_time - spend_time
                remaining_steps = self.iter_per_epoch - step - 1
                remaining_time = step_time * remaining_steps
                self.logger.log(
                    f'Epoch:[{epoch + 1}/{self.args.epochs - 1}]({step + 1}/{self.iter_per_epoch}) '
                    f'loss:{loss.item():.4f} lr:{self.optimizer.param_groups[-1]["lr"]:.12f} '
                    f'step_time:{step_time:.2f}s '
                    f'remain:{remaining_time // 60:.0f}m{remaining_time % 60:.0f}s '
                    f'epoch_Time:{spend_time / (step + 1) * self.iter_per_epoch // 60 - spend_time // 60}min'
                )

                if self.writer is not None:
                    self.writer.add_scalar('Loss/train', loss.item(), 
                                         epoch * self.iter_per_epoch + step)
                    self.writer.add_scalar('LearningRate', 
                                         self.optimizer.param_groups[-1]['lr'],
                                         epoch * self.iter_per_epoch + step)
                    self.writer.add_scalar('Loss/CE', ce_loss.item(),
                                         epoch * self.iter_per_epoch + step)
                    self.writer.add_scalar('Loss/Distill', distill_loss.item(),
                                         epoch * self.iter_per_epoch + step)

            # 模型保存
            if (step + 1) % self.args.save_interval == 0 and (not self.args.ddp or dist.get_rank() == 0):
                self._save_model()
    
    def _get_lr(self, current_step, total_steps, lr):
        """计算当前学习率"""
        return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))
    
    def _save_model(self):
        """保存模型"""
        self.student_model.eval()
        moe_path = '_moe' if self.args.lm_config_student.use_moe else ''
        ckp = f'{self.args.save_dir}/full_dist_{self.args.lm_config_student.dim}{moe_path}.pth'

        if isinstance(self.student_model, DistributedDataParallel):
            state_dict = self.student_model.module.state_dict()
        else:
            state_dict = self.student_model.state_dict()

        torch.save(state_dict, ckp)
        self.student_model.train()
    
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
    config = DistillationConfig()
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
    trainer = DistillationTrainer(args)
    trainer.train()
