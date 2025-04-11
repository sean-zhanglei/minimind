import os
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from typing import Optional
from dataclasses import dataclass

from transformers import AutoTokenizer
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset
from torchinfo import summary

warnings.filterwarnings('ignore')

@dataclass
class PretrainConfig:
    out_dir: str = "out"  # 模型输出目录
    epochs: int = 1  # 训练总轮数
    # batch_size: int = 32
    batch_size: int = 32 * 2  # 每个batch的样本数
    learning_rate: float = 5e-4  # 初始学习率
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"  # 训练设备
    dtype: str = "bfloat16"  # 训练精度
    use_tensorboard: bool = True  # 是否使用TensorBoard记录
    tensorboard_log_dir: str = "./logs/tensorboard"  # TensorBoard日志目录
    base_seed: int = 1337  # 随机种子
    
    ddp: bool = True  # 是否启用分布式数据并行训练
    world_size: int = 4  # 分布式训练中的总进程数
    rank: int = 0  # 当前进程的全局排名(0到world_size-1)
    local_rank: int = 0  # 当前节点上的进程本地排名
    
    accumulation_steps: int = 8  # 梯度累积步数
    grad_clip: float = 1.0  # 梯度裁剪阈值
    warmup_iters: int = 0  # 学习率warmup步数
    log_interval: int = 100  # 日志记录间隔(step)
    save_interval: int = 100  # 模型保存间隔(step)
    dim: int = 512  # 模型隐藏层维度
    n_layers: int = 8  # 模型层数
    max_seq_len: int = 512  # 最大序列长度
    use_moe: bool = False  # 是否使用混合专家(MoE)结构
    data_path: str = "./dataset/pretrain_hq.jsonl"  # 训练数据路径

class PretrainLogger:
    def __init__(self, ddp: bool = False, rank: int = 0):
        self.ddp = ddp
        self.rank = rank
    
    def log(self, content: str, level: str = "INFO"):
        if not self.ddp or self.rank == 0:
            print(f"[{level}] {content}")

class PretrainTrainer:
    def __init__(self, config: PretrainConfig):
        self.config = config
        self.logger = PretrainLogger(config.ddp, getattr(config, 'local_rank', 0))
        self._setup()
        self._init_tensorboard()
    
    def _setup(self):
        self._setup_device()
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

    def _init_tensorboard(self):
        if not self.config.use_tensorboard or (self.config.ddp and dist.get_rank() != 0):
            return
            
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self.config.tensorboard_log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.config.tensorboard_log_dir)
            
            # Log model architecture and hyperparameters
            self.writer.add_text(
                "model_config", 
                str(self.lm_config.__dict__)
            )
            self.writer.add_text(
                "training_config",
                str({
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "epochs": self.config.epochs,
                    "warmup_iters": self.config.warmup_iters,
                    "grad_clip": self.config.grad_clip
                })
            )
        except ImportError:
            self.logger.log("tensorboard not installed, skipping initialization", "WARNING")
        
    def _setup_device(self):
        self.logger.log("_setup_device Configure device and distributed training...")
      
        torch.manual_seed(self.config.base_seed)
        torch.cuda.manual_seed(self.config.base_seed)
    
        if self.config.ddp and self._check_ddp_ready():
            rank = dist.get_rank()
            torch.manual_seed(self.config.base_seed + rank)
            torch.cuda.manual_seed(self.config.base_seed + rank)
            self._init_distributed()
        else:
            self.config.ddp = False
            self.logger.log("DDP not ready, using single GPU training") 
            
        self.logger.log(f"training with...device:{self.config.device}")
        self.config.device_type = "cuda" if "cuda" in self.config.device else "cpu"
        self.logger.log(f"training with...device_type:{self.config.device_type}")
        if(self.config.device_type == "cuda"):
             self.logger.log("Using GPU for training with...")
        self.ctx = nullcontext() if self.config.device_type == "cpu" else torch.cuda.amp.autocast()
    
    def _check_ddp_ready(self):
        """检查DDP环境是否就绪"""
        return (
            torch.distributed.is_initialized() and 
            torch.cuda.device_count() > 1 and
            "LOCAL_RANK" in os.environ
        )
    
    def _init_distributed(self):
        self.logger.log("_init_distributed Initialize distributed training...")
        
        dist.init_process_group(backend="nccl")
        self.config.local_rank = int(os.environ["LOCAL_RANK"])
        self.config.device = f"cuda:{self.config.local_rank}"
        torch.cuda.set_device(self.config.device)
    
    def _setup_model(self):
        self.logger.log("_setup_model Initialize model and tokenizer...")
        
        self.lm_config = LMConfig(
            dim=self.config.dim,
            n_layers=self.config.n_layers,
            max_seq_len=self.config.max_seq_len,
            use_moe=self.config.use_moe
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
        self.model = MiniMindLM(self.lm_config).to(self.config.device)
        
        self.logger.log('LLM：网络结构')
        summary(self.model)
        
        self.logger.log(f'LLM总参数量：{sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
        
        if self.config.ddp:
            self.model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
            self.model = DistributedDataParallel(
                self.model, 
                device_ids=[self.config.local_rank]
            )
    
    def _setup_data(self):
        self.logger.log("_setup_data Initialize data loading components...")
        
        train_ds = PretrainDataset(
            self.config.data_path, 
            self.tokenizer, 
            max_length=self.lm_config.max_seq_len
        )
        
        train_sampler = DistributedSampler(train_ds) if self.config.ddp else None
        
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            pin_memory=True,  # 启用内存锁定
            drop_last=False,
            shuffle=(train_sampler is None),  # 非分布式时启用shuffle
            num_workers=min(4, os.cpu_count()),  # 自动设置worker数量
            sampler=train_sampler,
            persistent_workers=True  # 保持worker进程
        )
        self.iter_per_epoch = len(self.train_loader)
    
    def _setup_optimizer(self):
        """初始化优化器和梯度缩放器
        
        1. 设置自动混合精度训练的梯度缩放器
        2. 创建AdamW优化器
        """
        self.logger.log("_setup_optimizer Initialize optimizer and scaler...")
        
        # 初始化梯度缩放器(用于混合精度训练)
        # 当使用float16或bfloat16精度时启用
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.config.dtype in ['float16', 'bfloat16'])
        )
        
        # 初始化AdamW优化器
        # AdamW是Adam优化器的变体，修正了权重衰减的实现
        self.optimizer = optim.AdamW(
            self.model.parameters(),  # 优化模型所有可训练参数
            lr=self.config.learning_rate  # 使用配置中的学习率
        )
    
    def _get_lr(self, current_step: int) -> float:
        """计算当前步数的学习率
        
        实现学习率调度策略，包含warmup阶段和余弦退火阶段
        
        Args:
            current_step: 当前训练步数
            
        Returns:
            float: 计算得到的学习率值
        """
        # 计算总训练步数(epoch数 × 每epoch的迭代次数)
        total_steps = self.config.epochs * self.iter_per_epoch
        # 获取warmup步数配置
        warmup_steps = self.config.warmup_iters
        
        # Warmup阶段：线性增加学习率
        if current_step < warmup_steps:
            return self.config.learning_rate * current_step / warmup_steps
        
        # 余弦退火阶段：使用余弦函数调整学习率
        # 计算当前进度(0-1之间的值)
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        # 应用余弦退火公式，最低学习率为初始学习率的10%
        return self.config.learning_rate * (0.1 + 0.5 * (1 + math.cos(math.pi * progress)))
    
    def _forward_pass(self, step, batch) -> torch.Tensor:
        """执行前向传播计算损失
        
        Args:
            step: 当前训练步数
            batch: 包含输入数据、标签和损失掩码的元组
            
        Returns:
            torch.Tensor: 计算得到的损失值(已考虑梯度累积)
        """
        # 解包批次数据: X是输入序列, Y是目标序列, loss_mask是损失掩码
        X, Y, loss_mask = batch
        
        # 定期记录输入数据的统计信息
        if step % self.config.log_interval == 0:
            self.logger.log(
                f"Forward pass - Batch size: {X.size(0)}, "  # 批次大小
                f"Sequence length: {X.size(1)}, "           # 序列长度
                f"Total elements: {X.numel() + Y.numel() + loss_mask.numel()}",  # 总元素数
                self.config.ddp
            )
        
        # 将数据移动到指定设备(GPU/CPU)
        X = X.to(self.config.device)
        Y = Y.to(self.config.device)
        loss_mask = loss_mask.to(self.config.device)
        
        # 使用自动混合精度上下文(如果启用)
        with self.ctx:
            # 前向传播获取模型输出
            res = self.model(X)
            
            # 计算交叉熵损失(不进行归约)
            loss = nn.CrossEntropyLoss(reduction='none')(
                res.logits.view(-1, res.logits.size(-1)),  # 展平预测logits
                Y.view(-1)                                 # 展平目标标签
            ).view(Y.size())                               # 恢复原始形状
            
            # 应用损失掩码并计算平均损失
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            
            # 添加辅助损失(如MoE中的专家负载平衡损失)
            loss += res.aux_loss
            
            # 返回损失值(考虑梯度累积步数)
            return loss / self.config.accumulation_steps
    
    def _save_checkpoint(self, epoch: int, step: int):
        self.logger.log(f"_save_checkpoint Save model checkpoint...epoch:{epoch + 1}, step:{step + 1}")
        
        if self.config.ddp and dist.get_rank() != 0:
            return
            
        self.model.eval()
        try:
            moe_path = '_moe' if self.lm_config.use_moe else ''
            os.makedirs(self.config.out_dir, exist_ok=True)
            ckp_path = f'{self.config.out_dir}/pretrain_{self.lm_config.dim}{moe_path}.pth'
            
            state_dict = (
                self.model.module.state_dict() 
                if isinstance(self.model, DistributedDataParallel) 
                else self.model.state_dict()
            )
            
            torch.save(state_dict, ckp_path)
            self.logger.log(f"Checkpoint saved to {ckp_path}")
        except Exception as e:
            self.logger.log(f"Failed to save checkpoint: {str(e)}", "ERROR")
        finally:
            self.model.train()
    
    def train(self):
        self.logger.log("train starting Main training loop...")
        try:
            for epoch in range(self.config.epochs):
                self._train_epoch(epoch)
            
            self.logger.log("Training completed successfully!", "SUCCESS")
            if hasattr(self, 'writer'):
                self.writer.close()
        except Exception as e:
            self.logger.log(f"Training failed: {str(e)}", "ERROR")
            if hasattr(self, 'writer'):
                self.writer.close()
            raise
    
    def _train_epoch(self, epoch: int):
        self.logger.log(f"_train_epoch Train one epoch:{epoch + 1}")
        
        self.model.train()
        start_time = time.time()
        
        for step, batch in enumerate(self.train_loader):
            # Forward pass and backward
            loss = self._forward_pass(step, batch)
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation and update
            if (step + 1) % self.config.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            # Update learning rate
            current_step = epoch * self.iter_per_epoch + step
            lr = self._get_lr(current_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Logging
            if step % self.config.log_interval == 0:
                spend_time = time.time() - start_time
                log_data = {
                    "epoch": epoch + 1,
                    "step": step,
                    "loss": loss.item() * self.config.accumulation_steps,
                    "lr": lr,
                    "time_per_step": spend_time / (step + 1),
                    "epoch_progress": step / self.iter_per_epoch
                }
                step_time = spend_time / (step + 1)
                remaining_time = step_time * (self.iter_per_epoch - step - 1)
                self.logger.log(
                    f"Epoch:[{epoch + 1}/{self.config.epochs}] "
                    f"Step:[{step + 1}/{self.iter_per_epoch}] "
                    f"Loss:{log_data['loss']:.3f} "
                    f"LR:{lr:.2e} "
                    f"Time:{spend_time//60:.0f}m{spend_time%60:.0f}s "
                    f"StepTime:{step_time:.2f}s "
                    f"Remaining_time:{remaining_time//60:.0f}m{remaining_time%60:.0f}s"
                )
                if hasattr(self, 'writer'):
                    for key, value in log_data.items():
                        self.writer.add_scalar(key, value, current_step)
                    
                    # Log parameter distributions periodically
                    if current_step % 100 == 0:
                        for name, param in self.model.named_parameters():
                            if 'weight' in name and ('attention' in name or 'feed_forward' in name):
                                self.writer.add_histogram(
                                    f'weights/{name.replace(".", "/")}',
                                    param.data,
                                    current_step
                                )
                                if param.grad is not None:
                                    self.writer.add_histogram(
                                        f'gradients/{name.replace(".", "/")}',
                                        param.grad.data,
                                        current_step
                                    )
            
            # Checkpoint saving
            if (step + 1) % self.config.save_interval == 0:
                self._save_checkpoint(epoch, step)

if __name__ == "__main__":
    config = PretrainConfig()
    trainer = PretrainTrainer(config)
    trainer.train()
