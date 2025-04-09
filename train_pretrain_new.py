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
    out_dir: str = "out"
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 5e-4
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    use_tensorboard: bool = True
    tensorboard_log_dir: str = "./logs/tensorboard"
    num_workers: int = 1
    base_seed: int = 1337
    
    ddp: bool = True
    world_size: int = 4
    rank: int = 0
    local_rank: int = 0
    local_world_size: int = 4
    master_addr: str = "localhost"
    master_port: str = "12345"
    
    accumulation_steps: int = 8
    grad_clip: float = 1.0
    warmup_iters: int = 0
    log_interval: int = 100
    save_interval: int = 100
    dim: int = 512
    n_layers: int = 8
    max_seq_len: int = 512
    use_moe: bool = False
    data_path: str = "./dataset/pretrain_hq.jsonl"

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
        """Initialize training components"""
        self._setup_device()
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

    def _init_tensorboard(self):
        """Initialize TensorBoard logging"""
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
            self.writer.add_graph(
                self.model,
            )
        except ImportError:
            self.logger.log("tensorboard not installed, skipping initialization", "WARNING")
        
    def _setup_device(self):
        self.logger.log("_setup_device Configure device and distributed training...")
      
        torch.manual_seed(self.config.base_seed)
        torch.cuda.manual_seed(self.config.base_seed)
    
        if self.config.ddp:
            self._init_distributed()
        
        self.device_type = "cuda" if "cuda" in self.config.device else "cpu"
        self.ctx = nullcontext() if self.device_type == "cpu" else torch.cuda.amp.autocast()
    
    def _init_distributed(self):
        self.logger.log("_init_distributed Initialize distributed training...")
        
        # 假设所有进程数即 world_size 为W，每个节点上的进程数即 local_world_size 为L，则每个进程上的两个ID：
        # rank的取值范围：[0, W-1]，rank=0 的进程为主进程，会负责一些同步分发的工作
        # local_rank的取值：[0, L-1]
        
        self.config.master_addr = str(self.config.master_addr)
        os.environ["MASTER_ADDR"] = self.config.master_addr
        
        self.config.master_port = str(self.config.master_port)
        os.environ["MASTER_PORT"] = self.config.master_port
        
        self.config.rank = int(os.environ['RANK'])
        
        self.config.world_size = int(os.environ['WORLD_SIZE'])
        
        self.config.local_rank = int(os.environ['LOCAL_RANK'])
        
        self.config.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        
        self.logger.log(f"Distributed training initialized: rank={self.config.rank}, world_size={self.config.world_size}, local_rank={self.config.local_rank}, local_world_size={self.config.local_world_size}")
        
        # ps 检查nccl是否可用
        if torch.distributed.is_nccl_available ():
            # GPU训练
        
            dist.init_process_group(backend="nccl")
            self.config.device = f"cuda:{self.config.local_rank}"
            torch.cuda.set_device(self.config.device)
            
            rank = dist.get_rank()
            torch.manual_seed(self.config.base_seed + rank)
            # 同时设置 CUDA 的随机种子
            torch.cuda.manual_seed(self.config.base_seed + rank)
        else:
            # CPU训练
            dist.init_process_group(backend="gloo")
            self.config.device = "cpu"
            torch.set_default_device(self.config.device)
            
            rank = dist.get_rank()
            torch.manual_seed(self.config.base_seed + rank)
        
    
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
            # ps 检查nccl是否可用
            if torch.distributed.is_nccl_available ():
                self.logger.log("Using GPU for training with DDP...")
                # GPU训练
                self.model = DistributedDataParallel(self.model, device_ids=[self.config.local_rank])
            else:
                # CPU训练
                self.logger.log("Using CPU for training with DDP...")
                self.model = DistributedDataParallel(self.model)
    
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
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=self.config.num_workers,
            sampler=train_sampler
        )
        self.iter_per_epoch = len(self.train_loader)
    
    def _setup_optimizer(self):
        self.logger.log("_setup_optimizer Initialize optimizer and scaler...")
        
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.config.dtype in ['float16', 'bfloat16'])
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
    
    def _get_lr(self, current_step: int) -> float:
        self.logger.log(f"_get_lr Calculate learning rate with warmup...current_step:{current_step}")
        
        total_steps = self.config.epochs * self.iter_per_epoch
        warmup_steps = self.config.warmup_iters
        
        if current_step < warmup_steps:
            return self.config.learning_rate * current_step / warmup_steps
        
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return self.config.learning_rate * (0.1 + 0.5 * (1 + math.cos(math.pi * progress)))
    
    def _forward_pass(self, batch) -> torch.Tensor:
        self.logger.log(f"_forward_pass Perform forward pass and compute loss...batch:{batch}")
        
        X, Y, loss_mask = batch
        X = X.to(self.config.device)
        Y = Y.to(self.config.device)
        loss_mask = loss_mask.to(self.config.device)
        
        with self.ctx:
            res = self.model(X)
            loss = nn.CrossEntropyLoss(reduction='none')(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            return loss / self.config.accumulation_steps
    
    def _save_checkpoint(self, epoch: int, step: int, loss: torch.Tensor):
        self.logger.log(f"_save_checkpoint Save model checkpoint...epoch:{epoch}, step:{step}, loss:{loss}")
        
        if self.config.ddp and dist.get_rank() != 0:
            return
            
        self.model.eval()
        try:
            moe_path = '_moe' if self.lm_config.use_moe else ''
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
        self.logger.log("_train_epoch Train one epoch...times:" , epoch)
        
        self.model.train()
        start_time = time.time()
        
        for step, batch in enumerate(self.train_loader):
            # Forward pass and backward
            loss = self._forward_pass(batch)
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
                self.logger.log(
                    f"Epoch:[{epoch+1}/{self.config.epochs}] "
                    f"Step:[{step}/{self.iter_per_epoch}] "
                    f"Loss:{log_data['loss']:.3f} "
                    f"LR:{lr:.2e} "
                    f"Time:{spend_time//60:.0f}m{spend_time%60:.0f}s"
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
                self._save_checkpoint(epoch, step, loss)

if __name__ == "__main__":
    config = PretrainConfig()
    trainer = PretrainTrainer(config)
    trainer.train()
