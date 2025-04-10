import os
import platform
import argparse
import time
import math
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch.utils.tensorboard import SummaryWriter

from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import SFTDataset
from torchinfo import summary

warnings.filterwarnings('ignore')


class TrainerConfig:
    def __init__(self):
        self.out_dir = "out"
        self.epochs = 1
        # self.batch_size = 32
        self.batch_size = 32 * 2
        self.learning_rate = 5e-5
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = "bfloat16"
        self.use_tensorboard = True
        self.ddp = True
        self.accumulation_steps = 1
        self.grad_clip = 1.0
        self.warmup_iters = 0
        self.log_interval = 100
        self.save_interval = 100
        self.local_rank = -1
        self.dim = 512
        self.n_layers = 8
        self.max_seq_len = 512
        self.use_moe = False
        self.data_path = "./dataset/sft_mini_512.jsonl"

    @classmethod
    def from_args(cls, args):
        config = cls()
        for key, value in vars(args).items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class ModelManager:
    def __init__(self, config):
        self.config = config
        self.lm_config = LMConfig(
            dim=config.dim,
            n_layers=config.n_layers,
            max_seq_len=config.max_seq_len,
            use_moe=config.use_moe
        )
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scaler = None

    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
        self.model = MiniMindLM(self.lm_config)
        moe_path = '_moe' if self.lm_config.use_moe else ''
        ckp = f'./out/pretrain_{self.lm_config.dim}{moe_path}.pth'
        state_dict = torch.load(ckp, map_location=self.config.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.config.device)
        return self.model, self.tokenizer

    def init_optimizer(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.config.dtype in ['float16', 'bfloat16'])
        )
        return self.optimizer, self.scaler

    def save_model(self, path):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        torch.save(state_dict, path)


class TrainerUtils:
    @staticmethod
    def log(content, ddp=False):
        if not ddp or dist.get_rank() == 0:
            print(content)

    @staticmethod
    def get_lr(current_step, total_steps, lr):
        return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


class Trainer:
    def __init__(self, config):
        self.config = config
        self.model_manager = ModelManager(config)
        self.utils = TrainerUtils()
        self.ddp = False
        self.ddp_local_rank = 0
        self.device = "cuda:0"
        self.iter_per_epoch = 0
        self.train_loader = None
        self.writer = None

    def init_distributed_mode(self):
        if not self.config.ddp:
            return

        dist.init_process_group(backend="nccl")
        self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
        self.device = f"cuda:{self.ddp_local_rank}"
        torch.cuda.set_device(self.device)
        self.config.device = torch.device(self.device)

    def init_tensorboard(self):
        if not self.ddp or self.ddp_local_rank == 0:
            log_dir = os.path.join(self.config.out_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

    def _check_ddp_ready(self):
        """检查DDP环境是否就绪"""
        return (
            torch.distributed.is_initialized() and 
            torch.cuda.device_count() > 1 and
            "LOCAL_RANK" in os.environ
        )
    
    def prepare_training(self):
        os.makedirs(self.config.out_dir, exist_ok=True)
        base_seed = 1337
        torch.manual_seed(base_seed)
        torch.cuda.manual_seed(base_seed)

        if self.ddp and self._check_ddp_ready():
            rank = dist.get_rank()
            torch.manual_seed(base_seed + rank)
            torch.cuda.manual_seed(base_seed + rank)
        else:
            self.ddp = False

        model, tokenizer = self.model_manager.init_model()
        
        self.utils.log('LLM：网络结构')
        summary(model)
        
        self.utils.log(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

        train_ds = SFTDataset(self.config.data_path, tokenizer, max_length=self.model_manager.lm_config.max_seq_len)
        train_sampler = DistributedSampler(train_ds) if self.ddp else None
        # 优化DataLoader配置
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

        if self.ddp:
            model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
            model = DistributedDataParallel(model, device_ids=[self.ddp_local_rank])

        return model

    def train_epoch(self, epoch, model):
        # 初始化损失函数和优化器
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        optimizer, scaler = self.model_manager.init_optimizer()
        
        # 记录训练开始时间和自动混合精度上下文
        start_time = time.time()
        ctx = nullcontext() if "cpu" in self.config.device else torch.cuda.amp.autocast()

        # 遍历训练数据
        for step, (X, Y, loss_mask) in enumerate(self.train_loader):
            # 将数据移动到指定设备并记录形状信息
            X = X.to(self.config.device)
            Y = Y.to(self.config.device)
            loss_mask = loss_mask.to(self.config.device)
            
            if step % self.config.log_interval == 0:
                # 记录输入数据形状和大小
                self.utils.log(
                    f"Forward pass - Batch size: {X.size(0)}, "
                    f"Sequence length: {X.size(1)}, "
                    f"Total elements: {X.numel() + Y.numel() + loss_mask.numel()}",
                    self.ddp
                )
            
            lr = self.utils.get_lr(
                epoch * self.iter_per_epoch + step,
                self.config.epochs * self.iter_per_epoch,
                self.config.learning_rate
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            with ctx:
                # 记录前向传播开始时间
                forward_start = time.time()
                res = model(X)
                forward_time = time.time() - forward_start
                
                if step % self.config.log_interval == 0:
                    # 记录前向传播耗时和输出形状
                    self.utils.log(
                        f"Forward pass completed - Time: {forward_time:.4f}s, "
                        f"Logits shape: {res.logits.shape if hasattr(res, 'logits') else 'N/A'}",
                        self.ddp
                    )
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())

                loss = (loss * loss_mask).sum() / loss_mask.sum()
                loss += res.aux_loss
                loss = loss / self.config.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % self.config.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if step % self.config.log_interval == 0:
                spend_time = time.time() - start_time
                avg_step_time = spend_time / (step + 1)
                remaining_time = (self.iter_per_epoch - step - 1) * avg_step_time
                step_min, step_sec = divmod(avg_step_time, 60)
                remain_min, remain_sec = divmod(remaining_time, 60)
                self.utils.log(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} step_time:{}m{:.0f}s remain:{}m{:.0f}s'.format(
                        epoch + 1,
                        self.config.epochs,
                        step,
                        self.iter_per_epoch,
                        loss.item(),
                        optimizer.param_groups[-1]['lr'],
                        int(step_min), step_sec,
                        int(remain_min), remain_sec),
                    self.ddp
                )

                if self.writer is not None:
                    self.writer.add_scalar('train/loss', loss.item(), epoch * self.iter_per_epoch + step)
                    self.writer.add_scalar('train/lr', optimizer.param_groups[-1]['lr'], epoch * self.iter_per_epoch + step)
                    self.writer.add_scalar('train/step_time', avg_step_time, epoch * self.iter_per_epoch + step)
                    self.writer.add_scalar('train/remaining_time', remaining_time, epoch * self.iter_per_epoch + step)

            if (step + 1) % self.config.save_interval == 0:
                model.eval()
                moe_path = '_moe' if self.model_manager.lm_config.use_moe else ''
                ckp = f'{self.config.out_dir}/full_sft_{self.model_manager.lm_config.dim}{moe_path}.pth'
                self.model_manager.save_model(ckp)
                model.train()

    def train(self):
        self.utils.log("train starting Main training loop...")
        try:
            if self.ddp:
                self.init_distributed_mode()

            if self.config.use_tensorboard:
                self.init_tensorboard()
                
            model = self.prepare_training()

            for epoch in range(self.config.epochs):
                self.train_epoch(epoch, model)
                
            self.utils.log("Training completed successfully!", "SUCCESS")
            if hasattr(self, 'writer'):
                self.writer.close()
        except Exception as e:
            self.utils.log(f"Training failed: {str(e)}", "ERROR")
            if hasattr(self, 'writer'):
                self.writer.close()
            raise

if __name__ == "__main__":
    config = TrainerConfig()
    trainer = Trainer(config)
    trainer.train()
