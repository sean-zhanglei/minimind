import os
import platform
import argparse
import random
import time
import math
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from contextlib import nullcontext
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import SFTDataset
from model.model_lora import apply_lora, save_lora

warnings.filterwarnings('ignore')

class Logger:
    def __init__(self, ddp=False, rank=0):
        self.ddp = ddp
        self.rank = rank
    
    def log(self, content):
        if not self.ddp or self.rank == 0:
            print(content)

class LoRAConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="MiniMind SFT with LoRA")
        self._setup_args()
    
    def _setup_args(self):
        self.parser.add_argument("--out_dir", type=str, default="out")
        self.parser.add_argument("--epochs", type=int, default=50)
        self.parser.add_argument("--batch_size", type=int, default=16)
        self.parser.add_argument("--learning_rate", type=float, default="5e-5")
        self.parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
        self.parser.add_argument("--dtype", type=str, default="bfloat16")
        self.parser.add_argument("--num_workers", type=int, default=1)
        self.parser.add_argument("--ddp", action="store_true")
        self.parser.add_argument("--accumulation_steps", type=int, default=1)
        self.parser.add_argument("--grad_clip", type=float, default=1.0)
        self.parser.add_argument("--warmup_iters", type=int, default=0)
        self.parser.add_argument("--log_interval", type=int, default=100)
        self.parser.add_argument("--save_interval", type=int, default=1)
        self.parser.add_argument('--local_rank', type=int, default=-1)
        self.parser.add_argument('--dim', default=512, type=int)
        self.parser.add_argument('--n_layers', default=8, type=int)
        self.parser.add_argument('--max_seq_len', default=512, type=int)
        self.parser.add_argument('--use_moe', default=False, type=bool)
        self.parser.add_argument("--data_path", type=str, default="./dataset/lora_medical.jsonl")
        self.parser.add_argument("--lora_name", type=str, default="lora_medical", help="根据任务保存成lora_(英文/医学/心理...)")
    
    def parse_args(self):
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
        args.tokens_per_iter = args.batch_size * args.lm_config.max_seq_len
        args.device_type = "cuda" if "cuda" in args.device else "cpu"
        return args

class ModelManager:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.lora_params = []
    
    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
        self.model = MiniMindLM(self.args.lm_config)
        moe_path = '_moe' if self.args.lm_config.use_moe else ''
        ckp = f'./out/rlhf_{self.args.lm_config.dim}{moe_path}.pth'
        state_dict = torch.load(ckp, map_location=self.args.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.args.device)
        return self.model, self.tokenizer
    
    def apply_lora(self):
        apply_lora(self.model)
        for name, param in self.model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
            else:
                self.lora_params.append(param)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        lora_params_count = sum(p.numel() for p in self.lora_params)
        return total_params, lora_params_count
    
    def save_lora(self, path):
        save_lora(self.model, path)

class DataLoaderManager:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
    
    def create_loader(self):
        train_ds = SFTDataset(self.args.data_path, self.tokenizer, max_length=self.args.lm_config.max_seq_len)
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

class LoRATrainer:
    def __init__(self, args, model_manager, data_loader_manager):
        self.args = args
        self.model_manager = model_manager
        self.data_loader_manager = data_loader_manager
        self.logger = Logger(args.ddp, dist.get_rank() if args.ddp else 0)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
        self.optimizer = optim.AdamW(self.model_manager.lora_params, lr=args.learning_rate)
        self.train_loader = self.data_loader_manager.create_loader()
        self.iter_per_epoch = len(self.train_loader)
        self.ctx = nullcontext() if args.device_type == "cpu" else torch.cuda.amp.autocast()
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        self.writer = None
        if not args.ddp or dist.get_rank() == 0:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(args.out_dir, 'runs')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
    
    def get_lr(self, current_step, total_steps, lr):
        return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))
    
    def train_epoch(self, epoch):
        start_time = time.time()
        for step, (X, Y, loss_mask) in enumerate(self.train_loader):
            X = X.to(self.args.device)
            Y = Y.to(self.args.device)
            loss_mask = loss_mask.to(self.args.device)
            
            lr = self.get_lr(epoch * self.iter_per_epoch + step, 
                            self.args.epochs * self.iter_per_epoch, 
                            self.args.learning_rate)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            with self.ctx:
                res = self.model_manager.model(X)
                loss = self.loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                loss = (loss * loss_mask).sum() / loss_mask.sum()
                loss += res.aux_loss
                loss = loss / self.args.accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model_manager.lora_params, self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            if step % self.args.log_interval == 0:
                spend_time = time.time() - start_time
                avg_step_time = spend_time / (step + 1)
                remaining_time = (self.iter_per_epoch - step - 1) * avg_step_time
                step_min, step_sec = divmod(avg_step_time, 60)
                remain_min, remain_sec = divmod(remaining_time, 60)
                self.logger.log(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} step_time:{}m{:.0f}s remain:{}m{:.0f}s'.format(
                        epoch + 1,
                        self.config.epochs,
                        step + 1,
                        self.iter_per_epoch,
                        loss.item(),
                        optimizer.param_groups[-1]['lr'],
                        int(step_min), step_sec,
                        int(remain_min), remain_sec),
                    self.ddp
                )

                if self.writer is not None:
                    self.writer.add_scalar('Loss/train', loss.item(), epoch * self.iter_per_epoch + step)
                    self.writer.add_scalar('LearningRate', self.optimizer.param_groups[-1]['lr'], epoch * self.iter_per_epoch + step)

            if (step + 1) % self.args.save_interval == 0 and (not self.args.ddp or dist.get_rank() == 0):
                self.model_manager.model.eval()
                save_path = f'{self.args.save_dir}/lora/{self.args.lora_name}_{self.args.lm_config.dim}.pth'
                os.makedirs(self.args.save_dir + '/lora', exist_ok=True)
                self.model_manager.save_lora(save_path)
                self.model_manager.model.train()

    def train(self):
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)

def init_distributed_mode(args):
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
    config = LoRAConfig()
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
    
    # 初始化模型
    model_manager = ModelManager(args)
    model, tokenizer = model_manager.init_model()
    total_params, lora_params_count = model_manager.apply_lora()
    
    if not args.ddp or dist.get_rank() == 0:
        print(f"LLM 总参数量: {total_params}")
        print(f"LoRA 参数量: {lora_params_count}")
        print(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    
    # 初始化数据加载器
    data_loader_manager = DataLoaderManager(args, tokenizer)
    
    # 创建训练器并开始训练
    trainer = LoRATrainer(args, model_manager, data_loader_manager)
    trainer.train()
