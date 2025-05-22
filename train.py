import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from model.model import ModelFactory
from utils.data_utils import create_dataset
from utils.metrics import calculate_metrics
from utils.logging_utils import setup_logging

def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--val_data_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--ddp', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_distributed():
    """设置分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        gpu = 0
    
    torch.cuda.set_device(gpu)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    return rank, world_size, gpu

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    args: argparse.Namespace,
    rank: int
) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(
        train_loader,
        desc=f'Epoch {epoch}',
        disable=rank != 0
    )
    
    for step, batch in enumerate(progress_bar):
        # 将数据移到GPU
        images = batch['images'].cuda()
        texts = batch['texts']
        targets = batch['targets'].cuda()
        
        # 前向传播
        with autocast(enabled=args.fp16):
            outputs = model(images, texts)
            losses = model.compute_loss(outputs, targets)
            loss = losses['total'] / args.gradient_accumulation_steps
        
        # 反向传播
        if args.fp16:
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        else:
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        # 更新进度条
        total_loss += loss.item() * args.gradient_accumulation_steps
        progress_bar.set_postfix({
            'loss': f'{total_loss / (step + 1):.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    return total_loss / num_batches

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    args: argparse.Namespace,
    rank: int
) -> Dict[str, float]:
    """验证模型"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating', disable=rank != 0):
            # 将数据移到GPU
            images = batch['images'].cuda()
            texts = batch['texts']
            targets = batch['targets'].cuda()
            
            # 前向传播
            outputs = model(images, texts)
            predictions, _ = model.post_process(outputs)
            
            # 收集预测结果
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算评估指标
    metrics = calculate_metrics(all_preds, all_targets)
    
    return metrics

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置分布式训练
    if args.ddp:
        rank, world_size, gpu = setup_distributed()
    else:
        rank, world_size, gpu = 0, 1, 0
    
    # 设置日志
    logger = setup_logging(args.log_dir, rank)
    
    # 加载配置
    config = load_config(args.config_path)
    
    # 创建数据集和数据加载器
    train_dataset = create_dataset(args.train_data_path, config)
    val_dataset = create_dataset(args.val_data_path, config)
    
    if args.ddp:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = ModelFactory.create_model(config)
    model = model.cuda()
    
    if args.ddp:
        model = DDP(model, device_ids=[gpu])
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 创建梯度缩放器
    scaler = GradScaler(enabled=args.fp16)
    
    # 训练循环
    best_f1 = 0
    for epoch in range(args.num_epochs):
        # 设置当前epoch
        if args.ddp:
            train_sampler.set_epoch(epoch)
        model.module.set_epoch(epoch) if args.ddp else model.set_epoch(epoch)
        
        # 训练一个epoch
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            epoch,
            args,
            rank
        )
        
        # 验证
        metrics = validate(model, val_loader, args, rank)
        
        # 更新最佳模型
        if metrics['f1'] > best_f1 and rank == 0:
            best_f1 = metrics['f1']
            if args.ddp:
                torch.save(model.module.state_dict(), 
                          os.path.join(args.checkpoint_dir, 'best_model.pth'))
            else:
                torch.save(model.state_dict(), 
                          os.path.join(args.checkpoint_dir, 'best_model.pth'))
        
        # 记录日志
        if rank == 0:
            logger.info(f'Epoch {epoch}:')
            logger.info(f'Train Loss: {train_loss:.4f}')
            logger.info(f'Validation Metrics: {metrics}')
    
    # 清理分布式环境
    if args.ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main() 