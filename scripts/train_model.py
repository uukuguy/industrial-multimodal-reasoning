#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练增强型多模态模型

此脚本用于训练增强型多模态模型，基于transformers库的Trainer类实现
支持多种训练策略，包括预训练、微调、端到端训练等。
"""

import os
import sys
import json
import time
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from model.optimization import (
    ComputationOptimizationConfig,
    MemoryOptimizationConfig
)
from model.model import EnhancedMultimodalModel

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入模型包
from model import (
    load_config, 
    save_config, 
    init_config,
    MultimodalQuestionAnsweringDataset,
    DataCollator,
    create_trainer
)

# 配置日志
logger = logging.getLogger("model_trainer")

def prepare_datasets(args: argparse.Namespace, config: Optional[Dict[str, Any]] = None) -> Tuple[Optional[MultimodalQuestionAnsweringDataset],
                                                       Optional[MultimodalQuestionAnsweringDataset],
                                                       Optional[MultimodalQuestionAnsweringDataset]]:
    """准备数据集，支持自动划分训练集和验证集"""
    import random
    from torch.utils.data import Subset, random_split
    
    # 获取数据集划分配置
    dataset_split_config = {}
    if config and 'dataset_split' in config:
        dataset_split_config = config.get('dataset_split', {})
    
    # 确定验证集划分参数，优先使用命令行参数，其次使用配置文件
    validation_split_ratio = args.validation_split_ratio
    validation_split_count = args.validation_split_count
    validation_split_seed = args.validation_split_seed
    
    # 训练集
    train_dataset = None
    if args.train_questions:
        full_dataset = MultimodalQuestionAnsweringDataset(
            questions_path=args.train_questions,
            documents_dir=args.documents,
            processed_data_dir=args.processed_data
        )
        logger.info(f"加载数据集: {len(full_dataset)} 个样本")
        
        # 验证集（如有）
        valid_dataset = None
        
        # 如果提供了验证集文件，直接加载
        if args.valid_questions:
            valid_dataset = MultimodalQuestionAnsweringDataset(
                questions_path=args.valid_questions,
                documents_dir=args.documents,
                processed_data_dir=args.processed_data
            )
            train_dataset = full_dataset
            logger.info(f"使用指定验证集: {len(valid_dataset)} 个样本")
        
        # 如果没有提供验证集文件，但提供了划分参数，则自动划分
        elif len(full_dataset) > 1:
            # 设置随机种子以确保可复现性
            random.seed(validation_split_seed)
            torch.manual_seed(validation_split_seed)
            
            # 确定验证集大小
            valid_size = 0
            
            # 检查是否提供了样本数量参数且大于0
            if validation_split_count is not None and validation_split_count > 0:
                # 使用指定的样本数量（优先）
                valid_size = min(validation_split_count, len(full_dataset) - 1)
                logger.info(f"使用指定的验证集样本数: {valid_size}")
            # 如果没有提供有效的样本数量，检查比例参数
            elif validation_split_ratio is not None and validation_split_ratio > 0:
                # 使用指定的比例
                valid_size = int(len(full_dataset) * validation_split_ratio)
                valid_size = max(1, min(valid_size, len(full_dataset) - 1))  # 确保至少有1个样本，且不超过总数-1
                logger.info(f"使用指定的验证集比例 {validation_split_ratio}: {valid_size} 个样本")
            
            if valid_size > 0:
                # 划分数据集
                train_size = len(full_dataset) - valid_size
                train_dataset, valid_dataset = random_split(
                    full_dataset,
                    [train_size, valid_size],
                    generator=torch.Generator().manual_seed(validation_split_seed)
                )
                logger.info(f"数据集已划分: 训练集 {len(train_dataset)} 个样本, 验证集 {len(valid_dataset)} 个样本")
            else:
                # 不划分验证集
                train_dataset = full_dataset
                logger.info(f"未划分验证集，使用全部 {len(train_dataset)} 个样本进行训练")
        else:
            # 没有验证集参数，使用全部数据进行训练
            train_dataset = full_dataset
            logger.info(f"训练集: {len(train_dataset)} 个样本")
    
    # 测试集（如有）
    test_dataset = None
    if args.test_questions:
        test_dataset = MultimodalQuestionAnsweringDataset(
            questions_path=args.test_questions,
            documents_dir=args.documents,
            processed_data_dir=args.processed_data,
            is_test=True
        )
        logger.info(f"测试集: {len(test_dataset)} 个样本")
    
    return train_dataset, valid_dataset, test_dataset

def update_args_from_config(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """从配置文件更新命令行参数（如果未在命令行中明确指定）"""
    # 数据路径参数
    if 'data' in config:
        data_config = config.get('data', {})
        if args.train_questions is None and 'train_questions' in data_config:
            args.train_questions = data_config['train_questions']
            logger.info(f"从配置文件设置train_questions: {args.train_questions}")
            
        if args.valid_questions is None and 'valid_questions' in data_config:
            args.valid_questions = data_config['valid_questions']
            logger.info(f"从配置文件设置valid_questions: {args.valid_questions}")
            
        if args.test_questions is None and 'test_questions' in data_config:
            args.test_questions = data_config['test_questions']
            logger.info(f"从配置文件设置test_questions: {args.test_questions}")
            
        if args.documents is None and 'documents_dir' in data_config:
            args.documents = data_config['documents_dir']
            logger.info(f"从配置文件设置documents: {args.documents}")
            
        if args.processed_data is None and 'processed_data_dir' in data_config:
            args.processed_data = data_config['processed_data_dir']
            logger.info(f"从配置文件设置processed_data: {args.processed_data}")
            
        if args.output_dir == f'outputs/{datetime.now().strftime("%Y%m%d_%H%M%S")}' and 'output_dir' in data_config:
            args.output_dir = data_config['output_dir']
            logger.info(f"从配置文件设置output_dir: {args.output_dir}")
    
    # 训练参数
    training_config = config.get('training', {})
    if 'num_epochs' in training_config and not hasattr(args, 'epochs_set_by_user'):
        args.epochs = training_config['num_epochs']
        logger.info(f"从配置文件设置epochs: {args.epochs}")
        
    if 'batch_size' in training_config and not hasattr(args, 'batch_size_set_by_user'):
        args.batch_size = training_config['batch_size']
        logger.info(f"从配置文件设置batch_size: {args.batch_size}")
        
    if 'gradient_accumulation_steps' in training_config and not hasattr(args, 'gradient_accumulation_steps_set_by_user'):
        args.gradient_accumulation_steps = training_config['gradient_accumulation_steps']
        logger.info(f"从配置文件设置gradient_accumulation_steps: {args.gradient_accumulation_steps}")
        
    if 'fp16' in training_config and not args.fp16:
        args.fp16 = training_config['fp16']
        if args.fp16:
            logger.info("从配置文件启用fp16")
    
    # 优化器参数
    optimizer_config = training_config.get('optimizer', {})
    if 'learning_rate' in optimizer_config and not hasattr(args, 'learning_rate_set_by_user'):
        args.learning_rate = optimizer_config['learning_rate']
        logger.info(f"从配置文件设置learning_rate: {args.learning_rate}")
        
    if 'weight_decay' in optimizer_config and not hasattr(args, 'weight_decay_set_by_user'):
        args.weight_decay = optimizer_config['weight_decay']
        logger.info(f"从配置文件设置weight_decay: {args.weight_decay}")
    
    # 调度器参数
    scheduler_config = training_config.get('scheduler', {})
    if 'warmup_ratio' in scheduler_config and not hasattr(args, 'warmup_ratio_set_by_user'):
        args.warmup_ratio = scheduler_config['warmup_ratio']
        logger.info(f"从配置文件设置warmup_ratio: {args.warmup_ratio}")
    
    # 检查点参数
    checkpoint_config = training_config.get('checkpoint', {})
    if 'save_total_limit' in checkpoint_config and not hasattr(args, 'save_total_limit_set_by_user'):
        args.save_total_limit = checkpoint_config['save_total_limit']
        logger.info(f"从配置文件设置save_total_limit: {args.save_total_limit}")
        
    if 'load_best_model_at_end' in checkpoint_config and not args.load_best_model_at_end:
        args.load_best_model_at_end = checkpoint_config['load_best_model_at_end']
        if args.load_best_model_at_end:
            logger.info("从配置文件启用load_best_model_at_end")
    
    # 日志参数
    logging_config = training_config.get('logging', {})
    if 'steps' in logging_config and not hasattr(args, 'logging_steps_set_by_user'):
        args.logging_steps = logging_config['steps']
        logger.info(f"从配置文件设置logging_steps: {args.logging_steps}")
    
    # PEFT参数
    peft_config = config.get('peft', {})
    if 'use_peft' in peft_config and not args.use_peft:
        args.use_peft = peft_config['use_peft']
        if args.use_peft:
            logger.info("从配置文件启用PEFT")
            
    if args.use_peft and 'peft_technique' in peft_config and not hasattr(args, 'peft_technique_set_by_user'):
        args.peft_technique = peft_config['peft_technique']
        logger.info(f"从配置文件设置peft_technique: {args.peft_technique}")
    
    # 数据增强参数
    data_aug_config = config.get('data_augmentation', {})
    if 'use_data_augmentation' in data_aug_config and not args.use_data_augmentation:
        args.use_data_augmentation = data_aug_config['use_data_augmentation']
        if args.use_data_augmentation:
            logger.info("从配置文件启用数据增强")
    
    # 损失函数参数
    loss_config = config.get('loss', {})
    if 'use_optimized_loss' in loss_config and not args.use_optimized_loss:
        args.use_optimized_loss = loss_config['use_optimized_loss']
        if args.use_optimized_loss:
            logger.info("从配置文件启用优化损失函数")
    
    # 数据集划分参数
    dataset_split_config = config.get('dataset_split', {})
    if 'validation_split_ratio' in dataset_split_config and not hasattr(args, 'validation_split_ratio_set_by_user'):
        args.validation_split_ratio = dataset_split_config['validation_split_ratio']
        logger.info(f"从配置文件设置validation_split_ratio: {args.validation_split_ratio}")
        
    if 'validation_split_count' in dataset_split_config and not hasattr(args, 'validation_split_count_set_by_user'):
        args.validation_split_count = dataset_split_config['validation_split_count']
        logger.info(f"从配置文件设置validation_split_count: {args.validation_split_count}")
        
    if 'validation_split_seed' in dataset_split_config and not hasattr(args, 'validation_split_seed_set_by_user'):
        args.validation_split_seed = dataset_split_config['validation_split_seed']
        logger.info(f"从配置文件设置validation_split_seed: {args.validation_split_seed}")

def update_config_from_args(config: Dict[str, Any], args: argparse.Namespace, report_to: List[str]) -> None:
    """从命令行参数更新配置（命令行参数优先级高于配置文件）"""
    # 确保配置中有必要的部分
    if 'training' not in config:
        config['training'] = {}
    
    # 从命令行参数更新训练配置
    if hasattr(args, 'epochs'):
        config['training']['num_epochs'] = args.epochs
    
    if hasattr(args, 'batch_size'):
        config['training']['batch_size'] = args.batch_size
    
    if hasattr(args, 'gradient_accumulation_steps'):
        config['training']['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    
    if hasattr(args, 'fp16'):
        config['training']['fp16'] = args.fp16
    
    # 优化器配置
    if 'optimizer' not in config['training']:
        config['training']['optimizer'] = {}
    
    if hasattr(args, 'learning_rate'):
        config['training']['optimizer']['learning_rate'] = args.learning_rate
    
    if hasattr(args, 'weight_decay'):
        config['training']['optimizer']['weight_decay'] = args.weight_decay
    
    # 调度器配置
    if 'scheduler' not in config['training']:
        config['training']['scheduler'] = {}
    
    if hasattr(args, 'warmup_ratio'):
        config['training']['scheduler']['warmup_ratio'] = args.warmup_ratio
    
    # 检查点配置
    if 'checkpoint' not in config['training']:
        config['training']['checkpoint'] = {}
    
    if hasattr(args, 'save_total_limit'):
        config['training']['checkpoint']['save_total_limit'] = args.save_total_limit
    
    if hasattr(args, 'load_best_model_at_end'):
        config['training']['checkpoint']['load_best_model_at_end'] = args.load_best_model_at_end
    
    # 日志配置
    if 'logging' not in config['training']:
        config['training']['logging'] = {}
    
    if hasattr(args, 'logging_steps'):
        config['training']['logging']['steps'] = args.logging_steps
    
    # 添加日志配置
    config['training']['report_to'] = report_to
    
    # PEFT配置
    if args.use_peft:
        logger.info(f"启用PEFT，技术: {args.peft_technique}")
        if 'peft' not in config:
            config['peft'] = {}
        config['peft']['use_peft'] = True
        config['peft']['peft_technique'] = args.peft_technique
    
    # 数据增强配置
    if args.use_data_augmentation:
        if 'data_augmentation' not in config:
            config['data_augmentation'] = {}
        config['data_augmentation']['use_data_augmentation'] = True
    
    # 损失函数配置
    if args.use_optimized_loss:
        if 'loss' not in config:
            config['loss'] = {}
        config['loss']['use_optimized_loss'] = True

class ModelTrainer:
    def __init__(
        self,
        model: EnhancedMultimodalModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        computation_config: Optional[ComputationOptimizationConfig] = None,
        memory_config: Optional[MemoryOptimizationConfig] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 初始化优化配置
        self.computation_config = computation_config or ComputationOptimizationConfig()
        self.memory_config = memory_config or MemoryOptimizationConfig()
        
        # 初始化优化器
        self.optimizer = self._init_optimizer()
        self.scaler = GradScaler()
        
        # 性能监控
        self.throughput_history = []
        self.latency_history = []
        
    def _init_optimizer(self):
        """初始化优化器"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.model.config.get('learning_rate', 1e-4),
            weight_decay=self.model.config.get('weight_decay', 0.01)
        )
        
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 1. 准备数据
            batch = self._prepare_batch(batch)
            
            # 2. 前向传播
            with autocast():
                loss = self._forward_pass(batch)
                
            # 3. 反向传播
            self._backward_pass(loss)
            
            # 4. 更新统计信息
            total_loss += loss.item()
            batch_count += 1
            
            # 5. 优化批处理大小
            if batch_count % 10 == 0:
                self._optimize_batch_size()
                
            # 6. 清理内存
            if batch_count % self.memory_config.clear_cache_frequency == 0:
                self.model.clear_memory()
                
        return total_loss / batch_count
        
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """准备批次数据"""
        return {k: v.to(self.model.device) for k, v in batch.items()}
        
    def _forward_pass(self, batch: Dict[str, Any]) -> torch.Tensor:
        """前向传播"""
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        outputs = self.model(**batch)
        end_time.record()
        
        # 记录延迟
        torch.cuda.synchronize()
        latency = start_time.elapsed_time(end_time)
        self.latency_history.append(latency)
        
        return outputs['loss']
        
    def _backward_pass(self, loss: torch.Tensor):
        """反向传播"""
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
    def _optimize_batch_size(self):
        """优化批处理大小"""
        if len(self.throughput_history) > 0 and len(self.latency_history) > 0:
            # 计算平均吞吐量和延迟
            avg_throughput = sum(self.throughput_history[-10:]) / min(10, len(self.throughput_history))
            avg_latency = sum(self.latency_history[-10:]) / min(10, len(self.latency_history))
            
            # 优化批处理大小
            new_batch_size = self.model.optimize_batch_size(avg_throughput, avg_latency)
            
            # 更新数据加载器
            if new_batch_size != self.train_loader.batch_size:
                self._update_dataloader(new_batch_size)
                
    def _update_dataloader(self, new_batch_size: int):
        """更新数据加载器的批处理大小"""
        self.train_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=new_batch_size,
            shuffle=True,
            num_workers=self.train_loader.num_workers,
            pin_memory=self.train_loader.pin_memory
        )
        
    def validate(self) -> float:
        """验证模型"""
        if not self.val_loader:
            return 0.0
            
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._prepare_batch(batch)
                with autocast():
                    loss = self._forward_pass(batch)
                total_loss += loss.item()
                batch_count += 1
                
        return total_loss / batch_count

def train_model(
    model: EnhancedMultimodalModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    computation_config: Optional[ComputationOptimizationConfig] = None,
    memory_config: Optional[MemoryOptimizationConfig] = None
):
    """训练模型"""
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        computation_config=computation_config,
        memory_config=memory_config
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练
        train_loss = trainer.train_epoch(epoch)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        
        # 验证
        if val_loader:
            val_loss = trainer.validate()
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save("best_model.pt")
                
    return model

def main():
    """主函数"""
    # 配置命令行参数解析
    parser = argparse.ArgumentParser(description='训练增强型多模态模型')
    
    # 配置文件参数
    parser.add_argument('--config', type=str, default=None,
                        help='YAML配置文件路径')
    
    # 数据参数
    parser.add_argument('--train_questions', type=str, default=None,
                        help='训练集问题JSONL文件路径')
    parser.add_argument('--valid_questions', type=str, default=None,
                        help='验证集问题JSONL文件路径')
    parser.add_argument('--test_questions', type=str, default=None,
                        help='测试集问题JSONL文件路径')
    parser.add_argument('--documents', type=str, default=None,
                        help='文档目录路径')
    parser.add_argument('--processed_data', type=str, default=None,
                        help='预处理数据目录路径')
    parser.add_argument('--output_file', type=str, default=None,
                        help='结果输出文件路径，如果提供测试集，则必填')
    
    # 数据集划分参数
    parser.add_argument('--validation_split_ratio', type=float, default=None,
                        help='验证集划分比例，0-1之间的浮点数，例如0.2表示20%的数据用于验证')
    parser.add_argument('--validation_split_count', type=int, default=None,
                        help='验证集样本数量，整数，优先级高于比例参数')
    parser.add_argument('--validation_split_seed', type=int, default=42,
                        help='数据集划分的随机种子，用于复现划分结果')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批大小')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--workers', type=int, default=4,
                        help='数据加载工作进程数')
    parser.add_argument('--fp16', action='store_true',
                        help='是否使用混合精度训练')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='梯度累积步数，用于增大有效批量大小')
    
    # 训练超参数
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup比例')
    parser.add_argument('--save_total_limit', type=int, default=3,
                        help='最多保存的检查点数量')
    parser.add_argument('--load_best_model_at_end', action='store_true',
                        help='训练结束时是否加载最优模型')

    # PEFT 参数
    parser.add_argument('--use_peft', action='store_true',
                        help='是否使用PEFT (Parameter-Efficient Fine-Tuning)')
    parser.add_argument('--peft_technique', type=str, default='lora',
                        help='使用的PEFT技术，例如lora')

    # 数据增强参数
    parser.add_argument('--use_data_augmentation', action='store_true',
                        help='是否使用数据增强')

    # 优化损失参数
    parser.add_argument('--use_optimized_loss', action='store_true',
                        help='是否使用优化后的损失函数')

    # 分布式训练参数
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='分布式训练的本地排名，由分布式启动器自动设置')
    parser.add_argument('--ddp', action='store_true',
                        help='启用分布式数据并行(DistributedDataParallel)')
    
    # 日志参数
    parser.add_argument('--logging_steps', type=int, default=50,
                        help='每多少步记录一次日志')
    parser.add_argument('--use_wandb', action='store_true',
                        help='启用Weights & Biases日志记录')
    parser.add_argument('--wandb_project', type=str, default='industrial-multimodal-qa',
                        help='Weights & Biases项目名称')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Weights & Biases运行名称，默认使用输出目录名')
    
    # 配置文件参数
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的模型路径')
    
    # 模式参数
    parser.add_argument('--predict_only', action='store_true',
                       help='仅预测，不进行训练')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default=f'outputs/{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, 'training.log'))
        ]
    )
    
    # 验证参数
    if args.test_questions and args.predict_only and not args.output_file:
        parser.error("当使用--predict_only且提供测试集时，必须指定--output_file")
    
    # 确保必要的参数存在
    # if (args.train_questions or args.test_questions) and not args.documents:
    #     parser.error("必须指定--documents参数或在配置文件的data部分中提供documents_dir参数")
    
    # 配置wandb
    if args.use_wandb:
        try:
            import wandb
            if not args.wandb_name:
                args.wandb_name = os.path.basename(args.output_dir)
            wandb.init(project=args.wandb_project, name=args.wandb_name, dir=args.output_dir)
            logger.info(f"Weights & Biases初始化完成，项目: {args.wandb_project}, 运行: {args.wandb_name}")
            report_to = ["wandb", "tensorboard"]
        except ImportError:
            logger.warning("未安装wandb包，无法使用Weights & Biases记录日志")
            report_to = ["tensorboard"]
    else:
        report_to = ["tensorboard"]
            
    # 加载配置
    if args.config:
        # 从YAML配置文件加载所有配置
        config = load_config(args.config)
        logger.info(f"从配置文件加载配置: {args.config}")
        
        # 从配置文件更新命令行参数（如果未在命令行中明确指定）
        update_args_from_config(args, config)
    else:
        config = init_config()
        logger.info("使用默认配置")
    
    # 从命令行参数更新配置（命令行参数优先级高于配置文件）
    update_config_from_args(config, args, report_to)
    
    # 准备数据集
    train_dataset, valid_dataset, test_dataset = prepare_datasets(args, config)
    
    # 创建数据收集器
    data_collator = DataCollator.collate_fn
    
    # 创建模型
    if args.resume:
        logger.info(f"从检查点加载模型: {args.resume}")
        model = EnhancedMultimodalModel.from_pretrained(args.resume)
    else:
        logger.info("创建新模型")
        model = EnhancedMultimodalModel(**config)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # 训练模型
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs
    )
    
    # 训练模型或预测
    if not args.predict_only:
        logger.info("开始训练模型")
        train_result = model.train()
        model.save()
        
        logger.info(f"训练完成，模型保存到: {args.output_dir}")
        
        # 最终评估
        if valid_dataset:
            logger.info("进行最终评估")
            metrics = model.evaluate()
            model.log_metrics("eval", metrics)
            model.save_metrics("eval", metrics)
    
    # 如果提供测试集，进行预测
    if test_dataset:
        logger.info("开始预测")
        predictions = model.predict(test_dataset)
        
        # 如果指定了输出文件，保存预测结果
        if args.output_file:
            model.save_predictions(predictions, args.output_file)
        else:
            output_file = os.path.join(args.output_dir, "predictions.jsonl")
            model.save_predictions(predictions, output_file)

if __name__ == "__main__":
    main()