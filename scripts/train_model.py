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

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入模型包
from model import (
    EnhancedMultiModalModel, 
    load_config, 
    save_config, 
    init_config,
    MultimodalQuestionAnsweringDataset,
    DataCollator,
    create_trainer
)

# 配置日志
logger = logging.getLogger("model_trainer")

def prepare_datasets(args: argparse.Namespace) -> Tuple[Optional[MultimodalQuestionAnsweringDataset], 
                                                      Optional[MultimodalQuestionAnsweringDataset], 
                                                      Optional[MultimodalQuestionAnsweringDataset]]:
    """准备数据集"""
    # 训练集
    train_dataset = None
    if args.train_questions:
        train_dataset = MultimodalQuestionAnsweringDataset(
            questions_path=args.train_questions,
            documents_dir=args.documents,
            processed_data_dir=args.processed_data
        )
        logger.info(f"训练集: {len(train_dataset)} 个样本")
    
    # 验证集（如有）
    valid_dataset = None
    if args.valid_questions:
        valid_dataset = MultimodalQuestionAnsweringDataset(
            questions_path=args.valid_questions,
            documents_dir=args.documents,
            processed_data_dir=args.processed_data
        )
        logger.info(f"验证集: {len(valid_dataset)} 个样本")
    
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

def main():
    """主函数"""
    # 配置命令行参数解析
    parser = argparse.ArgumentParser(description='训练增强型多模态模型')
    
    # 数据参数
    parser.add_argument('--train_questions', type=str, default=None,
                        help='训练集问题JSONL文件路径')
    parser.add_argument('--valid_questions', type=str, default=None,
                        help='验证集问题JSONL文件路径')
    parser.add_argument('--test_questions', type=str, default=None,
                        help='测试集问题JSONL文件路径')
    parser.add_argument('--documents', type=str, required=True,
                        help='文档目录路径')
    parser.add_argument('--processed_data', type=str, default=None,
                        help='预处理数据目录路径')
    parser.add_argument('--output_file', type=str, default=None,
                        help='结果输出文件路径，如果提供测试集，则必填')
    
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
    
    # 模型参数
    parser.add_argument('--config', type=str, default=None,
                        help='模型配置文件路径')
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
        config = load_config(args.config)
    else:
        config = init_config()
    
    # 添加训练相关配置
    if 'training' not in config:
        config['training'] = {
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
            'optimizer': {
                'type': 'AdamW',
                'learning_rate': 1e-4,
                'weight_decay': 0.01
            },
            'scheduler': {
                'type': 'cosine',
                'warmup_ratio': 0.1
            }
        }
    
    # 添加日志配置
    config['training']['report_to'] = report_to
    
    # 准备数据集
    train_dataset, valid_dataset, test_dataset = prepare_datasets(args)
    
    # 创建数据收集器
    data_collator = DataCollator.collate_fn
    
    # 创建模型
    if args.resume:
        logger.info(f"从检查点加载模型: {args.resume}")
        model = EnhancedMultiModalModel.from_pretrained(args.resume)
    else:
        logger.info("创建新模型")
        model = EnhancedMultiModalModel(**config)
    
    # 创建训练参数
    training_args = {
        "fp16": args.fp16,
        "dataloader_num_workers": args.workers,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
    }
    
    # 添加分布式训练参数
    if args.local_rank != -1 or args.ddp:
        logger.info("启用分布式训练")
        training_args["local_rank"] = args.local_rank
        training_args["ddp_backend"] = "nccl"  # GPU训练使用nccl后端
    
    # 创建训练器
    trainer = create_trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        output_dir=args.output_dir,
        data_collator=data_collator,
        training_args=training_args
    )
    
    # 训练模型或预测
    if not args.predict_only and train_dataset:
        # 训练模型
        logger.info("开始训练模型")
        train_result = trainer.train(resume_from_checkpoint=args.resume)
        trainer.save_model()
        
        # 保存训练结果
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        logger.info(f"训练完成，模型保存到: {args.output_dir}")
        
        # 最终评估
        if valid_dataset:
            logger.info("进行最终评估")
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
    
    # 如果提供测试集，进行预测
    if test_dataset:
        logger.info("开始预测")
        predictions = trainer.predict(test_dataset)
        
        # 如果指定了输出文件，保存预测结果
        if args.output_file:
            trainer.save_predictions(predictions, args.output_file)
        else:
            output_file = os.path.join(args.output_dir, "predictions.jsonl")
            trainer.save_predictions(predictions, output_file)

if __name__ == "__main__":
    main()