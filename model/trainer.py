# -*- coding: utf-8 -*-

import os
import json
import time
import torch
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from functools import wraps
from transformers import TrainingArguments

from .training.trainer import BaseTrainer
from .training.optimizer import create_optimizer
from .training.scheduler import create_scheduler
from .training.checkpoint import CheckpointManager
from .training.logger import Logger

# 导入优化策略模块
try:
    from .data_augmentation import DataAugmentation
    HAS_DATA_AUG = True
except ImportError:
    HAS_DATA_AUG = False
    logger = logging.getLogger(__name__)
    logger.warning("数据增强模块不可用，无法使用高级数据增强功能")

try:
    from .peft_training import PEFTHandler
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    logger = logging.getLogger(__name__)
    logger.warning("PEFT模块不可用，无法使用参数高效微调功能")

logger = logging.getLogger(__name__)

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def error_handler(func):
    """错误处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            # 可以添加自定义的错误恢复逻辑
            raise
    return wrapper

class EnhancedMultiModalTrainer(BaseTrainer):
    """扩展基础训练器，支持多模态模型训练"""
    
    @error_handler
    def __init__(self, 
                model=None, 
                args=None, 
                data_collator=None,
                train_dataset=None,
                eval_dataset=None,
                tokenizer=None,
                config=None,
                output_dir=None,
                use_peft=False,
                peft_config=None,
                use_data_augmentation=False,
                data_augmentation_config=None,
                use_optimized_loss=False,
                **kwargs):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型实例
            args: TrainingArguments实例，如果未提供，会从config创建
            data_collator: 数据收集器
            train_dataset: 训练数据集
            eval_dataset: 验证数据集
            tokenizer: 分词器，用于处理文本
            config: 配置字典，用于创建TrainingArguments
            output_dir: 输出目录，如果args未提供
            use_peft: 是否使用参数高效微调
            peft_config: PEFT配置
            use_data_augmentation: 是否使用数据增强
            data_augmentation_config: 数据增强配置
            use_optimized_loss: 是否使用优化后的损失函数
            **kwargs: 传递给父类的其他参数
        """
        logger.info("Initializing EnhancedMultiModalTrainer...")
        
        # 如果未提供args，从config创建
        if args is None and config is not None:
            logger.info("Creating TrainingArguments from config...")
            training_config = config.get('training', {})
            output_dir = output_dir or f'outputs/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            
            args = self._create_training_args(training_config, output_dir, eval_dataset, kwargs)
            
        # 保存配置
        self.config = config
        if config is not None and output_dir is not None:
            self._save_config(config, output_dir)
        
        # 存储是否使用优化损失的标志
        self.use_optimized_loss = use_optimized_loss
        
        # 应用数据增强
        self._setup_data_augmentation(use_data_augmentation, data_augmentation_config, train_dataset)
        
        # 应用PEFT
        self._setup_peft(use_peft, peft_config, model)
        
        # 创建数据加载器
        train_loader = self._create_data_loader(train_dataset, data_collator, args)
        val_loader = self._create_data_loader(eval_dataset, data_collator, args) if eval_dataset else None
            
        # 初始化父类
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_config=self._get_optimizer_config(args),
            scheduler_config=self._get_scheduler_config(args),
            checkpoint_config=self._get_checkpoint_config(args),
            logger_config=self._get_logger_config(args),
            device=args.device if args else None,
            **kwargs
        )
        
        logger.info("EnhancedMultiModalTrainer initialized successfully")

    def _create_data_loader(self, dataset, data_collator, args):
        """创建数据加载器"""
        if dataset is None:
            return None
            
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory
        )

    def _get_optimizer_config(self, args):
        """获取优化器配置"""
        return {
            "type": args.optim,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_epsilon": args.adam_epsilon
        }

    def _get_scheduler_config(self, args):
        """获取学习率调度器配置"""
        return {
            "type": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": args.warmup_steps,
            "num_training_steps": args.max_steps
        }

    def _get_checkpoint_config(self, args):
        """获取检查点配置"""
        return {
            "save_dir": args.output_dir,
            "save_strategy": args.save_strategy,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "load_best_model_at_end": args.load_best_model_at_end
        }

    def _get_logger_config(self, args):
        """获取日志配置"""
        return {
            "log_dir": args.logging_dir,
            "log_steps": args.logging_steps,
            "report_to": args.report_to
        }

    @error_handler
    def _create_training_args(self, training_config: Dict, output_dir: str, 
                            eval_dataset: Optional[Any], kwargs: Dict) -> TrainingArguments:
        """创建训练参数"""
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config.get('num_epochs', 10),
            per_device_train_batch_size=training_config.get('batch_size', 8),
            per_device_eval_batch_size=training_config.get('batch_size', 8),
            learning_rate=training_config.get('optimizer', {}).get('learning_rate', 1e-4),
            weight_decay=training_config.get('optimizer', {}).get('weight_decay', 0.01),
            warmup_ratio=training_config.get('scheduler', {}).get('warmup_ratio', 0.1),
            logging_dir=os.path.join(output_dir, 'logs'),
            report_to=training_config.get('report_to', ["tensorboard"]),
            ddp_find_unused_parameters=False,
            evaluation_strategy=training_config.get('evaluation', {}).get('strategy', 
                "epoch" if eval_dataset is not None else "no"),
            save_strategy=training_config.get('checkpoint', {}).get('save_strategy', "epoch"),
            save_steps=training_config.get('checkpoint', {}).get('save_steps', 500),
            save_total_limit=training_config.get('checkpoint', {}).get('save_total_limit', 3),
            load_best_model_at_end=training_config.get('checkpoint', {}).get('load_best_model_at_end', 
                True if eval_dataset is not None else False),
            logging_steps=training_config.get('logging', {}).get('steps', 50),
            **kwargs.pop('training_args', {})
        )
        
        # 检查配置兼容性
        if args.load_best_model_at_end and args.evaluation_strategy == "no":
            logger.warning("load_best_model_at_end requires evaluation_strategy to be non-'no'. "
                         "Setting load_best_model_at_end to False.")
            args.load_best_model_at_end = False
            
        return args

    @error_handler
    def _save_config(self, config: Dict, output_dir: str) -> None:
        """保存配置到文件"""
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        logger.info(f"Configuration saved to: {config_path}")

    @error_handler
    def _setup_data_augmentation(self, use_data_augmentation: bool, 
                               data_augmentation_config: Optional[Dict], 
                               train_dataset: Optional[Any]) -> None:
        """设置数据增强"""
        self.use_data_augmentation = use_data_augmentation and HAS_DATA_AUG
        self.data_augmentation_config = data_augmentation_config or {}
        
        if self.use_data_augmentation and train_dataset is not None:
            logger.info("Applying data augmentation...")
            train_dataset = self._apply_data_augmentation(train_dataset)
            logger.info(f"Data augmentation completed. Dataset size: {len(train_dataset)}")

    @error_handler
    def _setup_peft(self, use_peft: bool, peft_config: Optional[Dict], model: Optional[Any]) -> None:
        """设置PEFT"""
        self.use_peft = use_peft and HAS_PEFT
        self.peft_config = peft_config or {}
        
        if self.use_peft and model is not None:
            logger.info("Applying PEFT...")
            model = self._apply_peft(model)
            logger.info("PEFT applied successfully")

    def _apply_data_augmentation(self, dataset):
        """应用数据增强策略"""
        logger.info("应用数据增强...")
        
        # 获取数据增强配置
        augmentation_factor = self.data_augmentation_config.get('augmentation_factor', 1.5)
        min_samples_per_type = self.data_augmentation_config.get('min_samples_per_type', 20)
        
        # 创建数据增强器
        augmenter = DataAugmentation(
            seed=self.data_augmentation_config.get('seed', 42)
        )
        
        # 获取文档文本（简单实现，实际应从文档中提取）
        document_texts = {}
        for item in dataset:
            doc = item.get('document', '')
            if doc and doc not in document_texts:
                # 在实际实现中，这里应该从文档目录加载文本
                document_texts[doc] = f"示例文档内容: {doc}"
        
        # 应用数据增强
        augmented_dataset = augmenter.augment_dataset(
            dataset=dataset,
            document_texts=document_texts,
            augmentation_factor=augmentation_factor,
            min_samples_per_type=min_samples_per_type
        )
        
        logger.info(f"数据增强完成. 原始数据量: {len(dataset)}, 增强后数据量: {len(augmented_dataset)}")
        return augmented_dataset
    
    def _apply_peft(self, model):
        """应用参数高效微调"""
        logger.info("应用参数高效微调...")
        
        # 获取PEFT配置
        technique = self.peft_config.get('technique', 'lora')
        
        # 获取模型配置
        model_config = getattr(model, 'config', {})
        if hasattr(model_config, 'to_dict'):
            model_config = model_config.to_dict()
        
        # 创建PEFT处理器
        peft_handler = PEFTHandler(technique=technique)
        
        # 应用PEFT
        peft_model = peft_handler.prepare_model(
            model=model,
            peft_config=self.peft_config,
            model_config=model_config
        )
        
        logger.info(f"PEFT ({technique}) 已应用到模型")
        return peft_model
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """准备批次数据，处理多模态输入
        
        Args:
            batch: 原始批次数据
            
        Returns:
            处理后的批次数据
        """
        # 首先调用父类的方法处理基本数据
        batch = super()._prepare_batch(batch)
        
        # 处理文档编码
        if 'document' in batch:
            batch = self._prepare_document_encodings(batch)
            
        return batch
        
    def _prepare_document_encodings(self, inputs):
        """准备文档编码
        
        Args:
            inputs: 输入数据
            
        Returns:
            处理后的输入数据
        """
        # 这里实现文档编码的处理逻辑
        # 例如：文本编码、图像编码等
        return inputs
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """自定义损失计算方法，处理多模态输入"""
        if self.use_optimized_loss:
            return self._compute_custom_loss(model, inputs, return_outputs)
        return super().compute_loss(model, inputs, return_outputs)
        
    def _compute_custom_loss(self, model, inputs, return_outputs=False):
        """计算自定义损失
        
        Args:
            model: 模型
            inputs: 输入数据
            return_outputs: 是否返回输出
            
        Returns:
            损失值或(损失值, 输出)元组
        """
        # 实现自定义损失计算逻辑
        outputs = model(**inputs)
        loss = outputs.loss
        
        if return_outputs:
            return loss, outputs
        return loss
        
    def evaluate(self, 
                eval_dataset=None,
                ignore_keys=None,
                metric_key_prefix="eval",
                **kwargs):
        """评估模型
        
        Args:
            eval_dataset: 评估数据集
            ignore_keys: 忽略的键
            metric_key_prefix: 指标前缀
            **kwargs: 其他参数
            
        Returns:
            评估结果
        """
        # 实现评估逻辑
        pass
        
    def predict(self, test_dataset):
        """预测
        
        Args:
            test_dataset: 测试数据集
            
        Returns:
            预测结果
        """
        # 实现预测逻辑
        pass
        
    def save_predictions(self, predictions, output_file):
        """保存预测结果
        
        Args:
            predictions: 预测结果
            output_file: 输出文件路径
        """
        # 实现保存预测结果的逻辑
        pass


def create_trainer(
    model,
    config,
    train_dataset=None,
    eval_dataset=None,
    output_dir=None,
    data_collator=None,
    use_peft=False,
    peft_config=None,
    use_data_augmentation=False,
    data_augmentation_config=None,
    use_optimized_loss=False,
    **kwargs
):
    """
    创建训练器的工厂函数
    
    Args:
        model: 模型实例
        config: 配置字典
        train_dataset: 训练数据集
        eval_dataset: 验证数据集
        output_dir: 输出目录
        data_collator: 数据收集器
        use_peft: 是否使用参数高效微调
        peft_config: PEFT配置
        use_data_augmentation: 是否使用数据增强
        data_augmentation_config: 数据增强配置
        use_optimized_loss: 是否使用优化后的损失函数
        **kwargs: 其他参数
        
    Returns:
        EnhancedMultiModalTrainer实例
    """
    # 根据数据集大小自动决定是否使用PEFT
    if train_dataset is not None and not use_peft:
        dataset_size = len(train_dataset)
        if dataset_size < 1000 and HAS_PEFT:
            logger.info(f"数据集较小({dataset_size}样本)，自动启用PEFT优化")
            use_peft = True
            if peft_config is None:
                # 获取默认配置
                from .peft_training import PEFTHandler
                model_size = sum(p.numel() for p in model.parameters()) / 1e6
                technique = PEFTHandler.get_recommended_technique(dataset_size, model_size)
                model_type = getattr(model, 'config', {}).get('model_type', '')
                if hasattr(model.config, 'model_type'):
                    model_type = model.config.model_type
                peft_config = PEFTHandler.get_default_config(technique, model_type)
                peft_config['technique'] = technique
    
    # 根据数据集大小自动决定是否使用数据增强
    if train_dataset is not None and not use_data_augmentation:
        dataset_size = len(train_dataset)
        question_types = set()
        for item in train_dataset:
            question = item.get('question', '')
            for key, words in {
                '位置关系': ['位置', '相对', '方向'],
                '功能描述': ['功能', '作用', '用于'],
                '技术参数': ['参数', '角度', '数值'],
                '结构组成': ['结构', '组成', '部件'],
                '操作步骤': ['步骤', '操作', '首先']
            }.items():
                if any(word in question for word in words):
                    question_types.add(key)
                    break
        
        # 检查是否存在低频问题类型
        if len(question_types) >= 3 and HAS_DATA_AUG:
            type_min_count = 20  # 每种类型的最小样本数
            if dataset_size / len(question_types) < type_min_count:
                logger.info(f"检测到问题类型不平衡，自动启用数据增强")
                use_data_augmentation = True
                if data_augmentation_config is None:
                    data_augmentation_config = {
                        'augmentation_factor': 1.5,
                        'min_samples_per_type': type_min_count
                    }
    
    return EnhancedMultiModalTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        data_collator=data_collator,
        use_peft=use_peft,
        peft_config=peft_config,
        use_data_augmentation=use_data_augmentation,
        data_augmentation_config=data_augmentation_config,
        use_optimized_loss=use_optimized_loss,
        **kwargs
    )