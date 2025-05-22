# -*- coding: utf-8 -*-

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 默认配置路径
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "default_config.yaml")

# 加载默认配置
def load_default_config() -> Dict[str, Any]:
    """加载默认配置"""
    if os.path.exists(DEFAULT_CONFIG_PATH):
        try:
            with open(DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载默认配置文件失败: {e}，使用内置默认配置")
    
    # 内置默认配置（简化版本，完整版本在default_config.yaml文件中）
    return {
        # 模型总体配置
        "model": {
            "name": "IndustrialDocQA",
            "version": "1.0.0",
            "embedding_dim": 768,
            "fusion_dim": 512,
            "use_cache": True,
            "device": "auto",
            "use_reconstructor": False,
            "use_uncertainty": False
        },
        # 编码器配置
        "encoders": {
            "text": {
                "model_name": "bert-base-chinese",
                "pooling_strategy": "cls",
                "max_length": 512,
                "batch_size": 16
            },
            "vision": {
                "model_name": "google/vit-base-patch16-224",
                "image_size": 224,
                "batch_size": 8,
                "multiscale_fusion": True
            },
            "layout": {
                "feature_dim": 5,
                "use_positional_encoding": True
            }
        },
        # 融合模块配置
        "fusion": {
            "strategy": "hierarchical",
            "num_attention_heads": 8,
            "dropout": 0.1,
            "use_layer_norm": True
        },
        # 问答模块配置
        "qa_module": {
            "model_name": "placeholder/lmm-model",
            "temperature": 1.0,
            "max_answer_length": 100,
            "confidence_threshold": 0.5,
            "top_k": 1
        },
        # 训练配置
        "training": {
            "num_epochs": 10,
            "batch_size": 8,
            "optimizer": {
                "type": "AdamW",
                "learning_rate": 1e-4,
                "weight_decay": 0.01
            },
            "scheduler": {
                "type": "cosine",
                "warmup_ratio": 0.1
            }
        },
        # 系统配置
        "system": {
            "temp_dir": "temp_processed_data",
            "max_workers": 4,
            "batch_size": 8,
            "cache_dir": ".cache",
            "log_level": "INFO"
        }
    }

# 默认配置（延迟加载）
DEFAULT_CONFIG = None

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    global DEFAULT_CONFIG
    
    # 延迟加载默认配置
    if DEFAULT_CONFIG is None:
        DEFAULT_CONFIG = load_default_config()
    
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return DEFAULT_CONFIG.copy()
        
    _, ext = os.path.splitext(config_path)
    
    try:
        if ext.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif ext.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            logger.warning(f"不支持的配置文件格式: {ext}，使用默认配置")
            return DEFAULT_CONFIG.copy()
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}，使用默认配置")
        return DEFAULT_CONFIG.copy()
        
    # 合并默认配置和用户配置
    merged_config = merge_configs(DEFAULT_CONFIG, config)
    
    # 验证配置
    validate_config(merged_config)
    
    return merged_config

def merge_configs(default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并配置"""
    merged = default_config.copy()
    
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged

def validate_config(config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> bool:
    """验证配置有效性"""
    global DEFAULT_CONFIG
    
    # 延迟加载默认配置
    if DEFAULT_CONFIG is None:
        DEFAULT_CONFIG = load_default_config()
    
    # 基本配置验证
    required_sections = [
        'model', 'encoders', 'fusion', 'qa_module', 'system',
        'training', 'peft', 'data_augmentation', 'dataset_split', 'loss'
    ]
    
    for section in required_sections:
        if section not in config:
            logger.warning(f"配置缺少必要部分: {section}，使用默认值")
            if section in DEFAULT_CONFIG:
                config[section] = DEFAULT_CONFIG[section]
            else:
                logger.error(f"默认配置中也缺少部分: {section}")
    
    # 检查关键参数
    if not config.get('encoders', {}).get('text', {}).get('model_name'):
        raise ValueError("必须指定文本编码器模型名称")
    
    if not config.get('encoders', {}).get('vision', {}).get('model_name'):
        raise ValueError("必须指定视觉编码器模型名称")
    
    # 验证训练配置
    training_config = config.get('training', {})
    if training_config:
        # 验证优化器配置
        optimizer_config = training_config.get('optimizer', {})
        if optimizer_config:
            if 'type' not in optimizer_config:
                logger.warning("优化器类型未指定，使用默认值: AdamW")
                optimizer_config['type'] = 'AdamW'
            
            if 'learning_rate' not in optimizer_config:
                logger.warning("学习率未指定，使用默认值: 1e-4")
                optimizer_config['learning_rate'] = 1e-4
        
        # 验证调度器配置
        scheduler_config = training_config.get('scheduler', {})
        if scheduler_config:
            if 'type' not in scheduler_config:
                logger.warning("学习率调度器类型未指定，使用默认值: cosine")
                scheduler_config['type'] = 'cosine'
    
    # 验证PEFT配置
    peft_config = config.get('peft', {})
    if peft_config and peft_config.get('use_peft', False):
        if 'peft_technique' not in peft_config:
            logger.warning("PEFT技术未指定，使用默认值: lora")
            peft_config['peft_technique'] = 'lora'
        
        # 验证LoRA配置
        if peft_config['peft_technique'] == 'lora' and 'lora' not in peft_config:
            logger.warning("LoRA配置未指定，使用默认值")
            peft_config['lora'] = {
                'r': 8,
                'alpha': 16,
                'dropout': 0.1,
                'target_modules': ["query", "key", "value"]
            }
    
    # 验证数据增强配置
    data_aug_config = config.get('data_augmentation', {})
    if data_aug_config and data_aug_config.get('use_data_augmentation', False):
        if 'augmentation_factor' not in data_aug_config:
            logger.warning("数据增强因子未指定，使用默认值: 1.5")
            data_aug_config['augmentation_factor'] = 1.5
    
    # 验证损失函数配置
    loss_config = config.get('loss', {})
    if loss_config and loss_config.get('use_optimized_loss', False):
        if 'label_smoothing' not in loss_config:
            logger.warning("标签平滑系数未指定，使用默认值: 0.1")
            loss_config['label_smoothing'] = 0.1
    
    # 如果提供了schema，进行详细验证
    if schema:
        # 实现详细的schema验证
        pass
        
    return True

def save_config(config: Dict[str, Any], output_path: str) -> None:
    """保存配置到文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    _, ext = os.path.splitext(output_path)
    
    try:
        if ext.lower() in ['.yaml', '.yml']:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif ext.lower() == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的配置文件格式: {ext}")
        
        logger.info(f"配置已保存到: {output_path}")
    except Exception as e:
        logger.error(f"保存配置失败: {e}")

def set_logging_level(level: str) -> None:
    """设置日志级别"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'无效的日志级别: {level}')
    
    logging.getLogger().setLevel(numeric_level)
    logger.info(f"日志级别设置为: {level}")

def init_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """初始化配置"""
    global DEFAULT_CONFIG
    
    # 延迟加载默认配置
    if DEFAULT_CONFIG is None:
        DEFAULT_CONFIG = load_default_config()
    
    # 加载配置
    if config_path:
        config = load_config(config_path)
    else:
        config = DEFAULT_CONFIG.copy()
        
    # 验证配置
    validate_config(config)
    
    # 设置日志级别
    set_logging_level(config['system']['log_level'])
    
    # 设置随机种子
    if 'seed' in config['system']:
        seed = config['system']['seed']
        import random
        import numpy as np
        import torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if config['system'].get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        logger.info(f"随机种子设置为: {seed}")
    
    # 确保临时目录存在
    os.makedirs(config['system']['temp_dir'], exist_ok=True)
    os.makedirs(config['system']['cache_dir'], exist_ok=True)
    
    return config

if __name__ == "__main__":
    """测试配置模块"""
    # 测试加载默认配置
    config = init_config()
    print("默认配置加载成功")
    
    # 保存默认配置
    os.makedirs("config", exist_ok=True)
    save_config(config, "config/default_config.yaml")
    print("默认配置已保存到: config/default_config.yaml")