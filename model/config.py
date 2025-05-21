# -*- coding: utf-8 -*-

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_CONFIG = {
    # 模型总体配置
    "model": {
        "name": "IndustrialDocQA",
        "version": "1.0.0",
        "embedding_dim": 768,
        "fusion_dim": 512,
        "use_cache": True,
        "device": "auto",  # "auto", "cpu", "cuda", "cuda:0"
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
        "strategy": "hierarchical",  # "simple", "attention", "hierarchical"
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

    # 系统配置
    "system": {
        "temp_dir": "temp_processed_data",
        "max_workers": 4,
        "batch_size": 8,
        "cache_dir": ".cache",
        "log_level": "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    }
}

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return DEFAULT_CONFIG
        
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
            return DEFAULT_CONFIG
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}，使用默认配置")
        return DEFAULT_CONFIG
        
    # 合并默认配置和用户配置
    merged_config = merge_configs(DEFAULT_CONFIG, config)
    
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
    # 基本配置验证
    required_sections = ['model', 'encoders', 'fusion', 'qa_module', 'system']
    
    for section in required_sections:
        if section not in config:
            logger.warning(f"配置缺少必要部分: {section}，使用默认值")
            config[section] = DEFAULT_CONFIG[section]
    
    # 检查关键参数
    if not config['encoders']['text'].get('model_name'):
        raise ValueError("必须指定文本编码器模型名称")
    
    if not config['encoders']['vision'].get('model_name'):
        raise ValueError("必须指定视觉编码器模型名称")
        
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
    # 加载配置
    if config_path:
        config = load_config(config_path)
    else:
        config = DEFAULT_CONFIG.copy()
        
    # 验证配置
    validate_config(config)
    
    # 设置日志级别
    set_logging_level(config['system']['log_level'])
    
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