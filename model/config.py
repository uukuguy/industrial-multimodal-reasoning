# -*- coding: utf-8 -*-

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import torch

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 默认配置路径
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "default_config.yaml")

@dataclass
class ModelConfig:
    """模型配置类"""
    
    # 基础配置
    model_name: str = "enhanced_multimodal_model"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # 文本编码器配置
    text_encoder: str = "bert"  # bert, roberta
    text_model_name: str = "bert-base-chinese"
    text_hidden_size: int = 768
    text_max_length: int = 512
    text_dropout: float = 0.1
    
    # 图像编码器配置
    image_encoder: str = "resnet"  # resnet, efficientnet
    image_model_name: str = "resnet50"
    image_hidden_size: int = 2048
    image_dropout: float = 0.1
    
    # 布局编码器配置
    layout_encoder: str = "transformer"  # transformer, graph
    layout_hidden_size: int = 768
    layout_num_layers: int = 6
    layout_num_heads: int = 8
    layout_dropout: float = 0.1
    
    # 特征融合配置
    fusion_type: str = "attention"  # attention, concat, gate
    fusion_hidden_size: int = 768
    fusion_num_heads: int = 8
    fusion_dropout: float = 0.1
    
    # 输出头配置
    head_type: str = "classification"  # classification, regression
    num_classes: int = 2
    head_hidden_size: int = 768
    head_dropout: float = 0.1
    
    # 训练配置
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_epochs: int = 100
    batch_size: int = 32
    gradient_clip_val: float = 1.0
    
    # 损失函数配置
    loss_type: str = "cross_entropy"  # cross_entropy, mse
    label_smoothing: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """从字典创建配置实例"""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> None:
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
    
    def __str__(self) -> str:
        """返回配置的字符串表示"""
        config_str = "Model Configuration:\n"
        for key, value in self.to_dict().items():
            config_str += f"{key}: {value}\n"
        return config_str

@dataclass
class TrainingConfig:
    """训练配置类"""
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = True
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    logging_steps: int = 50
    report_to: list = field(default_factory=lambda: ["tensorboard"])

@dataclass
class DataConfig:
    """数据配置类"""
    train_file: Optional[str] = None
    eval_file: Optional[str] = None
    test_file: Optional[str] = None
    max_seq_length: int = 512
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    cache_dir: Optional[str] = None

@dataclass
class LoggingConfig:
    """日志配置类"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    max_log_files: int = 5
    max_log_size: int = 10 * 1024 * 1024  # 10MB

@dataclass
class Config:
    """总配置类"""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    logging: LoggingConfig
    output_dir: str = "outputs"
    seed: int = 42
    local_rank: int = -1
    world_size: int = 1

class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Config:
        """
        从文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Config对象
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        # 根据文件扩展名选择加载方法
        if config_path.suffix == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
        return ConfigManager._dict_to_config(config_dict)
    
    @staticmethod
    def save_config(config: Config, output_path: Union[str, Path], format: str = 'json') -> None:
        """
        保存配置到文件
        
        Args:
            config: Config对象
            output_path: 输出文件路径
            format: 输出格式 ('json' 或 'yaml')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = ConfigManager._config_to_dict(config)
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
        elif format == 'yaml':
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, allow_unicode=True, sort_keys=False)
        else:
            raise ValueError(f"不支持的输出格式: {format}")
            
        logger.info(f"配置已保存到: {output_path}")
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> Config:
        """将字典转换为Config对象"""
        return Config(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            output_dir=config_dict.get('output_dir', 'outputs'),
            seed=config_dict.get('seed', 42),
            local_rank=config_dict.get('local_rank', -1),
            world_size=config_dict.get('world_size', 1)
        )
    
    @staticmethod
    def _config_to_dict(config: Config) -> Dict[str, Any]:
        """将Config对象转换为字典"""
        return {
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'data': config.data.__dict__,
            'logging': config.logging.__dict__,
            'output_dir': config.output_dir,
            'seed': config.seed,
            'local_rank': config.local_rank,
            'world_size': config.world_size
        }
    
    @staticmethod
    def get_default_config() -> Config:
        """获取默认配置"""
        return Config(
            model=ModelConfig(
                model_name="bert-base-chinese",
                embedding_dim=512,
                temperature=1.0,
                confidence_threshold=0.7
            ),
            training=TrainingConfig(),
            data=DataConfig(),
            logging=LoggingConfig()
        )
    
    @staticmethod
    def validate_config(config: Config) -> bool:
        """
        验证配置的有效性
        
        Args:
            config: 要验证的Config对象
            
        Returns:
            配置是否有效
        """
        try:
            # 验证模型配置
            if not config.model.model_name:
                raise ValueError("模型名称不能为空")
            if config.model.embedding_dim <= 0:
                raise ValueError("嵌入维度必须大于0")
            if not 0 < config.model.temperature <= 2:
                raise ValueError("温度参数必须在0到2之间")
            if not 0 < config.model.confidence_threshold <= 1:
                raise ValueError("置信度阈值必须在0到1之间")
            
            # 验证训练配置
            if config.training.num_epochs <= 0:
                raise ValueError("训练轮数必须大于0")
            if config.training.batch_size <= 0:
                raise ValueError("批次大小必须大于0")
            if config.training.learning_rate <= 0:
                raise ValueError("学习率必须大于0")
            
            # 验证数据配置
            if config.data.max_seq_length <= 0:
                raise ValueError("最大序列长度必须大于0")
            
            # 验证日志配置
            valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if config.logging.level not in valid_log_levels:
                raise ValueError(f"无效的日志级别: {config.logging.level}")
            
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {str(e)}")
            return False

def create_config_from_args(args) -> Config:
    """
    从命令行参数创建配置
    
    Args:
        args: 命令行参数对象
        
    Returns:
        Config对象
    """
    config = ConfigManager.get_default_config()
    
    # 更新模型配置
    if hasattr(args, 'model_name'):
        config.model.model_name = args.model_name
    if hasattr(args, 'embedding_dim'):
        config.model.embedding_dim = args.embedding_dim
    if hasattr(args, 'temperature'):
        config.model.temperature = args.temperature
    if hasattr(args, 'confidence_threshold'):
        config.model.confidence_threshold = args.confidence_threshold
    
    # 更新训练配置
    if hasattr(args, 'num_epochs'):
        config.training.num_epochs = args.num_epochs
    if hasattr(args, 'batch_size'):
        config.training.batch_size = args.batch_size
    if hasattr(args, 'learning_rate'):
        config.training.learning_rate = args.learning_rate
    
    # 更新数据配置
    if hasattr(args, 'train_file'):
        config.data.train_file = args.train_file
    if hasattr(args, 'eval_file'):
        config.data.eval_file = args.eval_file
    if hasattr(args, 'test_file'):
        config.data.test_file = args.test_file
    
    # 更新日志配置
    if hasattr(args, 'log_level'):
        config.logging.level = args.log_level
    if hasattr(args, 'log_file'):
        config.logging.log_file = args.log_file
    
    # 更新其他配置
    if hasattr(args, 'output_dir'):
        config.output_dir = args.output_dir
    if hasattr(args, 'seed'):
        config.seed = args.seed
    
    return config

if __name__ == "__main__":
    # 测试配置管理
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-chinese')
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--confidence_threshold', type=float, default=0.7)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--eval_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 从命令行参数创建配置
    config = create_config_from_args(args)
    
    # 验证配置
    if ConfigManager.validate_config(config):
        print("配置验证通过")
        
        # 保存配置
        ConfigManager.save_config(config, 'config.json')
        ConfigManager.save_config(config, 'config.yaml', format='yaml')
        
        # 加载配置
        loaded_config = ConfigManager.load_config('config.json')
        print("配置加载成功")
    else:
        print("配置验证失败")