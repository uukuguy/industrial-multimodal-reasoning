from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
import json
import os
from .config_loader import ConfigLoader
import yaml
import logging

logger = logging.getLogger(__name__)

@dataclass
class EncoderConfig:
    """编码器配置"""
    model_name: str
    pretrained: bool = True
    max_length: int = 512
    dropout: float = 0.1
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    activation: str = "gelu"
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'EncoderConfig':
        """从字典创建配置"""
        return cls(**config)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "max_length": self.max_length,
            "dropout": self.dropout,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "activation": self.activation
        }

@dataclass
class FusionConfig:
    """特征融合配置"""
    method: str
    input_dims: Dict[str, int]
    hidden_size: int = 768
    num_layers: int = 2
    dropout: float = 0.1
    attention_heads: int = 8
    use_residual: bool = True
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'FusionConfig':
        """从字典创建配置"""
        return cls(**config)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "method": self.method,
            "input_dims": self.input_dims,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "attention_heads": self.attention_heads,
            "use_residual": self.use_residual
        }

@dataclass
class HeadConfig:
    """输出头配置"""
    type: str
    hidden_size: int = 768
    num_classes: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'HeadConfig':
        """从字典创建配置"""
        return cls(**config)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.type,
            "hidden_size": self.hidden_size,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "activation": self.activation
        }

@dataclass
class QAConfig:
    """问答模块配置"""
    enabled: bool = True
    max_question_length: int = 128
    max_answer_length: int = 64
    num_beams: int = 4
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'QAConfig':
        """从字典创建配置"""
        return cls(**config)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "enabled": self.enabled,
            "max_question_length": self.max_question_length,
            "max_answer_length": self.max_answer_length,
            "num_beams": self.num_beams,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p
        }

@dataclass
class UncertaintyConfig:
    """不确定性估计配置"""
    enabled: bool = True
    hidden_size: int = 768
    dropout: float = 0.1
    loss_weight: float = 0.1
    temperature: float = 1.0
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'UncertaintyConfig':
        """从字典创建配置"""
        return cls(**config)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "enabled": self.enabled,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "loss_weight": self.loss_weight,
            "temperature": self.temperature
        }

@dataclass
class ComputationConfig:
    """计算优化配置"""
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    torch_compile: bool = True
    compile_mode: str = "reduce-overhead"
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ComputationConfig':
        """从字典创建配置"""
        return cls(**config)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "mixed_precision": self.mixed_precision,
            "gradient_checkpointing": self.gradient_checkpointing,
            "torch_compile": self.torch_compile,
            "compile_mode": self.compile_mode
        }

@dataclass
class MemoryConfig:
    """内存优化配置"""
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    gradient_clipping: bool = True
    memory_efficient_attention: bool = True
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'MemoryConfig':
        """从字典创建配置"""
        return cls(**config)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "gradient_clipping": self.gradient_clipping,
            "memory_efficient_attention": self.memory_efficient_attention
        }

@dataclass
class BatchSizeConfig:
    """批处理大小配置"""
    training: int = 32
    evaluation: int = 64
    gradient_accumulation: int = 1
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'BatchSizeConfig':
        """从字典创建配置"""
        return cls(**config)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "training": self.training,
            "evaluation": self.evaluation,
            "gradient_accumulation": self.gradient_accumulation
        }

@dataclass
class OptimizationConfig:
    """优化配置"""
    computation: ComputationConfig
    memory: MemoryConfig
    batch_size: BatchSizeConfig
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'OptimizationConfig':
        """从字典创建配置"""
        return cls(
            computation=ComputationConfig.from_dict(config["computation"]),
            memory=MemoryConfig.from_dict(config["memory"]),
            batch_size=BatchSizeConfig.from_dict(config["batch_size"])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "computation": self.computation.to_dict(),
            "memory": self.memory.to_dict(),
            "batch_size": self.batch_size.to_dict()
        }

@dataclass
class PEFTConfig:
    """PEFT 配置"""
    use_peft: bool = False
    task_type: str = "seq_cls"  # "seq_cls" or "causal_lm"
    use_8bit: bool = False
    use_4bit: bool = False
    gradient_checkpointing: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"  # "none", "all", or "lora_only"
    modules_to_save: Optional[List[str]] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PEFTConfig":
        """从字典创建配置"""
        return cls(**config_dict)
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

@dataclass
class DistributedConfig:
    """分布式训练配置"""
    enabled: bool = False
    backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU
    world_size: int = -1  # -1 means use all available GPUs
    rank: int = -1
    local_rank: int = -1
    master_addr: str = "localhost"
    master_port: str = "12355"
    init_method: str = "env://"
    sync_bn: bool = True  # 是否同步BatchNorm
    find_unused_parameters: bool = False  # 是否查找未使用的参数
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'DistributedConfig':
        """从字典创建配置"""
        return cls(**config)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

@dataclass
class TrainingConfig:
    """训练配置"""
    output_dir: str = "outputs"
    num_epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_steps: int = -1
    warmup_steps: int = 0
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: Optional[int] = None
    seed: int = 42
    fp16: bool = False
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    metric_name: str = "accuracy"
    metric_mode: str = "max"  # "max" or "min"
    save_best_only: bool = True
    max_checkpoints: int = 5
    num_workers: int = 4
    task_type: str = "classification"  # "classification", "regression", "generation"
    use_peft: bool = False
    peft_config: Optional[PEFTConfig] = None
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """从字典创建配置"""
        if "peft_config" in config_dict and config_dict["peft_config"] is not None:
            config_dict["peft_config"] = PEFTConfig.from_dict(config_dict["peft_config"])
        if "distributed" in config_dict:
            config_dict["distributed"] = DistributedConfig.from_dict(config_dict["distributed"])
        return cls(**config_dict)
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        config_dict = asdict(self)
        if config_dict["peft_config"] is not None:
            config_dict["peft_config"] = config_dict["peft_config"].to_dict()
        if config_dict["distributed"] is not None:
            config_dict["distributed"] = config_dict["distributed"].to_dict()
        return config_dict

@dataclass
class EvaluationConfig:
    """评估配置"""
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])
    eval_steps: int = 1000
    save_best_only: bool = True
    save_last: bool = True
    save_strategy: str = "steps"
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'EvaluationConfig':
        """从字典创建配置"""
        return cls(**config)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "metrics": self.metrics,
            "eval_steps": self.eval_steps,
            "save_best_only": self.save_best_only,
            "save_last": self.save_last,
            "save_strategy": self.save_strategy
        }

@dataclass
class TrainingLaunchConfig:
    """训练启动配置"""
    # 基本配置
    config_path: str = "configs/train_config.yaml"  # 配置文件路径
    output_dir: str = "outputs"  # 输出目录
    seed: int = 42  # 随机种子
    
    # 分布式训练配置
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    
    # 日志配置
    logging_level: str = "INFO"  # 日志级别
    logging_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # 日志格式
    log_to_file: bool = True  # 是否输出到文件
    log_file: str = "training.log"  # 日志文件路径
    
    # 数据集配置
    train_data_path: str = "data/train.json"  # 训练数据路径
    eval_data_path: str = "data/eval.json"  # 验证数据路径
    cache_dir: str = "cache"  # 缓存目录
    
    # 模型配置
    model_name_or_path: str = "bert-base-chinese"  # 模型名称或路径
    model_type: str = "bert"  # 模型类型
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'TrainingLaunchConfig':
        """从字典创建配置"""
        if "distributed" in config:
            config["distributed"] = DistributedConfig.from_dict(config["distributed"])
        return cls(**config)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        config_dict = asdict(self)
        if config_dict["distributed"] is not None:
            config_dict["distributed"] = config_dict["distributed"].to_dict()
        return config_dict
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingLaunchConfig':
        """从YAML文件加载配置
        
        Args:
            yaml_path: YAML文件路径
            
        Returns:
            训练启动配置
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)
    
    def save(self, path: str):
        """保存配置到文件
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)
        logger.info(f"配置已保存到: {path}")

@dataclass
class ModelConfig:
    """模型配置"""
    # 编码器配置
    text_encoder: EncoderConfig
    image_encoder: EncoderConfig
    layout_encoder: Optional[EncoderConfig] = None
    ocr_encoder: Optional[EncoderConfig] = None
    
    # 特征融合配置
    fusion: FusionConfig
    
    # 输出头配置
    head: HeadConfig
    
    # 问答模块配置
    qa: QAConfig = field(default_factory=QAConfig)
    
    # 不确定性估计配置
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    
    # 优化配置
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        computation=ComputationConfig(),
        memory=MemoryConfig(),
        batch_size=BatchSizeConfig()
    ))
    
    # 训练配置
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 评估配置
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # 训练启动配置
    launch: TrainingLaunchConfig = field(default_factory=TrainingLaunchConfig)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ModelConfig':
        """从字典创建配置"""
        if "launch" in config:
            config["launch"] = TrainingLaunchConfig.from_dict(config["launch"])
        return cls(
            text_encoder=EncoderConfig.from_dict(config["text_encoder"]),
            image_encoder=EncoderConfig.from_dict(config["image_encoder"]),
            layout_encoder=EncoderConfig.from_dict(config["layout_encoder"]) if "layout_encoder" in config else None,
            ocr_encoder=EncoderConfig.from_dict(config["ocr_encoder"]) if "ocr_encoder" in config else None,
            fusion=FusionConfig.from_dict(config["fusion"]),
            head=HeadConfig.from_dict(config["head"]),
            qa=QAConfig.from_dict(config.get("qa", {})),
            uncertainty=UncertaintyConfig.from_dict(config.get("uncertainty", {})),
            optimization=OptimizationConfig.from_dict(config.get("optimization", {})),
            training=TrainingConfig.from_dict(config.get("training", {})),
            evaluation=EvaluationConfig.from_dict(config.get("evaluation", {})),
            launch=TrainingLaunchConfig.from_dict(config.get("launch", {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        config = {
            "text_encoder": self.text_encoder.to_dict(),
            "image_encoder": self.image_encoder.to_dict(),
            "fusion": self.fusion.to_dict(),
            "head": self.head.to_dict(),
            "qa": self.qa.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "optimization": self.optimization.to_dict(),
            "training": self.training.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "launch": self.launch.to_dict()
        }
        
        if self.layout_encoder:
            config["layout_encoder"] = self.layout_encoder.to_dict()
        if self.ocr_encoder:
            config["ocr_encoder"] = self.ocr_encoder.to_dict()
            
        return config
    
    def save(self, path: str):
        """保存配置到文件
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)
        logger.info(f"配置已保存到: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """从文件加载配置
        
        Args:
            path: 配置文件路径
            
        Returns:
            模型配置
        """
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"已从 {path} 加载配置")
        return cls.from_dict(config)

    @classmethod
    def from_pretrained(cls, model_path: str) -> 'ModelConfig':
        """从预训练模型加载配置"""
        config_path = model_path + ".config"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ModelConfig':
        """从YAML文件加载配置
        
        Args:
            yaml_path: YAML文件路径
            
        Returns:
            模型配置
        """
        config_loader = ConfigLoader(yaml_path)
        return cls.from_dict(config_loader.get_config()) 