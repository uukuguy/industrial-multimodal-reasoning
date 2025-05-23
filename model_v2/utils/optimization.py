import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    LambdaLR,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR
)
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Any, Optional, List, Union, Callable
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class OptimizerConfig:
    """优化器配置"""
    name: str
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    amsgrad: bool = False
    nesterov: bool = False
    centered: bool = False
    rho: float = 0.9
    alpha: float = 0.99
    lr_scale: float = 1.0
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    gradient_clipping: bool = True
    gradient_clipping_value: float = 1.0
    
    def __post_init__(self):
        """验证配置"""
        if self.name not in ["sgd", "adam", "adamw", "rmsprop", "lion", "adafactor"]:
            raise ValueError(f"Invalid optimizer name: {self.name}")

@dataclass
class SchedulerConfig:
    """学习率调度器配置"""
    name: str
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    total_steps: Optional[int] = None
    step_size: int = 1
    gamma: float = 0.1
    milestones: List[int] = field(default_factory=list)
    min_lr: float = 0.0
    max_lr: float = 1e-3
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    cooldown: int = 0
    min_delta: float = 0.0
    
    def __post_init__(self):
        """验证配置"""
        if self.name not in ["linear", "cosine", "step", "multistep", "exponential",
                           "plateau", "onecycle"]:
            raise ValueError(f"Invalid scheduler name: {self.name}")

@dataclass
class AMPConfig:
    """自动混合精度配置"""
    enabled: bool = False
    dtype: str = "float16"  # "float16", "bfloat16"
    scaler: bool = True
    scaler_init_scale: float = 2**16
    scaler_growth_factor: float = 2.0
    scaler_backoff_factor: float = 0.5
    scaler_growth_interval: int = 2000
    scaler_enabled: bool = True
    
    def __post_init__(self):
        """验证配置"""
        if self.dtype not in ["float16", "bfloat16"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

def get_optimizer(
    model: nn.Module,
    config: OptimizerConfig
) -> torch.optim.Optimizer:
    """获取优化器
    
    Args:
        model: 模型
        config: 优化器配置
        
    Returns:
        优化器
    """
    # 获取需要优化的参数
    if hasattr(model, "get_optimizer_params"):
        params = model.get_optimizer_params()
    else:
        params = model.parameters()
        
    # 创建优化器
    if config.name == "sgd":
        optimizer = optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov
        )
    elif config.name == "adam":
        optimizer = optim.Adam(
            params,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.name == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.name == "rmsprop":
        optimizer = optim.RMSprop(
            params,
            lr=config.lr,
            alpha=config.alpha,
            eps=config.eps,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            centered=config.centered
        )
    elif config.name == "lion":
        try:
            from lion_pytorch import Lion
            optimizer = Lion(
                params,
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay
            )
        except ImportError:
            logger.warning("Lion optimizer not available, falling back to AdamW")
            optimizer = optim.AdamW(
                params,
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay
            )
    elif config.name == "adafactor":
        try:
            from transformers import Adafactor, AdafactorSchedule
            optimizer = Adafactor(
                params,
                lr=config.lr,
                scale_parameter=True,
                relative_step=False,
                warmup_init=False,
                clip_threshold=1.0
            )
        except ImportError:
            logger.warning("Adafactor optimizer not available, falling back to AdamW")
            optimizer = optim.AdamW(
                params,
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay
            )
    else:
        raise ValueError(f"Unknown optimizer: {config.name}")
        
    return optimizer

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig
) -> torch.optim.lr_scheduler._LRScheduler:
    """获取学习率调度器
    
    Args:
        optimizer: 优化器
        config: 调度器配置
        
    Returns:
        学习率调度器
    """
    if config.name == "linear":
        def lr_lambda(step):
            if step < config.warmup_steps:
                return float(step) / float(max(1, config.warmup_steps))
            return max(
                0.0,
                float(config.total_steps - step) / float(max(1, config.total_steps - config.warmup_steps))
            )
        scheduler = LambdaLR(optimizer, lr_lambda)
    elif config.name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.total_steps,
            eta_min=config.min_lr
        )
    elif config.name == "step":
        scheduler = StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma
        )
    elif config.name == "multistep":
        scheduler = MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma
        )
    elif config.name == "exponential":
        scheduler = ExponentialLR(
            optimizer,
            gamma=config.gamma
        )
    elif config.name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            min_lr=config.min_lr,
            cooldown=config.cooldown,
            min_delta=config.min_delta
        )
    elif config.name == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.max_lr,
            total_steps=config.total_steps,
            pct_start=config.pct_start,
            div_factor=config.div_factor,
            final_div_factor=config.final_div_factor
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.name}")
        
    return scheduler

def get_amp_scaler(config: AMPConfig) -> Optional[GradScaler]:
    """获取梯度缩放器
    
    Args:
        config: AMP配置
        
    Returns:
        梯度缩放器
    """
    if config.enabled and config.scaler:
        return GradScaler(
            init_scale=config.scaler_init_scale,
            growth_factor=config.scaler_growth_factor,
            backoff_factor=config.scaler_backoff_factor,
            growth_interval=config.scaler_growth_interval,
            enabled=config.scaler_enabled
        )
    return None

def get_amp_dtype(config: AMPConfig) -> torch.dtype:
    """获取AMP数据类型
    
    Args:
        config: AMP配置
        
    Returns:
        AMP数据类型
    """
    if config.dtype == "float16":
        return torch.float16
    elif config.dtype == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Invalid dtype: {config.dtype}")

def get_amp_context(config: AMPConfig):
    """获取AMP上下文
    
    Args:
        config: AMP配置
        
    Returns:
        AMP上下文
    """
    if config.enabled:
        return autocast(dtype=get_amp_dtype(config))
    return nullcontext()

class nullcontext:
    """空上下文管理器"""
    def __enter__(self):
        return None
    def __exit__(self, *excinfo):
        pass

class ComputationOptimizationConfig:
    """计算优化配置"""
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_torch_compile: bool = True
    num_workers: int = 4
    pin_memory: bool = True

class MemoryOptimizationConfig:
    """内存优化配置"""
    use_gradient_accumulation: bool = True
    gradient_accumulation_steps: int = 4
    use_gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    use_memory_efficient_attention: bool = True

def get_grad_scaler() -> torch.cuda.amp.GradScaler:
    """获取梯度缩放器
    
    Returns:
        梯度缩放器实例
    """
    return torch.cuda.amp.GradScaler()

class ComputationOptimizer:
    """计算优化器"""
    
    @staticmethod
    def optimize(model: nn.Module) -> nn.Module:
        """优化模型计算
        
        Args:
            model: 模型
            
        Returns:
            优化后的模型
        """
        # 启用自动混合精度
        model = model.half()
        
        # 启用梯度检查点
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            
        # 启用JIT编译
        try:
            model = torch.jit.script(model)
        except Exception as e:
            logger.warning(f"Failed to enable JIT compilation: {e}")
            
        return model

class MemoryOptimizer:
    """内存优化器"""
    
    @staticmethod
    def optimize(model: nn.Module) -> nn.Module:
        """优化模型内存使用
        
        Args:
            model: 模型
            
        Returns:
            优化后的模型
        """
        # 启用梯度检查点
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            
        # 启用内存优化
        if hasattr(model, "enable_memory_efficient_sdp"):
            model.enable_memory_efficient_sdp()
            
        return model

class BatchSizeOptimizer:
    """批处理大小优化器"""
    
    @staticmethod
    def optimize(
        model: nn.Module,
        initial_batch_size: int = 32,
        target_throughput: float = 100.0,
        target_latency: float = 0.1,
        max_batch_size: int = 128,
        min_batch_size: int = 1
    ) -> int:
        """优化批处理大小
        
        Args:
            model: 模型
            initial_batch_size: 初始批处理大小
            target_throughput: 目标吞吐量
            target_latency: 目标延迟
            max_batch_size: 最大批处理大小
            min_batch_size: 最小批处理大小
            
        Returns:
            优化后的批处理大小
        """
        current_batch_size = initial_batch_size
        
        # 测量当前性能
        current_throughput, current_latency = BatchSizeOptimizer._measure_performance(
            model, current_batch_size
        )
        
        # 二分查找最优批处理大小
        while min_batch_size <= max_batch_size:
            mid_batch_size = (min_batch_size + max_batch_size) // 2
            
            # 测量性能
            throughput, latency = BatchSizeOptimizer._measure_performance(
                model, mid_batch_size
            )
            
            # 更新批处理大小
            if throughput >= target_throughput and latency <= target_latency:
                min_batch_size = mid_batch_size + 1
                current_batch_size = mid_batch_size
            else:
                max_batch_size = mid_batch_size - 1
                
        logger.info(f"优化后的批处理大小: {current_batch_size}")
        return current_batch_size
    
    @staticmethod
    def _measure_performance(
        model: nn.Module,
        batch_size: int
    ) -> Tuple[float, float]:
        """测量性能
        
        Args:
            model: 模型
            batch_size: 批处理大小
            
        Returns:
            吞吐量和延迟
        """
        # 创建示例输入
        inputs = torch.randn(batch_size, 3, 224, 224).to(next(model.parameters()).device)
        
        # 预热
        for _ in range(3):
            with torch.no_grad():
                _ = model(inputs)
                
        # 测量性能
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        with torch.no_grad():
            _ = model(inputs)
        end_time.record()
        
        torch.cuda.synchronize()
        latency = start_time.elapsed_time(end_time) / 1000.0  # 转换为秒
        throughput = batch_size / latency
        
        return throughput, latency 