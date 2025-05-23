import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union, Callable
import logging
from functools import lru_cache
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 指标注册表
METRIC_REGISTRY = {}

@dataclass
class MetricConfig:
    """指标配置"""
    name: str
    task_type: str
    threshold: float = 0.5
    average: str = "macro"  # "macro", "micro", "weighted"
    num_classes: Optional[int] = None
    ignore_index: Optional[int] = None
    reduction: str = "mean"  # "mean", "sum", "none"
    
    def __post_init__(self):
        """验证配置"""
        if self.average not in ["macro", "micro", "weighted"]:
            raise ValueError(f"Invalid average: {self.average}")
        if self.reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {self.reduction}")

def register_metric(name: str, task_type: str):
    """注册指标装饰器
    
    Args:
        name: 指标名称
        task_type: 任务类型
    """
    def decorator(func: Callable):
        METRIC_REGISTRY[name] = {
            "func": func,
            "task_type": task_type
        }
        return func
    return decorator

@register_metric("accuracy", "classification")
def calculate_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config: MetricConfig
) -> float:
    """计算准确率
    
    Args:
        logits: 模型输出
        labels: 标签
        config: 指标配置
        
    Returns:
        准确率
    """
    predictions = torch.argmax(logits, dim=-1)
    return (predictions == labels).float().mean().item()

@register_metric("f1", "classification")
def calculate_f1(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config: MetricConfig
) -> float:
    """计算F1分数
    
    Args:
        logits: 模型输出
        labels: 标签
        config: 指标配置
        
    Returns:
        F1分数
    """
    predictions = torch.argmax(logits, dim=-1)
    precision, recall, f1 = calculate_precision_recall_f1(predictions, labels)
    return f1

@register_metric("mse", "regression")
def calculate_mse(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config: MetricConfig
) -> float:
    """计算均方误差
    
    Args:
        logits: 模型输出
        labels: 标签
        config: 指标配置
        
    Returns:
        均方误差
    """
    return torch.mean((logits - labels) ** 2).item()

@lru_cache(maxsize=128)
def calculate_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_type: str = "classification",
    metric_names: Optional[List[str]] = None,
    config: Optional[MetricConfig] = None
) -> Dict[str, float]:
    """计算评估指标
    
    Args:
        logits: 模型输出
        labels: 标签
        task_type: 任务类型
        metric_names: 指标名称列表
        config: 指标配置
        
    Returns:
        指标字典
    """
    if config is None:
        config = MetricConfig(
            name="default",
            task_type=task_type
        )
        
    if metric_names is None:
        metric_names = [name for name, info in METRIC_REGISTRY.items()
                       if info["task_type"] == task_type]
        
    metrics = {}
    for name in metric_names:
        if name in METRIC_REGISTRY:
            metric_info = METRIC_REGISTRY[name]
            if metric_info["task_type"] == task_type:
                try:
                    metrics[name] = metric_info["func"](logits, labels, config)
                except Exception as e:
                    logger.warning(f"Failed to calculate metric {name}: {e}")
                    
    return metrics

def calculate_multi_label_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """计算多标签分类指标
    
    Args:
        logits: 模型输出
        labels: 标签
        threshold: 阈值
        
    Returns:
        指标字典
    """
    predictions = (torch.sigmoid(logits) > threshold).float()
    
    # 计算每个类别的指标
    metrics = {}
    for i in range(labels.size(1)):
        class_metrics = calculate_metrics(
            predictions[:, i],
            labels[:, i],
            task_type="classification"
        )
        for name, value in class_metrics.items():
            metrics[f"class_{i}_{name}"] = value
            
    # 计算平均指标
    for name in ["accuracy", "precision", "recall", "f1"]:
        values = [v for k, v in metrics.items() if k.endswith(name)]
        metrics[f"average_{name}"] = np.mean(values)
        
    return metrics

def calculate_multi_task_metrics(
    logits: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    task_configs: Dict[str, MetricConfig]
) -> Dict[str, Dict[str, float]]:
    """计算多任务指标
    
    Args:
        logits: 任务输出字典
        labels: 任务标签字典
        task_configs: 任务配置字典
        
    Returns:
        任务指标字典
    """
    metrics = {}
    for task_name, task_logits in logits.items():
        if task_name in task_configs:
            config = task_configs[task_name]
            metrics[task_name] = calculate_metrics(
                task_logits,
                labels[task_name],
                task_type=config.task_type,
                config=config
            )
    return metrics

def calculate_uncertainty_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    uncertainties: torch.Tensor,
    config: Optional[MetricConfig] = None
) -> Dict[str, float]:
    """计算不确定性指标
    
    Args:
        predictions: 预测结果
        labels: 标签
        uncertainties: 不确定性估计
        config: 指标配置
        
    Returns:
        指标字典
    """
    if config is None:
        config = MetricConfig(
            name="uncertainty",
            task_type="uncertainty"
        )
        
    # 计算预测误差
    errors = torch.abs(predictions - labels)
    
    # 计算不确定性校准误差
    calibration_error = torch.mean(torch.abs(uncertainties - errors)).item()
    
    # 计算不确定性相关性
    correlation = torch.corrcoef(torch.stack([uncertainties, errors]))[0, 1].item()
    
    # 计算可靠性图
    reliability = calculate_reliability_diagram(uncertainties, errors)
    
    return {
        "calibration_error": calibration_error,
        "correlation": correlation,
        "reliability": reliability
    }

def calculate_reliability_diagram(
    uncertainties: torch.Tensor,
    errors: torch.Tensor,
    num_bins: int = 10
) -> Dict[str, float]:
    """计算可靠性图
    
    Args:
        uncertainties: 不确定性估计
        errors: 预测误差
        num_bins: 分箱数量
        
    Returns:
        可靠性指标
    """
    # 计算分箱边界
    bin_edges = torch.linspace(0, 1, num_bins + 1)
    
    # 计算每个分箱的统计量
    bin_means = []
    bin_errors = []
    for i in range(num_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
        if mask.any():
            bin_means.append(uncertainties[mask].mean().item())
            bin_errors.append(errors[mask].mean().item())
            
    # 计算可靠性指标
    reliability = {
        "bin_means": bin_means,
        "bin_errors": bin_errors,
        "ece": np.mean(np.abs(np.array(bin_means) - np.array(bin_errors)))
    }
    
    return reliability 