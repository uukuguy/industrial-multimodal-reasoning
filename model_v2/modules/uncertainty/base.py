import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseUncertaintyModule(nn.Module, ABC):
    """不确定性估计模块基类"""
    
    def __init__(self, **kwargs):
        """初始化不确定性估计模块
        
        Args:
            **kwargs: 配置参数
        """
        super().__init__()
        self.config = kwargs
        
    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
        num_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            features: 输入特征，形状为 [batch_size, feature_dim]
            num_samples: 采样数量
            
        Returns:
            包含以下键的字典：
            - mean: 预测均值，形状为 [batch_size, output_dim]
            - variance: 预测方差，形状为 [batch_size, output_dim]
            - samples: 采样结果，形状为 [num_samples, batch_size, output_dim]
        """
        pass
        
    @abstractmethod
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            outputs: 模型输出
            targets: 目标值
            
        Returns:
            损失值
        """
        pass
        
    @abstractmethod
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """计算评估指标
        
        Args:
            outputs: 模型输出
            targets: 目标值
            
        Returns:
            指标字典
        """
        pass
        
    def predict(
        self,
        features: torch.Tensor,
        num_samples: int = 1,
        return_uncertainty: bool = True
    ) -> Dict[str, torch.Tensor]:
        """预测
        
        Args:
            features: 输入特征
            num_samples: 采样数量
            return_uncertainty: 是否返回不确定性估计
            
        Returns:
            包含以下键的字典：
            - predictions: 预测结果
            - uncertainty: 不确定性估计（如果return_uncertainty为True）
        """
        outputs = self.forward(features, num_samples)
        
        predictions = outputs["mean"]
        result = {"predictions": predictions}
        
        if return_uncertainty:
            result["uncertainty"] = outputs["variance"]
            
        return result
        
    def get_confidence_intervals(
        self,
        features: torch.Tensor,
        confidence_level: float = 0.95,
        num_samples: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """获取置信区间
        
        Args:
            features: 输入特征
            confidence_level: 置信水平
            num_samples: 采样数量
            
        Returns:
            包含以下键的字典：
            - lower: 置信区间下界
            - upper: 置信区间上界
        """
        outputs = self.forward(features, num_samples)
        samples = outputs["samples"]
        
        # 计算置信区间
        alpha = (1 - confidence_level) / 2
        lower = torch.quantile(samples, alpha, dim=0)
        upper = torch.quantile(samples, 1 - alpha, dim=0)
        
        return {
            "lower": lower,
            "upper": upper
        } 