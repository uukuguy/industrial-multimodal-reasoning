import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class UncertaintyEstimator:
    """不确定性估计器
    
    负责估计模型预测的不确定性，包括：
    1. 不确定性计算
    2. 不确定性损失
    3. 不确定性校准
    """
    
    def __init__(
        self,
        hidden_dim: int,
        dropout_rate: float = 0.1,
        uncertainty_weight: float = 1.0
    ):
        """初始化不确定性估计器
        
        Args:
            hidden_dim: 隐藏层维度
            dropout_rate: Dropout率
            uncertainty_weight: 不确定性损失权重
        """
        self.uncertainty_weight = uncertainty_weight
        
        # 创建不确定性估计层
        self.uncertainty_layer = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        logger.info("Uncertainty estimator initialized")
        
    def estimate(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """估计不确定性
        
        Args:
            hidden_states: 隐藏状态
            
        Returns:
            不确定性估计
        """
        return self.uncertainty_layer(hidden_states)
        
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """计算不确定性损失
        
        Args:
            logits: 模型输出
            labels: 标签
            uncertainty: 不确定性估计
            
        Returns:
            不确定性损失
        """
        # 计算预测误差
        error = torch.abs(logits - labels)
        
        # 计算不确定性损失
        loss = torch.mean(
            uncertainty * error - torch.log(uncertainty)
        )
        
        return self.uncertainty_weight * loss
        
    def calibrate(
        self,
        uncertainty: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """校准不确定性估计
        
        Args:
            uncertainty: 不确定性估计
            temperature: 温度参数
            
        Returns:
            校准后的不确定性估计
        """
        return torch.pow(uncertainty, temperature)
        
    def get_uncertainty_info(
        self,
        uncertainty: torch.Tensor
    ) -> Dict[str, float]:
        """获取不确定性信息
        
        Args:
            uncertainty: 不确定性估计
            
        Returns:
            不确定性信息字典
        """
        return {
            "mean": uncertainty.mean().item(),
            "std": uncertainty.std().item(),
            "min": uncertainty.min().item(),
            "max": uncertainty.max().item()
        } 