import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
import logging
from .base import BaseHead

logger = logging.getLogger(__name__)

class ClassificationHead(BaseHead):
    """分类输出头"""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        **kwargs
    ):
        """初始化分类输出头
        
        Args:
            input_dim: 输入维度
            num_classes: 类别数
            dropout: Dropout比率
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            features: 输入特征
            
        Returns:
            分类logits
        """
        return self.classifier(features)
        
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            logits: 模型输出
            labels: 标签
            
        Returns:
            损失值
        """
        return nn.CrossEntropyLoss()(logits, labels)
        
    def compute_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """计算评估指标
        
        Args:
            logits: 模型输出
            labels: 标签
            
        Returns:
            指标字典
        """
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean().item()
        return {"accuracy": accuracy}

class RegressionHead(BaseHead):
    """回归输出头"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        dropout: float = 0.1,
        **kwargs
    ):
        """初始化回归输出头
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            dropout: Dropout比率
        """
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            features: 输入特征
            
        Returns:
            回归值
        """
        return self.regressor(features)
        
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            logits: 模型输出
            labels: 标签
            
        Returns:
            损失值
        """
        return nn.MSELoss()(logits, labels)
        
    def compute_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """计算评估指标
        
        Args:
            logits: 模型输出
            labels: 标签
            
        Returns:
            指标字典
        """
        mse = nn.MSELoss()(logits, labels).item()
        rmse = torch.sqrt(torch.tensor(mse)).item()
        mae = nn.L1Loss()(logits, labels).item()
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae
        }

class MultiLabelHead(BaseHead):
    """多标签输出头"""
    
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        dropout: float = 0.1,
        **kwargs
    ):
        """初始化多标签输出头
        
        Args:
            input_dim: 输入维度
            num_labels: 标签数
            dropout: Dropout比率
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_labels),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            features: 输入特征
            
        Returns:
            多标签预测
        """
        return self.classifier(features)
        
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            logits: 模型输出
            labels: 标签
            
        Returns:
            损失值
        """
        return nn.BCELoss()(logits, labels)
        
    def compute_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """计算评估指标
        
        Args:
            logits: 模型输出
            labels: 标签
            
        Returns:
            指标字典
        """
        predictions = (logits > 0.5).float()
        accuracy = (predictions == labels).float().mean().item()
        return {"accuracy": accuracy} 