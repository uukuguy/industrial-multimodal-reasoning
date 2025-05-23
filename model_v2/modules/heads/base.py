import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseHead(nn.Module, ABC):
    """输出头基类"""
    
    def __init__(self, **kwargs):
        """初始化输出头
        
        Args:
            **kwargs: 配置参数
        """
        super().__init__()
        self.config = kwargs
        
    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            features: 输入特征，形状为 [batch_size, input_dim]
            
        Returns:
            输出logits，形状根据任务类型不同而不同：
            - 分类任务: [batch_size, num_classes]
            - 回归任务: [batch_size, output_dim]
            - 多标签任务: [batch_size, num_labels]
        """
        pass
        
    @abstractmethod
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
        pass
        
    @abstractmethod
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
        pass
        
    def get_input_dim(self) -> int:
        """获取输入维度
        
        Returns:
            输入维度
        """
        return self.config.get("input_dim")
        
    def get_output_dim(self) -> int:
        """获取输出维度
        
        Returns:
            输出维度
        """
        return self.config.get("num_classes", self.config.get("output_dim", 1))
        
    def get_config(self) -> Dict[str, Any]:
        """获取配置
        
        Returns:
            配置字典
        """
        return self.config.copy()
        
    def save_config(self, path: str) -> None:
        """保存配置
        
        Args:
            path: 保存路径
        """
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)
            
    @classmethod
    def from_config(cls, path: str) -> "BaseHead":
        """从配置加载
        
        Args:
            path: 配置路径
            
        Returns:
            输出头
        """
        import json
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(**config)
        
    def predict(
        self,
        features: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """预测
        
        Args:
            features: 输入特征
            threshold: 阈值（用于多标签任务）
            
        Returns:
            预测结果
        """
        logits = self.forward(features)
        
        if hasattr(self, "num_classes") and self.num_classes > 1:
            # 分类任务
            return torch.argmax(logits, dim=-1)
        elif hasattr(self, "num_labels"):
            # 多标签任务
            return (torch.sigmoid(logits) > threshold).float()
        else:
            # 回归任务
            return logits 