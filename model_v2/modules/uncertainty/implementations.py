import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union
import logging
from .base import BaseUncertaintyModule

logger = logging.getLogger(__name__)

class MCDropoutModule(BaseUncertaintyModule):
    """基于Monte Carlo Dropout的不确定性估计模块"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """初始化
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            dropout_rate: Dropout比率
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        # 构建MLP
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
    def forward(
        self,
        features: torch.Tensor,
        num_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            features: 输入特征，形状为 [batch_size, input_dim]
            num_samples: 采样数量
            
        Returns:
            包含以下键的字典：
            - mean: 预测均值
            - variance: 预测方差
            - samples: 采样结果
        """
        # 启用dropout
        self.train()
        
        # 生成多个样本
        samples = []
        for _ in range(num_samples):
            sample = self.mlp(features)
            samples.append(sample)
            
        # 堆叠样本
        samples = torch.stack(samples)  # [num_samples, batch_size, output_dim]
        
        # 计算均值和方差
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0)
        
        return {
            "mean": mean,
            "variance": variance,
            "samples": samples
        }
        
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
        mean = outputs["mean"]
        variance = outputs["variance"]
        
        # 计算MSE损失
        mse_loss = F.mse_loss(mean, targets)
        
        # 计算不确定性损失
        uncertainty_loss = torch.mean(torch.abs(variance))
        
        return mse_loss + 0.1 * uncertainty_loss
        
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
        mean = outputs["mean"]
        variance = outputs["variance"]
        
        # 计算MSE
        mse = F.mse_loss(mean, targets).item()
        
        # 计算MAE
        mae = F.l1_loss(mean, targets).item()
        
        # 计算R2分数
        ss_tot = torch.sum((targets - targets.mean()) ** 2)
        ss_res = torch.sum((targets - mean) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        # 计算不确定性校准
        error = torch.abs(mean - targets)
        uncertainty_calibration = torch.corrcoef(
            torch.stack([error.flatten(), variance.flatten()])
        )[0, 1].item()
        
        return {
            "mse": mse,
            "mae": mae,
            "r2": r2.item(),
            "uncertainty_calibration": uncertainty_calibration
        }

class DeepEnsembleModule(BaseUncertaintyModule):
    """基于深度集成的不确定性估计模块"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        num_models: int = 5,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """初始化
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            num_models: 集成模型数量
            dropout_rate: Dropout比率
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.num_models = num_models
        
        # 创建多个模型
        self.models = nn.ModuleList([
            self._create_model(input_dim, hidden_dims, output_dim, dropout_rate)
            for _ in range(num_models)
        ])
        
    def _create_model(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float
    ) -> nn.Module:
        """创建单个模型
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            dropout_rate: Dropout比率
            
        Returns:
            模型
        """
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
        
    def forward(
        self,
        features: torch.Tensor,
        num_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            features: 输入特征，形状为 [batch_size, input_dim]
            num_samples: 采样数量（对于集成模型，此参数被忽略）
            
        Returns:
            包含以下键的字典：
            - mean: 预测均值
            - variance: 预测方差
            - samples: 采样结果
        """
        # 获取每个模型的预测
        predictions = []
        for model in self.models:
            pred = model(features)
            predictions.append(pred)
            
        # 堆叠预测
        predictions = torch.stack(predictions)  # [num_models, batch_size, output_dim]
        
        # 计算均值和方差
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        return {
            "mean": mean,
            "variance": variance,
            "samples": predictions
        }
        
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
        samples = outputs["samples"]
        
        # 计算每个模型的损失
        losses = []
        for i in range(self.num_models):
            loss = F.mse_loss(samples[i], targets)
            losses.append(loss)
            
        # 返回平均损失
        return torch.stack(losses).mean()
        
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
        mean = outputs["mean"]
        variance = outputs["variance"]
        
        # 计算MSE
        mse = F.mse_loss(mean, targets).item()
        
        # 计算MAE
        mae = F.l1_loss(mean, targets).item()
        
        # 计算R2分数
        ss_tot = torch.sum((targets - targets.mean()) ** 2)
        ss_res = torch.sum((targets - mean) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        # 计算不确定性校准
        error = torch.abs(mean - targets)
        uncertainty_calibration = torch.corrcoef(
            torch.stack([error.flatten(), variance.flatten()])
        )[0, 1].item()
        
        # 计算集成多样性
        samples = outputs["samples"]
        diversity = torch.mean(torch.var(samples, dim=0))
        
        return {
            "mse": mse,
            "mae": mae,
            "r2": r2.item(),
            "uncertainty_calibration": uncertainty_calibration,
            "ensemble_diversity": diversity.item()
        } 