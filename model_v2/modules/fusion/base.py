import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseFusionModule(nn.Module, ABC):
    """特征融合模块基类"""
    
    def __init__(self, **kwargs):
        """初始化特征融合模块
        
        Args:
            **kwargs: 配置参数
        """
        super().__init__()
        self.config = kwargs
        
    @abstractmethod
    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            features: 多模态特征字典，包含以下键：
                - text: 文本特征，形状为 [batch_size, seq_len, text_dim]
                - image: 图像特征，形状为 [batch_size, image_dim]
                - layout: 布局特征，形状为 [batch_size, num_boxes, layout_dim]
                
        Returns:
            融合后的特征，形状为 [batch_size, output_dim]
        """
        pass
        
    def get_output_dim(self) -> int:
        """获取输出维度
        
        Returns:
            输出维度
        """
        return self.config.get("output_dim")
        
    def get_input_dims(self) -> Dict[str, int]:
        """获取输入维度
        
        Returns:
            输入维度字典
        """
        return self.config.get("input_dims", {})
        
    def validate_inputs(self, features: Dict[str, torch.Tensor]) -> None:
        """验证输入特征
        
        Args:
            features: 多模态特征字典
            
        Raises:
            ValueError: 输入特征无效
        """
        # 检查必需的特征
        required_features = {"text", "image", "layout"}
        missing_features = required_features - set(features.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        # 检查特征维度
        input_dims = self.get_input_dims()
        for name, feature in features.items():
            if name not in input_dims:
                continue
            expected_dim = input_dims[name]
            if feature.size(-1) != expected_dim:
                raise ValueError(
                    f"Invalid dimension for {name} feature: "
                    f"expected {expected_dim}, got {feature.size(-1)}"
                )
                
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
    def from_config(cls, path: str) -> "BaseFusionModule":
        """从配置加载
        
        Args:
            path: 配置路径
            
        Returns:
            特征融合模块
        """
        import json
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(**config) 