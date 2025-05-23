import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseTextEncoder(nn.Module, ABC):
    """文本编码器基类"""
    
    def __init__(self, **kwargs):
        """初始化文本编码器"""
        super().__init__()
        
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            token_type_ids: 标记类型ID
            
        Returns:
            文本特征
        """
        pass

class BaseImageEncoder(nn.Module, ABC):
    """图像编码器基类"""
    
    def __init__(self, **kwargs):
        """初始化图像编码器"""
        super().__init__()
        
    @abstractmethod
    def forward(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            images: 图像张量
            
        Returns:
            图像特征
        """
        pass

class BaseLayoutEncoder(nn.Module, ABC):
    """布局编码器基类"""
    
    def __init__(self, **kwargs):
        """初始化布局编码器"""
        super().__init__()
        
    @abstractmethod
    def forward(
        self,
        layout_features: torch.Tensor,
        layout_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            layout_features: 布局特征
            layout_mask: 布局掩码
            
        Returns:
            布局特征
        """
        pass

class BaseFusionModule(nn.Module, ABC):
    """特征融合模块基类"""
    
    def __init__(self, **kwargs):
        """初始化特征融合模块"""
        super().__init__()
        
    @abstractmethod
    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            features: 多模态特征
            
        Returns:
            融合后的特征
        """
        pass 