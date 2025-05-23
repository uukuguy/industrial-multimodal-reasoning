import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
import logging
from dataclasses import dataclass

from ...modules.encoders import (
    BaseTextEncoder,
    BaseImageEncoder,
    BaseLayoutEncoder
)
from ...modules.fusion import BaseFusionModule
from ...modules.heads import BaseHead
from ...config.model_config import ModelConfig

logger = logging.getLogger(__name__)

@dataclass
class ModelOutput:
    """模型输出数据类"""
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    metrics: Optional[Dict[str, float]] = None
    uncertainty: Optional[torch.Tensor] = None
    hidden_states: Optional[Dict[str, torch.Tensor]] = None

class BaseModel(nn.Module):
    """基础模型类
    
    定义了模型的核心功能：
    1. 多模态特征编码
    2. 特征融合
    3. 损失计算
    4. 指标计算
    """
    
    def __init__(
        self,
        config: Union[Dict[str, Any], ModelConfig],
        text_encoder: Optional[BaseTextEncoder] = None,
        image_encoder: Optional[BaseImageEncoder] = None,
        layout_encoder: Optional[BaseLayoutEncoder] = None,
        fusion_module: Optional[BaseFusionModule] = None,
        head: Optional[BaseHead] = None
    ):
        """初始化模型
        
        Args:
            config: 模型配置
            text_encoder: 文本编码器
            image_encoder: 图像编码器
            layout_encoder: 布局编码器
            fusion_module: 特征融合模块
            head: 输出头
        """
        super().__init__()
        
        # 设置配置
        if isinstance(config, dict):
            self.config = ModelConfig(**config)
        else:
            self.config = config
            
        # 设置组件
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.layout_encoder = layout_encoder
        self.fusion_module = fusion_module
        self.head = head
        
        # 验证配置
        self._validate_config()
            
    def _validate_config(self):
        """验证配置"""
        if self.config is None:
            logger.warning("No config provided")
            return
            
        # 验证编码器配置
        if self.text_encoder and not self.config.text_encoder:
            raise ValueError("Text encoder provided but no config")
        if self.image_encoder and not self.config.image_encoder:
            raise ValueError("Image encoder provided but no config")
        if self.layout_encoder and not self.config.layout_encoder:
            raise ValueError("Layout encoder provided but no config")
            
        # 验证融合模块配置
        if self.fusion_module and not self.config.fusion:
            raise ValueError("Fusion module provided but no config")
            
        # 验证输出头配置
        if self.head and not self.config.head:
            raise ValueError("Head provided but no config")
        
    def encode(
        self,
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
        image_inputs: Optional[Dict[str, torch.Tensor]] = None,
        layout_inputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """编码多模态输入
        
        Args:
            text_inputs: 文本输入
            image_inputs: 图像输入
            layout_inputs: 布局输入
            
        Returns:
            编码后的特征
        """
        features = {}
        
        # 编码文本
        if self.text_encoder and text_inputs:
            features["text"] = self.text_encoder(**text_inputs)
            
        # 编码图像
        if self.image_encoder and image_inputs:
            features["image"] = self.image_encoder(**image_inputs)
            
        # 编码布局
        if self.layout_encoder and layout_inputs:
            features["layout"] = self.layout_encoder(**layout_inputs)
            
        return features
        
    def fuse(
        self,
        features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """融合多模态特征
        
        Args:
            features: 多模态特征
            
        Returns:
            融合后的特征
        """
        if not self.fusion_module:
            raise ValueError("No fusion module")
            
        return self.fusion_module(features)
        
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
        if not self.head:
            raise ValueError("No head")
            
        return self.head.compute_loss(logits, labels)
        
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
        if not self.head:
            raise ValueError("No head")
            
        return self.head.compute_metrics(logits, labels)
        
    def forward(
        self,
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
        image_inputs: Optional[Dict[str, torch.Tensor]] = None,
        layout_inputs: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> ModelOutput:
        """前向传播
        
        Args:
            text_inputs: 文本输入
            image_inputs: 图像输入
            layout_inputs: 布局输入
            labels: 标签
            return_hidden_states: 是否返回隐藏状态
            
        Returns:
            模型输出
        """
        # 编码多模态输入
        features = self.encode(text_inputs, image_inputs, layout_inputs)
        
        # 融合特征
        hidden_states = self.fuse(features)
        
        # 计算输出
        logits = self.head(hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)
            
        # 计算指标
        metrics = None
        if labels is not None:
            metrics = self.compute_metrics(logits, labels)
            
        return ModelOutput(
            logits=logits,
            loss=loss,
            metrics=metrics,
            hidden_states=hidden_states if return_hidden_states else None
        ) 