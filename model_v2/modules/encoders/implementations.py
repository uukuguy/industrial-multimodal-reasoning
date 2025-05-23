import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
import logging
from transformers import AutoModel, AutoTokenizer
from .base import BaseTextEncoder, BaseImageEncoder, BaseLayoutEncoder

logger = logging.getLogger(__name__)

class BertTextEncoder(BaseTextEncoder):
    """BERT文本编码器"""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        **kwargs
    ):
        """初始化BERT文本编码器
        
        Args:
            model_name: 模型名称
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return outputs.last_hidden_state

class ResNetImageEncoder(BaseImageEncoder):
    """ResNet图像编码器"""
    
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        **kwargs
    ):
        """初始化ResNet图像编码器
        
        Args:
            model_name: 模型名称
            pretrained: 是否使用预训练模型
        """
        super().__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            model_name,
            pretrained=pretrained
        )
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            images: 图像张量
            
        Returns:
            图像特征
        """
        features = self.model(images)
        return features.squeeze(-1).squeeze(-1)

class LayoutEncoder(BaseLayoutEncoder):
    """布局编码器"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        """初始化布局编码器
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏维度
            num_layers: 层数
            dropout: Dropout比率
        """
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
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
        outputs, _ = self.encoder(layout_features)
        if layout_mask is not None:
            outputs = outputs * layout_mask.unsqueeze(-1)
        return outputs 