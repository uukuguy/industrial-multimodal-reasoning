import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union
import logging
from .base import BaseQAModule

logger = logging.getLogger(__name__)

class SpanQAModule(BaseQAModule):
    """基于跨度的问答模块"""
    
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        **kwargs
    ):
        """初始化
        
        Args:
            hidden_dim: 隐藏层维度
            dropout: Dropout比率
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        
        # 跨度预测层
        self.span_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # 2表示开始和结束位置
        )
        
    def forward(
        self,
        question_features: torch.Tensor,
        context_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            question_features: 问题特征，形状为 [batch_size, question_len, hidden_dim]
            context_features: 上下文特征，形状为 [batch_size, context_len, hidden_dim]
            attention_mask: 注意力掩码，形状为 [batch_size, context_len]
            
        Returns:
            包含以下键的字典：
            - start_logits: 答案开始位置的logits
            - end_logits: 答案结束位置的logits
        """
        # 计算问题表示
        question_repr = torch.mean(question_features, dim=1)  # [batch_size, hidden_dim]
        
        # 将问题表示与上下文特征拼接
        question_repr = question_repr.unsqueeze(1).expand(-1, context_features.size(1), -1)
        concat_features = torch.cat([context_features, question_repr], dim=-1)
        
        # 预测开始和结束位置
        span_logits = self.span_predictor(concat_features)  # [batch_size, context_len, 2]
        start_logits, end_logits = span_logits.split(1, dim=-1)
        
        return {
            "start_logits": start_logits.squeeze(-1),
            "end_logits": end_logits.squeeze(-1)
        }
        
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            outputs: 模型输出
            start_positions: 答案开始位置
            end_positions: 答案结束位置
            attention_mask: 注意力掩码
            
        Returns:
            损失值
        """
        start_logits = outputs["start_logits"]
        end_logits = outputs["end_logits"]
        
        # 应用注意力掩码
        if attention_mask is not None:
            start_logits = start_logits.masked_fill(~attention_mask.bool(), float("-inf"))
            end_logits = end_logits.masked_fill(~attention_mask.bool(), float("-inf"))
            
        # 计算交叉熵损失
        start_loss = F.cross_entropy(start_logits, start_positions)
        end_loss = F.cross_entropy(end_logits, end_positions)
        
        return (start_loss + end_loss) / 2
        
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """计算评估指标
        
        Args:
            outputs: 模型输出
            start_positions: 答案开始位置
            end_positions: 答案结束位置
            attention_mask: 注意力掩码
            
        Returns:
            指标字典
        """
        predictions = self.predict(
            outputs["question_features"],
            outputs["context_features"],
            attention_mask
        )
        
        # 计算准确率
        start_accuracy = (predictions["start_positions"] == start_positions).float().mean()
        end_accuracy = (predictions["end_positions"] == end_positions).float().mean()
        
        # 计算完全匹配率
        exact_match = ((predictions["start_positions"] == start_positions) & 
                      (predictions["end_positions"] == end_positions)).float().mean()
        
        return {
            "start_accuracy": start_accuracy.item(),
            "end_accuracy": end_accuracy.item(),
            "exact_match": exact_match.item()
        } 