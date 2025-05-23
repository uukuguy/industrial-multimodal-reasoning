import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseQAModule(nn.Module, ABC):
    """问答模块基类"""
    
    def __init__(self, **kwargs):
        """初始化问答模块
        
        Args:
            **kwargs: 配置参数
        """
        super().__init__()
        self.config = kwargs
        
    @abstractmethod
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
            - start_logits: 答案开始位置的logits，形状为 [batch_size, context_len]
            - end_logits: 答案结束位置的logits，形状为 [batch_size, context_len]
            - span_logits: 答案跨度的logits，形状为 [batch_size, context_len, context_len]
        """
        pass
        
    @abstractmethod
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
            start_positions: 答案开始位置，形状为 [batch_size]
            end_positions: 答案结束位置，形状为 [batch_size]
            attention_mask: 注意力掩码，形状为 [batch_size, context_len]
            
        Returns:
            损失值
        """
        pass
        
    @abstractmethod
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
            start_positions: 答案开始位置，形状为 [batch_size]
            end_positions: 答案结束位置，形状为 [batch_size]
            attention_mask: 注意力掩码，形状为 [batch_size, context_len]
            
        Returns:
            指标字典
        """
        pass
        
    def predict(
        self,
        question_features: torch.Tensor,
        context_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_answer_length: int = 30
    ) -> Dict[str, torch.Tensor]:
        """预测答案
        
        Args:
            question_features: 问题特征
            context_features: 上下文特征
            attention_mask: 注意力掩码
            max_answer_length: 最大答案长度
            
        Returns:
            包含以下键的字典：
            - start_positions: 预测的答案开始位置
            - end_positions: 预测的答案结束位置
            - answer_scores: 答案的置信度分数
        """
        outputs = self.forward(question_features, context_features, attention_mask)
        
        # 获取开始和结束位置的预测
        start_logits = outputs["start_logits"]
        end_logits = outputs["end_logits"]
        
        # 应用注意力掩码
        if attention_mask is not None:
            start_logits = start_logits.masked_fill(~attention_mask.bool(), float("-inf"))
            end_logits = end_logits.masked_fill(~attention_mask.bool(), float("-inf"))
            
        # 获取最可能的开始和结束位置
        start_positions = torch.argmax(start_logits, dim=-1)
        end_positions = torch.argmax(end_logits, dim=-1)
        
        # 计算答案分数
        start_scores = torch.softmax(start_logits, dim=-1)
        end_scores = torch.softmax(end_logits, dim=-1)
        answer_scores = start_scores.gather(1, start_positions.unsqueeze(-1)) * \
                       end_scores.gather(1, end_positions.unsqueeze(-1))
        
        # 应用最大答案长度约束
        invalid_mask = (end_positions - start_positions) > max_answer_length
        if invalid_mask.any():
            # 对于超出长度限制的答案，重新选择结束位置
            for i in range(len(start_positions)):
                if invalid_mask[i]:
                    valid_end_positions = torch.arange(
                        start_positions[i],
                        min(start_positions[i] + max_answer_length, len(end_logits[i]))
                    )
                    end_scores_i = end_scores[i, valid_end_positions]
                    end_positions[i] = valid_end_positions[torch.argmax(end_scores_i)]
        
        return {
            "start_positions": start_positions,
            "end_positions": end_positions,
            "answer_scores": answer_scores.squeeze(-1)
        } 