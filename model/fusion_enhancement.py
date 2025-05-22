import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math

class GraphAttention(nn.Module):
    """图注意力层"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_heads = config.get('num_heads', 4)
        self.dropout = config.get('dropout', 0.1)
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        # 自注意力
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x, attn_mask=mask)
        x = residual + x
        
        # 前馈网络
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x

class RelationAwareAttention(nn.Module):
    """关系感知注意力层"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_heads = config.get('num_heads', 4)
        self.dropout = config.get('dropout', 0.1)
        
        # 关系编码器
        self.relation_encoder = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 注意力权重
        self.attention_weights = nn.Parameter(
            torch.randn(self.num_heads, self.hidden_dim, self.hidden_dim))
        
        # 输出投影
        self.output_proj = nn.Linear(self.hidden_dim * self.num_heads, self.hidden_dim)
    
    def forward(self, x: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len, _ = x.size()
        
        # 编码关系
        relation_features = self.relation_encoder(relations)
        
        # 计算注意力分数
        scores = torch.matmul(x, self.attention_weights)
        scores = scores.view(batch_size, seq_len, self.num_heads, -1)
        scores = scores.transpose(1, 2)
        
        # 应用关系感知
        relation_scores = torch.matmul(relation_features, scores.transpose(-2, -1))
        attention_weights = F.softmax(relation_scores, dim=-1)
        
        # 加权求和
        context = torch.matmul(attention_weights, x)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, -1)
        
        # 输出投影
        output = self.output_proj(context)
        
        return output

class CrossModalFusion(nn.Module):
    """跨模态融合层"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.hidden_dim = config.get('hidden_dim', 256)
        self.dropout = config.get('dropout', 0.1)
        
        # 模态特定投影
        self.vision_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.text_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            dropout=self.dropout
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 投影
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)
        
        # 跨模态注意力
        vision_attended, _ = self.cross_attention(vision_proj, text_proj, text_proj)
        text_attended, _ = self.cross_attention(text_proj, vision_proj, vision_proj)
        
        # 特征融合
        fused = torch.cat([vision_attended, text_attended], dim=-1)
        output = self.fusion_layer(fused)
        
        return output

class EnhancedFusionModule(nn.Module):
    """增强型融合模块"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # 图注意力层
        self.graph_attention = GraphAttention(config)
        
        # 关系感知注意力层
        self.relation_attention = RelationAwareAttention(config)
        
        # 跨模态融合层
        self.cross_modal_fusion = CrossModalFusion(config)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def forward(self, 
                vision_features: torch.Tensor,
                text_features: torch.Tensor,
                layout_features: torch.Tensor,
                relations: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        # 应用图注意力
        layout_enhanced = self.graph_attention(layout_features)
        
        # 应用关系感知注意力
        if relations is not None:
            layout_enhanced = self.relation_attention(layout_enhanced, relations)
        
        # 跨模态融合
        vision_text_fused = self.cross_modal_fusion(vision_features, text_features)
        
        # 与布局特征融合
        all_features = torch.cat([vision_text_fused, layout_enhanced], dim=1)
        output = self.output_layer(all_features)
        
        return output

class UncertaintyEstimation(nn.Module):
    """不确定性估计模块"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # 不确定性估计器
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 估计不确定性
        uncertainty = self.uncertainty_estimator(features)
        
        # 应用不确定性加权
        weighted_features = features * (1 - uncertainty)
        
        return weighted_features, uncertainty 