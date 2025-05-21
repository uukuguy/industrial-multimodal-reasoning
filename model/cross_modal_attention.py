# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块，用于增强模态间的信息交流
    """
    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        # 自适应门控机制
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        
        # 输出层标准化
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, query_modal: torch.Tensor, key_modal: torch.Tensor, 
                value_modal: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query_modal: 查询模态的嵌入 [batch_size, seq_len_q, embedding_dim]
            key_modal: 键模态的嵌入 [batch_size, seq_len_k, embedding_dim]
            value_modal: 值模态的嵌入，默认等于key_modal
            
        Returns:
            output: 注意力输出 [batch_size, seq_len_q, embedding_dim]
            attn_weights: 注意力权重
        """
        if value_modal is None:
            value_modal = key_modal
            
        # 调整维度顺序以适应多头注意力机制
        query_modal_t = query_modal.transpose(0, 1)  # [seq_len_q, batch_size, embedding_dim]
        key_modal_t = key_modal.transpose(0, 1)      # [seq_len_k, batch_size, embedding_dim]
        value_modal_t = value_modal.transpose(0, 1)  # [seq_len_k, batch_size, embedding_dim]
        
        # 计算跨模态注意力
        attn_output, attn_weights = self.attention(
            query=query_modal_t,
            key=key_modal_t,
            value=value_modal_t
        )
        
        # 转回原始维度顺序
        attn_output = attn_output.transpose(0, 1)  # [batch_size, seq_len_q, embedding_dim]
        
        # 计算自适应门控
        gate_input = torch.cat([query_modal, attn_output], dim=-1)
        gate_value = self.gate(gate_input)
        
        # 应用门控机制
        output = gate_value * attn_output + (1 - gate_value) * query_modal
        
        # 应用层标准化
        output = self.layer_norm(output)
        
        return output, attn_weights

class MultiModalAttentionHub(nn.Module):
    """
    多模态注意力中枢，集成多个跨模态注意力模块
    """
    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 创建三个跨模态注意力模块，用于不同模态对之间的交互
        # 文本->视觉注意力
        self.text_to_vision = CrossModalAttention(embedding_dim, num_heads, dropout)
        
        # 视觉->文本注意力
        self.vision_to_text = CrossModalAttention(embedding_dim, num_heads, dropout)
        
        # 文本/视觉->布局注意力
        self.content_to_layout = CrossModalAttention(embedding_dim, num_heads, dropout)
        
        # 布局->内容注意力
        self.layout_to_content = CrossModalAttention(embedding_dim, num_heads, dropout)
        
        # 输出融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, text_embed: torch.Tensor, vision_embed: torch.Tensor, 
                layout_embed: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            text_embed: 文本嵌入 [batch_size, seq_len_t, embedding_dim]
            vision_embed: 视觉嵌入 [batch_size, seq_len_v, embedding_dim]
            layout_embed: 布局嵌入 [batch_size, seq_len_l, embedding_dim]
            
        Returns:
            包含注意力结果和权重的字典
        """
        results = {}
        
        # 文本->视觉注意力
        enhanced_vision, t2v_weights = self.text_to_vision(vision_embed, text_embed)
        results['enhanced_vision'] = enhanced_vision
        results['text_to_vision_weights'] = t2v_weights
        
        # 视觉->文本注意力
        enhanced_text, v2t_weights = self.vision_to_text(text_embed, vision_embed)
        results['enhanced_text'] = enhanced_text
        results['vision_to_text_weights'] = v2t_weights
        
        # 如果有布局嵌入
        if layout_embed is not None:
            # 融合文本和视觉
            content_embed = torch.cat([enhanced_text, enhanced_vision], dim=1)
            
            # 内容->布局注意力
            enhanced_layout, c2l_weights = self.content_to_layout(layout_embed, content_embed)
            results['enhanced_layout'] = enhanced_layout
            results['content_to_layout_weights'] = c2l_weights
            
            # 布局->内容注意力
            enhanced_content, l2c_weights = self.layout_to_content(content_embed, layout_embed)
            results['enhanced_content'] = enhanced_content
            results['layout_to_content_weights'] = l2c_weights
            
            # 计算紧凑的全局表示
            # 获取每个模态的池化表示（取每个序列的平均值）
            pooled_text = enhanced_text.mean(dim=1)  # [batch_size, embedding_dim]
            pooled_vision = enhanced_vision.mean(dim=1)  # [batch_size, embedding_dim]
            pooled_layout = enhanced_layout.mean(dim=1)  # [batch_size, embedding_dim]
            
            # 拼接并融合
            concat_embed = torch.cat([pooled_text, pooled_vision, pooled_layout], dim=1)
            fused_embed = self.fusion_layer(concat_embed)
            
            results['fused_embedding'] = fused_embed
            
        else:
            # 如果没有布局嵌入，只融合文本和视觉
            pooled_text = enhanced_text.mean(dim=1)  # [batch_size, embedding_dim]
            pooled_vision = enhanced_vision.mean(dim=1)  # [batch_size, embedding_dim]
            
            # 创建一个零向量作为布局嵌入的占位符
            batch_size = pooled_text.size(0)
            pooled_layout = torch.zeros(batch_size, self.embedding_dim, device=pooled_text.device)
            
            # 拼接并融合
            concat_embed = torch.cat([pooled_text, pooled_vision, pooled_layout], dim=1)
            fused_embed = self.fusion_layer(concat_embed)
            
            results['fused_embedding'] = fused_embed
        
        return results


if __name__ == "__main__":
    # 测试跨模态注意力模块
    embedding_dim = 768
    batch_size = 2
    
    # 创建模拟数据
    text_embed = torch.randn(batch_size, 10, embedding_dim)  # 2个样本，10个文本token
    vision_embed = torch.randn(batch_size, 5, embedding_dim)  # 2个样本，5个视觉区域
    layout_embed = torch.randn(batch_size, 8, embedding_dim)  # 2个样本，8个布局元素
    
    # 测试单个注意力模块
    print("测试单个跨模态注意力模块...")
    attention = CrossModalAttention(embedding_dim)
    output, weights = attention(text_embed, vision_embed)
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")
    
    # 测试注意力中枢
    print("\n测试多模态注意力中枢...")
    hub = MultiModalAttentionHub(embedding_dim)
    results = hub(text_embed, vision_embed, layout_embed)
    
    print(f"增强后的文本形状: {results['enhanced_text'].shape}")
    print(f"增强后的视觉形状: {results['enhanced_vision'].shape}")
    print(f"增强后的布局形状: {results['enhanced_layout'].shape}")
    print(f"融合嵌入形状: {results['fused_embedding'].shape}")
    
    # 测试没有布局嵌入的情况
    print("\n测试没有布局嵌入的情况...")
    results_no_layout = hub(text_embed, vision_embed)
    print(f"融合嵌入形状 (无布局): {results_no_layout['fused_embedding'].shape}")