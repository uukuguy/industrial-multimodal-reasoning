# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

from .cross_modal_attention import CrossModalAttention

logger = logging.getLogger(__name__)

class MultimodalFusion(nn.Module):
    """
    基础多模态融合模块，将文本、图像和版面信息融合为统一的表示。
    """
    def __init__(self, embedding_dim: int, fused_embedding_dim: int = 512, dropout: float = 0.1):
        """
        初始化多模态融合模块。

        Args:
            embedding_dim: 输入嵌入的维度。
            fused_embedding_dim: 输出融合嵌入的维度。
            dropout: Dropout比率。
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fused_embedding_dim = fused_embedding_dim
        
        # 投影层，将不同模态的嵌入投影到相同的语义空间
        self.text_projection = nn.Linear(embedding_dim, embedding_dim)
        self.image_projection = nn.Linear(embedding_dim, embedding_dim)
        self.layout_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # 注意力融合层
        self.attention_pooling = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, fused_embedding_dim * 2),
            nn.LayerNorm(fused_embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_embedding_dim * 2, fused_embedding_dim),
            nn.LayerNorm(fused_embedding_dim)
        )
        
    def forward(self, 
               text_embeddings: Union[torch.Tensor, List[torch.Tensor]], 
               visual_embeddings: Optional[torch.Tensor] = None, 
               layout_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播，融合多模态嵌入。

        Args:
            text_embeddings: 文本嵌入，可以是单个张量或张量列表。
            visual_embeddings: 图像嵌入，可选。
            layout_embeddings: 版面嵌入，可选。

        Returns:
            融合后的嵌入向量。
        """
        # 处理文本嵌入
        if isinstance(text_embeddings, list) and text_embeddings:
            if len(text_embeddings) == 1:
                text_embed = text_embeddings[0]
            else:
                # 如果有多个文本嵌入，使用注意力池化
                stacked_text = torch.cat(text_embeddings, dim=0)  # [num_texts, embedding_dim]
                attention_weights = self.attention_pooling(stacked_text)  # [num_texts, 1]
                text_embed = (stacked_text * attention_weights).sum(dim=0, keepdim=True)  # [1, embedding_dim]
        elif isinstance(text_embeddings, torch.Tensor):
            text_embed = text_embeddings
        else:
            # 如果没有文本嵌入，创建一个零嵌入
            batch_size = 1
            if visual_embeddings is not None:
                batch_size = visual_embeddings.size(0)
            elif layout_embeddings is not None:
                batch_size = layout_embeddings.size(0)
            
            text_embed = torch.zeros(batch_size, self.embedding_dim, device=self._get_device())
        
        # 投影文本嵌入
        text_embed = self.text_projection(text_embed)  # [batch_size, embedding_dim]
        
        # 处理图像嵌入
        if visual_embeddings is not None:
            visual_embed = self.image_projection(visual_embeddings)  # [batch_size, embedding_dim]
        else:
            visual_embed = torch.zeros_like(text_embed)  # [batch_size, embedding_dim]
        
        # 处理版面嵌入
        if layout_embeddings is not None:
            layout_embed = self.layout_projection(layout_embeddings)  # [batch_size, embedding_dim]
        else:
            layout_embed = torch.zeros_like(text_embed)  # [batch_size, embedding_dim]
        
        # 融合所有模态
        # 拼接所有嵌入
        combined_embed = torch.cat([text_embed, visual_embed, layout_embed], dim=1)  # [batch_size, 3*embedding_dim]
        
        # 通过融合层
        fused_embed = self.fusion_layer(combined_embed)  # [batch_size, fused_embedding_dim]
        
        return fused_embed
    
    def _get_device(self):
        """获取当前设备"""
        return next(self.parameters()).device


class HierarchicalMultimodalFusion(nn.Module):
    """
    层次化多模态融合模块，实现低级、中级和高级特征的逐步融合
    """
    def __init__(self, embedding_dim: int, hidden_dims: Optional[List[int]] = None, 
                dropout: float = 0.1, use_attention: bool = True):
        super().__init__()
        # 设置合理的融合层次结构
        if hidden_dims is None:
            self.hidden_dims = [embedding_dim, embedding_dim//2, embedding_dim//4]
        else:
            self.hidden_dims = hidden_dims
            
        self.embedding_dim = embedding_dim
        self.use_attention = use_attention
        
        # 跨模态注意力模块（如果启用）
        if use_attention:
            self.text_vision_attention = CrossModalAttention(embedding_dim)
            self.vision_text_attention = CrossModalAttention(embedding_dim)
            self.layout_content_attention = CrossModalAttention(embedding_dim)
        
        # 低级特征融合层（基本对齐）
        self.low_level_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, self.hidden_dims[0]),
            nn.LayerNorm(self.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 中级特征融合层（语义和空间关系）
        self.mid_level_fusion = nn.Sequential(
            nn.Linear(self.hidden_dims[0] * 2, self.hidden_dims[1]),
            nn.LayerNorm(self.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 高级特征融合层（统一表示）
        self.high_level_fusion = nn.Sequential(
            nn.Linear(self.hidden_dims[1] * 2, self.hidden_dims[2]),
            nn.LayerNorm(self.hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(self.hidden_dims[2], embedding_dim)
        
        # 模态特定处理
        self.text_projection = nn.Linear(embedding_dim, embedding_dim)
        self.vision_projection = nn.Linear(embedding_dim, embedding_dim)
        self.layout_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # 模态缺失指示器嵌入
        self.modality_indicators = nn.Parameter(torch.randn(3, embedding_dim))  # [text, vision, layout]
        
    def _prepare_text_embedding(self, text_embeddings: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """准备文本嵌入"""
        if isinstance(text_embeddings, list) and text_embeddings:
            if len(text_embeddings) == 1:
                text_embed = text_embeddings[0]
            else:
                # 如果有多个文本嵌入，计算平均值
                text_embed = torch.stack(text_embeddings).mean(dim=0, keepdim=True)
        elif isinstance(text_embeddings, torch.Tensor):
            text_embed = text_embeddings
        else:
            # 如果没有文本嵌入，使用指示器嵌入
            device = self.modality_indicators.device
            batch_size = 1
            text_embed = self.modality_indicators[0].unsqueeze(0).expand(batch_size, -1)
            
        return self.text_projection(text_embed)
        
    def forward(self, 
               text_embeddings: Union[torch.Tensor, List[torch.Tensor], None], 
               visual_embeddings: Optional[torch.Tensor] = None, 
               layout_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播，融合多模态嵌入。

        Args:
            text_embeddings: 文本嵌入，可以是单个张量或张量列表。
            visual_embeddings: 图像嵌入，可选。
            layout_embeddings: 版面嵌入，可选。

        Returns:
            包含各级融合结果的字典。
        """
        results = {}
        
        # 记录模态存在情况
        modality_presence = {
            'text': text_embeddings is not None,
            'vision': visual_embeddings is not None,
            'layout': layout_embeddings is not None
        }
        results['modality_presence'] = modality_presence
        
        # 准备每个模态的嵌入
        device = self._get_device()
        batch_size = 1
        
        # 确定批次大小
        if visual_embeddings is not None:
            batch_size = visual_embeddings.size(0)
        elif isinstance(text_embeddings, torch.Tensor):
            batch_size = text_embeddings.size(0)
        elif layout_embeddings is not None:
            batch_size = layout_embeddings.size(0)
        
        # 准备文本嵌入
        if text_embeddings is not None:
            text_embed = self._prepare_text_embedding(text_embeddings)
        else:
            text_embed = self.modality_indicators[0].unsqueeze(0).expand(batch_size, -1).to(device)
        
        # 准备视觉嵌入
        if visual_embeddings is not None:
            vision_embed = self.vision_projection(visual_embeddings)
        else:
            vision_embed = self.modality_indicators[1].unsqueeze(0).expand(batch_size, -1).to(device)
        
        # 准备版面嵌入
        if layout_embeddings is not None:
            layout_embed = self.layout_projection(layout_embeddings)
        else:
            layout_embed = self.modality_indicators[2].unsqueeze(0).expand(batch_size, -1).to(device)
        
        # 应用跨模态注意力（如果启用）
        if self.use_attention:
            if modality_presence['text'] and modality_presence['vision']:
                text_embed_att, _ = self.vision_text_attention(text_embed, vision_embed)
                vision_embed_att, _ = self.text_vision_attention(vision_embed, text_embed)
                text_embed = text_embed_att
                vision_embed = vision_embed_att
                
            if modality_presence['layout'] and (modality_presence['text'] or modality_presence['vision']):
                # 创建内容嵌入（文本+视觉）
                if modality_presence['text'] and modality_presence['vision']:
                    content_embed = torch.cat([text_embed, vision_embed], dim=0)
                elif modality_presence['text']:
                    content_embed = text_embed
                else:
                    content_embed = vision_embed
                
                layout_embed_att, _ = self.layout_content_attention(layout_embed, content_embed)
                layout_embed = layout_embed_att
        
        # 低级融合：文本和视觉
        low_fusion = self.low_level_fusion(torch.cat([text_embed, vision_embed], dim=1))
        results['low_fusion'] = low_fusion
        
        # 中级融合：低级融合结果和布局信息
        mid_fusion = self.mid_level_fusion(torch.cat([low_fusion, layout_embed], dim=1))
        results['mid_fusion'] = mid_fusion
        
        # 高级融合：中级融合结果和低级融合结果（残差连接）
        high_fusion = self.high_level_fusion(torch.cat([mid_fusion, low_fusion], dim=1))
        results['high_fusion'] = high_fusion
        
        # 输出投影
        output = self.output_projection(high_fusion)
        results['fused_embedding'] = output
        
        return results
    
    def _get_device(self):
        """获取当前设备"""
        return next(self.parameters()).device


if __name__ == "__main__":
    # 测试多模态融合模块
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    embedding_dim = 768
    batch_size = 2
    
    text_embeddings = [torch.randn(1, embedding_dim) for _ in range(3)]  # 3个文本嵌入
    visual_embedding = torch.randn(batch_size, embedding_dim)  # 视觉嵌入
    layout_embedding = torch.randn(batch_size, embedding_dim)  # 版面嵌入
    
    # 测试基础融合模块
    print("测试基础多模态融合模块...")
    fusion = MultimodalFusion(embedding_dim=embedding_dim, fused_embedding_dim=512)
    fused_embed = fusion(text_embeddings, visual_embedding, layout_embedding)
    print(f"融合嵌入形状: {fused_embed.shape}")
    
    # 测试层次化融合模块
    print("\n测试层次化多模态融合模块...")
    hierarchical_fusion = HierarchicalMultimodalFusion(embedding_dim=embedding_dim)
    results = hierarchical_fusion(text_embeddings, visual_embedding, layout_embedding)
    
    print(f"低级融合形状: {results['low_fusion'].shape}")
    print(f"中级融合形状: {results['mid_fusion'].shape}")
    print(f"高级融合形状: {results['high_fusion'].shape}")
    print(f"最终融合嵌入形状: {results['fused_embedding'].shape}")
    
    # 测试模态缺失情况
    print("\n测试模态缺失情况...")
    results_missing = hierarchical_fusion(text_embeddings, None, layout_embedding)
    print(f"缺失视觉模态的融合嵌入形状: {results_missing['fused_embedding'].shape}")