# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import logging
from typing import Dict, Tuple, Optional, List, Any, Union

logger = logging.getLogger(__name__)

class ModalityReconstructor(nn.Module):
    """
    模态重构模块，用于从可用模态重建缺失模态
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        """
        初始化模态重构模块

        Args:
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            dropout: Dropout率
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 视觉模态重构网络
        self.visual_reconstructor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # 文本模态重构网络
        self.text_reconstructor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # 布局模态重构网络
        self.layout_reconstructor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # 重构质量评估
        self.quality_estimator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def _ensure_tensor(self, embedding: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """确保输入是张量"""
        if isinstance(embedding, list):
            if not embedding:
                raise ValueError("空的嵌入列表")
            if len(embedding) == 1:
                return embedding[0]
            else:
                return torch.cat(embedding, dim=0).mean(dim=0, keepdim=True)
        return embedding
        
    def reconstruct_visual(self, text_embedding: Union[torch.Tensor, List[torch.Tensor]], 
                          layout_embedding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从文本和布局重构视觉特征

        Args:
            text_embedding: 文本嵌入
            layout_embedding: 布局嵌入（可选）

        Returns:
            重构的视觉特征和重构质量评分
        """
        text_embedding = self._ensure_tensor(text_embedding)
        
        if layout_embedding is None:
            # 如果没有布局嵌入，创建一个零张量
            layout_embedding = torch.zeros_like(text_embedding)
        
        combined = torch.cat([text_embedding, layout_embedding], dim=1)
        reconstructed = self.visual_reconstructor(combined)
        quality = self.quality_estimator(reconstructed)
        
        return reconstructed, quality
        
    def reconstruct_text(self, visual_embedding: torch.Tensor, 
                        layout_embedding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从视觉和布局重构文本特征

        Args:
            visual_embedding: 视觉嵌入
            layout_embedding: 布局嵌入（可选）

        Returns:
            重构的文本特征和重构质量评分
        """
        if layout_embedding is None:
            # 如果没有布局嵌入，创建一个零张量
            layout_embedding = torch.zeros_like(visual_embedding)
            
        combined = torch.cat([visual_embedding, layout_embedding], dim=1)
        reconstructed = self.text_reconstructor(combined)
        quality = self.quality_estimator(reconstructed)
        
        return reconstructed, quality
        
    def reconstruct_layout(self, visual_embedding: torch.Tensor, 
                          text_embedding: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从视觉和文本重构布局特征

        Args:
            visual_embedding: 视觉嵌入
            text_embedding: 文本嵌入

        Returns:
            重构的布局特征和重构质量评分
        """
        text_embedding = self._ensure_tensor(text_embedding)
        combined = torch.cat([visual_embedding, text_embedding], dim=1)
        reconstructed = self.layout_reconstructor(combined)
        quality = self.quality_estimator(reconstructed)
        
        return reconstructed, quality
    
    def forward(self, target_modality: int, 
               visual_embedding: Optional[torch.Tensor] = None,
               text_embedding: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
               layout_embedding: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播，根据目标模态调用相应的重构函数

        Args:
            target_modality: 目标模态 (0=视觉, 1=文本, 2=布局)
            visual_embedding: 视觉嵌入 (可选)
            text_embedding: 文本嵌入 (可选)
            layout_embedding: 布局嵌入 (可选)

        Returns:
            包含重构结果和质量评分的字典
        """
        results = {}
        
        if target_modality == 0:  # 重构视觉
            if text_embedding is None:
                raise ValueError("重构视觉模态需要文本嵌入")
                
            reconstructed, quality = self.reconstruct_visual(text_embedding, layout_embedding)
            results['reconstructed'] = reconstructed
            results['quality'] = quality
            results['modality'] = 'visual'
            
        elif target_modality == 1:  # 重构文本
            if visual_embedding is None:
                raise ValueError("重构文本模态需要视觉嵌入")
                
            reconstructed, quality = self.reconstruct_text(visual_embedding, layout_embedding)
            results['reconstructed'] = reconstructed
            results['quality'] = quality
            results['modality'] = 'text'
            
        elif target_modality == 2:  # 重构布局
            if visual_embedding is None or text_embedding is None:
                raise ValueError("重构布局模态需要视觉和文本嵌入")
                
            reconstructed, quality = self.reconstruct_layout(visual_embedding, text_embedding)
            results['reconstructed'] = reconstructed
            results['quality'] = quality
            results['modality'] = 'layout'
            
        else:
            raise ValueError(f"无效的目标模态: {target_modality}")
            
        return results
    

if __name__ == "__main__":
    # 测试模态重构模块
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    embedding_dim = 768
    batch_size = 2
    
    text_embedding = torch.randn(batch_size, embedding_dim)
    visual_embedding = torch.randn(batch_size, embedding_dim)
    layout_embedding = torch.randn(batch_size, embedding_dim)
    
    # 初始化重构器
    reconstructor = ModalityReconstructor(embedding_dim)
    
    # 测试视觉重构
    print("测试视觉重构...")
    reconstructed_visual, quality = reconstructor.reconstruct_visual(text_embedding, layout_embedding)
    print(f"重构视觉嵌入形状: {reconstructed_visual.shape}")
    print(f"重构质量: {quality.item():.4f}")
    
    # 测试文本重构
    print("\n测试文本重构...")
    reconstructed_text, quality = reconstructor.reconstruct_text(visual_embedding, layout_embedding)
    print(f"重构文本嵌入形状: {reconstructed_text.shape}")
    print(f"重构质量: {quality.item():.4f}")
    
    # 测试布局重构
    print("\n测试布局重构...")
    reconstructed_layout, quality = reconstructor.reconstruct_layout(visual_embedding, text_embedding)
    print(f"重构布局嵌入形状: {reconstructed_layout.shape}")
    print(f"重构质量: {quality.item():.4f}")
    
    # 测试前向传播
    print("\n测试前向传播...")
    results = reconstructor(0, None, text_embedding, layout_embedding)
    print(f"重构模态: {results['modality']}")
    print(f"重构形状: {results['reconstructed'].shape}")
    print(f"重构质量: {results['quality'].item():.4f}")