# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Tuple, Optional, List, Any, Union

logger = logging.getLogger(__name__)

class BaseReconstructor(nn.Module):
    """基础重构器"""
    
    def __init__(self, input_dims: Dict[str, int], output_dim: int):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        
    def can_reconstruct(self, available_modalities: Dict[str, torch.Tensor]) -> bool:
        """检查是否可以使用可用模态进行重构"""
        return all(modality in available_modalities for modality in self.input_dims.keys())

class VisualReconstructor(BaseReconstructor):
    """视觉特征重构器"""
    
    def __init__(self, text_dim: int, layout_dim: int, visual_dim: int):
        super().__init__(
            input_dims={'text': text_dim, 'layout': layout_dim},
            output_dim=visual_dim
        )
        
        # 文本到视觉的转换
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, visual_dim),
            nn.LayerNorm(visual_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 版面到视觉的转换
        self.layout_encoder = nn.Sequential(
            nn.Linear(layout_dim, visual_dim),
            nn.LayerNorm(visual_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim * 2, visual_dim),
            nn.LayerNorm(visual_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(visual_dim, visual_dim)
        )
        
        # 对抗训练组件
        self.discriminator = nn.Sequential(
            nn.Linear(visual_dim, visual_dim // 2),
            nn.ReLU(),
            nn.Linear(visual_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, text_features: torch.Tensor, layout_features: torch.Tensor) -> torch.Tensor:
        # 编码各个模态
        text_visual = self.text_encoder(text_features)
        layout_visual = self.layout_encoder(layout_features)
        
        # 特征融合
        combined = torch.cat([text_visual, layout_visual], dim=-1)
        reconstructed = self.fusion(combined)
        
        return reconstructed
    
    def discriminate(self, features: torch.Tensor) -> torch.Tensor:
        """判别器前向传播"""
        return self.discriminator(features)

class TextReconstructor(BaseReconstructor):
    """文本特征重构器"""
    
    def __init__(self, visual_dim: int, layout_dim: int, text_dim: int):
        super().__init__(
            input_dims={'visual': visual_dim, 'layout': layout_dim},
            output_dim=text_dim
        )
        
        # 视觉到文本的转换
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 版面到文本的转换
        self.layout_encoder = nn.Sequential(
            nn.Linear(layout_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(text_dim * 2, text_dim),
            nn.LayerNorm(text_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(text_dim, text_dim)
        )
        
        # 对抗训练组件
        self.discriminator = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.ReLU(),
            nn.Linear(text_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, visual_features: torch.Tensor, layout_features: torch.Tensor) -> torch.Tensor:
        # 编码各个模态
        visual_text = self.visual_encoder(visual_features)
        layout_text = self.layout_encoder(layout_features)
        
        # 特征融合
        combined = torch.cat([visual_text, layout_text], dim=-1)
        reconstructed = self.fusion(combined)
        
        return reconstructed
    
    def discriminate(self, features: torch.Tensor) -> torch.Tensor:
        """判别器前向传播"""
        return self.discriminator(features)

class LayoutReconstructor(BaseReconstructor):
    """版面特征重构器"""
    
    def __init__(self, visual_dim: int, text_dim: int, layout_dim: int):
        super().__init__(
            input_dims={'visual': visual_dim, 'text': text_dim},
            output_dim=layout_dim
        )
        
        # 视觉到版面的转换
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, layout_dim),
            nn.LayerNorm(layout_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 文本到版面的转换
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, layout_dim),
            nn.LayerNorm(layout_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(layout_dim * 2, layout_dim),
            nn.LayerNorm(layout_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(layout_dim, layout_dim)
        )
        
        # 对抗训练组件
        self.discriminator = nn.Sequential(
            nn.Linear(layout_dim, layout_dim // 2),
            nn.ReLU(),
            nn.Linear(layout_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        # 编码各个模态
        visual_layout = self.visual_encoder(visual_features)
        text_layout = self.text_encoder(text_features)
        
        # 特征融合
        combined = torch.cat([visual_layout, text_layout], dim=-1)
        reconstructed = self.fusion(combined)
        
        return reconstructed
    
    def discriminate(self, features: torch.Tensor) -> torch.Tensor:
        """判别器前向传播"""
        return self.discriminator(features)

class ReconstructionEvaluator:
    """重构质量评估器"""
    
    def __init__(self):
        self.metrics = {
            'mse': nn.MSELoss(),
            'cosine': nn.CosineSimilarity(dim=-1),
            'l1': nn.L1Loss()
        }
    
    def evaluate(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        """
        评估重构质量
        
        Args:
            original: 原始特征
            reconstructed: 重构特征
            
        Returns:
            包含各项指标的字典
        """
        results = {}
        for name, metric in self.metrics.items():
            if name == 'cosine':
                results[name] = 1 - metric(original, reconstructed).mean().item()
            else:
                results[name] = metric(original, reconstructed).item()
        return results

class AdaptiveReconstructor:
    """自适应重构器"""
    
    def __init__(self, reconstructors: Dict[str, BaseReconstructor], evaluator: ReconstructionEvaluator):
        self.reconstructors = reconstructors
        self.evaluator = evaluator
        
    def reconstruct(self, available_modalities: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, str]:
        """
        根据可用模态选择最佳重构路径
        
        Args:
            available_modalities: 可用模态的特征字典
            
        Returns:
            重构特征和使用的重构器名称
        """
        best_reconstruction = None
        best_score = float('inf')
        best_reconstructor = None
        
        for name, reconstructor in self.reconstructors.items():
            if reconstructor.can_reconstruct(available_modalities):
                # 执行重构
                reconstruction = reconstructor(**available_modalities)
                
                # 评估重构质量
                score = self.evaluator.evaluate(
                    reconstruction,
                    self._get_ground_truth(available_modalities)
                )['mse']
                
                if score < best_score:
                    best_score = score
                    best_reconstruction = reconstruction
                    best_reconstructor = name
        
        return best_reconstruction, best_reconstructor
    
    def _get_ground_truth(self, available_modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """获取地面真值（用于评估）"""
        # 这里需要根据实际情况实现
        raise NotImplementedError

class ReconstructionTrainer:
    """重构训练器"""
    
    def __init__(self, reconstructor: BaseReconstructor, lr: float = 1e-4):
        self.reconstructor = reconstructor
        self.optimizer = torch.optim.Adam(reconstructor.parameters(), lr=lr)
        
    def train_step(self, available_modalities: Dict[str, torch.Tensor], target: torch.Tensor):
        """
        训练步骤
        
        Args:
            available_modalities: 可用模态的特征字典
            target: 目标特征
        """
        # 重构
        reconstructed = self.reconstructor(**available_modalities)
        
        # 重构损失
        reconstruction_loss = F.mse_loss(reconstructed, target)
        
        # 对抗损失
        real_pred = self.reconstructor.discriminate(target)
        fake_pred = self.reconstructor.discriminate(reconstructed.detach())
        
        d_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred)) + \
                F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
        
        g_loss = F.binary_cross_entropy(
            self.reconstructor.discriminate(reconstructed),
            torch.ones_like(fake_pred)
        )
        
        # 总损失
        total_loss = reconstruction_loss + 0.1 * (d_loss + g_loss)
        
        # 优化
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'reconstruction_loss': reconstruction_loss.item(),
            'discriminator_loss': d_loss.item(),
            'generator_loss': g_loss.item(),
            'total_loss': total_loss.item()
        }

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