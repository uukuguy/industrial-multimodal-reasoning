"""多模态特征融合模块

该包包含多种特征融合策略：
- 简单融合 (SimpleFusion): 特征拼接
- 注意力融合 (AttentionFusion): 多头注意力机制
- 交叉注意力融合 (CrossAttentionFusion): 跨模态注意力机制
- 门控融合 (GatedFusion): 门控机制
- 双线性融合 (BilinearFusion): 双线性交互
- Tucker分解融合 (TuckerFusion): Tucker张量分解
- 层次融合 (HierarchicalFusion): 多层网络融合
"""

from .base import BaseFusionModule
from .implementations import (
    SimpleFusion,
    AttentionFusion,
    CrossAttentionFusion,
    GatedFusion,
    BilinearFusion,
    TuckerFusion,
    HierarchicalFusion
)

__all__ = [
    'BaseFusionModule',
    'SimpleFusion',
    'AttentionFusion',
    'CrossAttentionFusion',
    'GatedFusion',
    'BilinearFusion',
    'TuckerFusion',
    'HierarchicalFusion'
] 