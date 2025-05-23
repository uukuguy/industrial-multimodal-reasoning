import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
from .base import BaseFusionModule

logger = logging.getLogger(__name__)

class SimpleFusion(BaseFusionModule):
    """简单特征融合模块（拼接）
    
    适用场景：
    1. 输入模态特征维度相近，且模态间关系简单
    2. 需要保留所有原始特征信息
    3. 计算资源有限，需要轻量级融合方案
    4. 作为其他复杂融合策略的基准线
    
    优点：
    - 实现简单，计算效率高
    - 保留所有原始特征信息
    - 不需要额外的参数学习
    
    缺点：
    - 无法捕获模态间的复杂交互
    - 输出维度会随输入模态数量线性增长
    - 对特征尺度敏感
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int,
        dropout: float = 0.1,
        **kwargs
    ):
        """初始化简单特征融合模块
        
        Args:
            input_dims: 输入维度字典
            output_dim: 输出维度
            dropout: Dropout比率
        """
        super().__init__(input_dims=input_dims, output_dim=output_dim, dropout=dropout, **kwargs)
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in input_dims.items()
        })
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播
        
        Args:
            features: 多模态特征
            
        Returns:
            融合后的特征
        """
        # 投影到相同维度
        projected = {
            name: self.projections[name](feat)
            for name, feat in features.items()
        }
        
        # 拼接特征
        fused = torch.cat(list(projected.values()), dim=-1)
        
        # 应用Dropout
        fused = self.dropout(fused)
        
        return fused

class AttentionFusion(BaseFusionModule):
    """注意力特征融合模块
    
    适用场景：
    1. 需要捕获全局依赖关系的场景
    2. 不同模态特征的重要性需要动态调整
    3. 输入序列长度可变
    4. 需要并行处理多个模态
    
    优点：
    - 可以捕获长距离依赖
    - 支持并行计算
    - 可以动态调整特征权重
    
    缺点：
    - 计算复杂度随序列长度平方增长
    - 需要较多的注意力头来捕获不同尺度的关系
    - 对超参数（如头数）敏感
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        **kwargs
    ):
        """初始化注意力特征融合模块
        
        Args:
            input_dims: 输入维度字典
            output_dim: 输出维度
            num_heads: 注意力头数
            dropout: Dropout比率
            use_layer_norm: 是否使用层归一化
        """
        super().__init__(input_dims=input_dims, output_dim=output_dim, num_heads=num_heads, 
                        dropout=dropout, use_layer_norm=use_layer_norm, **kwargs)
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in input_dims.items()
        })
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播
        
        Args:
            features: 多模态特征
            
        Returns:
            融合后的特征
        """
        # 投影到相同维度
        projected = {
            name: self.projections[name](feat)
            for name, feat in features.items()
        }
        
        # 拼接特征
        fused = torch.cat(list(projected.values()), dim=1)
        
        # 自注意力
        fused, _ = self.attention(fused, fused, fused)
        
        # 应用层归一化和Dropout
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused

class CrossAttentionFusion(BaseFusionModule):
    """交叉注意力特征融合模块
    
    适用场景：
    1. 需要显式建模跨模态交互
    2. 不同模态间存在强相关性
    3. 需要对齐不同模态的信息
    4. 模态间存在互补信息
    
    优点：
    - 显式建模跨模态关系
    - 支持非对称的模态交互
    - 可以捕获模态间的互补信息
    
    缺点：
    - 计算复杂度较高
    - 需要仔细设计注意力机制
    - 对模态质量差异敏感
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        **kwargs
    ):
        """初始化交叉注意力特征融合模块
        
        Args:
            input_dims: 输入维度字典
            output_dim: 输出维度
            num_heads: 注意力头数
            dropout: Dropout比率
            use_layer_norm: 是否使用层归一化
        """
        super().__init__(input_dims=input_dims, output_dim=output_dim, num_heads=num_heads,
                        dropout=dropout, use_layer_norm=use_layer_norm, **kwargs)
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in input_dims.items()
        })
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播
        
        Args:
            features: 多模态特征
            
        Returns:
            融合后的特征
        """
        # 投影到相同维度
        projected = {
            name: self.projections[name](feat)
            for name, feat in features.items()
        }
        
        # 对每对模态进行交叉注意力
        fused = list(projected.values())[0]
        for i in range(1, len(projected)):
            fused, _ = self.attention(
                query=fused,
                key=list(projected.values())[i],
                value=list(projected.values())[i]
            )
        
        # 应用层归一化和Dropout
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused

class GatedFusion(BaseFusionModule):
    """门控特征融合模块
    
    适用场景：
    1. 需要动态调整不同模态的重要性
    2. 某些模态可能包含噪声或无关信息
    3. 不同模态的可靠性不同
    4. 需要自适应特征选择
    
    优点：
    - 可以动态调整特征权重
    - 对噪声具有鲁棒性
    - 支持多种门控机制
    
    缺点：
    - 需要额外的门控参数
    - 可能过度依赖某些模态
    - 训练可能不稳定
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int,
        dropout: float = 0.1,
        gating_type: str = "sigmoid",
        use_layer_norm: bool = True,
        **kwargs
    ):
        """初始化门控特征融合模块
        
        Args:
            input_dims: 输入维度字典
            output_dim: 输出维度
            dropout: Dropout比率
            gating_type: 门控类型 ("sigmoid", "tanh", "relu")
            use_layer_norm: 是否使用层归一化
        """
        super().__init__(input_dims=input_dims, output_dim=output_dim, dropout=dropout,
                        gating_type=gating_type, use_layer_norm=use_layer_norm, **kwargs)
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in input_dims.items()
        })
        self.gate = nn.Linear(output_dim * 2, output_dim)
        self.gate_activation = self._get_gate_activation(gating_type)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()
        
    def _get_gate_activation(self, gating_type: str) -> nn.Module:
        """获取门控激活函数
        
        Args:
            gating_type: 门控类型
            
        Returns:
            激活函数模块
        """
        if gating_type == "sigmoid":
            return nn.Sigmoid()
        elif gating_type == "tanh":
            return nn.Tanh()
        elif gating_type == "relu":
            return nn.ReLU()
        else:
            raise ValueError(f"Invalid gating type: {gating_type}")
            
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播
        
        Args:
            features: 多模态特征
            
        Returns:
            融合后的特征
        """
        # 投影到相同维度
        projected = {
            name: self.projections[name](feat)
            for name, feat in features.items()
        }
        
        # 对每对模态进行门控融合
        fused = list(projected.values())[0]
        for i in range(1, len(projected)):
            gate = self.gate_activation(
                self.gate(torch.cat([fused, list(projected.values())[i]], dim=-1))
            )
            fused = gate * fused + (1 - gate) * list(projected.values())[i]
        
        # 应用层归一化和Dropout
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused

class BilinearFusion(BaseFusionModule):
    """双线性特征融合模块
    
    适用场景：
    1. 需要捕获特征间的二阶交互
    2. 模态间存在复杂的非线性关系
    3. 需要细粒度的特征交互
    4. 输入特征维度适中
    
    优点：
    - 可以捕获二阶交互
    - 计算效率较高
    - 参数数量适中
    
    缺点：
    - 无法捕获高阶交互
    - 对特征维度敏感
    - 需要较大的输出维度
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        **kwargs
    ):
        """初始化双线性特征融合模块
        
        Args:
            input_dims: 输入维度字典
            output_dim: 输出维度
            dropout: Dropout比率
            use_layer_norm: 是否使用层归一化
        """
        super().__init__(input_dims=input_dims, output_dim=output_dim, dropout=dropout,
                        use_layer_norm=use_layer_norm, **kwargs)
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in input_dims.items()
        })
        self.bilinear = nn.Bilinear(output_dim, output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播
        
        Args:
            features: 多模态特征
            
        Returns:
            融合后的特征
        """
        # 投影到相同维度
        projected = {
            name: self.projections[name](feat)
            for name, feat in features.items()
        }
        
        # 对每对模态进行双线性融合
        fused = list(projected.values())[0]
        for i in range(1, len(projected)):
            fused = self.bilinear(fused, list(projected.values())[i])
        
        # 应用层归一化和Dropout
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused

class TuckerFusion(BaseFusionModule):
    """Tucker分解特征融合模块
    
    适用场景：
    1. 需要降维和捕获高阶交互
    2. 输入特征维度较高
    3. 需要压缩模型参数量
    4. 模态数量固定（目前仅支持3个模态）
    
    优点：
    - 可以捕获高阶交互
    - 支持特征降维
    - 参数量可控
    
    缺点：
    - 仅支持固定数量的模态
    - 计算复杂度较高
    - 需要仔细选择分解秩
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int,
        rank: int = 32,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        **kwargs
    ):
        """初始化Tucker分解特征融合模块
        
        Args:
            input_dims: 输入维度字典
            output_dim: 输出维度
            rank: Tucker分解的秩
            dropout: Dropout比率
            use_layer_norm: 是否使用层归一化
        """
        super().__init__(input_dims=input_dims, output_dim=output_dim, rank=rank,
                        dropout=dropout, use_layer_norm=use_layer_norm, **kwargs)
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in input_dims.items()
        })
        self.core = nn.Parameter(torch.randn(rank, rank, rank))
        self.factors = nn.ModuleList([
            nn.Linear(output_dim, rank) for _ in range(3)
        ])
        self.output = nn.Linear(rank, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播
        
        Args:
            features: 多模态特征
            
        Returns:
            融合后的特征
        """
        if len(features) != 3:
            raise ValueError("Tucker fusion requires exactly three modalities")
            
        # 投影到相同维度
        projected = {
            name: self.projections[name](feat)
            for name, feat in features.items()
        }
        
        # 投影到低维空间
        factors = [factor(mod) for factor, mod in zip(self.factors, projected.values())]
        
        # 计算Tucker分解
        tucker = torch.einsum('ijk,i,j,k->ijk', self.core, *factors)
        
        # 投影回原始空间
        fused = self.output(tucker)
        
        # 应用层归一化和Dropout
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused

class HierarchicalFusion(BaseFusionModule):
    """层次特征融合模块
    
    适用场景：
    1. 需要多层次的特征提取
    2. 不同模态的特征需要不同层次的处理
    3. 需要渐进式的特征融合
    4. 输入特征维度差异较大
    
    优点：
    - 支持多层次特征提取
    - 可以处理不同维度的特征
    - 结构灵活可配置
    
    缺点：
    - 需要仔细设计网络结构
    - 训练时间较长
    - 可能出现过拟合
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        **kwargs
    ):
        """初始化层次特征融合模块
        
        Args:
            input_dims: 输入维度字典
            output_dim: 输出维度
            hidden_dims: 隐藏维度列表
            dropout: Dropout比率
            use_layer_norm: 是否使用层归一化
        """
        super().__init__(input_dims=input_dims, output_dim=output_dim, hidden_dims=hidden_dims,
                        dropout=dropout, use_layer_norm=use_layer_norm, **kwargs)
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dims[0])
            for name, dim in input_dims.items()
        })
        
        # 构建层次融合网络
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.fusion_network = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播
        
        Args:
            features: 多模态特征
            
        Returns:
            融合后的特征
        """
        # 投影到相同维度
        projected = {
            name: self.projections[name](feat)
            for name, feat in features.items()
        }
        
        # 拼接特征
        fused = torch.cat(list(projected.values()), dim=-1)
        
        # 层次融合
        fused = self.fusion_network(fused)
        
        # 应用层归一化
        fused = self.layer_norm(fused)
        
        return fused 