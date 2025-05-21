# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Union, List, Tuple

from .config import load_config, validate_config
from .multimodal_encoder import MultimodalEncoder
from .multimodal_fusion import MultimodalFusion, HierarchicalMultimodalFusion
from .cross_modal_attention import MultiModalAttentionHub
from .modality_reconstructor import ModalityReconstructor
from .uncertainty_estimator import UncertaintyEstimator
from .qa_module import ReasoningQAModule, EnhancedReasoningQAModule

logger = logging.getLogger(__name__)

class EnhancedMultiModalModel(nn.Module):
    """
    增强型多模态模型，集成了多模态编码、融合和问答能力
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        初始化增强型多模态模型
        
        Args:
            config_path: 配置文件路径
            **kwargs: 配置参数覆盖
        """
        super().__init__()
        
        # 加载配置
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = kwargs
        
        # 验证配置
        validate_config(self.config)
        
        # 获取设备
        self.device = self._get_device(self.config.get('model', {}).get('device', 'auto'))
        
        # 初始化编码器
        self.encoder = self._init_encoder()
        
        # 初始化多模态注意力中枢
        self.attention_hub = self._init_attention_hub()
        
        # 初始化融合模块
        self.fusion = self._init_fusion()
        
        # 初始化问答模块
        self.qa_module = self._init_qa_module()
        
        # 初始化可选组件：模态重构器
        self.reconstructor = self._init_reconstructor()
        
        # 初始化可选组件：不确定性估计器
        self.uncertainty_estimator = self._init_uncertainty_estimator()
        
        # 移动到指定设备
        self.to(self.device)
        
        logger.info(f"增强型多模态模型初始化完成，使用设备: {self.device}")
    
    def _get_device(self, device_preference: str) -> torch.device:
        """获取计算设备"""
        if device_preference == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"自动选择GPU设备: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                logger.info("自动选择CPU设备")
        else:
            device = torch.device(device_preference)
            logger.info(f"使用指定设备: {device}")
            
        return device
    
    def _init_encoder(self) -> MultimodalEncoder:
        """初始化编码器"""
        encoder_config = self.config.get('encoders', {})
        model_config = self.config.get('model', {})
        
        text_model = encoder_config.get('text', {}).get('model_name', 'bert-base-chinese')
        vision_model = encoder_config.get('vision', {}).get('model_name', 'google/vit-base-patch16-224')
        
        # 其他编码器参数
        use_cache = model_config.get('use_cache', True)
        batch_size = encoder_config.get('text', {}).get('batch_size', 16)
        
        encoder = MultimodalEncoder(
            text_model_name=text_model,
            vision_model_name=vision_model,
            use_cache=use_cache,
            device=self.device,
            batch_size=batch_size
        )
        
        return encoder
    
    def _init_attention_hub(self) -> Optional[MultiModalAttentionHub]:
        """初始化多模态注意力中枢"""
        fusion_config = self.config.get('fusion', {})
        model_config = self.config.get('model', {})
        
        if fusion_config.get('strategy', 'hierarchical') != 'attention':
            # 如果不使用注意力策略，返回None
            return None
        
        embedding_dim = model_config.get('embedding_dim', 768)
        num_heads = fusion_config.get('num_attention_heads', 8)
        dropout = fusion_config.get('dropout', 0.1)
        
        attention_hub = MultiModalAttentionHub(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        return attention_hub
    
    def _init_fusion(self) -> Union[MultimodalFusion, HierarchicalMultimodalFusion]:
        """初始化融合模块"""
        fusion_config = self.config.get('fusion', {})
        model_config = self.config.get('model', {})
        
        embedding_dim = model_config.get('embedding_dim', 768)
        fusion_dim = model_config.get('fusion_dim', 512)
        strategy = fusion_config.get('strategy', 'hierarchical')
        dropout = fusion_config.get('dropout', 0.1)
        
        if strategy == 'hierarchical':
            fusion = HierarchicalMultimodalFusion(
                embedding_dim=embedding_dim,
                hidden_dims=[embedding_dim, embedding_dim//2, fusion_dim],
                dropout=dropout,
                use_attention=True
            )
        else:
            # 默认使用简单融合
            fusion = MultimodalFusion(
                embedding_dim=embedding_dim,
                fused_embedding_dim=fusion_dim,
                dropout=dropout
            )
            
        return fusion
    
    def _init_qa_module(self) -> Union[ReasoningQAModule, EnhancedReasoningQAModule]:
        """初始化问答模块"""
        qa_config = self.config.get('qa_module', {})
        model_config = self.config.get('model', {})
        
        lmm_model_name = qa_config.get('model_name', 'placeholder/lmm-model')
        fusion_dim = model_config.get('fusion_dim', 512)
        temperature = qa_config.get('temperature', 1.0)
        confidence_threshold = qa_config.get('confidence_threshold', 0.7)
        
        # 使用增强型问答模块
        max_answer_length = qa_config.get('max_answer_length', 100)
        top_k = qa_config.get('top_k', 5)
        
        qa_module = EnhancedReasoningQAModule(
            lmm_model_name=lmm_model_name,
            embedding_dim=fusion_dim,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            max_answer_length=max_answer_length,
            top_k=top_k
        )
            
        return qa_module
    
    def _init_reconstructor(self) -> Optional[ModalityReconstructor]:
        """初始化模态重构器（可选）"""
        model_config = self.config.get('model', {})
        
        if not model_config.get('use_reconstructor', False):
            return None
            
        embedding_dim = model_config.get('embedding_dim', 768)
        
        reconstructor = ModalityReconstructor(
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim
        )
            
        return reconstructor
    
    def _init_uncertainty_estimator(self) -> Optional[UncertaintyEstimator]:
        """初始化不确定性估计器（可选）"""
        model_config = self.config.get('model', {})
        
        if not model_config.get('use_uncertainty', False):
            return None
            
        fusion_dim = model_config.get('fusion_dim', 512)
        num_classes = 4  # 对于初赛的 A/B/C/D 选择题
        
        estimator = UncertaintyEstimator(
            embedding_dim=fusion_dim,
            num_classes=num_classes
        )
            
        return estimator
    
    def encode_document(self, page_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        编码文档的所有页面
        
        Args:
            page_data_list: 包含页面数据的列表（每个页面是一个字典）
            
        Returns:
            包含编码结果的字典
        """
        results = {'page_encodings': []}
        
        for page_data in page_data_list:
            # 使用编码器编码页面
            encoded_page = self.encoder.encode_document_page(page_data)
            results['page_encodings'].append(encoded_page)
        
        return results
    
    def forward(self, document_encodings: Dict[str, Any], question: str, 
               options: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            document_encodings: 文档编码结果
            question: 问题文本
            options: 选项列表（如果是选择题）
            
        Returns:
            包含预测结果的字典
        """
        results = {}
        document_fused_embeddings = []
        
        # 处理每个页面的编码
        page_encodings = document_encodings.get('page_encodings', [])
        
        for encoded_page in page_encodings:
            # 提取不同模态的嵌入
            text_embeddings = encoded_page.get('text_embeddings', [])
            page_image_embedding = encoded_page.get('page_image_embedding')
            layout_embedding = encoded_page.get('layout_embedding')
            
            # 如果使用注意力中枢
            if self.attention_hub is not None:
                # 准备输入
                batch_size = 1
                seq_len_text = len(text_embeddings)
                
                # 将文本嵌入转换为张量
                if text_embeddings:
                    text_tensor = torch.cat([emb for emb in text_embeddings], dim=0)  # [seq_len_text, embed_dim]
                    text_tensor = text_tensor.unsqueeze(0)  # [1, seq_len_text, embed_dim]
                else:
                    # 如果没有文本嵌入，创建一个空张量
                    embed_dim = page_image_embedding.shape[-1] if page_image_embedding is not None else 768
                    text_tensor = torch.zeros(batch_size, 1, embed_dim, device=self.device)
                
                # 处理图像嵌入
                if page_image_embedding is not None:
                    # 确保形状是 [batch_size, 1, embed_dim]
                    if page_image_embedding.dim() == 2:
                        page_image_embedding = page_image_embedding.unsqueeze(1)  # [batch_size, 1, embed_dim]
                else:
                    # 如果没有图像嵌入，创建一个空张量
                    embed_dim = text_tensor.shape[-1]
                    page_image_embedding = torch.zeros(batch_size, 1, embed_dim, device=self.device)
                
                # 处理布局嵌入
                if layout_embedding is not None:
                    # 确保形状是 [batch_size, 1, embed_dim]
                    if layout_embedding.dim() == 2:
                        layout_embedding = layout_embedding.unsqueeze(1)  # [batch_size, 1, embed_dim]
                else:
                    # 如果没有布局嵌入，创建一个空张量
                    embed_dim = text_tensor.shape[-1]
                    layout_embedding = torch.zeros(batch_size, 1, embed_dim, device=self.device)
                
                # 应用注意力中枢
                attention_results = self.attention_hub(text_tensor, page_image_embedding, layout_embedding)
                fused_embedding = attention_results['fused_embedding']
                
                # 记录注意力权重
                results['attention_weights'] = {
                    'text_to_vision': attention_results.get('text_to_vision_weights'),
                    'vision_to_text': attention_results.get('vision_to_text_weights')
                }
            else:
                # 使用融合模块直接融合
                fusion_results = self.fusion(text_embeddings, page_image_embedding, layout_embedding)
                
                if isinstance(fusion_results, dict):
                    fused_embedding = fusion_results.get('fused_embedding')
                    # 记录其他融合结果
                    for k, v in fusion_results.items():
                        if k != 'fused_embedding':
                            results[f'fusion_{k}'] = v
                else:
                    fused_embedding = fusion_results
            
            # 收集每个页面的融合嵌入
            document_fused_embeddings.append(fused_embedding)
        
        # 整合所有页面的融合嵌入
        if document_fused_embeddings:
            # 堆叠所有页面的融合嵌入
            all_pages_embeddings = torch.cat(document_fused_embeddings, dim=0)  # [num_pages, fused_dim]
            
            # 计算页面权重（可以是简单的平均或注意力加权）
            page_weights = torch.ones(all_pages_embeddings.size(0), 1, device=self.device)
            page_weights = page_weights / page_weights.sum()  # 归一化
            
            # 加权平均得到文档表示
            document_representation = (all_pages_embeddings * page_weights).sum(dim=0, keepdim=True)  # [1, fused_dim]
            
            # 记录文档表示
            results['document_representation'] = document_representation
            
            # 如果启用不确定性估计
            if self.uncertainty_estimator is not None:
                is_classification = options is not None and len(options) > 0
                uncertainty_results = self.uncertainty_estimator(document_representation, is_classification)
                results['uncertainty'] = uncertainty_results
            
            # 生成答案
            if hasattr(self.qa_module, 'answer_question'):
                answer = self.qa_module.answer_question(document_representation, question, options)
                results['answer'] = answer
        else:
            # 如果没有有效的页面融合嵌入
            results['error'] = "没有有效的页面融合嵌入"
            results['answer'] = "A" if options and len(options) > 0 else "无法回答，未找到有效内容"
        
        return results
    
    def get_embeddings(self, document_encodings: Dict[str, Any]) -> torch.Tensor:
        """
        获取文档的统一嵌入表示
        
        Args:
            document_encodings: 文档编码结果
            
        Returns:
            文档统一嵌入表示
        """
        document_fused_embeddings = []
        
        # 处理每个页面的编码
        page_encodings = document_encodings.get('page_encodings', [])
        
        for encoded_page in page_encodings:
            # 提取不同模态的嵌入
            text_embeddings = encoded_page.get('text_embeddings', [])
            page_image_embedding = encoded_page.get('page_image_embedding')
            layout_embedding = encoded_page.get('layout_embedding')
            
            # 使用融合模块直接融合
            fusion_results = self.fusion(text_embeddings, page_image_embedding, layout_embedding)
            
            if isinstance(fusion_results, dict):
                fused_embedding = fusion_results.get('fused_embedding')
            else:
                fused_embedding = fusion_results
            
            # 收集每个页面的融合嵌入
            document_fused_embeddings.append(fused_embedding)
        
        # 整合所有页面的融合嵌入
        if document_fused_embeddings:
            # 堆叠所有页面的融合嵌入
            all_pages_embeddings = torch.cat(document_fused_embeddings, dim=0)  # [num_pages, fused_dim]
            
            # 计算平均值作为文档表示
            document_representation = all_pages_embeddings.mean(dim=0, keepdim=True)  # [1, fused_dim]
            
            return document_representation
        else:
            # 如果没有有效的页面融合嵌入，返回零向量
            embedding_dim = self.config.get('model', {}).get('fusion_dim', 512)
            return torch.zeros(1, embedding_dim, device=self.device)
    
    def reconstruct_modality(self, target_modality: int, document_encodings: Dict[str, Any]) -> Dict[str, Any]:
        """
        重构缺失的模态
        
        Args:
            target_modality: 目标模态 (0=视觉, 1=文本, 2=布局)
            document_encodings: 文档编码结果
            
        Returns:
            包含重构结果的字典
        """
        if self.reconstructor is None:
            raise ValueError("模态重构器未启用")
            
        results = {'reconstructed_pages': []}
        
        # 处理每个页面的编码
        page_encodings = document_encodings.get('page_encodings', [])
        
        for encoded_page in page_encodings:
            # 提取不同模态的嵌入
            text_embeddings = encoded_page.get('text_embeddings', [])
            page_image_embedding = encoded_page.get('page_image_embedding')
            layout_embedding = encoded_page.get('layout_embedding')
            
            # 重构目标模态
            if target_modality == 0:  # 重构视觉
                reconstructed, quality = self.reconstructor.reconstruct_visual(text_embeddings, layout_embedding)
                modality_name = 'visual'
            elif target_modality == 1:  # 重构文本
                reconstructed, quality = self.reconstructor.reconstruct_text(page_image_embedding, layout_embedding)
                modality_name = 'text'
            elif target_modality == 2:  # 重构布局
                reconstructed, quality = self.reconstructor.reconstruct_layout(page_image_embedding, text_embeddings)
                modality_name = 'layout'
            else:
                raise ValueError(f"无效的目标模态: {target_modality}")
                
            # 记录结果
            results['reconstructed_pages'].append({
                'reconstructed': reconstructed,
                'quality': quality,
                'modality': modality_name
            })
            
        # 计算平均重构质量
        avg_quality = torch.mean(torch.tensor([page['quality'] for page in results['reconstructed_pages']]))
        results['average_quality'] = avg_quality
            
        return results
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'EnhancedMultiModalModel':
        """
        加载模型
        
        Args:
            path: 模型路径
            device: 计算设备
            
        Returns:
            加载的模型
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        # 自动选择设备
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        # 加载状态
        state_dict = torch.load(path, map_location=device)
        
        # 版本兼容性检查
        version = state_dict.get('version', "0.0.0")
        logger.info(f"加载模型版本: {version}")
        
        # 创建模型实例
        config = state_dict['config']
        model = cls(config_path=None, **config)
        
        # 加载模型权重
        model.load_state_dict(state_dict['model'])
        model = model.to(device)
        
        return model
    
    @classmethod
    def from_pretrained(cls, model_path: str, device: Optional[str] = None,
                       config_override: Optional[Dict[str, Any]] = None) -> 'EnhancedMultiModalModel':
        """
        从预训练模型路径加载模型，遵循transformers风格的API
        
        Args:
            model_path: 模型路径，可以是checkpoint目录或模型文件路径
            device: 计算设备
            config_override: 可选的配置覆盖
            
        Returns:
            加载的模型实例
        """
        # 检查路径是目录还是文件
        if os.path.isdir(model_path):
            # 查找目录中的模型文件
            model_files = [f for f in os.listdir(model_path)
                          if f.endswith('.pt') or f.endswith('.bin') or f == 'pytorch_model.bin']
            if not model_files:
                config_path = os.path.join(model_path, 'config.json')
                if os.path.exists(config_path):
                    # 仅有配置文件但没有权重，创建一个新模型
                    with open(config_path, 'r', encoding='utf-8') as f:
                        import json
                        config = json.load(f)
                    if config_override:
                        config.update(config_override)
                    logger.info(f"没有找到模型权重，从配置创建新模型: {config_path}")
                    model = cls(config_path=None, **config)
                    if device:
                        model = model.to(device)
                    return model
                else:
                    raise FileNotFoundError(f"在目录中未找到模型文件或配置文件: {model_path}")
                
            # 使用找到的第一个模型文件
            model_file = os.path.join(model_path, model_files[0])
            logger.info(f"从目录中加载模型文件: {model_file}")
        else:
            # 直接使用提供的文件路径
            model_file = model_path
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"模型文件不存在: {model_file}")
        
        # 加载模型
        return cls.load(model_file, device)
    
    def save(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None,
            epoch: int = 0, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
            optimizer: 优化器（可选）
            epoch: 当前轮次
            metrics: 评估指标
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型状态
        state_dict = {
            'model': self.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'epoch': epoch,
            'config': self.config,
            'metrics': metrics or {},
            'version': "1.0.0"  # 版本号
        }
        
        torch.save(state_dict, path)
        logger.info(f"模型已保存到: {path}")


if __name__ == "__main__":
    # 测试增强型多模态模型
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建示例配置
    config = {
        'model': {
            'embedding_dim': 768,
            'fusion_dim': 512,
            'use_cache': True,
            'device': 'cpu',
            'use_reconstructor': True,
            'use_uncertainty': True
        },
        'encoders': {
            'text': {
                'model_name': 'bert-base-chinese',
                'batch_size': 4
            },
            'vision': {
                'model_name': 'google/vit-base-patch16-224',
                'batch_size': 2
            }
        },
        'fusion': {
            'strategy': 'hierarchical',
            'num_attention_heads': 4,
            'dropout': 0.1
        },
        'qa_module': {
            'model_name': 'placeholder/lmm-model',
            'temperature': 1.2,
            'confidence_threshold': 0.65
        },
        'system': {
            'log_level': 'INFO'
        }
    }
    
    # 初始化模型
    print("初始化增强型多模态模型...")
    model = EnhancedMultiModalModel(**config)
    
    # 创建模拟文档编码
    batch_size = 1
    embedding_dim = config['model']['embedding_dim']
    fusion_dim = config['model']['fusion_dim']
    
    # 模拟两个页面的编码结果
    page_encodings = []
    for i in range(2):
        text_embeddings = [torch.randn(1, embedding_dim) for _ in range(3)]
        page_image_embedding = torch.randn(batch_size, embedding_dim)
        layout_embedding = torch.randn(batch_size, embedding_dim)
        
        page_encodings.append({
            'text_embeddings': text_embeddings,
            'page_image_embedding': page_image_embedding,
            'layout_embedding': layout_embedding
        })
    
    document_encodings = {'page_encodings': page_encodings}
    
    # 测试前向传播
    print("\n测试前向传播...")
    question = "根据文本信息，以下哪个描述符合该静电除尘器的特征？"
    options = ["A", "B", "C", "D"]
    
    results = model(document_encodings, question, options)
    
    print(f"答案: {results['answer']}")
    if 'uncertainty' in results:
        print(f"不确定性: {results['uncertainty'].get('task_uncertainty', torch.tensor(0.0)).item():.4f}")
    
    # 测试获取嵌入
    print("\n测试获取嵌入...")
    document_embedding = model.get_embeddings(document_encodings)
    print(f"文档嵌入形状: {document_embedding.shape}")
    
    # 测试模态重构
    if model.reconstructor is not None:
        print("\n测试模态重构...")
        recon_results = model.reconstruct_modality(0, document_encodings)  # 重构视觉模态
        print(f"平均重构质量: {recon_results['average_quality'].item():.4f}")
    
    # 测试保存和加载
    print("\n测试保存和加载模型...")
    save_dir = "temp_model"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "enhanced_model.pt")
    
    model.save(save_path)
    print(f"模型已保存到: {save_path}")
    
    loaded_model = EnhancedMultiModalModel.load(save_path)
    print("模型加载成功")
    
    # 清理
    import shutil
    shutil.rmtree(save_dir)
    print(f"清理临时目录: {save_dir}")