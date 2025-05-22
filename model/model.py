import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
from transformers import AutoModel, AutoTokenizer
import logging

from .image_enhancement import ImageEnhancement, EnhancedVisionEncoder
from .layout_enhancement import EnhancedLayoutEncoder
from .ocr_enhancement import EnhancedOCR, OCRFeatureExtractor
from .fusion_enhancement import EnhancedFusionModule, UncertaintyEstimation
from .loss_enhancement import EnhancedLossModule
from .data_augmentation import CurriculumAugmentation
from .post_processing import PostProcessor

class EnhancedMultimodalModel(nn.Module):
    """增强型多模态模型"""
    
    def __init__(self, config: Dict):
        """初始化模型
        
        Args:
            config: 配置字典
        """
        super().__init__()
        self.config = config
        
        # 初始化编码器
        self._init_encoders()
        
        # 初始化融合模块
        self._init_fusion()
        
        # 初始化输出层
        self._init_output()
        
        # 初始化损失函数
        self._init_loss()
        
        # 初始化后处理
        self._init_post_process()
        
        # 图像增强
        self.image_enhancement = ImageEnhancement(config)
        
        # 数据增强
        self.data_augmentation = CurriculumAugmentation(config)
    
    def _init_encoders(self):
        """初始化编码器"""
        # 文本编码器
        self.text_encoder = AutoModel.from_pretrained(
            self.config['model']['text_encoder']['model_name']
        )
        
        # 视觉编码器
        self.vision_encoder = EnhancedVisionEncoder(
            self.config['model']['vision_encoder']
        )
        
        # 布局分析编码器
        self.layout_encoder = EnhancedLayoutEncoder(
            self.config['model']['layout_analysis']
        )
        
        # OCR编码器
        self.ocr_encoder = OCREncoder(
            self.config['model']['ocr_enhancement']
        )
    
    def _init_fusion(self):
        """初始化融合模块"""
        self.fusion = EnhancedFusionModule(
            self.config['model']['fusion']
        )
    
    def _init_output(self):
        """初始化输出层"""
        hidden_size = self.config['model']['fusion']['hidden_size']
        num_classes = self.config['model']['num_classes']
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )
    
    def _init_loss(self):
        """初始化损失函数"""
        self.loss_module = EnhancedLossModule(
            self.config['model']['loss']
        )
    
    def _init_post_process(self):
        """初始化后处理"""
        self.post_processor = PostProcessor(
            self.config['model']['post_processing']
        )
    
    def forward(
        self,
        images: torch.Tensor,
        texts: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            images: 图像张量 [batch_size, channels, height, width]
            texts: 文本或文本列表
        
        Returns:
            输出字典
        """
        # 编码文本
        text_outputs = self.text_encoder(
            input_ids=texts['input_ids'],
            attention_mask=texts['attention_mask']
        )
        text_features = text_outputs.last_hidden_state
        
        # 编码图像
        vision_features = self.vision_encoder(images)
        
        # 编码布局
        layout_features = self.layout_encoder(images)
        
        # 编码OCR
        ocr_features = self.ocr_encoder(images)
        
        # 特征融合
        fused_features = self.fusion(
            text_features=text_features,
            vision_features=vision_features,
            layout_features=layout_features,
            ocr_features=ocr_features
        )
        
        # 分类
        logits = self.classifier(fused_features)
        
        return {
            'logits': logits,
            'text_features': text_features,
            'vision_features': vision_features,
            'layout_features': layout_features,
            'ocr_features': ocr_features,
            'fused_features': fused_features
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """计算损失
        
        Args:
            outputs: 模型输出
            targets: 目标标签
        
        Returns:
            损失字典
        """
        return self.loss_module(outputs, targets)
    
    def post_process(
        self,
        outputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """后处理
        
        Args:
            outputs: 模型输出
        
        Returns:
            预测结果和置信度
        """
        return self.post_processor(outputs)
    
    def set_epoch(self, epoch: int):
        """设置当前epoch
        
        Args:
            epoch: 当前epoch
        """
        self.current_epoch = epoch
        if hasattr(self.fusion, 'set_epoch'):
            self.fusion.set_epoch(epoch)
        if hasattr(self.loss_module, 'set_epoch'):
            self.loss_module.set_epoch(epoch)
        if hasattr(self.post_processor, 'set_epoch'):
            self.post_processor.set_epoch(epoch)
    
    def update_augmentation_strength(self, performance: float):
        """更新数据增强强度"""
        if isinstance(self.data_augmentation, CurriculumAugmentation):
            self.data_augmentation.update_strength(performance)
    
    def evaluate(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor) -> Dict[str, float]:
        """评估模型性能"""
        return self.post_processor.evaluate(predictions, targets)

class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_model(config: Dict) -> nn.Module:
        """创建模型
        
        Args:
            config: 配置字典
        
        Returns:
            模型对象
        """
        model_type = config['model']['type']
        if model_type == 'enhanced_multimodal':
            return EnhancedMultimodalModel(config)
        else:
            raise ValueError(f'Unknown model type: {model_type}')
    
    @staticmethod
    def load_model(config: Dict, checkpoint_path: str) -> EnhancedMultimodalModel:
        """加载预训练模型"""
        model = EnhancedMultimodalModel(config)
        model.load_state_dict(torch.load(checkpoint_path))
        return model
    
    @staticmethod
    def save_model(model: EnhancedMultimodalModel, save_path: str):
        """保存模型"""
        torch.save(model.state_dict(), save_path) 