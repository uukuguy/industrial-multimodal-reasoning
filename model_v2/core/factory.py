import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Type, List
import logging
from pathlib import Path
import json

from ..config.model_config import ModelConfig
from ..modules.encoders import (
    BaseTextEncoder,
    BaseImageEncoder,
    BaseLayoutEncoder,
    BaseFusionModule
)
from ..modules.heads import BaseHead

logger = logging.getLogger(__name__)

class ModelFactory:
    """模型工厂类
    
    用于创建和注册模型组件，支持动态扩展和配置。
    """
    
    def __init__(self):
        """初始化模型工厂"""
        self._text_encoders: Dict[str, Type[BaseTextEncoder]] = {}
        self._image_encoders: Dict[str, Type[BaseImageEncoder]] = {}
        self._layout_encoders: Dict[str, Type[BaseLayoutEncoder]] = {}
        self._fusion_modules: Dict[str, Type[BaseFusionModule]] = {}
        self._heads: Dict[str, Type[BaseHead]] = {}
        
    def register_text_encoder(self, name: str, encoder_class: Type[BaseTextEncoder]):
        """注册文本编码器
        
        Args:
            name: 编码器名称
            encoder_class: 编码器类
        """
        if name in self._text_encoders:
            logger.warning(f"Text encoder {name} already registered, overwriting")
        self._text_encoders[name] = encoder_class
        logger.info(f"Registered text encoder: {name}")
        
    def register_image_encoder(self, name: str, encoder_class: Type[BaseImageEncoder]):
        """注册图像编码器
        
        Args:
            name: 编码器名称
            encoder_class: 编码器类
        """
        if name in self._image_encoders:
            logger.warning(f"Image encoder {name} already registered, overwriting")
        self._image_encoders[name] = encoder_class
        logger.info(f"Registered image encoder: {name}")
        
    def register_layout_encoder(self, name: str, encoder_class: Type[BaseLayoutEncoder]):
        """注册布局编码器
        
        Args:
            name: 编码器名称
            encoder_class: 编码器类
        """
        if name in self._layout_encoders:
            logger.warning(f"Layout encoder {name} already registered, overwriting")
        self._layout_encoders[name] = encoder_class
        logger.info(f"Registered layout encoder: {name}")
        
    def register_fusion_module(self, name: str, fusion_class: Type[BaseFusionModule]):
        """注册特征融合模块
        
        Args:
            name: 融合模块名称
            fusion_class: 融合模块类
        """
        if name in self._fusion_modules:
            logger.warning(f"Fusion module {name} already registered, overwriting")
        self._fusion_modules[name] = fusion_class
        logger.info(f"Registered fusion module: {name}")
        
    def register_head(self, name: str, head_class: Type[BaseHead]):
        """注册输出头
        
        Args:
            name: 输出头名称
            head_class: 输出头类
        """
        if name in self._heads:
            logger.warning(f"Head {name} already registered, overwriting")
        self._heads[name] = head_class
        logger.info(f"Registered head: {name}")
        
    def create_text_encoder(
        self,
        name: str,
        **kwargs
    ) -> BaseTextEncoder:
        """创建文本编码器
        
        Args:
            name: 编码器名称
            **kwargs: 编码器参数
            
        Returns:
            文本编码器实例
        """
        if name not in self._text_encoders:
            raise ValueError(f"Unknown text encoder: {name}")
        return self._text_encoders[name](**kwargs)
        
    def create_image_encoder(
        self,
        name: str,
        **kwargs
    ) -> BaseImageEncoder:
        """创建图像编码器
        
        Args:
            name: 编码器名称
            **kwargs: 编码器参数
            
        Returns:
            图像编码器实例
        """
        if name not in self._image_encoders:
            raise ValueError(f"Unknown image encoder: {name}")
        return self._image_encoders[name](**kwargs)
        
    def create_layout_encoder(
        self,
        name: str,
        **kwargs
    ) -> BaseLayoutEncoder:
        """创建布局编码器
        
        Args:
            name: 编码器名称
            **kwargs: 编码器参数
            
        Returns:
            布局编码器实例
        """
        if name not in self._layout_encoders:
            raise ValueError(f"Unknown layout encoder: {name}")
        return self._layout_encoders[name](**kwargs)
        
    def create_fusion_module(
        self,
        name: str,
        **kwargs
    ) -> BaseFusionModule:
        """创建特征融合模块
        
        Args:
            name: 融合模块名称
            **kwargs: 融合模块参数
            
        Returns:
            特征融合模块实例
        """
        if name not in self._fusion_modules:
            raise ValueError(f"Unknown fusion module: {name}")
        return self._fusion_modules[name](**kwargs)
        
    def create_head(
        self,
        name: str,
        **kwargs
    ) -> BaseHead:
        """创建输出头
        
        Args:
            name: 输出头名称
            **kwargs: 输出头参数
            
        Returns:
            输出头实例
        """
        if name not in self._heads:
            raise ValueError(f"Unknown head: {name}")
        return self._heads[name](**kwargs)
        
    def get_available_components(self) -> Dict[str, List[str]]:
        """获取可用的组件列表
        
        Returns:
            组件字典，包含各种类型的可用组件名称
        """
        return {
            "text_encoders": list(self._text_encoders.keys()),
            "image_encoders": list(self._image_encoders.keys()),
            "layout_encoders": list(self._layout_encoders.keys()),
            "fusion_modules": list(self._fusion_modules.keys()),
            "heads": list(self._heads.keys())
        }
        
    def create_model_from_config(self, config: ModelConfig) -> nn.Module:
        """根据配置创建完整模型
        
        Args:
            config: 模型配置
            
        Returns:
            模型实例
        """
        from ..core.model import BaseModel
        
        # 创建编码器
        text_encoder = self.create_text_encoder(
            config.text_encoder.name,
            **config.text_encoder.params
        ) if config.text_encoder else None
        
        image_encoder = self.create_image_encoder(
            config.image_encoder.name,
            **config.image_encoder.params
        ) if config.image_encoder else None
        
        layout_encoder = self.create_layout_encoder(
            config.layout_encoder.name,
            **config.layout_encoder.params
        ) if config.layout_encoder else None
        
        # 创建融合模块
        fusion_module = self.create_fusion_module(
            config.fusion.name,
            **config.fusion.params
        ) if config.fusion else None
        
        # 创建输出头
        head = self.create_head(
            config.head.name,
            **config.head.params
        ) if config.head else None
        
        # 创建模型
        model = BaseModel(
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            layout_encoder=layout_encoder,
            fusion_module=fusion_module,
            head=head,
            config=config
        )
        
        return model
        
    def save_model(
        self,
        model: nn.Module,
        save_path: str,
        config: Optional[ModelConfig] = None
    ):
        """保存模型
        
        Args:
            model: 模型实例
            save_path: 保存路径
            config: 模型配置
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型参数
        torch.save(model.state_dict(), save_path / "model.pt")
        
        # 保存配置
        if config:
            with open(save_path / "config.json", "w") as f:
                json.dump(config.to_dict(), f, indent=2)
                
        logger.info(f"Model saved to {save_path}")
        
    def load_model(
        self,
        load_path: str,
        config: Optional[ModelConfig] = None
    ) -> nn.Module:
        """加载模型
        
        Args:
            load_path: 加载路径
            config: 模型配置
            
        Returns:
            模型实例
        """
        load_path = Path(load_path)
        
        # 加载配置
        if config is None:
            config_path = load_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = ModelConfig.from_dict(json.load(f))
            else:
                raise ValueError("No config file found and no config provided")
                
        # 创建模型
        model = self.create_model_from_config(config)
        
        # 加载参数
        model.load_state_dict(torch.load(load_path / "model.pt"))
        
        logger.info(f"Model loaded from {load_path}")
        return model

# 创建全局模型工厂实例
model_factory = ModelFactory() 