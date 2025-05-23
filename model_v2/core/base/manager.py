import os
import json
import torch
import logging
from typing import Optional, Dict, Any, List
from ..base.model import BaseModel
from ..base.device import DeviceManager
from ...config.model_config import ModelConfig

logger = logging.getLogger(__name__)

class ModelManager:
    """模型管理器
    
    负责管理模型的生命周期，包括：
    1. 模型保存
    2. 模型加载
    3. 参数管理
    4. 状态管理
    """
    
    def __init__(self, model: BaseModel, device_manager: DeviceManager):
        """初始化模型管理器
        
        Args:
            model: 模型
            device_manager: 设备管理器
        """
        self.model = model
        self.device_manager = device_manager
        
    def save_pretrained(self, save_path: str):
        """保存模型
        
        Args:
            save_path: 保存路径
        """
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型配置
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.model.config.to_dict(), f, indent=2)
            
        # 保存模型权重
        weights_path = os.path.join(save_path, "pytorch_model.bin")
        torch.save(self.model.state_dict(), weights_path)
        
        logger.info(f"Model saved to {save_path}")
        
    @classmethod
    def from_pretrained(
        cls,
        load_path: str,
        model_class: type,
        config: Optional[ModelConfig] = None,
        device_manager: Optional[DeviceManager] = None
    ) -> "ModelManager":
        """从预训练模型加载
        
        Args:
            load_path: 加载路径
            model_class: 模型类
            config: 模型配置
            device_manager: 设备管理器
            
        Returns:
            模型管理器
        """
        # 加载配置
        if config is None:
            config_path = os.path.join(load_path, "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = ModelConfig(**json.load(f))
                
        # 创建设备管理器
        if device_manager is None:
            device_manager = DeviceManager()
            
        # 创建模型
        model = model_class(config=config)
        
        # 加载权重
        weights_path = os.path.join(load_path, "pytorch_model.bin")
        model.load_state_dict(
            torch.load(weights_path, map_location=device_manager.device)
        )
        
        # 移动到设备
        model = device_manager.to_device(model)
        
        logger.info(f"Model loaded from {load_path}")
        return cls(model, device_manager)
        
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """获取可训练参数
        
        Returns:
            可训练参数列表
        """
        return [p for p in self.model.parameters() if p.requires_grad]
        
    def get_num_params(self) -> int:
        """获取参数总数
        
        Returns:
            参数总数
        """
        return sum(p.numel() for p in self.model.parameters())
        
    def get_num_trainable_params(self) -> int:
        """获取可训练参数数量
        
        Returns:
            可训练参数数量
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "num_params": self.get_num_params(),
            "num_trainable_params": self.get_num_trainable_params(),
            "device_info": self.device_manager.get_device_info()
        } 