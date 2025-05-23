import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DeviceManager:
    """设备管理器
    
    负责管理模型的计算设备，包括：
    1. 设备选择
    2. 设备迁移
    3. 设备状态监控
    """
    
    def __init__(self, device: Optional[str] = None):
        """初始化设备管理器
        
        Args:
            device: 指定的设备
        """
        self.device = self._get_device(device)
        logger.info(f"Device manager initialized on {self.device}")
        
    def _get_device(self, device: Optional[str] = None) -> torch.device:
        """获取设备
        
        Args:
            device: 指定的设备
            
        Returns:
            torch.device对象
        """
        if device is not None:
            return torch.device(device)
            
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
        
    def to_device(self, model: torch.nn.Module) -> torch.nn.Module:
        """将模型移动到指定设备
        
        Args:
            model: 模型
            
        Returns:
            移动后的模型
        """
        return model.to(self.device)
        
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息
        
        Returns:
            设备信息字典
        """
        info = {
            "device": str(self.device),
            "is_cuda": self.device.type == "cuda"
        }
        
        if self.device.type == "cuda":
            info.update({
                "device_name": torch.cuda.get_device_name(self.device),
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "memory_allocated": torch.cuda.memory_allocated(self.device),
                "memory_reserved": torch.cuda.memory_reserved(self.device)
            })
            
        return info 