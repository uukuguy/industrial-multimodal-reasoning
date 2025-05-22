import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

class ImageEnhancement:
    """图像增强类"""
    
    def __init__(self, config: Dict):
        """初始化图像增强
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.preprocessing_config = config['model']['vision_encoder']['preprocessing']
    
    def enhance(self, images: torch.Tensor) -> torch.Tensor:
        """增强图像
        
        Args:
            images: 图像张量 [batch_size, channels, height, width]
        
        Returns:
            增强后的图像张量
        """
        # 转换为numpy数组
        images_np = images.cpu().numpy()
        enhanced_images = []
        
        for image in images_np:
            # 转换为OpenCV格式
            image = np.transpose(image, (1, 2, 0))
            image = (image * 255).astype(np.uint8)
            
            # 应用预处理
            if self.preprocessing_config.get('contrast_enhancement', False):
                image = self._enhance_contrast(image)
            
            if self.preprocessing_config.get('denoising', False):
                image = self._denoise(image)
            
            if self.preprocessing_config.get('sharpening', False):
                image = self._sharpen(image)
            
            if self.preprocessing_config.get('adaptive_thresholding', False):
                image = self._adaptive_threshold(image)
            
            # 转换回PyTorch张量
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            enhanced_images.append(image)
        
        return torch.tensor(enhanced_images, device=images.device)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """增强对比度
        
        Args:
            image: 输入图像
        
        Returns:
            增强后的图像
        """
        # 转换到LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # 应用CLAHE到L通道
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        l = clahe.apply(l)
        
        # 合并通道
        lab = cv2.merge([l, a, b])
        
        # 转换回RGB
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """降噪
        
        Args:
            image: 输入图像
        
        Returns:
            降噪后的图像
        """
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
    
    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """锐化
        
        Args:
            image: 输入图像
        
        Returns:
            锐化后的图像
        """
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """自适应阈值
        
        Args:
            image: 输入图像
        
        Returns:
            处理后的图像
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

class FeaturePyramidNetwork(nn.Module):
    """特征金字塔网络"""
    
    def __init__(self, in_channels: int, out_channels: int):
        """初始化特征金字塔网络
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super().__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1)
            for _ in range(4)
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(4)
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """前向传播
        
        Args:
            features: 特征列表
        
        Returns:
            处理后的特征列表
        """
        laterals = [
            lateral_conv(feature)
            for feature, lateral_conv in zip(features, self.lateral_convs)
        ]
        
        # 自顶向下路径
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )
        
        # 输出卷积
        outputs = [
            fpn_conv(lateral)
            for lateral, fpn_conv in zip(laterals, self.fpn_convs)
        ]
        
        return outputs

class ROIPooling(nn.Module):
    """区域兴趣池化"""
    
    def __init__(self, output_size: Tuple[int, int]):
        """初始化区域兴趣池化
        
        Args:
            output_size: 输出大小
        """
        super().__init__()
        self.output_size = output_size
    
    def forward(
        self,
        features: torch.Tensor,
        rois: torch.Tensor
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            features: 特征图
            rois: 感兴趣区域 [N, 5]，每行格式为 [batch_idx, x1, y1, x2, y2]
        
        Returns:
            池化后的特征
        """
        return torchvision.ops.roi_align(
            features,
            rois,
            self.output_size,
            spatial_scale=1.0,
            sampling_ratio=-1
        )

class EnhancedVisionEncoder(nn.Module):
    """增强的视觉编码器"""
    
    def __init__(self, config: Dict):
        """初始化视觉编码器
        
        Args:
            config: 配置字典
        """
        super().__init__()
        self.config = config
        
        # 图像增强
        self.image_enhancement = ImageEnhancement(config)
        
        # 基础特征提取器
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # 特征金字塔网络
        self.fpn = FeaturePyramidNetwork(2048, 256)
        
        # 区域兴趣池化
        self.roi_pooling = ROIPooling((7, 7))
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(256, 8)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512)
        )
    
    def forward(
        self,
        images: torch.Tensor,
        rois: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            images: 输入图像 [batch_size, channels, height, width]
            rois: 感兴趣区域 [N, 5]
        
        Returns:
            图像特征
        """
        # 图像增强
        enhanced_images = self.image_enhancement.enhance(images)
        
        # 特征提取
        features = self.backbone(enhanced_images)
        
        # 特征金字塔
        fpn_features = self.fpn([features])
        
        if rois is not None:
            # 区域兴趣池化
            pooled_features = self.roi_pooling(fpn_features[0], rois)
        else:
            # 全局池化
            pooled_features = F.adaptive_avg_pool2d(fpn_features[0], (7, 7))
        
        # 注意力机制
        b, c, h, w = pooled_features.shape
        pooled_features = pooled_features.view(b, c, -1).permute(2, 0, 1)
        attended_features, _ = self.attention(
            pooled_features,
            pooled_features,
            pooled_features
        )
        attended_features = attended_features.permute(1, 2, 0).view(b, c, h, w)
        
        # 输出层
        features = attended_features.view(b, -1)
        features = self.output_layer(features)
        
        return features 