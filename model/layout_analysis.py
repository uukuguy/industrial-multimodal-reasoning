import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

class ArrowDetector:
    """箭头检测器"""
    
    def __init__(self, config: Dict):
        """初始化箭头检测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.min_arrow_length = config.get('min_arrow_length', 20)
        self.max_arrow_length = config.get('max_arrow_length', 200)
        self.arrow_thickness = config.get('arrow_thickness', 2)
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """检测箭头
        
        Args:
            image: 输入图像
        
        Returns:
            箭头列表，每个箭头包含起点、终点和方向
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 霍夫线变换
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=self.min_arrow_length,
            maxLineGap=10
        )
        
        if lines is None:
            return []
        
        # 检测箭头
        arrows = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 计算线段长度
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > self.max_arrow_length:
                continue
            
            # 计算方向
            angle = np.arctan2(y2 - y1, x2 - x1)
            
            # 检测箭头头部
            head = self._detect_arrow_head(image, (x1, y1), (x2, y2))
            
            if head is not None:
                arrows.append({
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'head': head,
                    'angle': angle,
                    'length': length
                })
        
        return arrows
    
    def _detect_arrow_head(
        self,
        image: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """检测箭头头部
        
        Args:
            image: 输入图像
            start: 线段起点
            end: 线段终点
        
        Returns:
            箭头头部坐标
        """
        # 在终点附近搜索箭头头部
        x, y = end
        window_size = 10
        
        # 计算搜索区域
        x1 = max(0, x - window_size)
        y1 = max(0, y - window_size)
        x2 = min(image.shape[1], x + window_size)
        y2 = min(image.shape[0], y + window_size)
        
        # 在搜索区域内查找最亮的点
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # 转换为灰度图
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # 查找最亮的点
        _, _, _, max_loc = cv2.minMaxLoc(roi)
        head_x = x1 + max_loc[0]
        head_y = y1 + max_loc[1]
        
        return (head_x, head_y)

class TextBoxDetector:
    """文本框检测器"""
    
    def __init__(self, config: Dict):
        """初始化文本框检测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.min_box_size = config.get('min_box_size', 20)
        self.max_box_size = config.get('max_box_size', 200)
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """检测文本框
        
        Args:
            image: 输入图像
        
        Returns:
            文本框列表，每个文本框包含位置和大小
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 二值化
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 过滤和提取文本框
        text_boxes = []
        for contour in contours:
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 过滤太小的框
            if w < self.min_box_size or h < self.min_box_size:
                continue
            
            # 过滤太大的框
            if w > self.max_box_size or h > self.max_box_size:
                continue
            
            # 计算宽高比
            aspect_ratio = w / h
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue
            
            text_boxes.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'center': (x + w//2, y + h//2)
            })
        
        return text_boxes

class SpatialRelationshipAnalyzer:
    """空间关系分析器"""
    
    def __init__(self, config: Dict):
        """初始化空间关系分析器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.max_distance = config.get('max_distance', 100)
    
    def analyze(
        self,
        arrows: List[Dict],
        text_boxes: List[Dict]
    ) -> List[Dict]:
        """分析空间关系
        
        Args:
            arrows: 箭头列表
            text_boxes: 文本框列表
        
        Returns:
            关系列表，每个关系包含箭头和文本框的对应关系
        """
        relationships = []
        
        for arrow in arrows:
            # 找到最近的文本框
            nearest_box = self._find_nearest_box(
                arrow['end'],
                text_boxes
            )
            
            if nearest_box is not None:
                relationships.append({
                    'arrow': arrow,
                    'text_box': nearest_box,
                    'distance': self._calculate_distance(
                        arrow['end'],
                        nearest_box['center']
                    )
                })
        
        return relationships
    
    def _find_nearest_box(
        self,
        point: Tuple[int, int],
        text_boxes: List[Dict]
    ) -> Optional[Dict]:
        """找到最近的文本框
        
        Args:
            point: 查询点
            text_boxes: 文本框列表
        
        Returns:
            最近的文本框
        """
        min_distance = float('inf')
        nearest_box = None
        
        for box in text_boxes:
            distance = self._calculate_distance(point, box['center'])
            if distance < min_distance and distance < self.max_distance:
                min_distance = distance
                nearest_box = box
        
        return nearest_box
    
    def _calculate_distance(
        self,
        point1: Tuple[int, int],
        point2: Tuple[int, int]
    ) -> float:
        """计算两点之间的距离
        
        Args:
            point1: 第一个点
            point2: 第二个点
        
        Returns:
            距离
        """
        return np.sqrt(
            (point1[0] - point2[0])**2 +
            (point1[1] - point2[1])**2
        )

class LayoutAnalysisEncoder(nn.Module):
    """布局分析编码器"""
    
    def __init__(self, config: Dict):
        """初始化布局分析编码器
        
        Args:
            config: 配置字典
        """
        super().__init__()
        self.config = config
        
        # 箭头检测器
        self.arrow_detector = ArrowDetector(config)
        
        # 文本框检测器
        self.text_box_detector = TextBoxDetector(config)
        
        # 空间关系分析器
        self.spatial_analyzer = SpatialRelationshipAnalyzer(config)
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            images: 输入图像 [batch_size, channels, height, width]
        
        Returns:
            布局特征
        """
        batch_size = images.shape[0]
        layout_features = []
        
        for i in range(batch_size):
            # 转换为numpy数组
            image = images[i].cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
            image = (image * 255).astype(np.uint8)
            
            # 检测箭头
            arrows = self.arrow_detector.detect(image)
            
            # 检测文本框
            text_boxes = self.text_box_detector.detect(image)
            
            # 分析空间关系
            relationships = self.spatial_analyzer.analyze(arrows, text_boxes)
            
            # 提取特征
            feature = self._extract_layout_feature(relationships)
            layout_features.append(feature)
        
        # 堆叠特征
        layout_features = torch.stack(layout_features)
        
        return layout_features
    
    def _extract_layout_feature(
        self,
        relationships: List[Dict]
    ) -> torch.Tensor:
        """提取布局特征
        
        Args:
            relationships: 关系列表
        
        Returns:
            布局特征
        """
        # 初始化特征向量
        feature = torch.zeros(512)
        
        if not relationships:
            return feature
        
        # 提取箭头特征
        arrow_features = []
        for rel in relationships:
            arrow = rel['arrow']
            arrow_feature = torch.tensor([
                arrow['start'][0],
                arrow['start'][1],
                arrow['end'][0],
                arrow['end'][1],
                arrow['angle'],
                arrow['length']
            ])
            arrow_features.append(arrow_feature)
        
        # 提取文本框特征
        box_features = []
        for rel in relationships:
            box = rel['text_box']
            box_feature = torch.tensor([
                box['x'],
                box['y'],
                box['width'],
                box['height']
            ])
            box_features.append(box_feature)
        
        # 提取关系特征
        rel_features = []
        for rel in relationships:
            rel_feature = torch.tensor([
                rel['distance']
            ])
            rel_features.append(rel_feature)
        
        # 合并特征
        if arrow_features:
            arrow_features = torch.stack(arrow_features)
            feature[:256] = arrow_features.mean(dim=0)
        
        if box_features:
            box_features = torch.stack(box_features)
            feature[256:384] = box_features.mean(dim=0)
        
        if rel_features:
            rel_features = torch.stack(rel_features)
            feature[384:] = rel_features.mean(dim=0)
        
        # 通过特征提取器
        feature = self.feature_extractor(feature)
        
        return feature 