import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import networkx as nx

class ArrowDetector:
    """箭头检测器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.min_length = config.get('min_length', 10)
        self.max_length = config.get('max_length', 100)
        self.angle_threshold = config.get('angle_threshold', 30)
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """检测图像中的箭头"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 霍夫线变换
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                               minLineLength=self.min_length,
                               maxLineGap=10)
        
        arrows = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # 长度过滤
                if self.min_length <= length <= self.max_length:
                    # 计算角度
                    angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                    
                    # 角度过滤
                    if angle <= self.angle_threshold:
                        arrows.append({
                            'start': (x1, y1),
                            'end': (x2, y2),
                            'length': length,
                            'angle': angle
                        })
        
        return arrows

class TextBoxDetector:
    """文本框检测器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.min_area = config.get('min_area', 100)
        self.max_area = config.get('max_area', 10000)
        self.aspect_ratio_threshold = config.get('aspect_ratio_threshold', 0.2)
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """检测图像中的文本框"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_boxes = []
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 面积过滤
            if self.min_area <= area <= self.max_area:
                # 获取最小外接矩形
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # 计算宽高比
                width = rect[1][0]
                height = rect[1][1]
                aspect_ratio = min(width, height) / max(width, height)
                
                # 宽高比过滤
                if aspect_ratio >= self.aspect_ratio_threshold:
                    text_boxes.append({
                        'box': box,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'center': rect[0]
                    })
        
        return text_boxes

class GraphConvolution(nn.Module):
    """图卷积网络"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_layers = config.get('num_layers', 3)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.dropout = config.get('dropout', 0.1)
        
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ))
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(torch.matmul(adj, x))
        return x

class SpatialRelationAnalyzer:
    """空间关系分析器"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def analyze(self, elements: List[Dict]) -> nx.Graph:
        """分析元素间的空间关系"""
        G = nx.Graph()
        
        # 添加节点
        for i, element in enumerate(elements):
            G.add_node(i, **element)
        
        # 分析空间关系
        for i in range(len(elements)):
            for j in range(i+1, len(elements)):
                relation = self._get_spatial_relation(elements[i], elements[j])
                if relation:
                    G.add_edge(i, j, relation=relation)
        
        return G
    
    def _get_spatial_relation(self, elem1: Dict, elem2: Dict) -> Optional[str]:
        """获取两个元素间的空间关系"""
        # 获取中心点
        center1 = elem1.get('center', (0, 0))
        center2 = elem2.get('center', (0, 0))
        
        # 计算相对位置
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        
        # 判断空间关系
        if abs(dx) < 10 and abs(dy) < 10:
            return 'overlap'
        elif abs(dx) > abs(dy):
            return 'left' if dx < 0 else 'right'
        else:
            return 'above' if dy < 0 else 'below'

class EnhancedLayoutEncoder(nn.Module):
    """增强型版面编码器"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.enhanced_layout_config = config.get('enhanced_layout', {})
        
        # 箭头检测器
        if self.enhanced_layout_config.get('detect_arrows'):
            self.arrow_detector = ArrowDetector(
                self.enhanced_layout_config.get('arrow_detection', {}))
        
        # 文本框检测器
        if self.enhanced_layout_config.get('detect_text_boxes'):
            self.text_box_detector = TextBoxDetector(
                self.enhanced_layout_config.get('text_box_detection', {}))
        
        # 图卷积网络
        if self.enhanced_layout_config.get('use_graph_conv'):
            self.graph_conv = GraphConvolution(
                self.enhanced_layout_config.get('graph_conv', {}))
        
        # 空间关系分析器
        if self.enhanced_layout_config.get('spatial_relation'):
            self.spatial_analyzer = SpatialRelationAnalyzer({})
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # 转换为numpy数组
        image_np = image.cpu().numpy()
        
        # 检测箭头
        arrows = []
        if hasattr(self, 'arrow_detector'):
            arrows = self.arrow_detector.detect(image_np)
        
        # 检测文本框
        text_boxes = []
        if hasattr(self, 'text_box_detector'):
            text_boxes = self.text_box_detector.detect(image_np)
        
        # 分析空间关系
        elements = arrows + text_boxes
        if hasattr(self, 'spatial_analyzer'):
            graph = self.spatial_analyzer.analyze(elements)
            adj_matrix = nx.adjacency_matrix(graph).toarray()
            adj_tensor = torch.from_numpy(adj_matrix).float().to(image.device)
        
        # 特征提取
        features = self._extract_features(elements)
        
        # 图卷积
        if hasattr(self, 'graph_conv'):
            features = self.graph_conv(features, adj_tensor)
        
        return features
    
    def _extract_features(self, elements: List[Dict]) -> torch.Tensor:
        """提取元素特征"""
        features = []
        for element in elements:
            # 提取位置特征
            pos_features = [
                element.get('center', (0, 0))[0],
                element.get('center', (0, 0))[1],
                element.get('area', 0),
                element.get('aspect_ratio', 0)
            ]
            
            # 提取类型特征
            type_features = [1 if 'arrow' in element else 0,
                           1 if 'box' in element else 0]
            
            features.append(pos_features + type_features)
        
        return torch.tensor(features, dtype=torch.float32) 