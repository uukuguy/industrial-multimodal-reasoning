import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import paddleocr
from paddleocr import PaddleOCR
import easyocr
from PIL import Image
import os
import torchvision.transforms as transforms
import logging

try:
    from paddleocr import PaddleOCR
except ImportError:
    logging.warning('PaddleOCR not installed, using EasyOCR only')
    PaddleOCR = None

try:
    import easyocr
except ImportError:
    logging.warning('EasyOCR not installed, using PaddleOCR only')
    easyocr = None

class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.target_size = config.get('target_size', (384, 384))
        self.denoise_strength = config.get('denoise_strength', 10)
        self.sharpen_strength = config.get('sharpen_strength', 1.5)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # 调整大小
        image = cv2.resize(image, self.target_size)
        
        # 降噪
        image = cv2.fastNlMeansDenoisingColored(
            image, None, self.denoise_strength, self.denoise_strength, 7, 21)
        
        # 锐化
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) * self.sharpen_strength
        image = cv2.filter2D(image, -1, kernel)
        
        # 对比度增强
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return image

class OCRPostprocessor:
    """OCR后处理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.min_text_length = config.get('min_text_length', 2)
        self.max_text_length = config.get('max_text_length', 50)
    
    def postprocess(self, results: List[Dict]) -> List[Dict]:
        """后处理OCR结果"""
        processed_results = []
        
        for result in results:
            # 置信度过滤
            if result['confidence'] < self.confidence_threshold:
                continue
            
            # 文本长度过滤
            text = result['text']
            if not (self.min_text_length <= len(text) <= self.max_text_length):
                continue
            
            # 文本清理
            text = self._clean_text(text)
            if not text:
                continue
            
            result['text'] = text
            processed_results.append(result)
        
        return processed_results
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除特殊字符
        text = ''.join(c for c in text if c.isprintable())
        
        # 移除多余空格
        text = ' '.join(text.split())
        
        return text

class EnhancedOCR:
    """增强型OCR"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ocr_config = config.get('ocr', {})
        
        # 初始化预处理器
        self.preprocessor = ImagePreprocessor(
            self.ocr_config.get('preprocessing', {}))
        
        # 初始化后处理器
        self.postprocessor = OCRPostprocessor(
            self.ocr_config.get('postprocessing', {}))
        
        # 初始化PaddleOCR
        if self.ocr_config.get('use_paddleocr', True):
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                use_gpu=torch.cuda.is_available(),
                show_log=False
            )
        
        # 初始化EasyOCR
        if self.ocr_config.get('use_easyocr', True):
            self.easy_ocr = easyocr.Reader(
                ['ch_sim', 'en'],
                gpu=torch.cuda.is_available()
            )
    
    def recognize(self, image: np.ndarray) -> List[Dict]:
        """识别图像中的文本"""
        # 预处理
        processed_image = self.preprocessor.preprocess(image)
        
        results = []
        
        # PaddleOCR识别
        if hasattr(self, 'paddle_ocr'):
            paddle_results = self.paddle_ocr.ocr(processed_image, cls=True)
            if paddle_results:
                for line in paddle_results[0]:
                    box, (text, confidence) = line
                    results.append({
                        'text': text,
                        'confidence': confidence,
                        'box': box,
                        'source': 'paddle'
                    })
        
        # EasyOCR识别
        if hasattr(self, 'easy_ocr'):
            easy_results = self.easy_ocr.readtext(processed_image)
            for box, text, confidence in easy_results:
                results.append({
                    'text': text,
                    'confidence': confidence,
                    'box': box,
                    'source': 'easy'
                })
        
        # 后处理
        processed_results = self.postprocessor.postprocess(results)
        
        return processed_results

class OCRFeatureExtractor(nn.Module):
    """OCR特征提取器"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # 文本编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 位置编码器
        self.position_encoder = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def forward(self, ocr_results: List[Dict]) -> torch.Tensor:
        """提取OCR结果的特征"""
        features = []
        
        for result in ocr_results:
            # 文本特征
            text = result['text']
            text_feature = self._encode_text(text)
            
            # 位置特征
            box = result['box']
            position_feature = self._encode_position(box)
            
            # 合并特征
            feature = text_feature + position_feature
            features.append(feature)
        
        return torch.stack(features)
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """编码文本"""
        # 这里使用简单的字符编码，实际应用中可以使用更复杂的文本编码器
        chars = list(text)
        char_ids = [ord(c) for c in chars]
        char_tensor = torch.tensor(char_ids, dtype=torch.float32)
        return self.text_encoder(char_tensor)
    
    def _encode_position(self, box: List[Tuple[float, float]]) -> torch.Tensor:
        """编码位置信息"""
        # 计算边界框的中心点和宽高
        x1, y1 = box[0]
        x2, y2 = box[2]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        position = torch.tensor([center_x, center_y, width, height], dtype=torch.float32)
        return self.position_encoder(position)

class OCRPreprocessor:
    """OCR预处理器"""
    
    def __init__(self, config: Dict):
        """初始化OCR预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.preprocessing_config = config['model']['ocr_enhancement']['preprocessing']
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像
        
        Args:
            image: 输入图像
        
        Returns:
            预处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 应用预处理
        if self.preprocessing_config.get('denoising', False):
            gray = self._denoise(gray)
        
        if self.preprocessing_config.get('contrast_enhancement', False):
            gray = self._enhance_contrast(gray)
        
        if self.preprocessing_config.get('binarization', False):
            gray = self._binarize(gray)
        
        if self.preprocessing_config.get('deskewing', False):
            gray = self._deskew(gray)
        
        return gray
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """降噪
        
        Args:
            image: 输入图像
        
        Returns:
            降噪后的图像
        """
        return cv2.fastNlMeansDenoising(
            image,
            None,
            h=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """增强对比度
        
        Args:
            image: 输入图像
        
        Returns:
            增强后的图像
        """
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        return clahe.apply(image)
    
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """二值化
        
        Args:
            image: 输入图像
        
        Returns:
            二值化后的图像
        """
        return cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """倾斜校正
        
        Args:
            image: 输入图像
        
        Returns:
            校正后的图像
        """
        # 查找轮廓
        contours, _ = cv2.findContours(
            image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return image
        
        # 找到最大的轮廓
        max_contour = max(contours, key=cv2.contourArea)
        
        # 计算最小外接矩形
        rect = cv2.minAreaRect(max_contour)
        angle = rect[2]
        
        # 旋转图像
        if angle < -45:
            angle = 90 + angle
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated

class OCREncoder(nn.Module):
    """OCR编码器"""
    
    def __init__(self, config: Dict):
        """初始化OCR编码器
        
        Args:
            config: 配置字典
        """
        super().__init__()
        self.config = config
        
        # 初始化OCR引擎
        self._init_ocr_engines()
        
        # 初始化预处理器和后处理器
        self.preprocessor = OCRPreprocessor(config)
        self.postprocessor = OCRPostprocessor(config)
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
    
    def _init_ocr_engines(self):
        """初始化OCR引擎"""
        self.engines = []
        
        # 初始化PaddleOCR
        if PaddleOCR is not None:
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                show_log=False
            )
            self.engines.append('paddle')
        
        # 初始化EasyOCR
        if easyocr is not None:
            self.easy_ocr = easyocr.Reader(['ch_sim', 'en'])
            self.engines.append('easy')
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            images: 输入图像 [batch_size, channels, height, width]
        
        Returns:
            OCR特征
        """
        batch_size = images.shape[0]
        ocr_features = []
        
        for i in range(batch_size):
            # 转换为numpy数组
            image = images[i].cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
            image = (image * 255).astype(np.uint8)
            
            # 预处理
            processed_image = self.preprocessor.preprocess(image)
            
            # OCR识别
            results = self._recognize_text(processed_image)
            
            # 后处理
            processed_results = self.postprocessor.postprocess(results)
            
            # 提取特征
            feature = self._extract_ocr_feature(processed_results)
            ocr_features.append(feature)
        
        # 堆叠特征
        ocr_features = torch.stack(ocr_features)
        
        return ocr_features
    
    def _recognize_text(self, image: np.ndarray) -> List[Dict]:
        """识别文本
        
        Args:
            image: 输入图像
        
        Returns:
            OCR结果列表
        """
        all_results = []
        
        # PaddleOCR识别
        if 'paddle' in self.engines:
            paddle_results = self.paddle_ocr.ocr(image, cls=True)
            if paddle_results is not None:
                for line in paddle_results:
                    for result in line:
                        box, (text, confidence) = result
                        all_results.append({
                            'text': text,
                            'box': box,
                            'confidence': confidence
                        })
        
        # EasyOCR识别
        if 'easy' in self.engines:
            easy_results = self.easy_ocr.readtext(image)
            for box, text, confidence in easy_results:
                all_results.append({
                    'text': text,
                    'box': box,
                    'confidence': confidence
                })
        
        return all_results
    
    def _extract_ocr_feature(
        self,
        results: List[Dict]
    ) -> torch.Tensor:
        """提取OCR特征
        
        Args:
            results: OCR结果列表
        
        Returns:
            OCR特征
        """
        # 初始化特征向量
        feature = torch.zeros(512)
        
        if not results:
            return feature
        
        # 提取文本特征
        text_features = []
        for result in results:
            text = result['text']
            # 使用简单的字符统计作为特征
            char_counts = torch.zeros(256)
            for c in text:
                if ord(c) < 256:
                    char_counts[ord(c)] += 1
            text_features.append(char_counts)
        
        # 提取位置特征
        box_features = []
        for result in results:
            box = result['box']
            # 计算边界框的中心点和大小
            center_x = sum(p[0] for p in box) / 4
            center_y = sum(p[1] for p in box) / 4
            width = max(p[0] for p in box) - min(p[0] for p in box)
            height = max(p[1] for p in box) - min(p[1] for p in box)
            box_features.append(torch.tensor([
                center_x,
                center_y,
                width,
                height
            ]))
        
        # 合并特征
        if text_features:
            text_features = torch.stack(text_features)
            feature[:256] = text_features.mean(dim=0)
        
        if box_features:
            box_features = torch.stack(box_features)
            feature[256:] = box_features.mean(dim=0)
        
        # 通过特征提取器
        feature = self.feature_extractor(feature)
        
        return feature 