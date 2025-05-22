import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class EnsembleMethod:
    """集成方法"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ensemble_config = config.get('ensemble', {})
        
        # 集成方法类型
        self.method = self.ensemble_config.get('method', 'voting')
        
        # 模型权重
        self.weights = self.ensemble_config.get('weights', None)
        
        # 投票阈值
        self.voting_threshold = self.ensemble_config.get('voting_threshold', 0.5)
    
    def ensemble(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """集成多个模型的预测结果"""
        if self.method == 'voting':
            return self._voting_ensemble(predictions)
        elif self.method == 'weighted':
            return self._weighted_ensemble(predictions)
        else:
            raise ValueError(f"Unsupported ensemble method: {self.method}")
    
    def _voting_ensemble(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """投票集成"""
        # 转换为numpy数组
        pred_np = [p.cpu().numpy() for p in predictions]
        
        # 创建投票分类器
        voting_clf = VotingClassifier(
            estimators=[(f'model_{i}', p) for i, p in enumerate(pred_np)],
            voting='soft',
            weights=self.weights
        )
        
        # 进行投票
        ensemble_pred = voting_clf.predict_proba(pred_np[0])
        
        return torch.from_numpy(ensemble_pred)
    
    def _weighted_ensemble(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """加权集成"""
        if self.weights is None:
            self.weights = [1.0 / len(predictions)] * len(predictions)
        
        # 加权求和
        weighted_sum = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_sum += weight * pred
        
        return weighted_sum

class ConfidenceThresholding:
    """置信度阈值处理"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.threshold_config = config.get('confidence_thresholding', {})
        
        # 置信度阈值
        self.threshold = self.threshold_config.get('threshold', 0.5)
        
        # 是否使用动态阈值
        self.use_dynamic = self.threshold_config.get('use_dynamic', False)
        
        # 动态阈值参数
        self.adaptive_rate = self.threshold_config.get('adaptive_rate', 0.1)
        self.min_threshold = self.threshold_config.get('min_threshold', 0.3)
        self.max_threshold = self.threshold_config.get('max_threshold', 0.9)
    
    def process(self, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """处理预测结果"""
        # 获取置信度
        confidence = F.softmax(predictions, dim=1)
        
        # 应用阈值
        if self.use_dynamic:
            threshold = self._get_dynamic_threshold(confidence)
        else:
            threshold = self.threshold
        
        # 过滤低置信度预测
        mask = confidence.max(dim=1)[0] >= threshold
        filtered_pred = predictions[mask]
        filtered_conf = confidence[mask]
        
        return filtered_pred, filtered_conf
    
    def _get_dynamic_threshold(self, confidence: torch.Tensor) -> float:
        """获取动态阈值"""
        # 计算当前批次的平均置信度
        mean_conf = confidence.mean().item()
        
        # 根据平均置信度调整阈值
        if mean_conf < self.min_threshold:
            threshold = self.min_threshold
        elif mean_conf > self.max_threshold:
            threshold = self.max_threshold
        else:
            threshold = mean_conf
        
        # 更新阈值
        self.threshold = (1 - self.adaptive_rate) * self.threshold + self.adaptive_rate * threshold
        
        return self.threshold

class PostProcessor:
    """后处理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 初始化集成方法
        self.ensemble = EnsembleMethod(config)
        
        # 初始化置信度阈值处理
        self.thresholding = ConfidenceThresholding(config)
        
        # 后处理配置
        self.post_config = config.get('post_processing', {})
        
        # 是否使用集成
        self.use_ensemble = self.post_config.get('use_ensemble', True)
        
        # 是否使用置信度阈值
        self.use_thresholding = self.post_config.get('use_thresholding', True)
    
    def process(self, predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """处理预测结果"""
        # 集成多个模型的预测
        if self.use_ensemble and len(predictions) > 1:
            predictions = [self.ensemble.ensemble(predictions)]
        
        # 应用置信度阈值
        if self.use_thresholding:
            processed_pred, confidence = self.thresholding.process(predictions[0])
        else:
            processed_pred = predictions[0]
            confidence = F.softmax(processed_pred, dim=1)
        
        return processed_pred, confidence
    
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """评估后处理效果"""
        # 转换为numpy数组
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(target_np, pred_np),
            'precision': precision_score(target_np, pred_np, average='weighted'),
            'recall': recall_score(target_np, pred_np, average='weighted'),
            'f1': f1_score(target_np, pred_np, average='weighted')
        }
        
        return metrics 