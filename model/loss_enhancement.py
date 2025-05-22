import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math

class ContrastiveLoss(nn.Module):
    """对比损失"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.temperature = config.get('temperature', 0.07)
        self.margin = config.get('margin', 0.5)
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T)
        
        # 创建标签矩阵
        labels = labels.view(-1, 1)
        label_matrix = (labels == labels.T).float()
        
        # 计算正样本对的损失
        positive_pairs = similarity_matrix * label_matrix
        positive_loss = -torch.log(torch.exp(positive_pairs / self.temperature) + 1e-8)
        
        # 计算负样本对的损失
        negative_pairs = similarity_matrix * (1 - label_matrix)
        negative_loss = torch.clamp(self.margin - negative_pairs, min=0)
        
        # 计算总损失
        loss = positive_loss.mean() + negative_loss.mean()
        
        return loss

class TripletLoss(nn.Module):
    """三元组损失"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.margin = config.get('margin', 1.0)
        self.distance_metric = config.get('distance_metric', 'euclidean')
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 计算距离矩阵
        if self.distance_metric == 'euclidean':
            dist_matrix = torch.cdist(features, features)
        else:  # cosine
            features = F.normalize(features, dim=1)
            dist_matrix = 1 - torch.matmul(features, features.T)
        
        # 创建标签矩阵
        labels = labels.view(-1, 1)
        label_matrix = (labels == labels.T).float()
        
        # 找到每个样本的正样本和负样本
        positive_mask = label_matrix
        negative_mask = 1 - label_matrix
        
        # 计算三元组损失
        loss = 0
        for i in range(len(features)):
            # 找到正样本
            positive_indices = torch.where(positive_mask[i])[0]
            if len(positive_indices) == 0:
                continue
            
            # 找到负样本
            negative_indices = torch.where(negative_mask[i])[0]
            if len(negative_indices) == 0:
                continue
            
            # 计算与正样本的距离
            positive_dist = dist_matrix[i, positive_indices]
            
            # 计算与负样本的距离
            negative_dist = dist_matrix[i, negative_indices]
            
            # 计算三元组损失
            for pos_dist in positive_dist:
                for neg_dist in negative_dist:
                    loss += F.relu(pos_dist - neg_dist + self.margin)
        
        return loss / (len(features) + 1e-8)

class FocalLoss(nn.Module):
    """Focal Loss"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.gamma = config.get('gamma', 2.0)
        self.alpha = config.get('alpha', 0.25)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 计算交叉熵
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # 计算pt
        pt = torch.exp(-ce_loss)
        
        # 计算focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.smoothing = config.get('smoothing', 0.1)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        n_classes = pred.size(1)
        
        # 创建平滑标签
        smooth_target = torch.zeros_like(pred)
        smooth_target.fill_(self.smoothing / (n_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        
        # 计算交叉熵
        log_prob = F.log_softmax(pred, dim=1)
        loss = (-smooth_target * log_prob).sum(dim=1).mean()
        
        return loss

class EnhancedLossModule(nn.Module):
    """增强型损失模块"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 初始化各种损失函数
        self.contrastive_loss = ContrastiveLoss(config.get('contrastive', {}))
        self.triplet_loss = TripletLoss(config.get('triplet', {}))
        self.focal_loss = FocalLoss(config.get('focal', {}))
        self.label_smoothing_loss = LabelSmoothingLoss(config.get('label_smoothing', {}))
        
        # 损失权重
        self.contrastive_weight = config.get('contrastive_weight', 0.1)
        self.triplet_weight = config.get('triplet_weight', 0.1)
        self.focal_weight = config.get('focal_weight', 0.4)
        self.label_smoothing_weight = config.get('label_smoothing_weight', 0.4)
    
    def forward(self, 
                pred: torch.Tensor,
                target: torch.Tensor,
                features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        losses = {}
        
        # 计算各种损失
        if self.contrastive_weight > 0:
            losses['contrastive'] = self.contrastive_loss(features, target)
        
        if self.triplet_weight > 0:
            losses['triplet'] = self.triplet_loss(features, target)
        
        if self.focal_weight > 0:
            losses['focal'] = self.focal_loss(pred, target)
        
        if self.label_smoothing_weight > 0:
            losses['label_smoothing'] = self.label_smoothing_loss(pred, target)
        
        # 计算总损失
        total_loss = sum(weight * loss for weight, loss in zip(
            [self.contrastive_weight, self.triplet_weight, 
             self.focal_weight, self.label_smoothing_weight],
            [losses.get('contrastive', 0), losses.get('triplet', 0),
             losses.get('focal', 0), losses.get('label_smoothing', 0)]
        ))
        
        losses['total'] = total_loss
        
        return losses 