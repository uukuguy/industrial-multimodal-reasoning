# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class UncertaintyEstimator(nn.Module):
    """
    不确定性估计模块，为预测结果提供置信度评分
    """
    
    def __init__(self, embedding_dim: int, num_classes: Optional[int] = 4, 
                hidden_dim: int = 256, dropout: float = 0.1, temperature: float = 1.5):
        """
        初始化不确定性估计模块
        
        Args:
            embedding_dim: 输入嵌入的维度
            num_classes: 分类任务的类别数量（如初赛的ABCD选项）
            hidden_dim: 隐藏层维度
            dropout: Dropout率
            temperature: 温度缩放参数，用于校准置信度
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.temperature = temperature
        
        # 任务不确定性估计
        self.task_uncertainty = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 确保不确定性为正值
        )
        
        # 对于分类任务的不确定性（适用于初赛-单选题）
        if num_classes:
            # 分类头
            self.classification_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
            
            # 每个选项的不确定性估计
            self.classification_uncertainty = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
                nn.Softplus()  # 确保不确定性为正值
            )
            
            # 用于开放式问答的复赛任务
            self.qa_confidence = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()  # 归一化到 [0, 1]
            )
    
    def forward(self, embedding: torch.Tensor, is_classification: bool = True) -> Dict[str, torch.Tensor]:
        """
        前向传播，计算不确定性估计
        
        Args:
            embedding: 输入嵌入 [batch_size, embedding_dim]
            is_classification: 是否为分类任务（初赛）
            
        Returns:
            包含不确定性估计的字典
        """
        result = {}
        
        # 任务级不确定性
        task_uncertainty = self.task_uncertainty(embedding)
        result['task_uncertainty'] = task_uncertainty
        
        # 分类任务不确定性（如果适用）
        if is_classification and self.num_classes:
            # 计算选项的logits
            logits = self.classification_head(embedding)
            result['logits'] = logits
            
            # 计算每个选项的不确定性
            option_uncertainties = self.classification_uncertainty(embedding)
            result['option_uncertainties'] = option_uncertainties
            
            # 温度缩放校准
            calibrated_probs = F.softmax(logits / self.temperature, dim=1)
            result['calibrated_probs'] = calibrated_probs
            
            # 最可能的答案及其置信度
            max_prob_indices = torch.argmax(calibrated_probs, dim=1)
            result['predicted_class'] = max_prob_indices
            
            # 对应到 A, B, C, D 选项
            option_names = ['A', 'B', 'C', 'D']
            if self.num_classes <= len(option_names):
                result['predicted_option'] = [option_names[idx.item()] for idx in max_prob_indices]
                
            # 收集每个样本的最高置信度
            max_probs = torch.gather(calibrated_probs, 1, max_prob_indices.unsqueeze(1))
            result['max_confidence'] = max_probs
            
        else:
            # 开放式问答的置信度（复赛）
            qa_confidence = self.qa_confidence(embedding)
            result['qa_confidence'] = qa_confidence
            
        return result
    
    def predict_with_uncertainty(self, 
                               embedding: torch.Tensor, 
                               options: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        根据嵌入生成预测结果和不确定性估计
        
        Args:
            embedding: 输入嵌入 [batch_size, embedding_dim]
            options: 选项列表 （如 ['A', 'B', 'C', 'D']）
            
        Returns:
            包含预测结果和不确定性的字典
        """
        # 判断任务类型
        is_classification = options is not None and len(options) > 0
        
        # 获取基本不确定性估计
        result = self.forward(embedding, is_classification)
        
        # 处理分类结果（初赛）
        if is_classification and self.num_classes:
            # 确保选项数量与模型匹配
            if len(options) != self.num_classes:
                logger.warning(f"选项数量 ({len(options)}) 与模型类别数 ({self.num_classes}) 不匹配")
            
            # 获取预测的选项索引
            pred_indices = result['predicted_class']
            
            # 获取对应的选项
            pred_options = [options[min(idx.item(), len(options)-1)] for idx in pred_indices]
            result['prediction'] = pred_options[0] if len(pred_options) == 1 else pred_options
            
            # 添加每个选项的置信度
            for i, opt in enumerate(options[:self.num_classes]):
                result[f'confidence_{opt}'] = result['calibrated_probs'][0, i].item()
                
            # 判断是否需要人工审核（低置信度）
            confidence_threshold = 0.7  # 可配置的阈值
            result['needs_review'] = result['max_confidence'].item() < confidence_threshold
            
        else:
            # 开放式问答（复赛）
            result['confidence'] = result['qa_confidence'].item()
            
        return result
    
    def calibrate(self, validation_embeddings: torch.Tensor, 
                validation_labels: torch.Tensor, 
                max_iter: int = 100, lr: float = 0.01) -> float:
        """
        使用温度缩放校准模型的置信度
        
        Args:
            validation_embeddings: 验证集嵌入
            validation_labels: 验证集标签
            max_iter: 最大迭代次数
            lr: 学习率
            
        Returns:
            校准后的温度参数
        """
        if not self.num_classes:
            logger.warning("无法校准非分类模型")
            return self.temperature
            
        # 将嵌入传入分类头获取logits
        with torch.no_grad():
            logits = self.classification_head(validation_embeddings)
        
        # 创建一个可学习的温度参数
        temperature = nn.Parameter(torch.ones(1) * self.temperature)
        
        # 使用优化器
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            # 应用温度缩放
            scaled_logits = logits / temperature
            # 计算负对数似然损失
            loss = F.cross_entropy(scaled_logits, validation_labels)
            loss.backward()
            return loss
            
        # 优化温度参数
        optimizer.step(eval_loss)
        
        # 更新模型的温度参数
        self.temperature = temperature.item()
        logger.info(f"置信度校准完成，温度参数: {self.temperature:.4f}")
        
        return self.temperature


class OptionSelector:
    """选项选择器，用于将模型输出映射到选项"""
    
    @staticmethod
    def select_option(output: str, options: List[str]) -> str:
        """
        从选项中选择最匹配的一个
        
        Args:
            output: 模型输出文本
            options: 选项列表，如 ['A', 'B', 'C', 'D'] 或完整选项文本
            
        Returns:
            选中的选项
        """
        # 如果输出直接包含选项字母（如 "A", "B"）
        if len(output) == 1 and output.upper() in "ABCD":
            option_idx = ord(output.upper()) - ord('A')
            if option_idx < len(options):
                return options[option_idx]
            
        # 如果输出是数字
        if output.isdigit():
            option_idx = int(output) - 1  # 转换为0-based索引
            if 0 <= option_idx < len(options):
                return options[option_idx]
                
        # 否则，寻找与选项文本的最佳匹配
        # 简单方法：检查输出是否包含选项文本
        for option in options:
            if option in output or output in option:
                return option
                
        # 如果无法确定，返回第一个选项
        return options[0]


if __name__ == "__main__":
    # 测试不确定性估计模块
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    embedding_dim = 768
    batch_size = 2
    num_classes = 4  # A, B, C, D
    
    embeddings = torch.randn(batch_size, embedding_dim)
    options = ['A', 'B', 'C', 'D']
    
    # 初始化不确定性估计器
    estimator = UncertaintyEstimator(embedding_dim, num_classes)
    
    # 测试前向传播
    print("测试不确定性估计...")
    results = estimator(embeddings)
    
    print(f"任务不确定性形状: {results['task_uncertainty'].shape}")
    print(f"logits形状: {results['logits'].shape}")
    print(f"选项不确定性形状: {results['option_uncertainties'].shape}")
    print(f"校准概率形状: {results['calibrated_probs'].shape}")
    print(f"预测类别: {results['predicted_class']}")
    
    # 测试预测
    print("\n测试预测与不确定性...")
    pred_results = estimator.predict_with_uncertainty(embeddings[0:1], options)
    
    print(f"预测结果: {pred_results['prediction']}")
    print(f"最大置信度: {pred_results['max_confidence'].item():.4f}")
    for opt in options:
        print(f"选项 {opt} 置信度: {pred_results[f'confidence_{opt}']:.4f}")
    print(f"需要人工审核: {pred_results['needs_review']}")
    
    # 测试选项选择器
    print("\n测试选项选择器...")
    outputs = ["A", "选项B似乎是最合适的", "3", "我认为D是正确答案"]
    
    for output in outputs:
        selected = OptionSelector.select_option(output, options)
        print(f"输出: '{output}' -> 选择: '{selected}'")