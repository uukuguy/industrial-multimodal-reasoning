# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class UncertaintyEstimator(nn.Module):
    """
    增强型不确定性估计模块，提供多层次的不确定性评估
    """
    
    def __init__(self, embedding_dim: int, num_classes: Optional[int] = 4, 
                hidden_dim: int = 256, dropout: float = 0.1, temperature: float = 1.5,
                use_mc_dropout: bool = True, n_mc_samples: int = 10,
                use_evidential: bool = True):
        """
        初始化增强型不确定性估计模块
        
        Args:
            embedding_dim: 输入嵌入的维度
            num_classes: 分类任务的类别数量
            hidden_dim: 隐藏层维度
            dropout: Dropout率
            temperature: 温度缩放参数
            use_mc_dropout: 是否使用Monte Carlo Dropout
            n_mc_samples: Monte Carlo采样次数
            use_evidential: 是否使用证据深度学习
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.temperature = temperature
        self.use_mc_dropout = use_mc_dropout
        self.n_mc_samples = n_mc_samples
        self.use_evidential = use_evidential
        
        # 任务级不确定性
        self.task_uncertainty = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # 模态级不确定性估计
        self.modal_uncertainty = nn.ModuleDict({
            'text': self._build_uncertainty_net(embedding_dim, hidden_dim, dropout),
            'visual': self._build_uncertainty_net(embedding_dim, hidden_dim, dropout),
            'layout': self._build_uncertainty_net(embedding_dim, hidden_dim, dropout)
        })
        
        # 特征级不确定性估计
        self.feature_uncertainty = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Softplus()
        )
        
        # 对于分类任务的不确定性
        if num_classes:
            # 分类头
            self.classification_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
            
            # 证据深度学习分类头
            if use_evidential:
                self.evidential_head = nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes * 2)  # 均值和方差
                )
            
            # 每个选项的不确定性估计
            self.classification_uncertainty = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
                nn.Softplus()
            )
            
            # 推理路径不确定性
            self.reasoning_uncertainty = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
    
    def _build_uncertainty_net(self, input_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
        """构建不确定性估计网络"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
    
    def _monte_carlo_dropout(self, x: torch.Tensor, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行Monte Carlo Dropout采样"""
        predictions = []
        for _ in range(self.n_mc_samples):
            pred = model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)
        
        return mean_pred, uncertainty
    
    def _evidential_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算证据深度学习的不确定性"""
        evidence = self.evidential_head(x)
        alpha = evidence + 1  # 证据参数
        S = alpha.sum(dim=-1, keepdim=True)  # 总证据
        
        # 计算不确定性
        uncertainty = self.num_classes / S
        
        # 计算概率
        probs = alpha / S
        
        return {
            'evidence': evidence,
            'alpha': alpha,
            'uncertainty': uncertainty,
            'probs': probs
        }
    
    def forward(self, embedding: torch.Tensor, modal_embeddings: Optional[Dict[str, torch.Tensor]] = None,
                is_classification: bool = True) -> Dict[str, torch.Tensor]:
        """
        前向传播，计算多层次的不确定性估计
        
        Args:
            embedding: 输入嵌入 [batch_size, embedding_dim]
            modal_embeddings: 各模态的嵌入字典
            is_classification: 是否为分类任务
            
        Returns:
            包含多层次不确定性估计的字典
        """
        result = {}
        
        # 1. 任务级不确定性
        task_uncertainty = self.task_uncertainty(embedding)
        result['task_uncertainty'] = task_uncertainty
        
        # 2. 模态级不确定性
        if modal_embeddings:
            modal_uncertainties = {}
            for modal_name, modal_embedding in modal_embeddings.items():
                if modal_name in self.modal_uncertainty:
                    if self.use_mc_dropout:
                        mean_pred, uncertainty = self._monte_carlo_dropout(
                            modal_embedding, 
                            self.modal_uncertainty[modal_name]
                        )
                        modal_uncertainties[modal_name] = {
                            'mean': mean_pred,
                            'uncertainty': uncertainty
                        }
                    else:
                        modal_uncertainties[modal_name] = self.modal_uncertainty[modal_name](modal_embedding)
            result['modal_uncertainties'] = modal_uncertainties
        
        # 3. 特征级不确定性
        feature_uncertainty = self.feature_uncertainty(embedding)
        result['feature_uncertainty'] = feature_uncertainty
        
        # 4. 分类任务不确定性
        if is_classification and self.num_classes:
            # 基础分类预测
            logits = self.classification_head(embedding)
            result['logits'] = logits
            
            # 证据深度学习不确定性
            if self.use_evidential:
                evidential_result = self._evidential_uncertainty(embedding)
                result.update(evidential_result)
            
            # 选项不确定性
            option_uncertainties = self.classification_uncertainty(embedding)
            result['option_uncertainties'] = option_uncertainties
            
            # 推理路径不确定性
            reasoning_uncertainty = self.reasoning_uncertainty(embedding)
            result['reasoning_uncertainty'] = reasoning_uncertainty
            
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
        
        return result
    
    def predict_with_uncertainty(self, 
                               embedding: torch.Tensor, 
                               modal_embeddings: Optional[Dict[str, torch.Tensor]] = None,
                               options: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        根据嵌入生成预测结果和不确定性估计
        
        Args:
            embedding: 输入嵌入
            modal_embeddings: 各模态的嵌入字典
            options: 选项列表
            
        Returns:
            包含预测结果和不确定性的字典
        """
        # 判断任务类型
        is_classification = options is not None and len(options) > 0
        
        # 获取基本不确定性估计
        result = self.forward(embedding, modal_embeddings, is_classification)
        
        # 处理分类结果
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
            
            # 判断是否需要人工审核
            confidence_threshold = 0.7  # 可配置的阈值
            result['needs_review'] = result['max_confidence'].item() < confidence_threshold
            
            # 添加模态贡献度分析
            if 'modal_uncertainties' in result:
                modal_contributions = {}
                for modal_name, modal_uncertainty in result['modal_uncertainties'].items():
                    if isinstance(modal_uncertainty, dict):
                        uncertainty = modal_uncertainty['uncertainty'].item()
                    else:
                        uncertainty = modal_uncertainty.item()
                    modal_contributions[modal_name] = 1.0 - uncertainty
                result['modal_contributions'] = modal_contributions
        
        return result
    
    def visualize_uncertainty(self, 
                            predictions: torch.Tensor,
                            uncertainties: Dict[str, torch.Tensor],
                            save_path: Optional[str] = None) -> None:
        """
        可视化不确定性分析结果
        
        Args:
            predictions: 预测结果
            uncertainties: 不确定性估计字典
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 创建图形
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 预测与不确定性关系图
        ax1 = plt.subplot(221)
        ax1.errorbar(
            range(len(predictions)),
            predictions.cpu().numpy(),
            yerr=uncertainties['task_uncertainty'].cpu().numpy(),
            fmt='o',
            ecolor='red',
            capsize=5
        )
        ax1.set_title('Predictions with Task Uncertainty')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Predicted Value')
        
        # 2. 模态贡献度热力图
        if 'modal_contributions' in uncertainties:
            ax2 = plt.subplot(222)
            modal_data = pd.DataFrame(uncertainties['modal_contributions'])
            sns.heatmap(modal_data, annot=True, cmap='YlOrRd', ax=ax2)
            ax2.set_title('Modal Contributions')
        
        # 3. 特征不确定性分布
        ax3 = plt.subplot(223)
        sns.histplot(uncertainties['feature_uncertainty'].cpu().numpy().flatten(), ax=ax3)
        ax3.set_title('Feature Uncertainty Distribution')
        
        # 4. 推理路径不确定性
        if 'reasoning_uncertainty' in uncertainties:
            ax4 = plt.subplot(224)
            sns.histplot(uncertainties['reasoning_uncertainty'].cpu().numpy().flatten(), ax=ax4)
            ax4.set_title('Reasoning Path Uncertainty')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"不确定性分析图已保存到: {save_path}")
        
        plt.close()


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