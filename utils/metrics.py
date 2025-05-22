import numpy as np
from typing import Dict, List, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(
    predictions: Union[List, np.ndarray],
    targets: Union[List, np.ndarray],
    average: str = 'weighted'
) -> Dict[str, float]:
    """计算评估指标
    
    Args:
        predictions: 模型预测结果
        targets: 真实标签
        average: 计算多分类指标时的平均方式，可选 'micro', 'macro', 'weighted'
    
    Returns:
        包含各项指标的字典
    """
    # 转换为numpy数组
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 计算各项指标
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average=average, zero_division=0)
    recall = recall_score(targets, predictions, average=average, zero_division=0)
    f1 = f1_score(targets, predictions, average=average, zero_division=0)
    
    # 返回指标字典
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_class_metrics(
    predictions: Union[List, np.ndarray],
    targets: Union[List, np.ndarray],
    labels: List[str]
) -> Dict[str, Dict[str, float]]:
    """计算每个类别的评估指标
    
    Args:
        predictions: 模型预测结果
        targets: 真实标签
        labels: 类别标签列表
    
    Returns:
        包含每个类别指标的字典
    """
    # 转换为numpy数组
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 计算每个类别的指标
    class_metrics = {}
    for i, label in enumerate(labels):
        # 将当前类别设为正类，其他类别设为负类
        pred_binary = (predictions == i).astype(int)
        target_binary = (targets == i).astype(int)
        
        # 计算二分类指标
        precision = precision_score(target_binary, pred_binary, zero_division=0)
        recall = recall_score(target_binary, pred_binary, zero_division=0)
        f1 = f1_score(target_binary, pred_binary, zero_division=0)
        
        # 存储当前类别的指标
        class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return class_metrics

def calculate_confusion_matrix(
    predictions: Union[List, np.ndarray],
    targets: Union[List, np.ndarray],
    labels: List[str]
) -> np.ndarray:
    """计算混淆矩阵
    
    Args:
        predictions: 模型预测结果
        targets: 真实标签
        labels: 类别标签列表
    
    Returns:
        混淆矩阵
    """
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(targets, predictions, labels=range(len(labels)))

def print_metrics(metrics: Dict[str, float], prefix: str = ''):
    """打印评估指标
    
    Args:
        metrics: 评估指标字典
        prefix: 打印前缀
    """
    print(f'{prefix}Accuracy: {metrics["accuracy"]:.4f}')
    print(f'{prefix}Precision: {metrics["precision"]:.4f}')
    print(f'{prefix}Recall: {metrics["recall"]:.4f}')
    print(f'{prefix}F1 Score: {metrics["f1"]:.4f}')

def print_class_metrics(class_metrics: Dict[str, Dict[str, float]], prefix: str = ''):
    """打印每个类别的评估指标
    
    Args:
        class_metrics: 类别指标字典
        prefix: 打印前缀
    """
    for label, metrics in class_metrics.items():
        print(f'\n{prefix}Class: {label}')
        print(f'{prefix}Precision: {metrics["precision"]:.4f}')
        print(f'{prefix}Recall: {metrics["recall"]:.4f}')
        print(f'{prefix}F1 Score: {metrics["f1"]:.4f}')

def save_metrics(
    metrics: Dict[str, float],
    class_metrics: Dict[str, Dict[str, float]],
    confusion_matrix: np.ndarray,
    labels: List[str],
    save_path: str
):
    """保存评估指标到文件
    
    Args:
        metrics: 总体评估指标
        class_metrics: 类别指标
        confusion_matrix: 混淆矩阵
        labels: 类别标签列表
        save_path: 保存路径
    """
    import json
    import numpy as np
    
    # 将numpy数组转换为列表
    confusion_matrix = confusion_matrix.tolist()
    
    # 保存指标
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({
            'overall_metrics': metrics,
            'class_metrics': class_metrics,
            'confusion_matrix': confusion_matrix,
            'labels': labels
        }, f, indent=4, ensure_ascii=False) 