import torch
import torchvision.transforms as T
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
from PIL import Image
import numpy as np
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def get_image_transforms(
    image_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    is_train: bool = True
) -> T.Compose:
    """获取图像变换
    
    Args:
        image_size: 图像大小
        mean: 均值
        std: 标准差
        is_train: 是否为训练模式
        
    Returns:
        图像变换
    """
    if is_train:
        return T.Compose([
            T.RandomResizedCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

def get_layout_transforms(
    max_boxes: int = 100,
    normalize: bool = True,
    augment: bool = True
) -> Dict[str, Any]:
    """获取布局变换
    
    Args:
        max_boxes: 最大框数
        normalize: 是否归一化
        augment: 是否进行数据增强
        
    Returns:
        布局变换
    """
    return {
        "max_boxes": max_boxes,
        "normalize": normalize,
        "augment": augment
    }

def process_layout(
    layout: List[Dict[str, float]],
    max_boxes: int = 100,
    normalize: bool = True,
    augment: bool = False
) -> torch.Tensor:
    """处理布局
    
    Args:
        layout: 布局列表
        max_boxes: 最大框数
        normalize: 是否归一化
        augment: 是否进行数据增强
        
    Returns:
        处理后的布局
    """
    # 提取坐标
    boxes = []
    for box in layout:
        boxes.append([
            box["x"],
            box["y"],
            box["width"],
            box["height"]
        ])
    
    # 转换为张量
    boxes = torch.tensor(boxes, dtype=torch.float32)
    
    # 数据增强
    if augment:
        # 随机缩放
        scale = np.random.uniform(0.9, 1.1)
        boxes = boxes * scale
        
        # 随机平移
        shift = np.random.uniform(-0.05, 0.05, size=(2,))
        boxes[:, :2] = boxes[:, :2] + shift
    
    # 填充或截断
    if len(boxes) < max_boxes:
        padding = torch.zeros((max_boxes - len(boxes), 4))
        boxes = torch.cat([boxes, padding], dim=0)
    else:
        boxes = boxes[:max_boxes]
    
    # 归一化
    if normalize:
        boxes[:, 0] = boxes[:, 0] / 100.0  # x
        boxes[:, 1] = boxes[:, 1] / 100.0  # y
        boxes[:, 2] = boxes[:, 2] / 100.0  # width
        boxes[:, 3] = boxes[:, 3] / 100.0  # height
    
    return boxes

def process_text(
    text: str,
    tokenizer_name: str = "bert-base-chinese",
    max_length: int = 512,
    padding: str = "max_length",
    truncation: bool = True,
    augment: bool = False
) -> Dict[str, torch.Tensor]:
    """处理文本
    
    Args:
        text: 文本
        tokenizer_name: 分词器名称
        max_length: 最大长度
        padding: 填充方式
        truncation: 是否截断
        augment: 是否进行数据增强
        
    Returns:
        处理后的文本
    """
    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # 数据增强
    if augment:
        # 随机删除字符
        if len(text) > 10:
            delete_ratio = np.random.uniform(0, 0.1)
            delete_num = int(len(text) * delete_ratio)
            delete_indices = np.random.choice(len(text), delete_num, replace=False)
            text = "".join([c for i, c in enumerate(text) if i not in delete_indices])
        
        # 随机替换字符
        if len(text) > 10:
            replace_ratio = np.random.uniform(0, 0.1)
            replace_num = int(len(text) * replace_ratio)
            replace_indices = np.random.choice(len(text), replace_num, replace=False)
            for idx in replace_indices:
                text = text[:idx] + " " + text[idx+1:]
    
    # 分词
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors="pt"
    )
    
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "token_type_ids": encoding["token_type_ids"].squeeze(0) if "token_type_ids" in encoding else None
    }

def process_label(
    label: Union[int, List[int]],
    num_classes: Optional[int] = None,
    smoothing: float = 0.0
) -> torch.Tensor:
    """处理标签
    
    Args:
        label: 标签
        num_classes: 类别数
        smoothing: 标签平滑系数
        
    Returns:
        处理后的标签
    """
    if isinstance(label, int):
        # 单标签
        if num_classes is None:
            raise ValueError("单标签模式下必须指定类别数")
            
        # 标签平滑
        if smoothing > 0:
            one_hot = torch.zeros(num_classes)
            one_hot[label] = 1 - smoothing
            one_hot += smoothing / num_classes
            return one_hot
        else:
            return torch.tensor(label, dtype=torch.long)
    else:
        # 多标签
        if num_classes is None:
            num_classes = max(label) + 1
            
        one_hot = torch.zeros(num_classes)
        one_hot[label] = 1
        
        # 标签平滑
        if smoothing > 0:
            one_hot = one_hot * (1 - smoothing) + smoothing / num_classes
            
        return one_hot 