import os
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import hashlib
import pickle
from PIL import Image
from transformers import AutoTokenizer
from .transforms import process_text, process_layout, process_label

logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """数据验证错误"""
    pass

class DataCache:
    """数据缓存"""
    
    def __init__(self, cache_dir: str):
        """初始化缓存
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_path(self, key: str) -> str:
        """获取缓存路径
        
        Args:
            key: 缓存键
            
        Returns:
            缓存路径
        """
        cache_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{cache_hash}.pkl")
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的数据
        """
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"读取缓存失败：{str(e)}")
        return None
        
    def set(self, key: str, value: Any):
        """设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"保存缓存失败：{str(e)}")
            
    def clear(self):
        """清除缓存"""
        try:
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
        except Exception as e:
            logger.warning(f"清除缓存失败：{str(e)}")

def validate_data_file(data_file: str) -> bool:
    """验证数据文件
    
    Args:
        data_file: 数据文件路径
        
    Returns:
        是否有效
    """
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise DataValidationError("数据文件格式错误：应为列表")
            
        for item in data:
            # 验证必要字段
            required_fields = ["id", "text", "image", "label"]
            for field in required_fields:
                if field not in item:
                    raise DataValidationError(f"缺少必要字段：{field}")
            
            # 验证文本
            if not isinstance(item["text"], str) or not item["text"].strip():
                raise DataValidationError("文本格式错误")
            
            # 验证标签
            if not isinstance(item["label"], (int, list)):
                raise DataValidationError("标签格式错误")
            
            # 验证布局（如果存在）
            if "layout" in item:
                if not isinstance(item["layout"], list):
                    raise DataValidationError("布局格式错误")
                for box in item["layout"]:
                    if not all(k in box for k in ["x", "y", "width", "height"]):
                        raise DataValidationError("布局框格式错误")
        
        return True
        
    except Exception as e:
        logger.error(f"验证数据文件失败：{str(e)}")
        return False

def validate_image_file(image_path: str) -> bool:
    """验证图像文件
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        是否有效
    """
    try:
        if not os.path.exists(image_path):
            raise DataValidationError(f"图像文件不存在：{image_path}")
            
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return True
        
    except Exception as e:
        logger.error(f"验证图像文件失败：{str(e)}")
        return False

def create_data_loader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        pin_memory: 是否使用固定内存
        drop_last: 是否丢弃最后一个不完整的批次
        
    Returns:
        数据加载器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """整理批次数据
    
    Args:
        batch: 批次数据
        
    Returns:
        整理后的批次数据
    """
    # 获取批次大小
    batch_size = len(batch)
    
    # 整理文本数据
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    token_type_ids = torch.stack([item["token_type_ids"] for item in batch])
    
    # 整理图像数据
    images = torch.stack([item["image"] for item in batch])
    
    # 整理标签
    labels = torch.stack([item["label"] for item in batch])
    
    # 组装数据
    collated = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "image": images,
        "label": labels
    }
    
    # 整理布局数据（如果存在）
    if "layout" in batch[0]:
        layouts = torch.stack([item["layout"] for item in batch])
        collated["layout"] = layouts
    
    return collated

def get_dataset_stats(data_file: str) -> Dict[str, Any]:
    """获取数据集统计信息
    
    Args:
        data_file: 数据文件路径
        
    Returns:
        统计信息
    """
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        stats = {
            "total": len(data),
            "text_length": {
                "min": float("inf"),
                "max": 0,
                "avg": 0
            },
            "label_dist": {},
            "has_layout": 0
        }
        
        total_text_length = 0
        
        for item in data:
            # 文本长度统计
            text_length = len(item["text"])
            stats["text_length"]["min"] = min(stats["text_length"]["min"], text_length)
            stats["text_length"]["max"] = max(stats["text_length"]["max"], text_length)
            total_text_length += text_length
            
            # 标签分布统计
            if isinstance(item["label"], int):
                label = str(item["label"])
            else:
                label = ",".join(map(str, sorted(item["label"])))
            stats["label_dist"][label] = stats["label_dist"].get(label, 0) + 1
            
            # 布局统计
            if "layout" in item:
                stats["has_layout"] += 1
                
        # 计算平均文本长度
        stats["text_length"]["avg"] = total_text_length / len(data)
        
        return stats
        
    except Exception as e:
        logger.error(f"获取数据集统计信息失败：{str(e)}")
        raise

def save_dataset_stats(stats: Dict[str, Any], output_file: str):
    """保存数据集统计信息
    
    Args:
        stats: 统计信息
        output_file: 输出文件路径
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存数据集统计信息失败：{str(e)}")
        raise

class DataProcessor:
    """数据处理器"""
    
    def __init__(
        self,
        image_dir: str,
        max_length: int = 512,
        image_size: int = 224,
        tokenizer_name: str = "google/bert-base-chinese"
    ):
        """初始化数据处理器
        
        Args:
            image_dir: 图像目录路径
            max_length: 最大文本长度
            image_size: 图像大小
            tokenizer_name: 分词器名称
        """
        self.image_dir = image_dir
        self.max_length = max_length
        self.image_size = image_size
        
        # 初始化分词器
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 初始化图像转换
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def process_data(
        self,
        data_file: str,
        output_file: str,
        split: str = "train"
    ):
        """处理数据
        
        Args:
            data_file: 输入数据文件路径
            output_file: 输出数据文件路径
            split: 数据集划分
        """
        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 处理数据
        processed_data = []
        for item in data:
            try:
                processed_item = self._process_item(item)
                processed_data.append(processed_item)
            except Exception as e:
                logger.warning(f"处理数据项失败: {item['id']}, 错误: {str(e)}")
                continue
                
        # 保存处理后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"数据处理完成，共处理 {len(processed_data)} 条数据")
        
    def _process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据项
        
        Args:
            item: 数据项
            
        Returns:
            处理后的数据项
        """
        # 处理图像
        image_path = os.path.join(self.image_dir, item['image'])
        if not os.path.exists(image_path):
            raise ValueError(f"图像文件不存在: {image_path}")
            
        # 处理问题
        question = self._process_text(item['question'])
        
        # 处理选项
        options = [self._process_text(option) for option in item['options']]
        
        # 处理答案
        answer = self._process_answer(item['answer'])
        
        return {
            'id': item['id'],
            'image': item['image'],
            'question': question,
            'options': options,
            'answer': answer
        }
        
    def _process_text(self, text: str) -> Dict[str, List[int]]:
        """处理文本
        
        Args:
            text: 文本
            
        Returns:
            处理后的文本
        """
        # 分词
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze().tolist(),
            'attention_mask': encoding['attention_mask'].squeeze().tolist()
        }
        
    def _process_answer(self, answer: Union[str, int]) -> Union[Dict[str, List[int]], int]:
        """处理答案
        
        Args:
            answer: 答案
            
        Returns:
            处理后的答案
        """
        if isinstance(answer, str):
            # 文本答案
            encoding = self.tokenizer(
                answer,
                max_length=self.max_length // 2,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze().tolist(),
                'attention_mask': encoding['attention_mask'].squeeze().tolist()
            }
        else:
            # 选项答案
            return answer

def split_dataset(
    data_file: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    output_dir: str = "data"
):
    """划分数据集
    
    Args:
        data_file: 数据文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        output_dir: 输出目录
    """
    # 检查比例
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("数据集比例之和必须为1")
        
    # 加载数据
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 打乱数据
    np.random.shuffle(data)
    
    # 计算划分点
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # 划分数据集
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存数据集
    with open(output_dir / "train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
        
    with open(output_dir / "val.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
        
    with open(output_dir / "test.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
        
    logger.info(
        f"数据集划分完成:\n"
        f"训练集: {len(train_data)} 条\n"
        f"验证集: {len(val_data)} 条\n"
        f"测试集: {len(test_data)} 条"
    ) 