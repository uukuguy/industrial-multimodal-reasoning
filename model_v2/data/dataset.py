import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from PIL import Image
import os
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import hashlib
import pickle
from .transforms import process_text, process_layout, process_label

logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """数据验证错误"""
    pass

class MultimodalDataset(Dataset):
    """多模态数据集"""
    
    def __init__(
        self,
        data_file: str,
        image_dir: str,
        text_encoder_name: str = "bert-base-chinese",
        max_length: int = 512,
        image_size: int = 224,
        transform: Optional[Any] = None,
        cache_dir: Optional[str] = None,
        is_train: bool = True,
        validate_data: bool = True
    ):
        """初始化数据集
        
        Args:
            data_file: 数据文件路径
            image_dir: 图像目录
            text_encoder_name: 文本编码器名称
            max_length: 最大文本长度
            image_size: 图像大小
            transform: 图像变换
            cache_dir: 缓存目录
            is_train: 是否为训练模式
            validate_data: 是否验证数据
        """
        self.data_file = data_file
        self.image_dir = image_dir
        self.max_length = max_length
        self.image_size = image_size
        self.transform = transform
        self.cache_dir = cache_dir
        self.is_train = is_train
        self.validate_data = validate_data
        
        # 初始化文本编码器
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        
        # 加载数据
        self.data = self._load_data()
        
        # 验证数据
        if validate_data:
            self._validate_data()
            
        # 创建缓存目录
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载数据
        
        Returns:
            数据列表
        """
        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                raise DataValidationError("数据文件格式错误：应为列表")
                
            logger.info(f"成功加载数据文件：{self.data_file}，共 {len(data)} 条数据")
            return data
            
        except Exception as e:
            logger.error(f"加载数据文件失败：{str(e)}")
            raise
            
    def _validate_data(self):
        """验证数据"""
        logger.info("开始验证数据...")
        valid_data = []
        
        for item in tqdm(self.data, desc="验证数据"):
            try:
                # 验证必要字段
                required_fields = ["id", "text", "image", "label"]
                for field in required_fields:
                    if field not in item:
                        raise DataValidationError(f"缺少必要字段：{field}")
                
                # 验证图像文件
                image_path = os.path.join(self.image_dir, item["image"])
                if not os.path.exists(image_path):
                    raise DataValidationError(f"图像文件不存在：{image_path}")
                    
                # 验证图像格式
                try:
                    image = Image.open(image_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                except Exception as e:
                    raise DataValidationError(f"图像文件格式错误：{str(e)}")
                
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
                
                valid_data.append(item)
                
            except DataValidationError as e:
                logger.warning(f"数据验证失败：{str(e)}，跳过该条数据")
                continue
                
        self.data = valid_data
        logger.info(f"数据验证完成，有效数据：{len(valid_data)} 条")
        
    def _get_cache_path(self, item_id: str) -> Optional[str]:
        """获取缓存路径
        
        Args:
            item_id: 数据项ID
            
        Returns:
            缓存路径
        """
        if not self.cache_dir:
            return None
            
        # 生成缓存文件名
        cache_key = f"{item_id}_{self.max_length}_{self.image_size}_{self.is_train}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{cache_hash}.pkl")
        
    def _load_from_cache(self, cache_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """从缓存加载数据
        
        Args:
            cache_path: 缓存路径
            
        Returns:
            缓存的数据
        """
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
            
    def _save_to_cache(self, cache_path: str, data: Dict[str, torch.Tensor]):
        """保存数据到缓存
        
        Args:
            cache_path: 缓存路径
            data: 要缓存的数据
        """
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"保存缓存失败：{str(e)}")
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            数据项
        """
        item = self.data[idx]
        
        # 尝试从缓存加载
        if self.cache_dir:
            cache_path = self._get_cache_path(item["id"])
            if cache_path and os.path.exists(cache_path):
                cached_data = self._load_from_cache(cache_path)
                if cached_data is not None:
                    return cached_data
        
        try:
            # 处理文本
            text_encoding = process_text(
                text=item["text"],
                tokenizer_name=self.tokenizer.name_or_path,
                max_length=self.max_length,
                augment=self.is_train
            )
            
            # 处理图像
            image_path = os.path.join(self.image_dir, item["image"])
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            
            # 处理布局（如果存在）
            layout = None
            if "layout" in item:
                layout = process_layout(
                    layout=item["layout"],
                    augment=self.is_train
                )
            
            # 处理标签
            label = process_label(
                label=item["label"],
                smoothing=0.1 if self.is_train else 0.0
            )
            
            # 组装数据
            data = {
                "input_ids": text_encoding["input_ids"],
                "attention_mask": text_encoding["attention_mask"],
                "token_type_ids": text_encoding["token_type_ids"],
                "image": image,
                "label": label
            }
            
            if layout is not None:
                data["layout"] = layout
            
            # 保存到缓存
            if self.cache_dir and cache_path:
                self._save_to_cache(cache_path, data)
            
            return data
            
        except Exception as e:
            logger.error(f"处理数据项失败：{str(e)}")
            raise

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