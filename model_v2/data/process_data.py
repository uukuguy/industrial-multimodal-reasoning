import os
import argparse
import logging
from pathlib import Path
import json
from typing import Dict, List, Optional, Any, Union
from tqdm import tqdm
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
from .transforms import process_text, process_layout, process_label

from .data_utils import DataProcessor, split_dataset

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """数据验证错误"""
    pass

class DataProcessor:
    """数据处理器"""
    
    def __init__(
        self,
        image_dir: str,
        max_length: int = 512,
        image_size: int = 224,
        text_encoder_name: str = "bert-base-chinese",
        validate_data: bool = True,
        show_progress: bool = True
    ):
        """初始化数据处理器
        
        Args:
            image_dir: 图像目录
            max_length: 最大文本长度
            image_size: 图像大小
            text_encoder_name: 文本编码器名称
            validate_data: 是否验证数据
            show_progress: 是否显示进度
        """
        self.image_dir = image_dir
        self.max_length = max_length
        self.image_size = image_size
        self.text_encoder_name = text_encoder_name
        self.validate_data = validate_data
        self.show_progress = show_progress
        
        # 初始化文本编码器
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        
    def _validate_item(self, item: Dict[str, Any]) -> bool:
        """验证数据项
        
        Args:
            item: 数据项
            
        Returns:
            是否有效
        """
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
            
            return True
            
        except DataValidationError as e:
            logger.warning(f"数据验证失败：{str(e)}")
            return False
            
    def _process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理数据项
        
        Args:
            item: 数据项
            
        Returns:
            处理后的数据项
        """
        try:
            # 处理文本
            text_encoding = process_text(
                text=item["text"],
                tokenizer_name=self.text_encoder_name,
                max_length=self.max_length
            )
            
            # 处理图像
            image_path = os.path.join(self.image_dir, item["image"])
            image = Image.open(image_path).convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            
            # 处理布局（如果存在）
            layout = None
            if "layout" in item:
                layout = process_layout(
                    layout=item["layout"],
                    augment=False
                )
            
            # 处理标签
            label = process_label(
                label=item["label"],
                smoothing=0.0
            )
            
            # 组装数据
            processed = {
                "id": item["id"],
                "input_ids": text_encoding["input_ids"].numpy(),
                "attention_mask": text_encoding["attention_mask"].numpy(),
                "token_type_ids": text_encoding["token_type_ids"].numpy() if text_encoding["token_type_ids"] is not None else None,
                "image": np.array(image),
                "label": label.numpy()
            }
            
            if layout is not None:
                processed["layout"] = layout.numpy()
            
            return processed
            
        except Exception as e:
            logger.error(f"处理数据项失败：{str(e)}")
            return None
            
    def process_data(
        self,
        input_file: str,
        output_file: str,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """处理数据
        
        Args:
            input_file: 输入文件
            output_file: 输出文件
            batch_size: 批处理大小
            
        Returns:
            处理统计信息
        """
        try:
            # 加载数据
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                raise DataValidationError("数据文件格式错误：应为列表")
                
            logger.info(f"成功加载数据文件：{input_file}，共 {len(data)} 条数据")
            
            # 处理数据
            processed_data = []
            stats = {
                "total": len(data),
                "valid": 0,
                "invalid": 0,
                "processed": 0,
                "failed": 0
            }
            
            # 创建进度条
            pbar = tqdm(
                total=len(data),
                desc="处理数据",
                disable=not self.show_progress
            )
            
            # 分批处理
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                for item in batch:
                    # 验证数据
                    if self.validate_data and not self._validate_item(item):
                        stats["invalid"] += 1
                        pbar.update(1)
                        continue
                        
                    stats["valid"] += 1
                    
                    # 处理数据
                    processed = self._process_item(item)
                    if processed is not None:
                        processed_data.append(processed)
                        stats["processed"] += 1
                    else:
                        stats["failed"] += 1
                        
                    pbar.update(1)
                    
            pbar.close()
            
            # 保存处理后的数据
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"数据处理完成，统计信息：")
            logger.info(f"- 总数据量：{stats['total']}")
            logger.info(f"- 有效数据：{stats['valid']}")
            logger.info(f"- 无效数据：{stats['invalid']}")
            logger.info(f"- 处理成功：{stats['processed']}")
            logger.info(f"- 处理失败：{stats['failed']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"处理数据失败：{str(e)}")
            raise

def split_dataset(
    data_file: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    show_progress: bool = True
) -> Dict[str, int]:
    """划分数据集
    
    Args:
        data_file: 数据文件
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
        show_progress: 是否显示进度
        
    Returns:
        各集合大小
    """
    try:
        # 加载数据
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise DataValidationError("数据文件格式错误：应为列表")
            
        logger.info(f"成功加载数据文件：{data_file}，共 {len(data)} 条数据")
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 打乱数据
        indices = np.random.permutation(len(data))
        
        # 计算划分点
        train_size = int(len(data) * train_ratio)
        val_size = int(len(data) * val_ratio)
        
        # 划分数据
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存数据
        splits = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices
        }
        
        for split_name, split_indices in splits.items():
            split_data = [data[i] for i in split_indices]
            output_file = os.path.join(output_dir, f"{split_name}.json")
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"保存{split_name}集：{output_file}，共 {len(split_data)} 条数据")
            
        return {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices)
        }
        
    except Exception as e:
        logger.error(f"划分数据集失败：{str(e)}")
        raise

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="处理多模态数据")
    
    # 数据参数
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="输入数据文件路径"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="图像目录路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="输出目录"
    )
    
    # 处理参数
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="最大文本长度"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="图像大小"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="google/bert-base-chinese",
        help="分词器名称"
    )
    
    # 数据集划分参数
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="验证集比例"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="测试集比例"
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据处理器
    processor = DataProcessor(
        image_dir=args.image_dir,
        max_length=args.max_length,
        image_size=args.image_size,
        tokenizer_name=args.tokenizer_name
    )
    
    # 处理数据
    logger.info("开始处理数据...")
    processor.process_data(
        data_file=args.data_file,
        output_file=output_dir / "processed_data.json"
    )
    
    # 划分数据集
    logger.info("开始划分数据集...")
    split_dataset(
        data_file=output_dir / "processed_data.json",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        output_dir=output_dir
    )
    
    logger.info("数据处理完成")

if __name__ == "__main__":
    main() 