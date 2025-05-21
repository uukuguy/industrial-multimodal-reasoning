# -*- coding: utf-8 -*-

import os
import json
import logging
from typing import Dict, List, Any, Optional
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class MultimodalQuestionAnsweringDataset(Dataset):
    """多模态问答数据集"""
    
    def __init__(self, 
                questions_path: str, 
                documents_dir: str,
                processed_data_dir: Optional[str] = None,
                transform=None,
                is_test: bool = False):
        """
        初始化数据集
        
        Args:
            questions_path: JSONL格式问题文件的路径
            documents_dir: PDF文档目录路径
            processed_data_dir: 预处理数据目录(可选，加速训练)
            transform: 数据增强转换(可选)
            is_test: 是否为测试集
        """
        self.questions = self._load_questions(questions_path)
        self.documents_dir = documents_dir
        self.processed_data_dir = processed_data_dir
        self.transform = transform
        self.is_test = is_test
        logger.info(f"数据集初始化完成，加载了 {len(self.questions)} 个问题")
    
    def _load_questions(self, questions_path: str) -> List[Dict[str, Any]]:
        """加载问题数据"""
        questions = []
        with open(questions_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        return questions
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        question_data = self.questions[idx].copy()
        
        # 构建文档路径
        doc_name = question_data.get("document", "")
        doc_path = os.path.join(self.documents_dir, doc_name)
        
        # 检查预处理数据
        processed_data = None
        if self.processed_data_dir:
            doc_id = os.path.splitext(doc_name)[0]
            processed_path = os.path.join(self.processed_data_dir, doc_id, "metadata.json")
            if os.path.exists(processed_path):
                try:
                    with open(processed_path, 'r', encoding='utf-8') as f:
                        processed_data = json.load(f)
                except Exception as e:
                    logger.warning(f"无法加载预处理数据 {processed_path}: {e}")
        
        # 构建样本
        sample = {
            "id": question_data.get("id", f"sample_{idx}"),
            "question": question_data.get("question", ""),
            "document_path": doc_path,
            "processed_data": processed_data,
            "options": question_data.get("options", [])
        }
        
        # 如果不是测试集，添加答案
        if not self.is_test:
            sample["answer"] = question_data.get("answer", "")
        
        # 应用数据增强
        if self.transform:
            sample = self.transform(sample)
            
        return sample


# 数据变换/增强类
class DocumentTransforms:
    """文档数据变换/增强"""
    
    @staticmethod
    def text_augmentation(sample: Dict[str, Any], aug_ratio: float = 0.1) -> Dict[str, Any]:
        """文本数据增强"""
        # 实际实现中可随机替换词语、插入同义词等
        return sample
    
    @staticmethod
    def image_augmentation(sample: Dict[str, Any]) -> Dict[str, Any]:
        """图像数据增强"""
        # 实际实现中可应用随机裁剪、旋转、调整亮度等
        return sample


# 数据批处理工具
class DataCollator:
    """数据批处理"""
    
    @staticmethod
    def collate_fn(batch):
        """批处理函数"""
        # 可以根据需要实现自定义的批处理逻辑
        batch_dict = {
            'id': [item['id'] for item in batch],
            'question': [item['question'] for item in batch],
            'document_path': [item['document_path'] for item in batch],
            'processed_data': [item.get('processed_data') for item in batch],
            'options': [item.get('options', []) for item in batch]
        }
        
        # 如果有答案，也收集
        if 'answer' in batch[0]:
            batch_dict['answer'] = [item['answer'] for item in batch]
            
        return batch_dict