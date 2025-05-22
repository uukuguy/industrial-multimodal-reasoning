# -*- coding: utf-8 -*-

import os
import json
import time
import torch
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from functools import wraps
from transformers import Trainer, TrainingArguments

# 导入优化策略模块
try:
    from .data_augmentation import DataAugmentation
    HAS_DATA_AUG = True
except ImportError:
    HAS_DATA_AUG = False
    logger = logging.getLogger(__name__)
    logger.warning("数据增强模块不可用，无法使用高级数据增强功能")

try:
    from .peft_training import PEFTHandler
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    logger = logging.getLogger(__name__)
    logger.warning("PEFT模块不可用，无法使用参数高效微调功能")

logger = logging.getLogger(__name__)

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def error_handler(func):
    """错误处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            # 可以添加自定义的错误恢复逻辑
            raise
    return wrapper

class EnhancedMultiModalTrainer(Trainer):
    """扩展transformers的Trainer类，支持多模态模型训练"""
    
    @error_handler
    def __init__(self, 
                model=None, 
                args=None, 
                data_collator=None,
                train_dataset=None,
                eval_dataset=None,
                tokenizer=None,
                config=None,
                output_dir=None,
                use_peft=False,
                peft_config=None,
                use_data_augmentation=False,
                data_augmentation_config=None,
                use_optimized_loss=False,
                **kwargs):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型实例
            args: TrainingArguments实例，如果未提供，会从config创建
            data_collator: 数据收集器
            train_dataset: 训练数据集
            eval_dataset: 验证数据集
            tokenizer: 分词器，用于处理文本
            config: 配置字典，用于创建TrainingArguments
            output_dir: 输出目录，如果args未提供
            use_peft: 是否使用参数高效微调
            peft_config: PEFT配置
            use_data_augmentation: 是否使用数据增强
            data_augmentation_config: 数据增强配置
            use_optimized_loss: 是否使用优化后的损失函数
            **kwargs: 传递给父类的其他参数
        """
        logger.info("Initializing EnhancedMultiModalTrainer...")
        
        # 如果未提供args，从config创建
        if args is None and config is not None:
            logger.info("Creating TrainingArguments from config...")
            training_config = config.get('training', {})
            output_dir = output_dir or f'outputs/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            
            args = self._create_training_args(training_config, output_dir, eval_dataset, kwargs)
            
        # 保存配置
        self.config = config
        if config is not None and output_dir is not None:
            self._save_config(config, output_dir)
        
        # 存储是否使用优化损失的标志
        self.use_optimized_loss = use_optimized_loss
        
        # 应用数据增强
        self._setup_data_augmentation(use_data_augmentation, data_augmentation_config, train_dataset)
        
        # 应用PEFT
        self._setup_peft(use_peft, peft_config, model)
            
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs
        )
        logger.info("EnhancedMultiModalTrainer initialized successfully")

    @error_handler
    def _create_training_args(self, training_config: Dict, output_dir: str, 
                            eval_dataset: Optional[Any], kwargs: Dict) -> TrainingArguments:
        """创建训练参数"""
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config.get('num_epochs', 10),
            per_device_train_batch_size=training_config.get('batch_size', 8),
            per_device_eval_batch_size=training_config.get('batch_size', 8),
            learning_rate=training_config.get('optimizer', {}).get('learning_rate', 1e-4),
            weight_decay=training_config.get('optimizer', {}).get('weight_decay', 0.01),
            warmup_ratio=training_config.get('scheduler', {}).get('warmup_ratio', 0.1),
            logging_dir=os.path.join(output_dir, 'logs'),
            report_to=training_config.get('report_to', ["tensorboard"]),
            ddp_find_unused_parameters=False,
            evaluation_strategy=training_config.get('evaluation', {}).get('strategy', 
                "epoch" if eval_dataset is not None else "no"),
            save_strategy=training_config.get('checkpoint', {}).get('save_strategy', "epoch"),
            save_steps=training_config.get('checkpoint', {}).get('save_steps', 500),
            save_total_limit=training_config.get('checkpoint', {}).get('save_total_limit', 3),
            load_best_model_at_end=training_config.get('checkpoint', {}).get('load_best_model_at_end', 
                True if eval_dataset is not None else False),
            logging_steps=training_config.get('logging', {}).get('steps', 50),
            **kwargs.pop('training_args', {})
        )
        
        # 检查配置兼容性
        if args.load_best_model_at_end and args.evaluation_strategy == "no":
            logger.warning("load_best_model_at_end requires evaluation_strategy to be non-'no'. "
                         "Setting load_best_model_at_end to False.")
            args.load_best_model_at_end = False
            
        return args

    @error_handler
    def _save_config(self, config: Dict, output_dir: str) -> None:
        """保存配置到文件"""
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        logger.info(f"Configuration saved to: {config_path}")

    @error_handler
    def _setup_data_augmentation(self, use_data_augmentation: bool, 
                               data_augmentation_config: Optional[Dict], 
                               train_dataset: Optional[Any]) -> None:
        """设置数据增强"""
        self.use_data_augmentation = use_data_augmentation and HAS_DATA_AUG
        self.data_augmentation_config = data_augmentation_config or {}
        
        if self.use_data_augmentation and train_dataset is not None:
            logger.info("Applying data augmentation...")
            train_dataset = self._apply_data_augmentation(train_dataset)
            logger.info(f"Data augmentation completed. Dataset size: {len(train_dataset)}")

    @error_handler
    def _setup_peft(self, use_peft: bool, peft_config: Optional[Dict], model: Optional[Any]) -> None:
        """设置PEFT"""
        self.use_peft = use_peft and HAS_PEFT
        self.peft_config = peft_config or {}
        
        if self.use_peft and model is not None:
            logger.info("Applying PEFT...")
            model = self._apply_peft(model)
            logger.info("PEFT applied successfully")

    def _apply_data_augmentation(self, dataset):
        """应用数据增强策略"""
        logger.info("应用数据增强...")
        
        # 获取数据增强配置
        augmentation_factor = self.data_augmentation_config.get('augmentation_factor', 1.5)
        min_samples_per_type = self.data_augmentation_config.get('min_samples_per_type', 20)
        
        # 创建数据增强器
        augmenter = DataAugmentation(
            seed=self.data_augmentation_config.get('seed', 42)
        )
        
        # 获取文档文本（简单实现，实际应从文档中提取）
        document_texts = {}
        for item in dataset:
            doc = item.get('document', '')
            if doc and doc not in document_texts:
                # 在实际实现中，这里应该从文档目录加载文本
                document_texts[doc] = f"示例文档内容: {doc}"
        
        # 应用数据增强
        augmented_dataset = augmenter.augment_dataset(
            dataset=dataset,
            document_texts=document_texts,
            augmentation_factor=augmentation_factor,
            min_samples_per_type=min_samples_per_type
        )
        
        logger.info(f"数据增强完成. 原始数据量: {len(dataset)}, 增强后数据量: {len(augmented_dataset)}")
        return augmented_dataset
    
    def _apply_peft(self, model):
        """应用参数高效微调"""
        logger.info("应用参数高效微调...")
        
        # 获取PEFT配置
        technique = self.peft_config.get('technique', 'lora')
        
        # 获取模型配置
        model_config = getattr(model, 'config', {})
        if hasattr(model_config, 'to_dict'):
            model_config = model_config.to_dict()
        
        # 创建PEFT处理器
        peft_handler = PEFTHandler(technique=technique)
        
        # 应用PEFT
        peft_model = peft_handler.prepare_model(
            model=model,
            peft_config=self.peft_config,
            model_config=model_config
        )
        
        logger.info(f"PEFT ({technique}) 已应用到模型")
        return peft_model
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """自定义损失计算方法，处理多模态输入"""
        # 处理文档编码
        document_encodings = self._prepare_document_encodings(inputs)
        
        # 获取问题和选项
        questions = inputs.get("question", [""])
        question = questions[0] if questions else ""
        
        options_list = inputs.get("options", [[]])
        options = options_list[0] if options_list else []
        
        # 模型前向传播
        outputs = model(document_encodings, question, options)
        
        # 获取答案
        answers = inputs.get("answer", [""])
        answer = answers[0] if answers else ""
        
        # 损失计算逻辑
        if self.use_optimized_loss:
            logger.debug("使用优化后的损失函数")
            loss = self._compute_custom_loss(outputs, answer)
        else:
            logger.debug("使用默认损失函数")
            # 使用默认损失
            if 'loss' in outputs:
                loss = outputs['loss']
            elif 'logits' in outputs and answer in ['A', 'B', 'C', 'D']:
                target_idx = ord(answer) - ord('A')
                target = torch.tensor(target_idx, device=self.model.device)
                loss = torch.nn.functional.cross_entropy(outputs['logits'], target)
            else:
                # 如果没有明确的损失，使用默认值
                loss = torch.tensor(0.0, requires_grad=True, device=self.model.device)
        
        return (loss, outputs) if return_outputs else loss
    
    def _prepare_document_encodings(self, inputs):
        """
        准备文档编码，处理多模态输入
        
        将各种格式的输入（预处理数据或文档路径）转换为模型可直接使用的张量格式。
        本函数主要负责批处理、张量转换和设备管理，底层PDF处理委托给pdf_processor模块。
        """
        import os
        import torch
        
        # 尝试导入PDF处理器
        try:
            from .pdf_processor import process_pdf, convert_processed_data_to_tensors
            HAS_PDF_PROCESSOR = True
        except ImportError:
            HAS_PDF_PROCESSOR = False
            logger.warning("PDF处理器不可用，仅能使用预处理数据")
        
        batch_size = len(inputs.get('question', []))
        if batch_size == 0:
            return None
        
        # 准备存储编码结果
        batch_encodings = []
        
        for i in range(batch_size):
            # 默认编码结构
            document_encoding = {
                'page_encodings': [],
                'document_features': None,
                'metadata': {}
            }
            
            # 1. 如果processed_data已提供，直接使用
            if 'processed_data' in inputs and inputs['processed_data'][i] is not None:
                processed_data = inputs['processed_data'][i]
                
                if HAS_PDF_PROCESSOR and callable(convert_processed_data_to_tensors):
                    # 使用pdf_processor模块的函数将处理数据转换为张量
                    try:
                        document_encoding = convert_processed_data_to_tensors(
                            processed_data,
                            device=self.model.device
                        )
                    except Exception as e:
                        logger.error(f"数据转换错误: {e}")
                        # 使用默认空编码作为回退方案
                else:
                    # 如果没有转换函数，手动转换（基础版本）
                    if isinstance(processed_data, dict) and 'pages' in processed_data:
                        for page in processed_data['pages']:
                            document_encoding['page_encodings'].append({
                                'text_embeddings': torch.tensor(
                                    page.get('text_embeddings', [0.0]*768),
                                    dtype=torch.float32,
                                    device=self.model.device
                                ),
                                'page_image_embedding': torch.tensor(
                                    page.get('image_embedding', [0.0]*768),
                                    dtype=torch.float32,
                                    device=self.model.device
                                ),
                                'layout_embedding': torch.tensor(
                                    page.get('layout_embedding', [0.0]*256),
                                    dtype=torch.float32,
                                    device=self.model.device
                                )
                            })
                        
                        # 保存元数据
                        if 'metadata' in processed_data:
                            document_encoding['metadata'] = processed_data['metadata']
            
            # 2. 如果没有预处理数据，但有文档路径，处理文档
            elif 'document_path' in inputs and HAS_PDF_PROCESSOR:
                doc_path = inputs['document_path'][i]
                if os.path.exists(doc_path):
                    try:
                        # 获取问题相关信息，帮助处理PDF
                        question = inputs['question'][i] if 'question' in inputs else ""
                        
                        # 确定目标页面 (如果问题中有明确的页码引用)
                        target_page = None
                        page_references = [
                            "第1页", "第2页", "第3页", "第4页", "第5页",
                            "第6页", "第7页", "第8页", "第9页", "第10页"
                        ]
                        for idx, ref in enumerate(page_references, 1):
                            if ref in question:
                                target_page = idx
                                break
                        
                        # 委托给pdf_processor进行处理
                        processed_result = process_pdf(
                            pdf_path=doc_path,
                            target_pages=target_page,
                            extract_text=True,
                            extract_images=True,
                            extract_layout=True,
                            compute_embeddings=True
                        )
                        
                        # 将处理结果转换为张量
                        if processed_result and isinstance(processed_result, dict):
                            try:
                                document_encoding = convert_processed_data_to_tensors(
                                    processed_result,
                                    device=self.model.device
                                )
                            except Exception as e:
                                logger.error(f"处理结果转换错误: {e}")
                    except Exception as e:
                        logger.error(f"PDF处理错误: {e}")
            
            # 3. 如果没有有效数据，创建占位符编码
            if not document_encoding['page_encodings']:
                # 创建一个占位符编码，具有预期的形状
                document_encoding['page_encodings'] = [{
                    'text_embeddings': torch.zeros(768, device=self.model.device),
                    'page_image_embedding': torch.zeros(768, device=self.model.device),
                    'layout_embedding': torch.zeros(256, device=self.model.device)
                }]
                document_encoding['metadata'] = document_encoding.get('metadata', {})
                document_encoding['metadata']['placeholder'] = True
            
            batch_encodings.append(document_encoding)
        
        # 如果是单个样本，直接返回编码
        if batch_size == 1:
            return batch_encodings[0]
        
        return batch_encodings
        
    def _compute_custom_loss(self, outputs, answer):
        """
        计算适合工业技术文档多模态推理的自定义损失函数
        
        针对不同问题类型采用不同的损失策略：
        1. 选择题：增强型交叉熵损失，带标签平滑和类别加权
        2. 开放题：基于语义相似度和关键词匹配的损失
        3. 多任务学习：结合重构损失和不确定性估计
        """
        import torch
        import torch.nn.functional as F
        import re
        
        # 获取设备
        device = self.model.device
        
        # 记录总损失和各部分损失，便于日志记录
        total_loss = 0.0
        loss_components = {}
        
        # 1. 分类任务（选择题）
        if 'logits' in outputs:
            logits = outputs['logits']
            num_classes = logits.size(-1)
            
            # 1.1 确定目标类别
            if answer in ['A', 'B', 'C', 'D']:
                target_idx = ord(answer) - ord('A')
                target = torch.tensor(target_idx, device=device)
                
                # 1.2 应用标签平滑处理，减少过拟合
                alpha = 0.1  # 平滑系数
                smooth_target = torch.zeros(num_classes, device=device)
                smooth_target.fill_(alpha / (num_classes - 1))
                smooth_target[target_idx] = 1.0 - alpha
                
                # 1.3 计算交叉熵损失
                ce_loss = -torch.sum(F.log_softmax(logits, dim=-1) * smooth_target)
                
                # 1.4 考虑类别不平衡（如果有类别权重）
                if hasattr(self, 'class_weights'):
                    weight = self.class_weights[target_idx]
                    ce_loss = ce_loss * weight
                    
                # 1.5 考虑预测置信度
                confidence_penalty = 0.0
                if 'uncertainty' in outputs:
                    uncertainty = outputs['uncertainty']
                    # 低不确定性高置信度应对应低损失
                    confidence_penalty = -0.1 * torch.log(uncertainty + 1e-10)
                
                classification_loss = ce_loss + confidence_penalty
                loss_components['classification'] = classification_loss.item()
                total_loss = classification_loss
            else:
                # 标准交叉熵损失作为后备方案
                target = torch.tensor(0, device=device)  # 默认值
                classification_loss = F.cross_entropy(logits, target)
                loss_components['classification'] = classification_loss.item()
                total_loss = classification_loss
        
        # 2. 生成任务（开放题）
        elif 'generated_text' in outputs and isinstance(answer, str) and len(answer) > 0:
            generated_text = outputs['generated_text']
            
            # 2.1 基于字符匹配的损失（简化版本）
            # 在实际实现中，这里可以使用更先进的文本匹配评估指标如ROUGE或BLEU
            if isinstance(generated_text, str):
                char_match = sum(1 for c in answer if c in generated_text) / max(len(answer), 1)
                gen_loss = torch.tensor(1.0 - char_match,
                                       device=device,
                                       requires_grad=True)
                loss_components['generation'] = gen_loss.item()
                total_loss = gen_loss
            
            # 2.2 处理技术参数匹配
            # 特别处理数值型答案，如尺寸、角度等
            if re.search(r'\d+\.?\d*', answer):
                # 提取答案中的数值
                answer_numbers = re.findall(r'\d+\.?\d*', answer)
                generated_numbers = re.findall(r'\d+\.?\d*', generated_text) if isinstance(generated_text, str) else []
                
                if answer_numbers and generated_numbers:
                    # 计算数值差异
                    number_diffs = []
                    for a_num in answer_numbers:
                        a_val = float(a_num)
                        closest_diff = min((abs(a_val - float(g_num)) / max(a_val, 1.0)
                                          for g_num in generated_numbers), default=1.0)
                        number_diffs.append(closest_diff)
                    
                    # 数值匹配损失
                    numeric_loss = torch.tensor(sum(number_diffs) / len(number_diffs),
                                              device=device,
                                              requires_grad=True)
                    loss_components['numeric'] = numeric_loss.item()
                    
                    # 如果已有总损失，添加数值损失；否则设置为数值损失
                    if total_loss > 0:
                        total_loss = total_loss * 0.7 + numeric_loss * 0.3
                    else:
                        total_loss = numeric_loss
        
        # 3. 如果模型已计算损失，使用它作为主损失或附加损失
        if 'loss' in outputs:
            model_loss = outputs['loss']
            loss_components['model_loss'] = model_loss.item()
            
            if total_loss > 0:
                # 组合自定义损失和模型损失
                total_loss = total_loss * 0.7 + model_loss * 0.3
            else:
                total_loss = model_loss
        
        # 4. 添加多任务辅助损失
        # 4.1 重构损失（如果有）
        if 'reconstruction_loss' in outputs:
            recon_loss = outputs['reconstruction_loss']
            loss_components['reconstruction'] = recon_loss.item()
            total_loss = total_loss + 0.1 * recon_loss  # 权重较小
        
        # 4.2 跨模态对齐损失
        if 'alignment_loss' in outputs:
            align_loss = outputs['alignment_loss']
            loss_components['alignment'] = align_loss.item()
            total_loss = total_loss + 0.2 * align_loss
        
        # 如果仍然没有有效损失，使用默认损失
        if not isinstance(total_loss, torch.Tensor) or total_loss == 0:
            total_loss = torch.tensor(1.0, requires_grad=True, device=device)
            loss_components['default'] = 1.0
        
        # 记录损失组件，便于调试和监控
        if hasattr(self, 'log_history') and self.log_history:
            current_step = len(self.log_history)
            if current_step % 10 == 0:  # 每10步记录一次详细损失
                for name, value in loss_components.items():
                    self.log({f"loss/{name}": value})
        
        return total_loss
    
    def evaluate(self, 
                eval_dataset=None,
                ignore_keys=None,
                metric_key_prefix="eval",
                **kwargs):
        """重写评估方法，支持自定义指标计算"""
        # 调用父类评估方法
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **kwargs
        )
        
        # 添加自定义指标计算，如准确率等
        # 可以在这里添加特定于工业文档QA的指标
        
        return metrics
    
    def predict(self, test_dataset):
        """预测函数，生成符合评测格式的结果"""
        # 使用模型生成预测结果
        pred_output = super().predict(test_dataset)
        
        # 获取问题ID
        question_ids = [item['id'] for item in test_dataset]
        
        # 处理预测结果，转换为要求的格式
        results = []
        
        for i, pred in enumerate(pred_output.predictions):
            # 根据模型输出类型处理预测结果
            # 对于选择题，选择概率最高的选项
            if isinstance(pred, dict) and 'logits' in pred:
                logits = pred['logits']
                pred_idx = torch.argmax(torch.tensor(logits)).item()
                answer = chr(ord('A') + pred_idx)
            # 对于开放题，使用生成的文本
            elif isinstance(pred, dict) and 'answer' in pred:
                answer = pred['answer']
            else:
                # 默认处理
                answer = "无法预测"
            
            # 构建结果
            result = {
                "id": question_ids[i],
                "answer": answer
            }
            results.append(result)
        
        return results
    
    def save_predictions(self, predictions, output_file):
        """保存预测结果为JSONL格式"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + '\n')
        
        logger.info(f"预测结果已保存到: {output_file}")


def create_trainer(
    model,
    config,
    train_dataset=None,
    eval_dataset=None,
    output_dir=None,
    data_collator=None,
    use_peft=False,
    peft_config=None,
    use_data_augmentation=False,
    data_augmentation_config=None,
    use_optimized_loss=False,
    **kwargs
):
    """
    创建训练器的工厂函数
    
    Args:
        model: 模型实例
        config: 配置字典
        train_dataset: 训练数据集
        eval_dataset: 验证数据集
        output_dir: 输出目录
        data_collator: 数据收集器
        use_peft: 是否使用参数高效微调
        peft_config: PEFT配置
        use_data_augmentation: 是否使用数据增强
        data_augmentation_config: 数据增强配置
        use_optimized_loss: 是否使用优化后的损失函数
        **kwargs: 其他参数
        
    Returns:
        EnhancedMultiModalTrainer实例
    """
    # 根据数据集大小自动决定是否使用PEFT
    if train_dataset is not None and not use_peft:
        dataset_size = len(train_dataset)
        if dataset_size < 1000 and HAS_PEFT:
            logger.info(f"数据集较小({dataset_size}样本)，自动启用PEFT优化")
            use_peft = True
            if peft_config is None:
                # 获取默认配置
                from .peft_training import PEFTHandler
                model_size = sum(p.numel() for p in model.parameters()) / 1e6
                technique = PEFTHandler.get_recommended_technique(dataset_size, model_size)
                model_type = getattr(model, 'config', {}).get('model_type', '')
                if hasattr(model.config, 'model_type'):
                    model_type = model.config.model_type
                peft_config = PEFTHandler.get_default_config(technique, model_type)
                peft_config['technique'] = technique
    
    # 根据数据集大小自动决定是否使用数据增强
    if train_dataset is not None and not use_data_augmentation:
        dataset_size = len(train_dataset)
        question_types = set()
        for item in train_dataset:
            question = item.get('question', '')
            for key, words in {
                '位置关系': ['位置', '相对', '方向'],
                '功能描述': ['功能', '作用', '用于'],
                '技术参数': ['参数', '角度', '数值'],
                '结构组成': ['结构', '组成', '部件'],
                '操作步骤': ['步骤', '操作', '首先']
            }.items():
                if any(word in question for word in words):
                    question_types.add(key)
                    break
        
        # 检查是否存在低频问题类型
        if len(question_types) >= 3 and HAS_DATA_AUG:
            type_min_count = 20  # 每种类型的最小样本数
            if dataset_size / len(question_types) < type_min_count:
                logger.info(f"检测到问题类型不平衡，自动启用数据增强")
                use_data_augmentation = True
                if data_augmentation_config is None:
                    data_augmentation_config = {
                        'augmentation_factor': 1.5,
                        'min_samples_per_type': type_min_count
                    }
    
    return EnhancedMultiModalTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        data_collator=data_collator,
        use_peft=use_peft,
        peft_config=peft_config,
        use_data_augmentation=use_data_augmentation,
        data_augmentation_config=data_augmentation_config,
        use_optimized_loss=use_optimized_loss,
        **kwargs
    )