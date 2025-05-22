# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import logging
import time
from typing import Dict, Any, Optional, Union, List, Tuple
from functools import wraps

from .uncertainty_estimator import UncertaintyEstimator, OptionSelector
from .optimization import ModelOptimizer, OptimizationConfig, AttentionOptimizationConfig

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def performance_monitor(func):
    """性能监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            execution_time = end_time - start_time
            memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
            
            logger.info(f"Performance metrics for {func.__name__}:")
            logger.info(f"  - Execution time: {execution_time:.2f} seconds")
            logger.info(f"  - Memory usage: {memory_used:.2f} MB")
            
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper

def error_handler(func):
    """错误处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper

class ModelFactory:
    """模型工厂类，负责模型的加载和初始化"""
    
    @staticmethod
    @error_handler
    @performance_monitor
    def load_model(model_path: str, optimization_config: Optional[OptimizationConfig] = None) -> Tuple[Any, Any]:
        """
        加载模型和分词器
        
        Args:
            model_path: 模型路径或名称
            optimization_config: 优化配置
            
        Returns:
            Tuple[model, tokenizer]: 加载的模型和分词器
        """
        logger.info(f"Loading model from: {model_path}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # 创建优化器
            if optimization_config is None:
                optimization_config = OptimizationConfig(
                    use_quantization=True,
                    quantization_bits=8,
                    use_cache=True,
                    use_mixed_precision=True
                )
            optimizer = ModelOptimizer(optimization_config)
            
            # 加载并优化模型
            model = optimizer.load_model(model_path)
            
            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            return model, tokenizer
            
        except ImportError:
            logger.error("Failed to import transformers library")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

class ReasoningQAModule(nn.Module):
    """
    推理与问答模块，负责根据融合的文档表示生成答案
    支持初赛（选择题）和复赛（开放题）
    """
    
    @error_handler
    @performance_monitor
    def __init__(self, lmm_model_name: str, embedding_dim: int = 512, 
                temperature: float = 1.0, confidence_threshold: float = 0.7,
                optimization_config: Optional[OptimizationConfig] = None):
        """
        初始化推理与问答模块
        
        Args:
            lmm_model_name: 大型多模态模型名称或路径
            embedding_dim: 嵌入维度
            temperature: 温度参数，影响输出的多样性
            confidence_threshold: 置信度阈值，低于此值的预测会被标记为不确定
            optimization_config: 优化配置
        """
        super().__init__()
        self.lmm_model_name = lmm_model_name
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        
        # 设置默认的注意力优化配置
        if optimization_config is None:
            optimization_config = OptimizationConfig(
                use_quantization=True,
                quantization_bits=8,
                use_cache=True,
                use_mixed_precision=True,
                attention_optimization=AttentionOptimizationConfig(
                    use_sparse_attention=True,
                    sparse_attention_threshold=0.1,
                    use_head_pruning=True,
                    head_pruning_ratio=0.3,
                    use_attention_cache=True,
                    use_flash_attention=True,
                    use_sliding_window=True,
                    window_size=512
                )
            )
        
        # 加载模型
        try:
            self.lmm_model, self.tokenizer = ModelFactory.load_model(
                lmm_model_name,
                optimization_config
            )
            self.lmm_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            self.lmm_model = None
            self.tokenizer = None
            self.lmm_initialized = False
            return
        
        # 创建不确定性估计器
        self.uncertainty_estimator = UncertaintyEstimator(
            embedding_dim=embedding_dim,
            num_classes=4,  # A, B, C, D
            temperature=temperature
        )
        
        # 添加分类头
        self.classification_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 4)  # 4个选项 A, B, C, D
        )
        
        # 添加文本生成投影层
        self.text_generation_projection = nn.Linear(embedding_dim, embedding_dim)
        
        logger.info("ReasoningQAModule initialized successfully")

    @error_handler
    @performance_monitor
    def preprocess_question(self, question: str) -> str:
        """
        预处理问题文本
        
        Args:
            question: 原始问题文本
            
        Returns:
            处理后的问题文本
        """
        # 移除多余的空白字符
        question = ' '.join(question.split())
        
        # 添加提示词
        if not question.startswith('根据文档'):
            question = f"根据文档内容，{question}"
            
        return question

    @error_handler
    @performance_monitor
    def postprocess_answer(self, answer: str, options: Optional[List[str]] = None) -> str:
        """
        后处理答案文本
        
        Args:
            answer: 原始答案文本
            options: 选项列表 (如果是选择题)
            
        Returns:
            处理后的答案文本
        """
        # 移除多余的空白字符
        answer = ' '.join(answer.split())
        
        # 如果是选择题，确保答案是选项之一
        if options and len(options) > 0:
            return OptionSelector.select_option(answer, options)
        
        return answer

    def is_initialized(self) -> bool:
        """检查模块是否成功初始化"""
        return self.lmm_initialized or hasattr(self, 'classification_head')
    
    @error_handler
    @performance_monitor
    def answer_classification_question(self, 
                                      document_embedding: torch.Tensor, 
                                      question: str,
                                      options: List[str]) -> Dict[str, Any]:
        """
        回答选择题（初赛）
        
        Args:
            document_embedding: 文档表示嵌入
            question: 问题文本
            options: 选项列表
            
        Returns:
            包含答案和置信度的字典
        """
        # 使用分类头预测答案
        logits = self.classification_head(document_embedding)
        probs = torch.softmax(logits / self.temperature, dim=1)
        
        # 获取预测类别
        pred_idx = torch.argmax(probs, dim=1).item()
        
        # 映射到选项
        if pred_idx < len(options):
            answer = options[pred_idx]
        else:
            answer = options[0]  # 默认选第一个
        
        # 计算置信度
        confidence = probs[0, pred_idx].item()
        
        # 判断是否需要使用LMM（如果可用）
        if self.lmm_initialized and confidence < self.confidence_threshold:
            logger.info(f"分类置信度 ({confidence:.4f}) 低于阈值 ({self.confidence_threshold})，尝试使用LMM生成答案")
            try:
                lmm_result = self._generate_with_lmm(document_embedding, question, options)
                return lmm_result
            except Exception as e:
                logger.error(f"LMM生成失败: {e}")
        
        # 返回结果
        return {
            'answer': answer,
            'confidence': confidence,
            'method': 'classification_head',
            'probabilities': {opt: probs[0, i].item() for i, opt in enumerate(options) if i < len(probs[0])}
        }
    
    @error_handler
    @performance_monitor
    def answer_open_question(self, 
                           document_embedding: torch.Tensor, 
                           question: str) -> Dict[str, Any]:
        """
        回答开放性问题（复赛）
        
        Args:
            document_embedding: 文档表示嵌入
            question: 问题文本
            
        Returns:
            包含答案和置信度的字典
        """
        # 如果LMM已初始化，使用LMM生成答案
        if self.lmm_initialized:
            return self._generate_with_lmm(document_embedding, question)
        
        # 否则，使用占位符答案
        # 在实际应用中，这里需要实现真正的答案生成逻辑
        answer = "该问题需要访问文档内容才能回答。"
        
        # 使用不确定性估计器获取置信度
        uncertainty_result = self.uncertainty_estimator.predict_with_uncertainty(
            document_embedding, 
            options=None  # 非分类任务
        )
        
        confidence = uncertainty_result.get('confidence', 0.5)  # 默认中等置信度
        
        return {
            'answer': answer,
            'confidence': confidence,
            'method': 'placeholder',
        }
    
    @error_handler
    @performance_monitor
    def _generate_with_lmm(self, 
                          document_embedding: torch.Tensor, 
                          question: str,
                          options: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        使用LMM生成答案
        
        Args:
            document_embedding: 文档表示嵌入
            question: 问题文本
            options: 选项列表 (如果是选择题)
            
        Returns:
            包含生成答案和置信度的字典
        """
        if not self.lmm_initialized:
            raise ValueError("LMM未初始化")
        
        # 投影嵌入以适应LMM
        projected_embedding = self.text_generation_projection(document_embedding)
        
        # 构建提示词
        prompt = self.preprocess_question(question)
        if options:
            prompt += f" 选项: {', '.join(options)}"
        
        try:
            # 使用实际的LMM模型生成答案
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                # 使用transformers API
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.lmm_model.device)
                
                # 将文档嵌入注入到模型中
                # 注意：具体实现取决于模型架构，这里使用一个通用方法
                if hasattr(self.lmm_model, 'condition_on_embeddings'):
                    # 如果模型支持条件嵌入
                    self.lmm_model.condition_on_embeddings(projected_embedding)
                
                # 生成参数
                gen_kwargs = {
                    "max_length": len(inputs.input_ids[0]) + self.max_answer_length if hasattr(self, 'max_answer_length') else 100,
                    "temperature": self.temperature,
                    "top_k": self.top_k if hasattr(self, 'top_k') else 50,
                    "top_p": 0.95,
                    "do_sample": True,
                    "num_return_sequences": 1,
                }
                
                # 生成答案
                with torch.no_grad():
                    outputs = self.lmm_model.generate(**inputs, **gen_kwargs)
                
                # 解码生成的文本
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 提取答案部分（去除提示词部分）
                answer = generated_text[len(prompt):].strip()
                
                # 计算置信度（使用生成概率）
                if hasattr(outputs, 'sequences_scores'):
                    confidence = torch.sigmoid(outputs.sequences_scores[0]).item()
                else:
                    confidence = 0.8  # 默认置信度
            else:
                # 回退到简单实现
                logger.warning("使用简单实现生成答案")
                if options:
                    # 选择题：随机选择一个选项
                    import random
                    answer = random.choice(options)
                else:
                    # 开放题：生成一个简单答案
                    answer = "根据文档内容，无法提供准确答案。"
                confidence = 0.5
        except Exception as e:
            logger.error(f"LMM生成答案失败: {e}")
            # 回退到简单实现
            if options:
                answer = options[0]  # 默认选第一个选项
            else:
                answer = "生成答案时发生错误。"
            confidence = 0.3
        
        # 后处理答案
        processed_answer = self.postprocess_answer(answer, options)
        
        return {
            'answer': processed_answer,
            'raw_answer': answer,
            'confidence': confidence,
            'method': 'lmm_generation',
        }
    
    @error_handler
    @performance_monitor
    def answer_question(self, 
                       document_embedding: torch.Tensor, 
                       question: str,
                       options: Optional[List[str]] = None) -> str:
        """
        回答问题的主入口函数
        
        Args:
            document_embedding: 文档表示嵌入
            question: 问题文本
            options: 选项列表 (如果是选择题)
            
        Returns:
            答案文本
        """
        try:
            # 预处理问题
            processed_question = self.preprocess_question(question)
            
            # 根据是否有选项决定问题类型
            if options and len(options) > 0:
                # 选择题（初赛）
                result = self.answer_classification_question(
                    document_embedding, 
                    processed_question,
                    options
                )
            else:
                # 开放性问题（复赛）
                result = self.answer_open_question(
                    document_embedding, 
                    processed_question
                )
            
            # 记录详细信息
            logger.info(f"问题: {question}")
            logger.info(f"答案: {result['answer']} (置信度: {result.get('confidence', 'N/A')}, 方法: {result.get('method', 'N/A')})")
            
            # 返回答案文本
            return result['answer']
            
        except Exception as e:
            logger.error(f"回答问题时发生错误: {e}")
            # 返回默认答案
            return "A" if options and len(options) > 0 else "处理问题时发生错误"


class EnhancedReasoningQAModule(ReasoningQAModule):
    """
    增强型推理与问答模块，添加了更多高级特性
    """
    
    def __init__(self, lmm_model_name: str, embedding_dim: int = 512, 
                temperature: float = 1.0, confidence_threshold: float = 0.7,
                max_answer_length: int = 100, top_k: int = 5,
                optimization_config: Optional[OptimizationConfig] = None):
        """
        初始化增强型推理与问答模块
        
        Args:
            lmm_model_name: 大型多模态模型名称或路径
            embedding_dim: 嵌入维度
            temperature: 温度参数，影响输出的多样性
            confidence_threshold: 置信度阈值
            max_answer_length: 最大答案长度
            top_k: 生成时的top-k采样参数
            optimization_config: 优化配置
        """
        super().__init__(
            lmm_model_name=lmm_model_name,
            embedding_dim=embedding_dim,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            optimization_config=optimization_config
        )
        self.max_answer_length = max_answer_length
        self.top_k = top_k
        
        # 添加自校正机制
        self.self_correction = True
        
        # 增强不确定性估计
        self.enhanced_uncertainty = True
        
        # 领域知识注入
        self.domain_knowledge_enhanced = False
        
        # 多样性生成
        self.diverse_generation = False
    
    def answer_with_self_correction(self, 
                                  document_embedding: torch.Tensor, 
                                  question: str,
                                  options: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        带自校正机制的问答
        
        Args:
            document_embedding: 文档表示嵌入
            question: 问题文本
            options: 选项列表 (如果是选择题)
            
        Returns:
            包含答案和置信度的字典
        """
        # 首次生成答案
        initial_result = super().answer_question(document_embedding, question, options)
        initial_answer = initial_result if isinstance(initial_result, str) else initial_result.get('answer', '')
        
        # 如果启用自校正且置信度较低，生成第二个答案并比较
        if self.self_correction and hasattr(self, 'uncertainty_estimator'):
            uncertainty_result = self.uncertainty_estimator.predict_with_uncertainty(
                document_embedding, 
                options
            )
            
            confidence = uncertainty_result.get('max_confidence', 
                         uncertainty_result.get('confidence', 0.9)).item()
            
            if confidence < self.confidence_threshold:
                # 修改问题措辞进行二次确认
                verification_question = f"请仔细检查并确认：{question}"
                second_result = super().answer_question(document_embedding, verification_question, options)
                second_answer = second_result if isinstance(second_result, str) else second_result.get('answer', '')
                
                # 如果两次答案不同，选择更高置信度的一个
                if second_answer != initial_answer:
                    logger.info(f"自校正激活：初始答案={initial_answer}，二次确认答案={second_answer}")
                    
                    # 再次评估第二个答案的置信度
                    second_uncertainty = self.uncertainty_estimator.predict_with_uncertainty(
                        document_embedding, 
                        options
                    )
                    
                    second_confidence = second_uncertainty.get('max_confidence',
                                       second_uncertainty.get('confidence', 0.5)).item()
                    
                    # 比较置信度，选择更高的
                    if second_confidence > confidence:
                        return {
                            'answer': second_answer,
                            'confidence': second_confidence,
                            'method': 'self_correction',
                            'initial_answer': initial_answer,
                            'initial_confidence': confidence
                        }
        
        # 返回初始答案
        return {
            'answer': initial_answer,
            'confidence': confidence if 'confidence' in locals() else 0.9,
            'method': 'standard'
        }
    
    @error_handler
    @performance_monitor
    def answer_question(self, 
                       document_embedding: torch.Tensor, 
                       question: str,
                       options: Optional[List[str]] = None) -> str:
        """
        增强型回答问题函数
        
        Args:
            document_embedding: 文档表示嵌入
            question: 问题文本
            options: 选项列表 (如果是选择题)
            
        Returns:
            答案文本
        """
        try:
            # 使用自校正机制
            if self.self_correction:
                result = self.answer_with_self_correction(document_embedding, question, options)
                return result['answer']
            else:
                # 使用基类的标准回答
                return super().answer_question(document_embedding, question, options)
                
        except Exception as e:
            logger.error(f"增强型回答问题时发生错误: {e}")
            # 返回默认答案
            return "A" if options and len(options) > 0 else "处理问题时发生错误"


if __name__ == "__main__":
    # 测试推理与问答模块
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    embedding_dim = 512
    batch_size = 1
    document_embedding = torch.randn(batch_size, embedding_dim)
    
    # 创建优化配置
    optimization_config = OptimizationConfig(
        use_quantization=True,
        quantization_bits=8,
        use_cache=True,
        use_mixed_precision=True,
        attention_optimization=AttentionOptimizationConfig(
            use_sparse_attention=True,
            sparse_attention_threshold=0.1,
            use_head_pruning=True,
            head_pruning_ratio=0.3,
            use_attention_cache=True,
            use_flash_attention=True,
            use_sliding_window=True,
            window_size=512
        )
    )
    
    # 初始化QA模块
    qa_module = ReasoningQAModule(
        lmm_model_name="placeholder/lmm-model",
        embedding_dim=embedding_dim,
        optimization_config=optimization_config
    )
    
    # 测试选择题（初赛）
    question = "根据文本信息，以下哪个描述符合该静电除尘器的特征？"
    options = ["A. 具有平行于外壳主轴线的垂直方向的片状沉积电极。", 
              "B. 具有管状入口和出口，它们分别由3种不同圆锥形部分所构成", 
              "C. 管状入口具有单个圆锥形部分，达到外壳直径的80至95%，剩余部分采用台阶形式。", 
              "D. 主要用于液体的除尘"]
    
    print("测试选择题答案...")
    answer = qa_module.answer_question(document_embedding, question, [opt[0] for opt in options])
    print(f"答案: {answer}")
    
    # 测试开放题（复赛）
    open_question = "在文件中第7页的图片中，部件4相对于部件5在图片中的位置关系是？"
    
    print("\n测试开放题答案...")
    open_answer = qa_module.answer_question(document_embedding, open_question)
    print(f"答案: {open_answer}")
    
    # 测试增强型QA模块
    print("\n测试增强型问答模块...")
    enhanced_qa = EnhancedReasoningQAModule(
        lmm_model_name="placeholder/lmm-model",
        embedding_dim=embedding_dim,
        optimization_config=optimization_config
    )
    
    enhanced_answer = enhanced_qa.answer_question(document_embedding, question, [opt[0] for opt in options])
    print(f"增强型答案: {enhanced_answer}")