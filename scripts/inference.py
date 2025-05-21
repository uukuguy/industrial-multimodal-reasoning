#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工业技术文档多模态推理问答系统的高效推理脚本

支持特性：
- 高性能批量推理
- 多种精度优化选项 (FP16/INT8)
- ONNX和TensorRT加速
- KV缓存和注意力优化
- 统计与性能监控
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm
import torch
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.inference_optimizer import InferenceOptimizer
from model.pdf_processor import process_pdf

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('inference.log')
    ]
)
logger = logging.getLogger(__name__)


def load_questions(questions_file: str) -> List[Dict[str, Any]]:
    """加载问题文件"""
    questions = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    question = json.loads(line)
                    questions.append(question)
                except json.JSONDecodeError:
                    logger.warning(f"跳过无效的JSON行: {line}")
    
    logger.info(f"已加载 {len(questions)} 个问题")
    return questions


def preprocess_pdf_documents(documents_dir: str, questions: List[Dict[str, Any]], 
                            processed_data_dir: Optional[str] = None,
                            use_cache: bool = True, workers: int = 1) -> Dict[str, Any]:
    """预处理PDF文档，支持缓存加速"""
    documents_data = {}
    unique_docs = set(q.get('document', '') for q in questions if q.get('document'))
    
    logger.info(f"需要处理 {len(unique_docs)} 个唯一文档")
    
    # 创建处理目录
    if processed_data_dir and not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir, exist_ok=True)
    
    # 处理每个文档
    for doc_name in tqdm(unique_docs, desc="处理文档"):
        doc_path = os.path.join(documents_dir, doc_name)
        
        # 检查缓存
        cache_path = None
        if processed_data_dir and use_cache:
            cache_path = os.path.join(processed_data_dir, f"{os.path.splitext(doc_name)[0]}.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        documents_data[doc_name] = json.load(f)
                    logger.info(f"从缓存加载文档: {doc_name}")
                    continue
                except Exception as e:
                    logger.warning(f"缓存加载失败: {e}, 将重新处理文档")
        
        # 处理文档
        try:
            if os.path.exists(doc_path):
                process_output_dir = processed_data_dir if processed_data_dir else None
                
                # 处理PDF
                result = process_pdf(
                    pdf_path=doc_path,
                    output_dir=process_output_dir,
                    extract_text=True,
                    extract_images=True,
                    extract_layout=True,
                    compute_embeddings=True,
                    visualize=False
                )
                
                documents_data[doc_name] = result
                
                # 保存到缓存
                if cache_path:
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    
                logger.info(f"处理文档完成: {doc_name}")
            else:
                logger.error(f"文档不存在: {doc_path}")
        except Exception as e:
            logger.error(f"处理文档失败 {doc_name}: {e}")
    
    logger.info(f"文档预处理完成，共 {len(documents_data)} 个文档")
    return documents_data


def prepare_inference_inputs(questions: List[Dict[str, Any]], documents_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """准备推理输入"""
    inference_inputs = []
    
    for q in questions:
        q_id = q.get('id', '')
        question_text = q.get('question', '')
        document_name = q.get('document', '')
        options = q.get('options', [])
        
        # 确保文档数据可用
        if document_name not in documents_data:
            logger.warning(f"问题 {q_id} 引用的文档 {document_name} 未找到，跳过")
            continue
        
        # 准备输入
        input_item = {
            'id': q_id,
            'question': question_text,
            'document_data': documents_data[document_name],
            'options': options
        }
        
        inference_inputs.append(input_item)
    
    logger.info(f"已准备 {len(inference_inputs)} 个推理输入")
    return inference_inputs


def run_inference(model, inference_inputs: List[Dict[str, Any]], 
                 batch_size: int = 1, device: str = None) -> List[Dict[str, Any]]:
    """运行批量推理"""
    results = []
    total_time = 0
    
    # 创建推理优化器
    optimizer = model if isinstance(model, InferenceOptimizer) else InferenceOptimizer(
        model=model,
        device=device,
        batch_size=batch_size,
        use_fp16=True,
        use_kv_cache=True
    )
    
    # 批量处理问题
    for i in range(0, len(inference_inputs), batch_size):
        batch = inference_inputs[i:i+batch_size]
        
        start_time = time.time()
        
        # 执行推理
        try:
            if batch_size > 1:
                # 动态批处理
                batch_outputs = optimizer.dynamic_batch_inference(batch)
            else:
                # 单样本推理
                batch_outputs = [optimizer.infer(item) for item in batch]
            
            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time
            
            # 处理输出
            for j, output in enumerate(batch_outputs):
                if i+j < len(inference_inputs):
                    question_item = inference_inputs[i+j]
                    
                    # 构建结果
                    result_item = {
                        'id': question_item['id'],
                        'output': output,
                        'inference_time': batch_time / len(batch)
                    }
                    
                    # 根据模型输出格式提取答案
                    if isinstance(output, dict):
                        if 'logits' in output:
                            # 分类任务（选择题）
                            logits = output['logits']
                            if isinstance(logits, torch.Tensor):
                                pred_idx = torch.argmax(logits, dim=-1).item()
                                result_item['answer'] = chr(ord('A') + pred_idx)
                        elif 'answer' in output:
                            # 直接提供答案
                            result_item['answer'] = output['answer']
                        elif 'generated_text' in output:
                            # 生成任务（开放题）
                            result_item['answer'] = output['generated_text']
                    
                    results.append(result_item)
            
            # 日志记录
            logger.info(f"批次 {i//batch_size + 1}/{(len(inference_inputs)-1)//batch_size + 1} 处理完成，"
                       f"用时 {batch_time:.4f}s，"
                       f"平均 {batch_time/len(batch):.4f}s/样本")
            
        except Exception as e:
            logger.error(f"批次推理失败: {e}")
            
            # 回退到逐个处理
            for item in batch:
                try:
                    output = optimizer.infer(item)
                    result_item = {
                        'id': item['id'],
                        'output': output
                    }
                    
                    # 提取答案
                    if isinstance(output, dict):
                        if 'logits' in output:
                            logits = output['logits']
                            if isinstance(logits, torch.Tensor):
                                pred_idx = torch.argmax(logits, dim=-1).item()
                                result_item['answer'] = chr(ord('A') + pred_idx)
                        elif 'answer' in output:
                            result_item['answer'] = output['answer']
                        elif 'generated_text' in output:
                            result_item['answer'] = output['generated_text']
                    
                    results.append(result_item)
                except Exception as e2:
                    logger.error(f"样本 {item['id']} 推理失败: {e2}")
    
    # 清理缓存
    optimizer.clear_cache()
    
    # 记录总体性能
    if inference_inputs:
        logger.info(f"推理完成，总时间: {total_time:.2f}s，"
                  f"平均: {total_time/len(inference_inputs):.4f}s/样本，"
                  f"处理速度: {len(inference_inputs)/total_time:.2f}样本/秒")
    
    return results


def format_results(results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """格式化结果为评测所需格式"""
    formatted = []
    
    for item in results:
        if 'id' not in item:
            logger.warning(f"结果项缺少id，跳过: {item}")
            continue
        
        # 提取答案
        answer = item.get('answer', '')
        if not answer and 'output' in item:
            output = item['output']
            if isinstance(output, dict):
                if 'answer' in output:
                    answer = output['answer']
                elif 'generated_text' in output:
                    answer = output['generated_text']
        
        # 添加格式化结果
        formatted.append({
            'id': item['id'],
            'answer': answer
        })
    
    return formatted


def save_results(results: List[Dict[str, str]], output_file: str) -> None:
    """保存结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"结果已保存至: {output_file}, 共 {len(results)} 条")


def load_model(model_path: str, config: Dict[str, Any]) -> InferenceOptimizer:
    """加载并优化模型"""
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    use_fp16 = config.get('fp16', True)
    use_int8 = config.get('int8', False)
    use_cuda_graph = config.get('cuda_graph', False)
    use_kv_cache = config.get('kv_cache', True)
    use_onnx = config.get('onnx', False)
    batch_size = config.get('batch_size', 1)
    
    logger.info(f"加载模型: {model_path}")
    logger.info(f"推理配置: 设备={device}, FP16={use_fp16}, INT8={use_int8}, "
              f"CUDA图={use_cuda_graph}, KV缓存={use_kv_cache}, ONNX={use_onnx}, "
              f"批处理大小={batch_size}")
    
    # 初始化推理优化器
    optimizer = InferenceOptimizer(
        model_path=model_path,
        device=device,
        use_fp16=use_fp16,
        use_int8=use_int8,
        use_cuda_graph=use_cuda_graph,
        use_kv_cache=use_kv_cache,
        use_onnx=use_onnx,
        batch_size=batch_size,
        max_batch_size=config.get('max_batch_size', 16),
        dynamic_batch=config.get('dynamic_batch', True),
    )
    
    return optimizer


def main(args):
    """主函数"""
    # 记录开始时间
    start_time = time.time()
    
    # 加载问题
    questions = load_questions(args.questions)
    
    # 预处理文档
    documents_data = preprocess_pdf_documents(
        documents_dir=args.documents,
        questions=questions,
        processed_data_dir=args.processed_data,
        use_cache=not args.no_cache,
        workers=args.workers
    )
    
    # 准备推理输入
    inference_inputs = prepare_inference_inputs(questions, documents_data)
    
    # 配置推理参数
    inference_config = {
        'device': args.device,
        'fp16': not args.no_fp16, 
        'int8': args.int8,
        'cuda_graph': args.cuda_graph,
        'kv_cache': not args.no_kv_cache,
        'onnx': args.use_onnx,
        'batch_size': args.batch_size,
        'max_batch_size': args.max_batch_size,
        'dynamic_batch': not args.no_dynamic_batch
    }
    
    # 加载模型
    model = load_model(args.model, inference_config)
    
    # 运行推理
    results = run_inference(
        model=model,
        inference_inputs=inference_inputs,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # 格式化结果
    formatted_results = format_results(results)
    
    # 保存结果
    save_results(formatted_results, args.output)
    
    # 记录总时间
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"全部处理完成，总耗时: {total_time:.2f}s")
    
    # 性能统计
    if args.perf_stats and inference_inputs:
        perf_stats = {
            'total_questions': len(questions),
            'processed_questions': len(inference_inputs),
            'total_time': total_time,
            'avg_time_per_question': total_time / len(inference_inputs) if inference_inputs else 0,
            'questions_per_second': len(inference_inputs) / total_time if total_time > 0 else 0,
        }
        
        # 如果使用GPU，添加内存使用情况
        if torch.cuda.is_available():
            perf_stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            perf_stats['gpu_max_memory_allocated'] = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        stats_file = f"{os.path.splitext(args.output)[0]}_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(perf_stats, f, indent=2)
            
        logger.info(f"性能统计已保存至: {stats_file}")
        logger.info(f"处理速度: {perf_stats['questions_per_second']:.2f} 问题/秒")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='工业技术文档多模态推理问答系统高效推理脚本')
    
    # 数据参数
    parser.add_argument('--questions', type=str, required=True, help='问题文件路径，JSONL格式')
    parser.add_argument('--documents', type=str, required=True, help='文档目录路径')
    parser.add_argument('--processed_data', type=str, default=None, help='预处理数据目录，用于缓存')
    parser.add_argument('--output', type=str, required=True, help='输出结果文件路径')
    
    # 模型参数
    parser.add_argument('--model', type=str, required=True, help='模型路径或目录')
    parser.add_argument('--device', type=str, default=None, help='推理设备，如cuda:0、cpu')
    
    # 批处理参数
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    parser.add_argument('--max_batch_size', type=int, default=16, help='最大批处理大小')
    parser.add_argument('--no_dynamic_batch', action='store_true', help='禁用动态批处理')
    
    # 性能优化选项
    parser.add_argument('--no_fp16', action='store_true', help='禁用FP16混合精度')
    parser.add_argument('--int8', action='store_true', help='启用INT8量化')
    parser.add_argument('--cuda_graph', action='store_true', help='启用CUDA图优化')
    parser.add_argument('--no_kv_cache', action='store_true', help='禁用KV缓存')
    parser.add_argument('--use_onnx', action='store_true', help='使用ONNX运行时')
    
    # 其他选项
    parser.add_argument('--workers', type=int, default=1, help='处理线程数')
    parser.add_argument('--no_cache', action='store_true', help='禁用缓存，强制重新处理文档')
    parser.add_argument('--perf_stats', action='store_true', help='输出性能统计')
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 执行主函数
    main(args)