# -*- coding: utf-8 -*-

import os
import json
import time
import torch
import shutil
import logging
import argparse
import traceback
import multiprocessing as mp
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List, Dict, Any, Optional, Union, Tuple

# 导入优化后的模块
from .config import load_config, validate_config, init_config
from .pdf_processor import process_pdf
from .enhanced_model import EnhancedMultiModalModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("main")

def load_questions(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    从 JSONL 文件加载问题数据。

    Args:
        jsonl_path: 输入 JSONL 文件的路径。

    Returns:
        包含问题数据的字典列表。
    """
    questions = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                questions.append(json.loads(line))
        logger.info(f"成功加载 {len(questions)} 条问题数据从 {jsonl_path}")
    except FileNotFoundError:
        logger.error(f"错误: 文件未找到 {jsonl_path}")
    except json.JSONDecodeError as e:
        logger.error(f"错误: 解析 JSONL 文件失败 {jsonl_path} - {e}")
    return questions

def save_results(results: List[Dict[str, Any]], output_jsonl_path: str):
    """
    将结果保存到 JSONL 文件。

    Args:
        results: 包含结果数据的字典列表 (每个字典应包含 "id" 和 "answer" 键)。
        output_jsonl_path: 输出 JSONL 文件的路径。
    """
    # 过滤结果，确保只包含评测所需的字段
    filtered_results = []
    for result in results:
        filtered_result = {
            "id": result.get("id", "unknown"),
            "answer": result.get("answer", "")
        }
        filtered_results.append(filtered_result)
    
    output_dir = os.path.dirname(output_jsonl_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for result in filtered_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        logger.info(f"结果已成功保存到 {output_jsonl_path}")
    except IOError as e:
        logger.error(f"错误: 保存结果到文件失败 {output_jsonl_path} - {e}")

def init_worker():
    """
    进程池工作进程的初始化函数。
    设置信号处理和其他必要的初始化。
    """
    import signal
    # 忽略 SIGINT 信号，让主进程处理
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # 配置工作进程日志
    worker_logger = logging.getLogger("worker")
    worker_logger.info(f"工作进程 {os.getpid()} 已初始化")

def process_single_question(
    question_data: Dict[str, Any], 
    pdf_documents_dir: str, 
    config: Dict[str, Any],
    shared_model_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    处理单个问题的独立函数，用于并行处理
    
    Args:
        question_data: 包含问题信息的字典
        pdf_documents_dir: PDF文档目录
        config: 配置字典
        shared_model_info: 共享模型信息（如果支持多进程共享）
        
    Returns:
        包含处理结果的字典
    """
    # 配置工作进程日志
    worker_logger = logging.getLogger(f"worker.{os.getpid()}")
    
    # 生成唯一的问题ID，如果原始数据中没有
    q_id = question_data.get("id", f"unknown_{int(time.time() * 1000)}")
    # 初始化结果模板
    result = {
        "id": q_id, 
        "answer": "", 
        "status": "success",
        "error": None,
        "processing_time": 0
    }
    
    start_time = time.time()
    
    # 创建临时处理目录
    temp_dir = config['system']['temp_dir']
    output_processing_dir = os.path.join(temp_dir, str(q_id))
    
    try:
        # 1. 获取问题信息
        question_text = question_data.get("question", "")
        document_filename = question_data.get("document", "")
        options = question_data.get("options", [])  # 对于初赛的选择题
        
        # 验证输入数据
        if not question_text or not document_filename:
            result.update({
                "answer": "A" if options else "输入数据不完整",
                "status": "invalid_input",
                "error": "问题文本或文档名称缺失"
            })
            worker_logger.warning(f"问题 {q_id} 输入数据不完整")
            return result
            
        # 2. 检查PDF文件是否存在
        pdf_path = os.path.join(pdf_documents_dir, document_filename)
        if not os.path.exists(pdf_path):
            result.update({
                "answer": "A" if options else "文档不存在",
                "status": "file_not_found",
                "error": f"PDF文件未找到: {pdf_path}"
            })
            worker_logger.error(f"问题 {q_id} 的PDF文件不存在: {pdf_path}")
            return result
            
        # 3. 创建临时目录
        os.makedirs(output_processing_dir, exist_ok=True)
        
        # 4. 调用PDF处理模块处理文档
        worker_logger.info(f"开始处理PDF文档: {document_filename}")
        processed_data = process_pdf(pdf_path, output_processing_dir)
        if not processed_data:
            result.update({
                "answer": "A" if options else "文档处理失败",
                "status": "preprocessing_failed",
                "error": "PDF处理返回空结果"
            })
            worker_logger.error(f"问题 {q_id} 的PDF文档处理失败")
            return result
        worker_logger.info(f"PDF处理完成，共处理了 {len(processed_data)} 页")
        
        # 5. 初始化增强型多模态模型
        # 注意：如果支持多进程共享模型，可以从shared_model_info中获取模型
        if shared_model_info and 'model' in shared_model_info:
            model = shared_model_info['model']
            worker_logger.info("使用共享模型")
        else:
            worker_logger.info("初始化新模型实例")
            model = EnhancedMultiModalModel(**config)
        
        # 6. 编码文档
        worker_logger.info("开始编码文档...")
        document_encodings = model.encode_document(processed_data)
        
        # 7. 进行推理和问答
        worker_logger.info("开始进行推理和问答...")
        if options:
            # 选择题（初赛）- 只保留选项字母 (A, B, C, D)
            option_letters = [opt[0] if opt else "" for opt in options]
            qa_results = model(document_encodings, question_text, option_letters)
        else:
            # 开放题（复赛）
            qa_results = model(document_encodings, question_text)
        
        # 8. 提取答案
        answer = qa_results.get('answer', "A" if options else "无法回答")
        
        # 9. 构建结果
        result.update({
            "answer": answer,
            "status": "success",
            "confidence": qa_results.get('confidence', 1.0) if isinstance(qa_results.get('confidence'), (int, float)) else 1.0,
            "processing_time": time.time() - start_time
        })
        
        worker_logger.info(f"问题 {q_id} 处理完成，答案: {answer}, 耗时: {result['processing_time']:.2f}秒")
        
        return result
        
    except Exception as e:
        error_msg = traceback.format_exc()
        worker_logger.error(f"处理问题 {q_id} 时发生严重错误:\n{error_msg}")
        
        # 更新结果
        result.update({
            "answer": "A" if options else "处理失败",
            "status": "exception",
            "error": str(e),
            "processing_time": time.time() - start_time
        })
        
        return result
        
    finally:
        # 清理临时文件
        if os.path.exists(output_processing_dir):
            try:
                shutil.rmtree(output_processing_dir)
                worker_logger.info(f"已清理临时目录: {output_processing_dir}")
            except Exception as e:
                worker_logger.error(f"清理临时目录失败: {output_processing_dir}, 错误: {e}")

def main(questions_jsonl_path: str, pdf_documents_dir: str, output_jsonl_path: str, 
         config_path: Optional[str] = None, parallel: bool = True, workers: int = None):
    """
    主处理流程。

    Args:
        questions_jsonl_path: 输入问题 JSONL 文件的路径。
        pdf_documents_dir: 原始 PDF 文档所在的目录。
        output_jsonl_path: 输出结果 JSONL 文件的路径。
        config_path: 配置文件路径。
        parallel: 是否启用并行处理。
        workers: 工作进程数量，None表示自动决定。
    """
    start_time = time.time()
    logger.info(f"开始处理任务... 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载和初始化配置
    config = init_config(config_path)
    
    # 更新配置参数（如果指定）
    if workers is not None:
        config['system']['max_workers'] = workers
    
    # 2. 创建临时目录
    temp_dir = config['system']['temp_dir']
    os.makedirs(temp_dir, exist_ok=True)
    
    # 3. 加载问题数据
    questions = load_questions(questions_jsonl_path)
    if not questions:
        logger.error("没有加载到问题数据，处理结束。")
        return
    logger.info(f"共加载了 {len(questions)} 个问题")
    
    # 4. 初始化模型（如果不并行处理或模型可以跨进程共享）
    model = None
    shared_model_info = None
    
    if not parallel:
        try:
            logger.info("初始化增强型多模态模型...")
            model = EnhancedMultiModalModel(**config)
            shared_model_info = {'model': model}
            logger.info("模型初始化成功")
        except Exception as e:
            logger.error(f"初始化模型失败: {e}")
            logger.error("将尝试在每个工作进程中单独初始化模型")
    
    # 5. 处理所有问题
    results = []
    total_questions = len(questions)
    
    # 处理单个问题的函数包装
    process_fn = partial(
        process_single_question,
        pdf_documents_dir=pdf_documents_dir,
        config=config,
        shared_model_info=shared_model_info
    )
    
    if parallel and total_questions > 1:
        # 并行处理
        max_workers = config['system']['max_workers']
        if max_workers is None or max_workers <= 0:
            max_workers = max(1, mp.cpu_count() - 1)  # 使用CPU核心数-1作为默认值
        
        logger.info(f"使用并行处理模式，工作进程数: {max_workers}")
        
        try:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=init_worker,
                mp_context=mp.get_context('spawn')
            ) as executor:
                # 提交所有任务
                futures = [executor.submit(process_fn, q) for q in questions]
                
                # 处理结果
                for i, future in enumerate(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"问题 {i+1}/{total_questions} 完成: ID={result['id']}, 状态={result['status']}")
                    except Exception as e:
                        logger.error(f"获取问题 {i+1} 的结果时出错: {e}")
                        # 添加错误结果
                        q_id = questions[i].get("id", f"unknown_{i}")
                        results.append({
                            "id": q_id,
                            "answer": "A" if "options" in questions[i] else "处理失败",
                            "status": "future_error",
                            "error": str(e)
                        })
                        
        except Exception as e:
            logger.error(f"并行处理出错: {e}")
            logger.info("切换到顺序处理模式...")
            # 如果并行处理失败，切换到顺序处理
            results = []
            parallel = False
    
    # 顺序处理（如果不并行或并行失败）
    if not parallel or not results:
        logger.info("使用顺序处理模式...")
        for i, question in enumerate(questions):
            logger.info(f"处理问题 {i+1}/{total_questions}...")
            result = process_fn(question)
            results.append(result)
            logger.info(f"问题 {i+1}/{total_questions} 完成: ID={result['id']}, 状态={result['status']}")
    
    # 6. 保存结果
    logger.info(f"所有问题处理完成，保存结果...")
    save_results(results, output_jsonl_path)
    
    # 7. 输出统计信息
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"任务完成! 总耗时: {total_time:.2f}秒")
    logger.info(f"平均每个问题耗时: {total_time/total_questions:.2f}秒")
    
    # 计算成功率
    success_count = sum(1 for r in results if r.get('status') == 'success')
    logger.info(f"成功处理: {success_count}/{total_questions} ({success_count/total_questions*100:.2f}%)")
    
    # 统计不同错误类型
    error_types = {}
    for r in results:
        status = r.get('status')
        if status != 'success':
            error_types[status] = error_types.get(status, 0) + 1
    
    if error_types:
        logger.info("错误统计:")
        for error_type, count in error_types.items():
            logger.info(f"  {error_type}: {count} ({count/total_questions*100:.2f}%)")
    
    # 8. 清理临时文件
    if os.path.exists(temp_dir) and os.listdir(temp_dir) == []:
        try:
            os.rmdir(temp_dir)
            logger.info(f"已清理临时目录: {temp_dir}")
        except Exception as e:
            logger.warning(f"清理临时目录 {temp_dir} 失败: {e}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='工业技术文档多模态推理问答系统')
    
    parser.add_argument('--questions', type=str, required=True,
                        help='输入问题 JSONL 文件的路径')
    parser.add_argument('--documents', type=str, required=True,
                        help='原始 PDF 文档所在的目录')
    parser.add_argument('--output', type=str, required=True,
                        help='输出结果 JSONL 文件的路径')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--workers', type=int, default=None,
                        help='工作进程数量，默认为CPU核心数-1')
    parser.add_argument('--sequential', action='store_true',
                        help='使用顺序处理而非并行处理')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别')
    
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 运行主流程
    main(
        questions_jsonl_path=args.questions,
        pdf_documents_dir=args.documents,
        output_jsonl_path=args.output,
        config_path=args.config,
        parallel=not args.sequential,
        workers=args.workers
    )