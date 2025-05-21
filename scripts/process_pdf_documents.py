#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF文档处理脚本

此脚本用于处理单个PDF文件或批量处理目录中的PDF文件，
将PDF文档转换为结构化的多模态信息，包括文本、图像和版面数据。

用法:
    python process_pdf_documents.py --input path/to/pdf_or_directory --output path/to/output_dir [--recursive] [--visualize]
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 添加项目根目录到Python路径，以便导入model包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.pdf_processor import process_pdf

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("pdf_processor_script")

def find_pdf_files(input_path: str, recursive: bool = False) -> List[str]:
    """查找PDF文件"""
    pdf_files = []
    
    if os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
        pdf_files.append(input_path)
    elif os.path.isdir(input_path):
        if recursive:
            # 递归查找所有PDF文件
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
        else:
            # 只查找当前目录下的PDF文件
            for file in os.listdir(input_path):
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(input_path, file))
    
    return pdf_files

def process_single_pdf(args: Dict[str, Any]) -> Dict[str, Any]:
    """处理单个PDF文件的包装函数，适用于并行处理"""
    pdf_path = args['pdf_path']
    output_dir = args['output_dir']
    
    # 创建PDF专属的输出目录
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_dir = os.path.join(output_dir, pdf_basename)
    
    start_time = time.time()
    logger.info(f"开始处理: {pdf_path}")
    
    try:
        # 调用PDF处理函数
        processed_data = process_pdf(pdf_path, pdf_output_dir)
        
        # 保存处理结果为JSON文件
        metadata_path = os.path.join(pdf_output_dir, "metadata.json")
        
        # 创建一个可序列化的结果
        serializable_data = []
        for page in processed_data if processed_data else []:
            # 克隆页面数据，移除不可序列化的部分
            page_copy = page.copy()
            # 处理layout和text_blocks中的数据
            for block_list in ['layout', 'text_blocks']:
                if block_list in page_copy:
                    page_copy[block_list] = [
                        {k: (list(v) if k == 'coordinates' else v) 
                         for k, v in block.items()}
                        for block in page_copy[block_list]
                    ]
            serializable_data.append(page_copy)
            
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        
        processing_time = time.time() - start_time
        result = {
            'pdf_path': pdf_path,
            'output_dir': pdf_output_dir,
            'success': processed_data is not None,
            'page_count': len(processed_data) if processed_data else 0,
            'processing_time': processing_time,
            'metadata_path': metadata_path if processed_data else None
        }
        
        logger.info(f"处理完成: {pdf_path} ({result['page_count']} 页, {processing_time:.2f} 秒)")
        return result
        
    except Exception as e:
        logger.error(f"处理 {pdf_path} 时出错: {e}")
        processing_time = time.time() - start_time
        return {
            'pdf_path': pdf_path,
            'output_dir': pdf_output_dir,
            'success': False,
            'error': str(e),
            'processing_time': processing_time
        }

def visualize_results(results: List[Dict[str, Any]]) -> None:
    """可视化处理结果"""
    # 统计成功和失败的数量
    success_count = sum(1 for r in results if r['success'])
    failure_count = len(results) - success_count
    
    # 计算总处理时间和页数
    total_time = sum(r['processing_time'] for r in results)
    total_pages = sum(r.get('page_count', 0) for r in results)
    
    # 打印摘要
    print("\n" + "="*50)
    print(f"PDF处理摘要:")
    print(f"总文件数: {len(results)}")
    print(f"成功处理: {success_count}")
    print(f"处理失败: {failure_count}")
    print(f"总页数: {total_pages}")
    print(f"总处理时间: {total_time:.2f} 秒")
    if total_pages > 0:
        print(f"平均每页处理时间: {total_time/total_pages:.2f} 秒")
    print("="*50)
    
    # 如果有失败的文件，列出它们
    if failure_count > 0:
        print("\n失败的文件:")
        for r in results:
            if not r['success']:
                print(f"- {r['pdf_path']}: {r.get('error', 'Unknown error')}")
        print()
    
    # 显示处理结果的位置
    print("处理结果保存在:")
    for r in results:
        if r['success']:
            print(f"- {r['pdf_path']} -> {r['output_dir']}")
    print()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='处理PDF文档，提取文本、图像和版面信息')
    parser.add_argument('--input', '-i', required=True, help='输入PDF文件或包含PDF文件的目录')
    parser.add_argument('--output', '-o', required=True, help='输出目录')
    parser.add_argument('--recursive', '-r', action='store_true', help='递归处理子目录中的PDF文件')
    parser.add_argument('--visualize', '-v', action='store_true', help='可视化处理结果')
    parser.add_argument('--workers', '-w', type=int, default=max(1, mp.cpu_count() - 1), 
                        help='并行处理的工作进程数')
    parser.add_argument('--sequential', '-s', action='store_true', 
                        help='使用顺序处理而非并行处理')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 查找PDF文件
    pdf_files = find_pdf_files(args.input, args.recursive)
    
    if not pdf_files:
        logger.error(f"未找到PDF文件: {args.input}")
        return
    
    logger.info(f"找到 {len(pdf_files)} 个PDF文件")
    
    # 准备处理任务
    tasks = [{'pdf_path': pdf, 'output_dir': args.output} for pdf in pdf_files]
    results = []
    
    start_time = time.time()
    
    # 处理PDF文件
    if not args.sequential and len(pdf_files) > 1 and args.workers > 1:
        # 并行处理
        logger.info(f"使用 {args.workers} 个工作进程并行处理")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # 提交所有任务
            futures = [executor.submit(process_single_pdf, task) for task in tasks]
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"任务执行失败: {e}")
    else:
        # 顺序处理
        logger.info("使用顺序处理模式")
        for task in tasks:
            result = process_single_pdf(task)
            results.append(result)
    
    total_time = time.time() - start_time
    logger.info(f"所有PDF处理完成，总耗时: {total_time:.2f} 秒")
    
    # 可视化结果
    if args.visualize:
        visualize_results(results)
    
    # 输出处理摘要
    success_count = sum(1 for r in results if r['success'])
    logger.info(f"成功: {success_count}/{len(results)}, 总耗时: {total_time:.2f} 秒")

if __name__ == "__main__":
    main()