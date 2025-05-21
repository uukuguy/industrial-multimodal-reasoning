#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练数据分析脚本
统计数据量、问题类型、文档分布等信息
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def count_documents(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """统计文档分布"""
    return Counter([item.get('document', '') for item in data])

def count_question_types(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """基于问题内容分析问题类型"""
    question_types = {
        '位置关系': 0,
        '功能描述': 0,
        '技术参数': 0,
        '结构组成': 0,
        '操作步骤': 0,
        '其他': 0
    }
    
    # 关键词匹配
    keywords = {
        '位置关系': ['位置', '相对', '方向', '上方', '下方', '左侧', '右侧'],
        '功能描述': ['功能', '作用', '用于', '目的', '解决', '优点'],
        '技术参数': ['参数', '角度', '数值', '大小', '尺寸', '温度', '压力', '速度'],
        '结构组成': ['结构', '组成', '部件', '组件', '构成', '装置'],
        '操作步骤': ['步骤', '操作', '首先', '然后', '之后', '调整', '安装', '使用'],
    }
    
    for item in data:
        question = item.get('question', '')
        matched = False
        
        for q_type, words in keywords.items():
            if any(word in question for word in words):
                question_types[q_type] += 1
                matched = True
                break
        
        if not matched:
            question_types['其他'] += 1
    
    return question_types

def count_groups(data: List[Dict[str, Any]]) -> Dict[int, int]:
    """统计问题组分布"""
    return Counter([item.get('group', 0) for item in data])

def count_options(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """统计答案选项分布"""
    return Counter([item.get('answer', '') for item in data])

def analyze_documents(data: List[Dict[str, Any]], documents_dir: str) -> Dict[str, Any]:
    """分析文档信息"""
    doc_stats = {}
    doc_counter = count_documents(data)
    
    for doc, count in doc_counter.items():
        doc_path = os.path.join(documents_dir, doc)
        doc_size = os.path.getsize(doc_path) if os.path.exists(doc_path) else 0
        
        doc_stats[doc] = {
            'questions_count': count,
            'file_size_kb': doc_size / 1024,
            'exists': os.path.exists(doc_path)
        }
    
    return doc_stats

def plot_statistics(data_stats: Dict[str, Any], output_dir: str) -> None:
    """绘制统计图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制问题类型分布
    plt.figure(figsize=(10, 6))
    plt.bar(data_stats['question_types'].keys(), data_stats['question_types'].values())
    plt.title('问题类型分布')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'question_types.png'))
    
    # 绘制答案分布
    plt.figure(figsize=(8, 6))
    plt.bar(data_stats['answer_distribution'].keys(), data_stats['answer_distribution'].values())
    plt.title('答案选项分布')
    plt.savefig(os.path.join(output_dir, 'answer_distribution.png'))
    
    # 绘制文档问题数量分布
    docs = sorted(data_stats['document_stats'].items(), key=lambda x: x[1]['questions_count'], reverse=True)
    doc_names = [doc[0] for doc in docs[:20]]  # 显示前20个文档
    question_counts = [doc[1]['questions_count'] for doc in docs[:20]]
    
    plt.figure(figsize=(12, 6))
    plt.bar(doc_names, question_counts)
    plt.title('文档问题数量分布（前20个文档）')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'document_distribution.png'))

def main():
    parser = argparse.ArgumentParser(description='分析训练数据')
    parser.add_argument('--questions', type=str, required=True,
                        help='问题JSONL文件路径')
    parser.add_argument('--documents', type=str, required=True,
                        help='文档目录路径')
    parser.add_argument('--output', type=str, default='data_analysis',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 加载数据
    data = load_jsonl(args.questions)
    
    # 统计数据
    data_stats = {
        'total_questions': len(data),
        'unique_documents': len(count_documents(data)),
        'question_types': count_question_types(data),
        'group_distribution': count_groups(data),
        'answer_distribution': count_options(data),
        'document_stats': analyze_documents(data, args.documents)
    }
    
    # 打印统计结果
    print(f"总问题数: {data_stats['total_questions']}")
    print(f"唯一文档数: {data_stats['unique_documents']}")
    print("\n问题类型分布:")
    for q_type, count in data_stats['question_types'].items():
        print(f"  {q_type}: {count} ({count/data_stats['total_questions']:.1%})")
    
    print("\n答案分布:")
    for answer, count in data_stats['answer_distribution'].items():
        print(f"  {answer}: {count} ({count/data_stats['total_questions']:.1%})")
    
    # 分析数据充分性
    print("\n数据充分性分析:")
    
    # 问题数量评估
    if data_stats['total_questions'] < 100:
        print("  问题数量严重不足，难以训练有效模型")
    elif data_stats['total_questions'] < 500:
        print("  问题数量较少，可能需要数据增强和迁移学习")
    elif data_stats['total_questions'] < 2000:
        print("  问题数量适中，使用预训练模型微调可能获得不错效果")
    else:
        print("  问题数量充足，适合大规模训练")
    
    # 文档多样性评估
    docs_per_question = data_stats['unique_documents'] / data_stats['total_questions']
    if docs_per_question < 0.1:
        print("  文档多样性低，可能存在严重过拟合风险")
    elif docs_per_question < 0.3:
        print("  文档多样性一般，需加强模型泛化能力")
    else:
        print("  文档多样性良好，有助于模型泛化")
    
    # 问题类型平衡性
    type_counts = list(data_stats['question_types'].values())
    max_type = max(type_counts)
    min_type = min([c for c in type_counts if c > 0])
    imbalance_ratio = max_type / min_type if min_type > 0 else float('inf')
    
    if imbalance_ratio > 10:
        print("  问题类型严重不平衡，建议进行均衡采样或加权训练")
    elif imbalance_ratio > 3:
        print("  问题类型分布不均，可能需要数据增强")
    else:
        print("  问题类型分布较均衡")
    
    # 输出结果到JSON
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'data_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(data_stats, f, ensure_ascii=False, indent=2)
    
    # 绘制图表
    plot_statistics(data_stats, args.output)
    
    print(f"\n分析结果已保存到: {args.output}")
    
    # 返回建议
    print("\n建议策略:")
    if data_stats['total_questions'] < 500:
        print("1. 采用强大的预训练基础模型")
        print("2. 使用数据增强技术生成更多训练样本")
        print("3. 实施低资源微调技术（如P-tuning、Adapter等）")
        print("4. 加入正则化策略防止过拟合")
    else:
        print("1. 针对不同问题类型建立专门的评估集")
        print("2. 使用多阶段微调策略")
        print("3. 考虑模型融合提高鲁棒性")
    
    # 具体问题类型建议
    if data_stats['question_types'].get('位置关系', 0) > data_stats['total_questions'] * 0.3:
        print("4. 强化视觉-文本对齐训练，提高空间关系理解能力")
    
    if data_stats['question_types'].get('技术参数', 0) > data_stats['total_questions'] * 0.3:
        print("5. 增强对数值、单位和技术参数的提取能力")

if __name__ == "__main__":
    main()