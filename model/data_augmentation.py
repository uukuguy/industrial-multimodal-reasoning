# -*- coding: utf-8 -*-

"""
数据增强模块

提供多种技术来扩充和增强训练数据，缓解数据量不足、类型不平衡等问题。
"""

import os
import json
import random
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple

# 尝试导入nlpaug库，用于文本增强
try:
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas
    import nlpaug.flow as naf
    HAS_NLPAUG = True
except ImportError:
    HAS_NLPAUG = False
    print("Warning: nlpaug library not found. Some augmentation features will be limited.")

# 尝试导入外部LLM调用库
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    
class DataAugmentation:
    """数据增强类，提供多种增强方法"""
    
    def __init__(self, seed: int = 42):
        """
        初始化数据增强类
        
        Args:
            seed: 随机种子，确保结果可复现
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # 初始化增强器
        self._init_augmenters()
    
    def _init_augmenters(self):
        """初始化文本增强器"""
        self.augmenters = {}
        
        if HAS_NLPAUG:
            # 同义词替换增强器
            self.augmenters['synonym'] = naw.SynonymAug(
                aug_src='wordnet', 
                lang='cmn' if os.path.exists('/usr/share/wordnet-cmn') else 'eng'
            )
            
            # 词语插入增强器
            self.augmenters['insert'] = naw.ContextualWordEmbsAug(
                model_path='bert-base-chinese' if os.path.exists('/usr/share/bert-base-chinese') else 'bert-base-uncased',
                action='insert'
            )
            
            # 回译增强器（如果可用）
            try:
                self.augmenters['back_trans'] = naf.Sequential([
                    nas.BackTranslationAug(
                        from_model_name='facebook/wmt19-zh-en',
                        to_model_name='facebook/wmt19-en-zh'
                    )
                ])
            except:
                pass
    
    def question_paraphrase(self, 
                           question: str, 
                           prob: float = 0.7, 
                           max_paraphrases: int = 2) -> List[str]:
        """
        问题改写增强
        
        Args:
            question: 原始问题
            prob: 应用增强的概率
            max_paraphrases: 每个问题生成的最大改写数量
            
        Returns:
            改写后的问题列表
        """
        if random.random() > prob:
            return [question]  # 按概率决定是否应用增强
            
        paraphrases = [question]  # 始终包含原始问题
        
        # 方法1: 使用nlpaug库进行改写
        if HAS_NLPAUG and self.augmenters:
            augmenters = list(self.augmenters.values())
            for _ in range(min(len(augmenters), max_paraphrases)):
                aug = random.choice(augmenters)
                try:
                    augmented = aug.augment(question)
                    if isinstance(augmented, list):
                        augmented = augmented[0]
                    if augmented != question and augmented not in paraphrases:
                        paraphrases.append(augmented)
                except Exception as e:
                    print(f"Augmentation error: {e}")
        
        # 方法2: 使用模板进行简单改写
        templates = [
            "请问{question}",
            "下列关于{question}的描述哪项正确？",
            "在文档中，{question}",
            "根据文档描述，{question}",
            "请根据图文信息判断：{question}"
        ]
        
        # 去除问题中可能已有的问号和模板词
        clean_q = question.replace("？", "").replace("?", "")
        for prefix in ["请问", "下列", "在文档中", "根据文档", "请根据"]:
            if clean_q.startswith(prefix):
                clean_q = clean_q[len(prefix):].lstrip("，,：: ")
        
        # 应用模板
        for template in templates:
            paraphrase = template.format(question=clean_q)
            if paraphrase not in paraphrases:
                paraphrases.append(paraphrase)
                if len(paraphrases) >= max_paraphrases + 1:  # +1是因为包含原始问题
                    break
        
        # 方法3: 如果有OpenAI API访问，使用GPT模型进行改写
        if HAS_OPENAI and len(paraphrases) < max_paraphrases + 1 and 'OPENAI_API_KEY' in os.environ:
            try:
                openai.api_key = os.environ['OPENAI_API_KEY']
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "你是一个中文问题改写助手。你需要保持原始问题的含义，但使用不同的表达方式。"},
                        {"role": "user", "content": f"请用2种不同的方式改写这个问题，保持原意但换种表达: {question}"}
                    ],
                    temperature=0.7,
                    max_tokens=100
                )
                
                # 处理响应
                if response and response.choices:
                    content = response.choices[0].message.content
                    for line in content.split('\n'):
                        line = line.strip()
                        if line and line not in paraphrases:
                            # 去除序号和其他格式字符
                            line = line.lstrip("1234567890.-) ")
                            paraphrases.append(line)
                        if len(paraphrases) >= max_paraphrases + 1:
                            break
            except Exception as e:
                print(f"OpenAI API error: {e}")
        
        return paraphrases[:max_paraphrases+1]  # 限制返回数量
    
    def augment_options(self, 
                       options: List[str], 
                       prob: float = 0.5) -> List[List[str]]:
        """
        选项增强（替换选项表述，保持含义）
        
        Args:
            options: 原始选项列表
            prob: 应用增强的概率
            
        Returns:
            增强后的选项列表（可能包含多组）
        """
        if random.random() > prob or not options:
            return [options]  # 按概率决定是否应用增强
            
        augmented_options_list = [options.copy()]  # 始终包含原始选项
        
        # 简单模板替换（适用于特定模式的选项）
        templates = {
            "位置": ["位于", "在", "处于", "放置在", "安装在"],
            "作用": ["用于", "功能是", "目的是", "作用是", "用途是"],
            "结构": ["由...组成", "包含", "构成有", "组成包括", "结构为"]
        }
        
        new_options = []
        for opt in options:
            new_opt = opt
            # 去除选项标签 (A. B. C. D.)
            if len(opt) > 2 and opt[0].isalpha() and opt[1] in ['.', '、', ' ', '：', ':']:
                new_opt = opt[2:].strip()
            
            # 应用模板替换
            for key, replacements in templates.items():
                if key in new_opt:
                    for replacement in replacements:
                        if replacement not in new_opt:
                            new_options.append(new_opt.replace(key, replacement))
            
        # 如果生成了新选项，随机替换一些现有选项
        if new_options:
            aug_options = options.copy()
            replace_indices = random.sample(range(len(options)), min(len(new_options), len(options)//2))
            for idx, new_opt in zip(replace_indices, new_options[:len(replace_indices)]):
                aug_options[idx] = new_opt
            
            augmented_options_list.append(aug_options)
        
        return augmented_options_list
    
    def generate_hard_negative_options(self, 
                                     correct_option: str, 
                                     document_text: str,
                                     num_options: int = 3) -> List[str]:
        """
        生成困难的负选项（干扰项）
        
        Args:
            correct_option: 正确选项
            document_text: 文档文本，用于提取相关信息生成干扰项
            num_options: 需要生成的选项数量
            
        Returns:
            生成的负选项列表
        """
        negative_options = []
        
        # 提取文档中的关键实体和数字
        import re
        entities = []
        
        # 提取数字和单位
        numbers = re.findall(r'\d+(?:\.\d+)?(?:mm|cm|m|kg|℃|度|%|秒|分钟|小时)?', document_text)
        
        # 提取可能的实体（假设是2-6个连续汉字）
        potential_entities = re.findall(r'[一-龥]{2,6}(?:装置|系统|模块|部件|组件|机构|工具|设备|机构|组|座|轴|板|管|杆)', document_text)
        entities.extend(potential_entities)
        
        # 基于文档内容创建负选项
        # 1. 数字修改 - 改变正确选项中的数字
        if any(num in correct_option for num in numbers):
            for num in numbers:
                if num in correct_option:
                    # 替换为文档中其他数字，或者修改数值
                    other_nums = [n for n in numbers if n != num]
                    if other_nums:
                        replacement = random.choice(other_nums)
                        negative_option = correct_option.replace(num, replacement)
                        if negative_option != correct_option:
                            negative_options.append(negative_option)
                    
                    # 修改数值
                    match = re.search(r'\d+(?:\.\d+)?', num)
                    if match:
                        original_num = float(match.group())
                        modified_num = original_num * (1.5 if random.random() > 0.5 else 0.5)
                        modified_str = str(int(modified_num) if modified_num.is_integer() else modified_num)
                        negative_option = correct_option.replace(match.group(), modified_str)
                        if negative_option != correct_option:
                            negative_options.append(negative_option)
        
        # 2. 实体替换 - 替换正确选项中的实体为其他实体
        for entity in entities:
            if entity in correct_option:
                other_entities = [e for e in entities if e != entity]
                if other_entities:
                    replacement = random.choice(other_entities)
                    negative_option = correct_option.replace(entity, replacement)
                    if negative_option != correct_option and negative_option not in negative_options:
                        negative_options.append(negative_option)
        
        # 3. 否定变换 - 添加或删除否定词
        negation_words = ["不", "没有", "无法", "不能"]
        has_negation = any(word in correct_option for word in negation_words)
        
        if has_negation:
            # 移除否定词
            for word in negation_words:
                if word in correct_option:
                    negative_option = correct_option.replace(word, "")
                    if negative_option != correct_option and negative_option not in negative_options:
                        negative_options.append(negative_option)
        else:
            # 添加否定词
            for word in negation_words:
                insert_positions = [m.start() for m in re.finditer(r'[是为能会]', correct_option)]
                if insert_positions:
                    pos = random.choice(insert_positions)
                    negative_option = correct_option[:pos] + word + correct_option[pos:]
                    if negative_option != correct_option and negative_option not in negative_options:
                        negative_options.append(negative_option)
        
        # 4. 如果生成的选项不足，使用简单变换
        if len(negative_options) < num_options:
            # 简单变换词语
            word_pairs = [
                ("上方", "下方"), ("左侧", "右侧"), ("内部", "外部"),
                ("水平", "垂直"), ("顺时针", "逆时针"), ("并联", "串联"),
                ("增加", "减少"), ("提高", "降低"), ("正向", "反向")
            ]
            
            for original, replacement in word_pairs:
                if original in correct_option:
                    negative_option = correct_option.replace(original, replacement)
                    if negative_option != correct_option and negative_option not in negative_options:
                        negative_options.append(negative_option)
                        if len(negative_options) >= num_options:
                            break
            
        # 随机选择指定数量的负选项返回
        random.shuffle(negative_options)
        return negative_options[:num_options]
    
    def document_based_generation(self, 
                                document_text: str, 
                                document_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        基于文档内容生成问题，无需外部大模型
        
        Args:
            document_text: 文档文本内容
            document_metadata: 文档元数据，包含标题、结构等信息
        
        Returns:
            生成的问题列表
        """
        questions = []
        
        # 基本信息提取
        title = document_metadata.get('title', '') if document_metadata else ''
        doc_type = document_metadata.get('type', '') if document_metadata else ''
        
        # 分段处理文档
        paragraphs = document_text.split('\n\n')
        paragraphs = [p for p in paragraphs if len(p.strip()) > 30]  # 过滤太短的段落
        
        # 关键词提取
        import re
        
        # 提取可能的部件名称
        parts = re.findall(r'[一-龥]{2,6}(?:装置|系统|模块|部件|组件|机构|工具|设备|组|座|轴|板|管|杆)', document_text)
        parts = list(set(parts))  # 去重
        
        # 提取技术参数
        parameters = re.findall(r'[一-龥]{2,6}(?:为|是|等于|大于|小于)[0-9]+(?:\.[0-9]+)?(?:mm|cm|m|kg|℃|度|%|秒|分钟|小时)?', document_text)
        parameters = list(set(parameters))  # 去重
        
        # 提取位置关系短语
        positions = re.findall(r'[一-龥]{2,6}(?:位于|安装在|设于|放置在|处于|固定于)[一-龥]{2,6}(?:上方|下方|左侧|右侧|内部|外部|前方|后方|之间)', document_text)
        positions = list(set(positions))  # 去重
        
        # 生成问题
        
        # 1. 生成位置关系问题
        if parts and positions:
            for pos in positions[:min(2, len(positions))]:
                # 找出位置描述中涉及的部件
                involved_parts = [part for part in parts if part in pos]
                if len(involved_parts) >= 1:
                    part = involved_parts[0]
                    for relation in ["上方", "下方", "左侧", "右侧", "内部", "外部", "前方", "后方"]:
                        if relation in pos:
                            # 提取位置关系中的另一个部件
                            other_part = pos.split(relation)[0].split("于")[-1] if "于" in pos.split(relation)[0] else ""
                            if other_part:
                                question = f"{part}相对于{other_part}的位置关系是？"
                                options = [
                                    f"位于{other_part}的{relation}",
                                    f"位于{other_part}的{'下方' if relation == '上方' else '上方' if relation == '下方' else '右侧' if relation == '左侧' else '左侧' if relation == '右侧' else '外部' if relation == '内部' else '内部' if relation == '外部' else '后方' if relation == '前方' else '前方'}",
                                    f"与{other_part}平行",
                                    f"与{other_part}垂直"
                                ]
                                
                                # 确定正确答案
                                answer = "A"  # 假设第一个选项是正确的
                                
                                questions.append({
                                    "question": question,
                                    "options": options,
                                    "answer": answer,
                                    "type": "位置关系"
                                })
        
        # 2. 生成结构组成问题
        if parts and len(parts) >= 3:
            # 随机选择2-3个部件
            selected_parts = random.sample(parts, min(3, len(parts)))
            part_str = "、".join(selected_parts)
            
            question = f"根据文档描述，以下哪项关于{title or '该装置'}结构组成的描述是正确的？"
            options = [
                f"包含{part_str}等部件",
                f"不包含{'和'.join(selected_parts[:2])}",
                f"主要由{'、'.join(random.sample(parts, min(2, len(parts))))}构成，不包含{selected_parts[-1]}",
                f"仅由{'和'.join(selected_parts[:2])}组成"
            ]
            
            # 假设第一个选项是正确的
            questions.append({
                "question": question,
                "options": options,
                "answer": "A",
                "type": "结构组成"
            })
        
        # 3. 生成功能描述问题
        if title and paragraphs:
            # 查找包含"作用"、"功能"、"目的"等词的段落
            function_paragraphs = [p for p in paragraphs if any(word in p for word in ["作用", "功能", "目的", "用于", "用途"])]
            
            if function_paragraphs:
                paragraph = function_paragraphs[0]
                question = f"根据文档描述，{title or '该技术'}的主要功能是什么？"
                
                # 从功能段落中提取关键短语
                functions = re.findall(r'用于[^，。；？！]+', paragraph)
                if not functions:
                    functions = re.findall(r'功能[是为][^，。；？！]+', paragraph)
                if not functions:
                    functions = re.findall(r'目的是[^，。；？！]+', paragraph)
                
                if functions:
                    correct_function = functions[0]
                    wrong_functions = []
                    
                    # 生成干扰选项
                    other_parts = [part for part in parts if part not in correct_function]
                    if other_parts:
                        wrong_functions.append(f"用于{random.choice(other_parts)}的组装")
                    
                    wrong_functions.append(f"不能{correct_function[2:]}" if correct_function.startswith("用于") else f"不是{correct_function[2:]}" if correct_function.startswith("功能是") else f"并非{correct_function[2:]}")
                    
                    while len(wrong_functions) < 3:
                        if other_parts:
                            wrong_functions.append(f"仅用于提高{random.choice(other_parts)}的稳定性")
                        else:
                            wrong_functions.append("用于降低生产成本")
                    
                    options = [correct_function] + wrong_functions[:3]
                    random.shuffle(options)
                    answer = chr(65 + options.index(correct_function))  # A, B, C, D
                    
                    questions.append({
                        "question": question,
                        "options": options,
                        "answer": answer,
                        "type": "功能描述"
                    })
        
        # 4. 生成技术参数问题
        if parameters:
            for param in parameters[:min(2, len(parameters))]:
                # 提取参数名称和数值
                param_name = re.search(r'[一-龥]{2,6}(?=为|是|等于|大于|小于)', param)
                param_value = re.search(r'[0-9]+(?:\.[0-9]+)?(?:mm|cm|m|kg|℃|度|%|秒|分钟|小时)?', param)
                
                if param_name and param_value:
                    param_name = param_name.group()
                    param_value = param_value.group()
                    
                    question = f"根据文档描述，{param_name}的数值是多少？"
                    
                    # 生成不同的数值选项
                    original_value = float(re.search(r'[0-9]+(?:\.[0-9]+)?', param_value).group())
                    unit = param_value[len(str(original_value)):] if len(param_value) > len(str(original_value)) else ""
                    
                    options = [
                        f"{original_value}{unit}",
                        f"{original_value * 0.5:.1f}{unit}",
                        f"{original_value * 1.5:.1f}{unit}",
                        f"{original_value * 2:.1f}{unit}"
                    ]
                    
                    questions.append({
                        "question": question,
                        "options": options,
                        "answer": "A",  # 假设第一个选项是正确的
                        "type": "技术参数"
                    })
        
        return questions
    
    def augment_dataset(self, 
                       dataset: List[Dict[str, Any]], 
                       document_texts: Dict[str, str],
                       augmentation_factor: float = 1.5,
                       min_samples_per_type: int = 20) -> List[Dict[str, Any]]:
        """
        增强整个数据集，确保每种问题类型都有足够样本
        
        Args:
            dataset: 原始数据集
            document_texts: 文档文本内容，以文档名为键
            augmentation_factor: 增强因子，控制生成数据量
            min_samples_per_type: 每种问题类型的最小样本数
            
        Returns:
            增强后的数据集
        """
        augmented_dataset = dataset.copy()
        
        # 问题类型统计
        type_counts = {
            '位置关系': 0,
            '功能描述': 0,
            '技术参数': 0,
            '结构组成': 0,
            '操作步骤': 0,
            '其他': 0
        }
        
        # 按文档组织数据，方便文档级增强
        docs_questions = {}
        
        # 关键词匹配
        keywords = {
            '位置关系': ['位置', '相对', '方向', '上方', '下方', '左侧', '右侧'],
            '功能描述': ['功能', '作用', '用于', '目的', '解决', '优点'],
            '技术参数': ['参数', '角度', '数值', '大小', '尺寸', '温度', '压力', '速度'],
            '结构组成': ['结构', '组成', '部件', '组件', '构成', '装置'],
            '操作步骤': ['步骤', '操作', '首先', '然后', '之后', '调整', '安装', '使用'],
        }
        
        # 分类现有问题和统计
        for item in dataset:
            doc = item.get('document', '')
            if doc not in docs_questions:
                docs_questions[doc] = []
            
            # 确定问题类型
            question = item.get('question', '')
            q_type = None
            for t, words in keywords.items():
                if any(word in question for word in words):
                    q_type = t
                    break
            
            if not q_type:
                q_type = '其他'
            
            # 更新统计
            type_counts[q_type] += 1
            
            # 将问题类型添加到项目中
            item_with_type = item.copy()
            item_with_type['type'] = q_type
            docs_questions[doc].append(item_with_type)
        
        # 为不足的问题类型生成额外样本
        for q_type, count in type_counts.items():
            if count < min_samples_per_type:
                needed = min_samples_per_type - count
                print(f"需要为 {q_type} 类型生成 {needed} 个额外样本")
                
                # 从现有同类型问题中抽样增强
                existing_questions = [item for item in augmented_dataset 
                                      if any(word in item.get('question', '') for word in keywords.get(q_type, []))]
                
                if existing_questions:
                    sample_size = min(len(existing_questions), needed)
                    for item in random.sample(existing_questions, sample_size):
                        # 问题改写增强
                        doc = item.get('document', '')
                        original_question = item.get('question', '')
                        options = item.get('options', [])
                        
                        paraphrases = self.question_paraphrase(original_question, prob=0.9, max_paraphrases=2)
                        for i, new_question in enumerate(paraphrases):
                            if new_question != original_question:
                                new_item = item.copy()
                                new_item['question'] = new_question
                                new_item['id'] = f"{item.get('id', 0)}_aug_{i}"
                                new_item['type'] = q_type
                                augmented_dataset.append(new_item)
                                needed -= 1
                                if needed <= 0:
                                    break
                
                # 如果还不足，尝试从文档生成新问题
                if needed > 0:
                    for doc, text in document_texts.items():
                        if needed <= 0:
                            break
                            
                        # 只为该文档生成少量问题，避免过于集中
                        doc_gen_count = min(3, needed)
                        
                        # 生成新问题
                        generated = self.document_based_generation(text, {'title': doc})
                        
                        # 筛选指定类型的问题
                        type_questions = [q for q in generated if q.get('type') == q_type]
                        
                        for i, gen_q in enumerate(type_questions[:doc_gen_count]):
                            new_item = {
                                'id': f"gen_{doc}_{i}",
                                'question': gen_q.get('question', ''),
                                'document': doc,
                                'options': gen_q.get('options', []),
                                'answer': gen_q.get('answer', ''),
                                'type': q_type,
                                'generated': True
                            }
                            augmented_dataset.append(new_item)
                            needed -= 1
                            if needed <= 0:
                                break
        
        # 对数据集进行平衡采样，确保最终大小合理
        target_size = int(len(dataset) * augmentation_factor)
        if len(augmented_dataset) > target_size:
            # 确保原始数据集中的所有样本都保留
            original_ids = {item.get('id') for item in dataset}
            original_samples = [item for item in augmented_dataset if item.get('id') in original_ids]
            augmented_samples = [item for item in augmented_dataset if item.get('id') not in original_ids]
            
            # 计算各类型需要的样本数
            type_targets = {}
            remaining = target_size - len(original_samples)
            for q_type in type_counts.keys():
                type_targets[q_type] = min_samples_per_type
                remaining -= min_samples_per_type
            
            # 剩余额度按比例分配
            total_count = sum(type_counts.values())
            for q_type, count in type_counts.items():
                if total_count > 0:
                    type_targets[q_type] += int(remaining * (count / total_count))
            
            # 按类型采样
            final_augmented = []
            for q_type, target in type_targets.items():
                type_samples = [item for item in augmented_samples if item.get('type') == q_type]
                if len(type_samples) > target:
                    final_augmented.extend(random.sample(type_samples, target))
                else:
                    final_augmented.extend(type_samples)
            
            augmented_dataset = original_samples + final_augmented
        
        return augmented_dataset


# 使用示例
if __name__ == "__main__":
    augmenter = DataAugmentation()
    
    # 示例问题
    question = "在图中，组件A相对于组件B的位置是什么？"
    
    # 问题改写
    paraphrases = augmenter.question_paraphrase(question)
    print("问题改写示例:")
    for p in paraphrases:
        print(f"  - {p}")
    
    # 选项增强
    options = ["A. 位于上方", "B. 位于下方", "C. 位于左侧", "D. 位于右侧"]
    augmented_options = augmenter.augment_options(options)
    print("\n选项增强示例:")
    for i, opts in enumerate(augmented_options):
        print(f"  选项组 {i+1}:")
        for opt in opts:
            print(f"    - {opt}")
    
    # 生成困难负选项
    document_text = """
    该装置由支架、传动轴、控制板和连接组件构成。支架位于装置底部，高度为50cm。
    传动轴安装在支架上方，直径为35mm，采用不锈钢材质。控制板位于传动轴右侧，
    用于调节转速，转速范围为0-1500rpm。连接组件将各部分固定在一起，确保系统稳定运行。
    """
    
    negative_options = augmenter.generate_hard_negative_options(
        "安装在支架上方，直径为35mm", document_text
    )
    print("\n困难负选项示例:")
    for opt in negative_options:
        print(f"  - {opt}")
    
    # 基于文档生成问题
    generated_questions = augmenter.document_based_generation(document_text, {"title": "传动装置"})
    print("\n基于文档生成的问题示例:")
    for q in generated_questions:
        print(f"  问题: {q['question']}")
        print(f"  选项: {q['options']}")
        print(f"  答案: {q['answer']}")
        print(f"  类型: {q['type']}")
        print()