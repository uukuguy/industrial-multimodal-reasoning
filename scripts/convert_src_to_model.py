#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将src目录迁移为model包，并修改内部引用为相对引用

此脚本会：
1. 将src目录下的所有Python文件复制到model目录
2. 更新内部导入语句，使用相对引用
3. 更新脚本中对src的引用为model
"""

import os
import re
import shutil
import sys
from typing import List, Dict, Set

def find_python_files(directory: str) -> List[str]:
    """查找目录下所有Python文件"""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def extract_module_names(directory: str) -> Set[str]:
    """提取模块名（不含.py后缀）"""
    module_names = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                module_name = file[:-3]  # 去除.py后缀
                module_names.add(module_name)
    return module_names

def convert_imports(content: str, module_names: Set[str]) -> str:
    """转换导入语句为相对导入"""
    # 替换直接导入，如 from pdf_processor import X
    for module in module_names:
        pattern = rf'from\s+{module}\s+import'
        replacement = f'from .{module} import'
        content = re.sub(pattern, replacement, content)
    
    # 替换 from model.X import Y
    content = re.sub(r'from\s+src\.', 'from .', content)
    
    # 替换 import model.X
    content = re.sub(r'import\s+src\.', 'from . import ', content)
    
    return content

def update_script_imports(content: str) -> str:
    """更新脚本中对src的引用"""
    # 替换 import model.X 为 import model.X
    content = re.sub(r'import\s+src\.', 'import model.', content)
    
    # 替换 from model.X import Y 为 from model.X import Y
    content = re.sub(r'from\s+src\.', 'from model.', content)
    
    # 替换 from model import X 为 from model import X
    content = re.sub(r'from\s+src\s+import', 'from model import', content)
    
    return content

def process_file(file_path: str, module_names: Set[str], is_module_file: bool) -> str:
    """处理单个文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 如果是模块文件，转换为相对导入
    if is_module_file:
        return convert_imports(content, module_names)
    # 如果是脚本文件，更新对src的引用
    else:
        return update_script_imports(content)

def main():
    """主函数"""
    src_dir = 'src'
    model_dir = 'model'
    scripts_dir = 'scripts'
    
    # 确保model目录存在
    os.makedirs(model_dir, exist_ok=True)
    
    # 获取模块名列表
    module_names = extract_module_names(src_dir)
    print(f"找到模块: {', '.join(sorted(module_names))}")
    
    # 处理src目录下的Python文件
    src_files = find_python_files(src_dir)
    for src_file in src_files:
        # 构造目标路径
        rel_path = os.path.relpath(src_file, src_dir)
        dst_file = os.path.join(model_dir, rel_path)
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        
        # 处理文件内容
        print(f"处理文件: {src_file} -> {dst_file}")
        modified_content = process_file(src_file, module_names, True)
        
        # 写入新文件
        with open(dst_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)
    
    # 处理scripts目录下的Python文件
    script_files = find_python_files(scripts_dir)
    for script_file in script_files:
        print(f"更新脚本: {script_file}")
        modified_content = process_file(script_file, module_names, False)
        
        # 更新脚本文件
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)
    
    print("转换完成！请检查model目录下的文件。")
    print("注意: 您仍然需要检查并更新README.md和其他文档中的引用。")

if __name__ == "__main__":
    main()