#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
推理优化模块，提供各种技术来加速模型推理并减少资源占用
"""

import os
import time
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class InferenceOptimizer:
    """
    模型推理优化器，提供多种推理加速技术
    
    支持的优化：
    - 模型量化 (INT8/FP16)
    - 批处理优化
    - CUDA图优化
    - 优化的注意力计算
    - 张量并行
    - KV缓存
    - 动态批处理
    """
    
    def __init__(self, 
                model=None,
                device: str = None,
                use_fp16: bool = True,
                use_int8: bool = False,
                use_cuda_graph: bool = False,
                use_kv_cache: bool = True,
                use_onnx: bool = False,
                batch_size: int = 1,
                max_batch_size: int = 16,
                dynamic_batch: bool = True,
                load_in_8bit: bool = False,
                onnx_path: str = None,
                model_path: str = None,
                onnx_provider: str = 'CUDAExecutionProvider'
                ):
        """
        初始化推理优化器
        
        Args:
            model: 预加载的模型（可选）
            device: 推理设备，默认自动选择
            use_fp16: 是否使用FP16推理
            use_int8: 是否使用INT8量化推理
            use_cuda_graph: 是否使用CUDA图优化静态输入形状
            use_kv_cache: 是否使用KV缓存
            use_onnx: 是否使用ONNX运行时
            batch_size: 默认批处理大小
            max_batch_size: 最大批处理大小
            dynamic_batch: 是否使用动态批处理
            load_in_8bit: 是否以8位精度加载模型
            onnx_path: ONNX模型路径
            model_path: 原始模型路径
            onnx_provider: ONNX执行提供程序
        """
        self.model = model
        self.model_path = model_path
        self.onnx_path = onnx_path
        self.onnx_provider = onnx_provider
        
        # 设备设置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 精度设置
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'  # FP16仅在GPU上可用
        self.use_int8 = use_int8
        self.load_in_8bit = load_in_8bit
        
        # 优化设置
        self.use_cuda_graph = use_cuda_graph and self.device.type == 'cuda'
        self.use_kv_cache = use_kv_cache
        self.use_onnx = use_onnx
        
        # 批处理设置
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.dynamic_batch = dynamic_batch
        
        # CUDA图存储
        self.cuda_graphs = {}
        # 缓存
        self.kv_cache = None
        
        # ONNX运行时
        self.onnx_session = None
        
        # 初始化
        self._setup()
        
    def _setup(self):
        """设置推理环境和加载模型"""
        # 加载模型（如果未提供）
        if self.model is None and self.model_path is not None:
            self._load_model()
            
        # 设置ONNX（如果启用）
        if self.use_onnx:
            self._setup_onnx()
            
        # 初始化FP16（如果启用）
        if self.use_fp16 and not self.use_onnx and hasattr(self, 'model') and self.model is not None:
            self._setup_fp16()
            
        # 初始化INT8量化（如果启用）
        if self.use_int8 and not self.use_onnx and not self.load_in_8bit and hasattr(self, 'model') and self.model is not None:
            self._setup_int8()
            
        # 创建CUDA图（如果启用）
        if self.use_cuda_graph and self.device.type == 'cuda':
            self._setup_cuda_graphs()
            
        logger.info(f"推理优化器初始化完成，设备: {self.device}, FP16: {self.use_fp16}, "
                  f"INT8: {self.use_int8}, CUDA图: {self.use_cuda_graph}, "
                  f"KV缓存: {self.use_kv_cache}, ONNX: {self.use_onnx}")
    
    def _load_model(self):
        """加载模型"""
        try:
            if self.load_in_8bit:
                # 使用8位精度加载
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=self.device,
                    load_in_8bit=True
                )
                logger.info(f"以8位精度加载模型: {self.model_path}")
            else:
                # 使用增强型模型的加载方法
                from .enhanced_model import EnhancedModel
                self.model = EnhancedModel.from_pretrained(self.model_path)
                self.model.to(self.device)
                self.model.eval()  # 设置为评估模式
                logger.info(f"标准精度加载模型: {self.model_path}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _setup_onnx(self):
        """设置ONNX运行时"""
        if not self.use_onnx:
            return
            
        try:
            import onnxruntime as ort
            
            # 配置ONNX运行时会话选项
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            # 为GPU推理设置额外选项
            if 'CUDA' in self.onnx_provider:
                providers = [(self.onnx_provider, {
                    'device_id': 0,
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                })]
            else:
                providers = [self.onnx_provider]
                
            # 创建ONNX运行时会话
            self.onnx_session = ort.InferenceSession(
                self.onnx_path, 
                sess_options=sess_options,
                providers=providers
            )
            
            # 获取模型输入输出
            self.onnx_inputs = [input.name for input in self.onnx_session.get_inputs()]
            self.onnx_outputs = [output.name for output in self.onnx_session.get_outputs()]
            
            logger.info(f"ONNX运行时初始化完成，提供程序: {self.onnx_provider}")
            logger.info(f"ONNX输入: {self.onnx_inputs}")
            logger.info(f"ONNX输出: {self.onnx_outputs}")
        except Exception as e:
            logger.error(f"ONNX运行时初始化失败: {e}")
            self.use_onnx = False
            raise
    
    def _setup_fp16(self):
        """设置FP16混合精度"""
        if self.model is None:
            return
            
        try:
            # 转换为FP16
            self.model = self.model.half()
            logger.info("模型已转换为FP16精度")
        except Exception as e:
            logger.error(f"FP16设置失败: {e}")
            self.use_fp16 = False
    
    def _setup_int8(self):
        """设置INT8量化"""
        if self.model is None or self.use_onnx:
            return
            
        try:
            # 使用PyTorch的量化功能
            from torch.quantization import quantize_dynamic
            self.model = quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.info("模型已进行INT8动态量化")
        except Exception as e:
            logger.error(f"INT8量化失败: {e}")
            self.use_int8 = False
    
    def _setup_cuda_graphs(self):
        """设置CUDA图优化"""
        if self.model is None or not self.use_cuda_graph or self.device.type != 'cuda':
            return
            
        try:
            # 为不同输入大小创建CUDA图
            # 这里我们为一些常见大小创建图
            common_sizes = [(1, 32), (1, 64), (1, 128), (4, 64), (8, 64)]
            
            for batch, seq_len in common_sizes:
                dummy_input = {
                    'input_ids': torch.zeros((batch, seq_len), dtype=torch.long, device=self.device),
                    'attention_mask': torch.ones((batch, seq_len), dtype=torch.long, device=self.device)
                }
                
                # 预热
                for _ in range(3):
                    with torch.no_grad():
                        _ = self.model(**dummy_input)
                
                # 捕获图
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    outputs = self.model(**dummy_input)
                
                self.cuda_graphs[(batch, seq_len)] = (g, dummy_input, outputs)
                
            logger.info(f"CUDA图创建完成，支持大小: {list(self.cuda_graphs.keys())}")
        except Exception as e:
            logger.error(f"CUDA图设置失败: {e}")
            self.use_cuda_graph = False
            self.cuda_graphs = {}
    
    @contextmanager
    def optimize_for_inference(self):
        """优化推理的上下文管理器"""
        if hasattr(self, 'model') and self.model is not None:
            # 保存原始状态
            training = self.model.training
            torch_grad = torch.is_grad_enabled()
            
            # 设置为推理模式
            self.model.eval()
            torch.set_grad_enabled(False)
            
            # 对于GPU，预热一下
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
                # 可选：锁定内存分配以防止碎片
                if hasattr(torch.cuda, 'memory_reserved'):
                    torch.cuda.memory_reserved()
                
            yield
            
            # 恢复原始状态
            self.model.train(training)
            torch.set_grad_enabled(torch_grad)
        else:
            yield
    
    def optimize_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """优化输入数据结构"""
        optimized_inputs = {}
        
        # 转换为张量并移动到正确设备
        for key, value in inputs.items():
            if isinstance(value, np.ndarray):
                # 对于NumPy数组，转换为张量
                tensor = torch.from_numpy(value)
            elif isinstance(value, list):
                # 对于列表，先转换为NumPy，再转为张量
                tensor = torch.tensor(value)
            elif isinstance(value, torch.Tensor):
                # 已经是张量
                tensor = value
            else:
                # 对于其他类型，保持不变
                optimized_inputs[key] = value
                continue
            
            # 移动到设备
            tensor = tensor.to(self.device)
            
            # 转换为FP16（如果适用）
            if self.use_fp16 and tensor.dtype in [torch.float32, torch.float]:
                tensor = tensor.half()
                
            optimized_inputs[key] = tensor
            
        return optimized_inputs
    
    def batch_inputs(self, inputs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """将输入列表批处理为单个批次"""
        # 如果只有一个输入，不需要批处理
        if len(inputs_list) == 1:
            return self.optimize_inputs(inputs_list[0])
        
        # 确定批处理大小
        batch_size = min(len(inputs_list), self.max_batch_size)
        
        # 按key分组
        batched_inputs = {}
        for key in inputs_list[0].keys():
            # 提取所有输入中的这个key
            values = [inputs[key] for inputs in inputs_list[:batch_size]]
            
            # 如果是张量，则堆叠
            if isinstance(values[0], torch.Tensor):
                batched_inputs[key] = torch.stack(values)
            elif isinstance(values[0], np.ndarray):
                batched_inputs[key] = torch.tensor(np.stack(values))
            elif isinstance(values[0], list):
                # 尝试转换为张量并堆叠
                try:
                    batched_inputs[key] = torch.tensor(values)
                except:
                    # 如果失败，保持为列表
                    batched_inputs[key] = values
            else:
                # 对于其他类型，保持为列表
                batched_inputs[key] = values
                
        # 优化批处理后的输入
        return self.optimize_inputs(batched_inputs)
    
    def dynamic_batch_inference(self, inputs_list: List[Dict[str, Any]]) -> List[Any]:
        """执行动态批处理推理"""
        # 检查输入列表是否为空
        if not inputs_list:
            return []
            
        results = []
        
        # 处理输入
        for i in range(0, len(inputs_list), self.max_batch_size):
            batch_inputs = inputs_list[i:i+self.max_batch_size]
            # 批处理
            batched = self.batch_inputs(batch_inputs)
            # 推理
            batch_results = self.infer(batched)
            # 拆分结果
            if isinstance(batch_results, list):
                results.extend(batch_results)
            elif isinstance(batch_results, torch.Tensor) and batch_results.dim() > 0:
                # 如果是批量张量，拆分为列表
                for j in range(len(batch_inputs)):
                    if j < batch_results.shape[0]:
                        results.append(batch_results[j])
            else:
                # 对于其他情况，直接添加
                results.append(batch_results)
                
        return results
    
    def infer(self, inputs: Dict[str, Any]) -> Any:
        """执行推理"""
        # 检查是否应该使用CUDA图
        if self.use_cuda_graph and hasattr(self, 'model'):
            # 确定输入形状
            if 'input_ids' in inputs:
                batch, seq_len = inputs['input_ids'].shape
                graph_key = (batch, seq_len)
                
                # 如果有CUDA图，使用它
                if graph_key in self.cuda_graphs:
                    g, dummy_input, outputs = self.cuda_graphs[graph_key]
                    
                    # 将实际输入复制到虚拟输入
                    for k, v in inputs.items():
                        if k in dummy_input:
                            dummy_input[k].copy_(v)
                    
                    # 重播图
                    g.replay()
                    
                    # 返回克隆的输出以避免被下一次调用覆盖
                    if isinstance(outputs, dict):
                        return {k: v.clone() for k, v in outputs.items()}
                    else:
                        return outputs.clone()
        
        # 使用ONNX运行时
        if self.use_onnx and self.onnx_session is not None:
            # 转换为ONNX输入格式
            onnx_inputs = {}
            for name in self.onnx_inputs:
                if name in inputs:
                    # 如果是张量，转换为NumPy
                    if isinstance(inputs[name], torch.Tensor):
                        onnx_inputs[name] = inputs[name].cpu().numpy()
                    else:
                        onnx_inputs[name] = inputs[name]
            
            # 执行ONNX推理
            outputs = self.onnx_session.run(self.onnx_outputs, onnx_inputs)
            
            # 处理输出
            if len(outputs) == 1:
                return outputs[0]
            else:
                return dict(zip(self.onnx_outputs, outputs))
                
        # 使用PyTorch模型
        if hasattr(self, 'model') and self.model is not None:
            with self.optimize_for_inference(), torch.no_grad():
                # 使用KV缓存
                if self.use_kv_cache and 'past_key_values' not in inputs and hasattr(self.model, 'forward'):
                    if self.kv_cache is not None:
                        inputs['past_key_values'] = self.kv_cache
                    outputs = self.model(**inputs)
                    if hasattr(outputs, 'past_key_values'):
                        self.kv_cache = outputs.past_key_values
                else:
                    outputs = self.model(**inputs)
                return outputs
        
        raise ValueError("没有可用的推理方法，请检查模型加载和配置")
    
    def clear_cache(self):
        """清除缓存"""
        self.kv_cache = None
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def benchmark(self, inputs: Dict[str, Any], iterations: int = 100, warmup: int = 10) -> Dict[str, float]:
        """对推理性能进行基准测试"""
        if not hasattr(self, 'model') and not self.use_onnx:
            raise ValueError("没有可用的模型进行基准测试")
            
        # 优化输入
        optimized_inputs = self.optimize_inputs(inputs)
        
        # 预热
        for _ in range(warmup):
            _ = self.infer(optimized_inputs)
            
        # 计时
        start_time = time.time()
        for _ in range(iterations):
            _ = self.infer(optimized_inputs)
        end_time = time.time()
        
        # 计算统计信息
        total_time = end_time - start_time
        avg_time = total_time / iterations
        throughput = iterations / total_time
        
        # 构建结果
        results = {
            'total_time': total_time,
            'average_time': avg_time,
            'throughput': throughput,
            'iterations': iterations
        }
        
        # 添加内存使用情况（如果在GPU上）
        if self.device.type == 'cuda':
            results['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            results['gpu_max_memory_allocated'] = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            
        return results
    
    def export_onnx(self, output_path: str, example_inputs: Dict[str, Any], opset_version: int = 12) -> str:
        """将模型导出为ONNX格式"""
        if not hasattr(self, 'model'):
            raise ValueError("没有可用的模型进行导出")
            
        try:
            # 优化输入
            optimized_inputs = self.optimize_inputs(example_inputs)
            
            # 设置导出参数
            input_names = list(optimized_inputs.keys())
            output_names = ['output']
            dynamic_axes = {
                key: {0: 'batch_size', 1: 'sequence_length'} 
                for key in input_names if isinstance(optimized_inputs[key], torch.Tensor) and optimized_inputs[key].dim() > 1
            }
            
            # 导出
            torch.onnx.export(
                self.model,
                (optimized_inputs,),
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )
            
            logger.info(f"模型已导出为ONNX格式: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"ONNX导出失败: {e}")
            raise

def optimize_model_for_inference(model_path: str, device: str = None, **kwargs) -> InferenceOptimizer:
    """
    优化模型用于推理的便捷函数
    
    Args:
        model_path: 模型路径
        device: 推理设备
        **kwargs: 传递给InferenceOptimizer的其他参数
        
    Returns:
        优化过的推理优化器实例
    """
    optimizer = InferenceOptimizer(model_path=model_path, device=device, **kwargs)
    return optimizer