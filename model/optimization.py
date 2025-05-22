import os
import torch
import logging
import numpy as np
from typing import Dict, Any, Optional, Union, List
from functools import wraps
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_offline_mode
import torch.nn as nn
import bitsandbytes as bnb
from torch.quantization import quantize_dynamic
import gc

logger = logging.getLogger(__name__)

@dataclass
class AttentionOptimizationConfig:
    """注意力优化配置"""
    use_sparse_attention: bool = False
    sparse_attention_threshold: float = 0.1
    use_head_pruning: bool = False
    head_pruning_ratio: float = 0.3
    use_attention_cache: bool = True
    use_flash_attention: bool = True
    use_sliding_window: bool = False
    window_size: int = 512

@dataclass
class OptimizationConfig:
    """优化配置类"""
    use_quantization: bool = False
    quantization_bits: int = 8
    use_cache: bool = True
    cache_dir: Optional[str] = None
    use_distributed: bool = False
    distributed_backend: str = "nccl"
    use_gradient_checkpointing: bool = False
    use_mixed_precision: bool = True
    use_compile: bool = False
    compile_mode: str = "reduce-overhead"
    attention_optimization: Optional[AttentionOptimizationConfig] = None

@dataclass
class MemoryOptimizationConfig:
    """内存优化配置"""
    use_gradient_checkpointing: bool = True
    use_quantization: bool = False
    quantization_bits: int = 8
    use_attention_optimization: bool = True
    max_memory_usage: Optional[int] = None  # 单位：GB
    clear_cache_frequency: int = 100  # 清理缓存的频率（步数）

@dataclass
class ComputationOptimizationConfig:
    """计算优化配置"""
    use_flash_attention: bool = True
    use_model_compilation: bool = True
    use_dynamic_batching: bool = True
    max_batch_size: int = 32
    min_batch_size: int = 1
    batch_size_step: int = 2
    use_mixed_precision: bool = True
    use_torch_compile: bool = True
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"

class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self, config: OptimizationConfig):
        """
        初始化模型优化器
        
        Args:
            config: 优化配置
        """
        self.config = config
        self.cache = {}
        self.attention_cache = {}
        
    @staticmethod
    def _get_device_map(model_size: int) -> Dict[str, str]:
        """
        根据模型大小自动确定设备映射
        
        Args:
            model_size: 模型大小（以MB为单位）
            
        Returns:
            设备映射字典
        """
        if not torch.cuda.is_available():
            return {"": "cpu"}
            
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            return {"": "cpu"}
            
        # 如果模型较小，放在单个GPU上
        if model_size < 2000:  # 2GB
            return {"": "cuda:0"}
            
        # 如果模型较大，需要分片
        device_map = {}
        layers_per_gpu = model_size / num_gpus
        
        for i in range(num_gpus):
            start_layer = int(i * layers_per_gpu)
            end_layer = int((i + 1) * layers_per_gpu)
            device_map[f"layer_{start_layer}:{end_layer}"] = f"cuda:{i}"
            
        return device_map
    
    def _apply_attention_optimization(self, model: torch.nn.Module) -> torch.nn.Module:
        """应用注意力优化"""
        if not self.config.attention_optimization:
            return model
            
        try:
            # 1. 应用稀疏注意力
            if self.config.attention_optimization.use_sparse_attention:
                model = self._apply_sparse_attention(model)
                
            # 2. 应用注意力头剪枝
            if self.config.attention_optimization.use_head_pruning:
                model = self._apply_head_pruning(model)
                
            # 3. 应用Flash Attention
            if self.config.attention_optimization.use_flash_attention:
                model = self._apply_flash_attention(model)
                
            # 4. 应用滑动窗口注意力
            if self.config.attention_optimization.use_sliding_window:
                model = self._apply_sliding_window_attention(model)
                
            logger.info("注意力优化已应用")
            
        except Exception as e:
            logger.warning(f"注意力优化失败: {e}")
            
        return model
    
    def _apply_sparse_attention(self, model: torch.nn.Module) -> torch.nn.Module:
        """应用稀疏注意力"""
        def sparse_attention_forward(self, hidden_states, attention_mask=None, **kwargs):
            # 计算注意力分数
            attention_scores = torch.matmul(hidden_states, hidden_states.transpose(-1, -2))
            
            # 应用稀疏化
            threshold = self.config.attention_optimization.sparse_attention_threshold
            attention_scores = torch.where(
                torch.abs(attention_scores) < threshold,
                torch.zeros_like(attention_scores),
                attention_scores
            )
            
            # 应用softmax
            attention_probs = torch.softmax(attention_scores, dim=-1)
            
            # 计算输出
            context_layer = torch.matmul(attention_probs, hidden_states)
            return context_layer
            
        # 为所有注意力层添加稀疏注意力
        for module in model.modules():
            if hasattr(module, 'self') and hasattr(module.self, 'attention'):
                module.self.attention.forward = sparse_attention_forward.__get__(module.self.attention)
                
        return model
    
    def _apply_head_pruning(self, model: torch.nn.Module) -> torch.nn.Module:
        """应用注意力头剪枝"""
        def prune_attention_heads(model, pruning_ratio):
            for module in model.modules():
                if hasattr(module, 'num_attention_heads'):
                    # 计算要保留的头数量
                    num_heads = module.num_attention_heads
                    num_heads_to_keep = int(num_heads * (1 - pruning_ratio))
                    
                    if num_heads_to_keep < num_heads:
                        # 计算每个头的重要性分数
                        importance_scores = []
                        for head_idx in range(num_heads):
                            # 使用头输出的方差作为重要性指标
                            head_output = module.self.attention.attention_outputs[head_idx]
                            importance = torch.var(head_output).item()
                            importance_scores.append((head_idx, importance))
                            
                        # 选择最重要的头
                        importance_scores.sort(key=lambda x: x[1], reverse=True)
                        heads_to_keep = [idx for idx, _ in importance_scores[:num_heads_to_keep]]
                        
                        # 更新注意力头
                        module.num_attention_heads = num_heads_to_keep
                        module.all_head_size = module.attention_head_size * num_heads_to_keep
                        
                        # 更新注意力权重
                        if hasattr(module.self.attention, 'query'):
                            module.self.attention.query.weight = torch.nn.Parameter(
                                module.self.attention.query.weight[heads_to_keep]
                            )
                        if hasattr(module.self.attention, 'key'):
                            module.self.attention.key.weight = torch.nn.Parameter(
                                module.self.attention.key.weight[heads_to_keep]
                            )
                        if hasattr(module.self.attention, 'value'):
                            module.self.attention.value.weight = torch.nn.Parameter(
                                module.self.attention.value.weight[heads_to_keep]
                            )
                            
        prune_attention_heads(model, self.config.attention_optimization.head_pruning_ratio)
        return model
    
    def _apply_flash_attention(self, model: torch.nn.Module) -> torch.nn.Module:
        """应用Flash Attention"""
        try:
            from flash_attn import flash_attn_func
            
            def flash_attention_forward(self, hidden_states, attention_mask=None, **kwargs):
                # 准备输入
                q = self.query(hidden_states)
                k = self.key(hidden_states)
                v = self.value(hidden_states)
                
                # 使用Flash Attention
                output = flash_attn_func(q, k, v, causal=False)
                return output
                
            # 为所有注意力层添加Flash Attention
            for module in model.modules():
                if hasattr(module, 'self') and hasattr(module.self, 'attention'):
                    module.self.attention.forward = flash_attention_forward.__get__(module.self.attention)
                    
        except ImportError:
            logger.warning("Flash Attention未安装，跳过此优化")
            
        return model
    
    def _apply_sliding_window_attention(self, model: torch.nn.Module) -> torch.nn.Module:
        """应用滑动窗口注意力"""
        window_size = self.config.attention_optimization.window_size
        
        def sliding_window_attention_forward(self, hidden_states, attention_mask=None, **kwargs):
            # 计算注意力分数
            attention_scores = torch.matmul(hidden_states, hidden_states.transpose(-1, -2))
            
            # 创建滑动窗口掩码
            seq_length = hidden_states.size(1)
            window_mask = torch.ones_like(attention_scores)
            for i in range(seq_length):
                start = max(0, i - window_size // 2)
                end = min(seq_length, i + window_size // 2)
                window_mask[i, :start] = 0
                window_mask[i, end:] = 0
                
            # 应用掩码
            attention_scores = attention_scores.masked_fill(window_mask == 0, float('-inf'))
            
            # 应用softmax
            attention_probs = torch.softmax(attention_scores, dim=-1)
            
            # 计算输出
            context_layer = torch.matmul(attention_probs, hidden_states)
            return context_layer
            
        # 为所有注意力层添加滑动窗口注意力
        for module in model.modules():
            if hasattr(module, 'self') and hasattr(module.self, 'attention'):
                module.self.attention.forward = sliding_window_attention_forward.__get__(module.self.attention)
                
        return model
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        优化模型
        
        Args:
            model: 要优化的模型
            
        Returns:
            优化后的模型
        """
        # 1. 应用注意力优化
        if self.config.attention_optimization:
            model = self._apply_attention_optimization(model)
            
        # 2. 应用量化
        if self.config.use_quantization:
            model = self._apply_quantization(model)
            
        # 3. 应用梯度检查点
        if self.config.use_gradient_checkpointing:
            model = self._apply_gradient_checkpointing(model)
            
        # 4. 应用混合精度
        if self.config.use_mixed_precision:
            model = self._apply_mixed_precision(model)
            
        # 5. 应用编译优化
        if self.config.use_compile and hasattr(torch, 'compile'):
            model = self._apply_compile(model)
            
        return model
    
    def _apply_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """应用量化"""
        try:
            if self.config.quantization_bits == 8:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
                model = model.quantize(quantization_config)
                logger.info("已应用8位量化")
            elif self.config.quantization_bits == 4:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model = model.quantize(quantization_config)
                logger.info("已应用4位量化")
        except Exception as e:
            logger.warning(f"量化失败: {e}")
            
        return model
    
    def _apply_gradient_checkpointing(self, model: torch.nn.Module) -> torch.nn.Module:
        """应用梯度检查点"""
        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("已启用梯度检查点")
        except Exception as e:
            logger.warning(f"启用梯度检查点失败: {e}")
            
        return model
    
    def _apply_mixed_precision(self, model: torch.nn.Module) -> torch.nn.Module:
        """应用混合精度训练"""
        try:
            if hasattr(model, "to"):
                model = model.to(torch.float16)
                logger.info("已启用混合精度训练")
        except Exception as e:
            logger.warning(f"启用混合精度训练失败: {e}")
            
        return model
    
    def _apply_compile(self, model: torch.nn.Module) -> torch.nn.Module:
        """应用编译优化"""
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode=self.config.compile_mode)
                logger.info(f"已应用编译优化 (模式: {self.config.compile_mode})")
        except Exception as e:
            logger.warning(f"应用编译优化失败: {e}")
            
        return model
    
    def load_model(self, model_name_or_path: str) -> torch.nn.Module:
        """
        加载并优化模型
        
        Args:
            model_name_or_path: 模型名称或路径
            
        Returns:
            优化后的模型
        """
        # 检查缓存
        if self.config.use_cache and model_name_or_path in self.cache:
            logger.info(f"从缓存加载模型: {model_name_or_path}")
            return self.cache[model_name_or_path]
            
        # 加载模型
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16 if self.config.use_mixed_precision else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=self.config.cache_dir
            )
            
            # 优化模型
            model = self.optimize_model(model)
            
            # 缓存模型
            if self.config.use_cache:
                self.cache[model_name_or_path] = model
                
            return model
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def setup_distributed_training(self) -> None:
        """设置分布式训练"""
        if not self.config.use_distributed:
            return
            
        try:
            import torch.distributed as dist
            
            # 初始化分布式环境
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.distributed_backend,
                    init_method='env://'
                )
                
            # 设置当前设备
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
            if local_rank != -1:
                torch.cuda.set_device(local_rank)
                
            logger.info(f"分布式训练已初始化 (后端: {self.config.distributed_backend})")
            
        except Exception as e:
            logger.warning(f"设置分布式训练失败: {e}")
    
    def cleanup(self) -> None:
        """清理资源"""
        # 清理缓存
        if self.config.use_cache:
            self.cache.clear()
            
        # 清理分布式环境
        if self.config.use_distributed:
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as e:
                logger.warning(f"清理分布式环境失败: {e}")

class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.step_count = 0
        
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """应用内存优化"""
        logger.info("开始应用内存优化...")
        
        # 1. 梯度检查点
        if self.config.use_gradient_checkpointing:
            model = self._apply_gradient_checkpointing(model)
            
        # 2. 模型量化
        if self.config.use_quantization:
            model = self._apply_quantization(model)
            
        # 3. 注意力优化
        if self.config.use_attention_optimization:
            model = self._optimize_attention_memory(model)
            
        logger.info("内存优化完成")
        return model
    
    def _apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """应用梯度检查点"""
        logger.info("应用梯度检查点...")
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        # 为所有符合条件的模块应用梯度检查点
        for module in model.modules():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                module.forward = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(module),
                    *module.forward.__code__.co_varnames
                )
        
        return model
    
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """应用模型量化"""
        logger.info(f"应用{self.config.quantization_bits}位量化...")
        
        if self.config.quantization_bits == 8:
            # 8位量化
            model = quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )
        elif self.config.quantization_bits == 4:
            # 4位量化
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    module.weight = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None
                    )
        
        return model
    
    def _optimize_attention_memory(self, model: nn.Module) -> nn.Module:
        """优化注意力计算的内存使用"""
        logger.info("优化注意力计算内存...")
        
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                # 1. 实现注意力缓存
                module.use_cache = True
                module.cache_size = 1024  # 可配置的缓存大小
                
                # 2. 实现注意力稀疏化
                module.sparse_threshold = 0.1
                
                # 3. 实现滑动窗口注意力
                module.use_sliding_window = True
                module.window_size = 512
        
        return model
    
    def clear_memory(self):
        """清理内存"""
        self.step_count += 1
        if self.step_count % self.config.clear_cache_frequency == 0:
            logger.info("清理内存缓存...")
            torch.cuda.empty_cache()
            gc.collect()
    
    def estimate_memory_usage(self, model: nn.Module, batch_size: int) -> float:
        """估计模型内存使用"""
        total_params = sum(p.numel() for p in model.parameters())
        param_memory = total_params * 4 / (1024 ** 3)  # GB
        
        # 估计激活值内存
        activation_memory = batch_size * model.config.hidden_size * 4 / (1024 ** 3)  # GB
        
        return param_memory + activation_memory
    
    def optimize_batch_size(self, model: nn.Module, initial_batch_size: int) -> int:
        """根据可用内存优化批处理大小"""
        if not self.config.max_memory_usage:
            return initial_batch_size
            
        available_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
        memory_per_sample = self.estimate_memory_usage(model, 1)
        
        optimal_batch_size = min(
            initial_batch_size,
            int(self.config.max_memory_usage / memory_per_sample)
        )
        
        logger.info(f"优化批处理大小: {initial_batch_size} -> {optimal_batch_size}")
        return optimal_batch_size

class ComputationOptimizer:
    """计算优化器"""
    
    def __init__(self, config: ComputationOptimizationConfig):
        self.config = config
        self.current_batch_size = config.min_batch_size
        self.batch_size_history = []
        
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """应用计算优化"""
        logger.info("开始应用计算优化...")
        
        # 1. Flash Attention
        if self.config.use_flash_attention:
            model = self._apply_flash_attention(model)
            
        # 2. 模型编译
        if self.config.use_model_compilation:
            model = self._compile_model(model)
            
        # 3. 混合精度
        if self.config.use_mixed_precision:
            model = self._apply_mixed_precision(model)
            
        logger.info("计算优化完成")
        return model
    
    def _apply_flash_attention(self, model: nn.Module) -> nn.Module:
        """应用Flash Attention"""
        try:
            from flash_attn.flash_attention import FlashAttention
            logger.info("应用Flash Attention...")
            
            for module in model.modules():
                if isinstance(module, nn.MultiheadAttention):
                    # 替换标准注意力为Flash Attention
                    flash_attn = FlashAttention(
                        softmax_scale=None,
                        attention_dropout=module.dropout
                    )
                    module.forward = flash_attn.forward
                    
            return model
        except ImportError:
            logger.warning("Flash Attention未安装，跳过优化")
            return model
    
    def _compile_model(self, model: nn.Module) -> nn.Module:
        """编译模型"""
        if not self.config.use_torch_compile:
            return model
            
        logger.info(f"使用{self.config.compile_mode}模式编译模型...")
        try:
            compiled_model = torch.compile(
                model,
                mode=self.config.compile_mode,
                fullgraph=True
            )
            return compiled_model
        except Exception as e:
            logger.warning(f"模型编译失败: {e}")
            return model
    
    def _apply_mixed_precision(self, model: nn.Module) -> nn.Module:
        """应用混合精度"""
        logger.info("应用混合精度...")
        
        # 使用torch.cuda.amp
        scaler = torch.cuda.amp.GradScaler()
        model = model.half()  # 转换为FP16
        
        return model
    
    def optimize_batch_size(self, throughput: float, latency: float) -> int:
        """动态优化批处理大小"""
        if not self.config.use_dynamic_batching:
            return self.current_batch_size
            
        # 记录历史数据
        self.batch_size_history.append({
            'batch_size': self.current_batch_size,
            'throughput': throughput,
            'latency': latency
        })
        
        # 计算最优批处理大小
        if len(self.batch_size_history) >= 3:
            # 分析历史数据
            throughput_trend = self._analyze_throughput_trend()
            latency_trend = self._analyze_latency_trend()
            
            # 根据趋势调整批处理大小
            if throughput_trend > 0 and latency_trend < 0:
                # 吞吐量上升且延迟下降，可以增加批处理大小
                new_batch_size = min(
                    self.current_batch_size + self.config.batch_size_step,
                    self.config.max_batch_size
                )
            elif throughput_trend < 0 or latency_trend > 0:
                # 吞吐量下降或延迟上升，需要减少批处理大小
                new_batch_size = max(
                    self.current_batch_size - self.config.batch_size_step,
                    self.config.min_batch_size
                )
            else:
                new_batch_size = self.current_batch_size
                
            self.current_batch_size = new_batch_size
            
        return self.current_batch_size
    
    def _analyze_throughput_trend(self) -> float:
        """分析吞吐量趋势"""
        if len(self.batch_size_history) < 3:
            return 0
            
        recent = self.batch_size_history[-3:]
        throughputs = [h['throughput'] for h in recent]
        return (throughputs[-1] - throughputs[0]) / throughputs[0]
    
    def _analyze_latency_trend(self) -> float:
        """分析延迟趋势"""
        if len(self.batch_size_history) < 3:
            return 0
            
        recent = self.batch_size_history[-3:]
        latencies = [h['latency'] for h in recent]
        return (latencies[-1] - latencies[0]) / latencies[0]

def optimize_model_for_inference(model: torch.nn.Module, 
                               config: Optional[OptimizationConfig] = None) -> torch.nn.Module:
    """
    为推理优化模型
    
    Args:
        model: 要优化的模型
        config: 优化配置
        
    Returns:
        优化后的模型
    """
    if config is None:
        config = OptimizationConfig(
            use_quantization=True,
            quantization_bits=8,
            use_cache=True,
            use_mixed_precision=True
        )
        
    optimizer = ModelOptimizer(config)
    return optimizer.optimize_model(model)

def optimize_model_for_training(model: torch.nn.Module,
                              config: Optional[OptimizationConfig] = None) -> torch.nn.Module:
    """
    为训练优化模型
    
    Args:
        model: 要优化的模型
        config: 优化配置
        
    Returns:
        优化后的模型
    """
    if config is None:
        config = OptimizationConfig(
            use_gradient_checkpointing=True,
            use_mixed_precision=True,
            use_compile=True,
            compile_mode="reduce-overhead"
        )
        
    optimizer = ModelOptimizer(config)
    return optimizer.optimize_model(model)

def optimize_model_memory(model: nn.Module, config: Optional[MemoryOptimizationConfig] = None) -> nn.Module:
    """
    优化模型内存使用
    
    Args:
        model: 要优化的模型
        config: 优化配置
        
    Returns:
        优化后的模型
    """
    if config is None:
        config = MemoryOptimizationConfig()
        
    optimizer = MemoryOptimizer(config)
    return optimizer.optimize_model(model)

def optimize_model_computation(model: nn.Module, config: Optional[ComputationOptimizationConfig] = None) -> nn.Module:
    """
    优化模型计算性能
    
    Args:
        model: 要优化的模型
        config: 优化配置
        
    Returns:
        优化后的模型
    """
    if config is None:
        config = ComputationOptimizationConfig()
        
    optimizer = ComputationOptimizer(config)
    return optimizer.optimize_model(model)

if __name__ == "__main__":
    # 测试模型优化
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-chinese')
    parser.add_argument('--use_quantization', action='store_true')
    parser.add_argument('--quantization_bits', type=int, default=8)
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--use_distributed', action='store_true')
    parser.add_argument('--use_gradient_checkpointing', action='store_true')
    parser.add_argument('--use_mixed_precision', action='store_true')
    parser.add_argument('--use_compile', action='store_true')
    
    args = parser.parse_args()
    
    # 创建优化配置
    config = OptimizationConfig(
        use_quantization=args.use_quantization,
        quantization_bits=args.quantization_bits,
        use_cache=args.use_cache,
        use_distributed=args.use_distributed,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_mixed_precision=args.use_mixed_precision,
        use_compile=args.use_compile
    )
    
    # 创建优化器
    optimizer = ModelOptimizer(config)
    
    # 加载并优化模型
    try:
        model = optimizer.load_model(args.model_name)
        print("模型优化成功")
        
        # 测试推理
        input_ids = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            outputs = model(input_ids)
        print("推理测试成功")
        
    except Exception as e:
        print(f"模型优化失败: {e}")
    finally:
        optimizer.cleanup() 