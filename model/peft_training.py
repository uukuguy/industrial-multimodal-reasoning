# -*- coding: utf-8 -*-

"""
参数高效微调(PEFT)模块

在数据量有限的情况下，使用PEFT技术对大型预训练模型进行高效微调，
避免过拟合并减少计算资源需求。
"""

import os
import math
import logging
import torch
from typing import Dict, Any, Optional, Union, List, Tuple

logger = logging.getLogger(__name__)

# 尝试导入PEFT库
try:
    from peft import (
        get_peft_model, 
        LoraConfig, 
        PrefixTuningConfig, 
        PromptTuningConfig,
        PromptEncoderConfig,
        TaskType
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    logger.warning("PEFT库未安装，无法使用参数高效微调功能。请安装: pip install peft")

class PEFTHandler:
    """参数高效微调处理器"""
    
    def __init__(self, technique: str = "lora"):
        """
        初始化PEFT处理器
        
        Args:
            technique: 微调技术，支持'lora'、'prefix_tuning'、'prompt_tuning'、'p_tuning'
        """
        if not HAS_PEFT:
            raise ImportError("请先安装PEFT库: pip install peft")
        
        self.technique = technique.lower()
        self.supported_techniques = {
            "lora": self._create_lora_config,
            "prefix_tuning": self._create_prefix_tuning_config,
            "prompt_tuning": self._create_prompt_tuning_config,
            "p_tuning": self._create_p_tuning_config
        }
        
        if self.technique not in self.supported_techniques:
            raise ValueError(f"不支持的PEFT技术: {technique}. 支持的技术有: {list(self.supported_techniques.keys())}")
    
    def _create_lora_config(self, 
                          config: Dict[str, Any], 
                          model_config: Optional[Dict[str, Any]] = None) -> LoraConfig:
        """
        创建LoRA配置
        
        Args:
            config: LoRA特定配置
            model_config: 模型配置
            
        Returns:
            LoRA配置对象
        """
        # 设置默认值
        r = config.get("r", 16)  # LoRA注意力维度
        alpha = config.get("alpha", 32)
        dropout = config.get("dropout", 0.05)
        bias = config.get("bias", "none")
        
        # 确定目标模块
        model_type = model_config.get("model_type", "") if model_config else ""
        
        # 为不同模型类型提供合理的默认目标模块
        if not config.get("target_modules"):
            if "bert" in model_type.lower():
                config["target_modules"] = ["query", "key", "value"]
            elif "t5" in model_type.lower():
                config["target_modules"] = ["q", "v"]
            elif "gpt" in model_type.lower() or "llama" in model_type.lower():
                config["target_modules"] = ["q_proj", "v_proj"]
            else:
                # 默认目标模块
                config["target_modules"] = ["query", "value"]
        
        # 推断任务类型
        task_type = None
        if model_config:
            if model_config.get("is_encoder_decoder", False):
                task_type = TaskType.SEQ_2_SEQ_LM
            elif "gpt" in model_type.lower() or "llama" in model_type.lower():
                task_type = TaskType.CAUSAL_LM
            else:
                task_type = TaskType.SEQUENCE_CLASSIFICATION
                
        return LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=config.get("target_modules"),
            lora_dropout=dropout,
            bias=bias,
            task_type=task_type,
            inference_mode=False,
            **{k: v for k, v in config.items() if k not in 
               ["r", "alpha", "target_modules", "dropout", "bias"]}
        )
    
    def _create_prefix_tuning_config(self, 
                                   config: Dict[str, Any], 
                                   model_config: Optional[Dict[str, Any]] = None) -> PrefixTuningConfig:
        """
        创建Prefix-tuning配置
        
        Args:
            config: Prefix-tuning特定配置
            model_config: 模型配置
            
        Returns:
            Prefix-tuning配置对象
        """
        # 设置默认值
        num_virtual_tokens = config.get("num_virtual_tokens", 20)
        
        # 确定模型类型和任务类型
        model_type = model_config.get("model_type", "") if model_config else ""
        task_type = None
        if model_config:
            if model_config.get("is_encoder_decoder", False):
                task_type = TaskType.SEQ_2_SEQ_LM
            elif "gpt" in model_type.lower() or "llama" in model_type.lower():
                task_type = TaskType.CAUSAL_LM
            else:
                task_type = TaskType.SEQUENCE_CLASSIFICATION
        
        return PrefixTuningConfig(
            num_virtual_tokens=num_virtual_tokens,
            encoder_hidden_size=config.get("encoder_hidden_size"),
            prefix_projection=config.get("prefix_projection", False),
            task_type=task_type,
            **{k: v for k, v in config.items() if k not in 
               ["num_virtual_tokens", "encoder_hidden_size", "prefix_projection"]}
        )
    
    def _create_prompt_tuning_config(self, 
                                  config: Dict[str, Any], 
                                  model_config: Optional[Dict[str, Any]] = None) -> PromptTuningConfig:
        """
        创建Prompt-tuning配置
        
        Args:
            config: Prompt-tuning特定配置
            model_config: 模型配置
            
        Returns:
            Prompt-tuning配置对象
        """
        # 设置默认值
        num_virtual_tokens = config.get("num_virtual_tokens", 20)
        
        # 确定任务类型
        model_type = model_config.get("model_type", "") if model_config else ""
        task_type = None
        if model_config:
            if model_config.get("is_encoder_decoder", False):
                task_type = TaskType.SEQ_2_SEQ_LM
            elif "gpt" in model_type.lower() or "llama" in model_type.lower():
                task_type = TaskType.CAUSAL_LM
            else:
                task_type = TaskType.SEQUENCE_CLASSIFICATION
        
        return PromptTuningConfig(
            num_virtual_tokens=num_virtual_tokens,
            task_type=task_type,
            **{k: v for k, v in config.items() if k not in ["num_virtual_tokens"]}
        )
    
    def _create_p_tuning_config(self, 
                             config: Dict[str, Any], 
                             model_config: Optional[Dict[str, Any]] = None) -> PromptEncoderConfig:
        """
        创建P-tuning配置
        
        Args:
            config: P-tuning特定配置
            model_config: 模型配置
            
        Returns:
            P-tuning配置对象
        """
        # 设置默认值
        num_virtual_tokens = config.get("num_virtual_tokens", 20)
        encoder_hidden_size = config.get("encoder_hidden_size", 128)
        
        # 确定任务类型
        model_type = model_config.get("model_type", "") if model_config else ""
        task_type = None
        if model_config:
            if model_config.get("is_encoder_decoder", False):
                task_type = TaskType.SEQ_2_SEQ_LM
            elif "gpt" in model_type.lower() or "llama" in model_type.lower():
                task_type = TaskType.CAUSAL_LM
            else:
                task_type = TaskType.SEQUENCE_CLASSIFICATION
        
        return PromptEncoderConfig(
            num_virtual_tokens=num_virtual_tokens,
            encoder_hidden_size=encoder_hidden_size,
            task_type=task_type,
            **{k: v for k, v in config.items() if k not in 
               ["num_virtual_tokens", "encoder_hidden_size"]}
        )
    
    def prepare_model(self, 
                     model, 
                     peft_config: Optional[Dict[str, Any]] = None, 
                     model_config: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
        """
        为模型应用PEFT，准备参数高效微调
        
        Args:
            model: 原始模型
            peft_config: PEFT配置参数
            model_config: 模型配置
            
        Returns:
            应用了PEFT的模型
        """
        if peft_config is None:
            peft_config = {}
        
        # 使用相应的配置创建函数生成PEFT配置
        config_fn = self.supported_techniques[self.technique]
        peft_config_obj = config_fn(peft_config, model_config)
        
        # 应用PEFT配置到模型
        peft_model = get_peft_model(model, peft_config_obj)
        
        # 打印可训练参数信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        
        logger.info(f"应用{self.technique.upper()}微调")
        logger.info(f"模型总参数: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,} ({trainable_params/total_params:.2%})")
        
        return peft_model
    
    @staticmethod
    def get_recommended_technique(dataset_size: int, model_size: int) -> str:
        """
        根据数据集大小和模型大小推荐PEFT技术
        
        Args:
            dataset_size: 数据集样本数
            model_size: 模型参数数量(以百万为单位)
            
        Returns:
            推荐的PEFT技术名称
        """
        # 基于简单启发式规则
        if dataset_size < 100:
            if model_size > 1000:  # 超大模型
                return "prompt_tuning"
            else:
                return "p_tuning"
        elif dataset_size < 500:
            if model_size > 1000:  # 超大模型
                return "prefix_tuning"
            else:
                return "lora"
        else:
            return "lora"  # LoRA通常在中等数据集上表现最佳
    
    @staticmethod
    def get_target_modules(model_type: str) -> List[str]:
        """
        获取特定模型类型的推荐目标模块
        
        Args:
            model_type: 模型类型名称
            
        Returns:
            推荐的目标模块列表
        """
        model_type = model_type.lower()
        
        if "bert" in model_type:
            return ["query", "key", "value"]
        elif "t5" in model_type:
            return ["q", "v"]
        elif "gpt" in model_type:
            return ["c_attn"]
        elif "llama" in model_type:
            return ["q_proj", "v_proj"]
        elif "vit" in model_type or "visual" in model_type:
            return ["attn.qkv"]
        else:
            return ["query", "value"]  # 默认值
    
    @staticmethod
    def get_default_config(technique: str, model_type: str) -> Dict[str, Any]:
        """
        获取PEFT技术的默认配置
        
        Args:
            technique: PEFT技术名称
            model_type: 模型类型
            
        Returns:
            默认配置字典
        """
        # 根据模型类型确定目标模块
        target_modules = PEFTHandler.get_target_modules(model_type)
        
        configs = {
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "bias": "none",
                "target_modules": target_modules
            },
            "prefix_tuning": {
                "num_virtual_tokens": 20,
                "prefix_projection": True
            },
            "prompt_tuning": {
                "num_virtual_tokens": 20,
                "prompt_tuning_init": "TEXT",
                "tokenizer_name_or_path": None  # 需要在使用时提供
            },
            "p_tuning": {
                "num_virtual_tokens": 20,
                "encoder_hidden_size": 128,
                "encoder_dropout": 0.1
            }
        }
        
        if technique in configs:
            return configs[technique]
        else:
            raise ValueError(f"未知的PEFT技术: {technique}")


# 使用示例
def apply_peft_to_model_example(model, dataset_size, model_config):
    """示例：如何将PEFT应用到模型"""
    
    # 根据数据集大小自动选择PEFT技术
    model_size = sum(p.numel() for p in model.parameters()) / 1e6  # 百万参数
    technique = PEFTHandler.get_recommended_technique(dataset_size, model_size)
    
    # 获取此技术的默认配置
    model_type = model_config.get("model_type", "")
    peft_config = PEFTHandler.get_default_config(technique, model_type)
    
    # 创建PEFT处理器并应用到模型
    peft_handler = PEFTHandler(technique=technique)
    peft_model = peft_handler.prepare_model(model, peft_config, model_config)
    
    return peft_model


# 不同PEFT技术的应用场景指南
PEFT_TECHNIQUES_GUIDE = """
参数高效微调(PEFT)技术选择指南:

1. LoRA (Low-Rank Adaptation)
   - 适用场景: 一般迁移学习场景，中等规模数据集(100-10000样本)
   - 优点: 参数高效，训练稳定，适用于多种模型架构
   - 可调参数: rank (r), alpha, target_modules
   - 推荐配置: r=16-32, alpha=32, target_modules=注意力层
   
2. Prefix-Tuning
   - 适用场景: 少样本任务(100-1000样本)，文本生成任务
   - 优点: 相比全参数微调效果更好，高度参数高效
   - 可调参数: num_virtual_tokens, prefix_projection
   - 推荐配置: num_virtual_tokens=10-30, prefix_projection=True
   
3. Prompt-Tuning
   - 适用场景: 极少样本场景(<100样本)，需要精确控制提示的任务
   - 优点: 极度参数高效，便于模型部署和层次融合
   - 可调参数: num_virtual_tokens, prompt_tuning_init
   - 推荐配置: num_virtual_tokens=5-20, prompt_tuning_init="TEXT"
   
4. P-Tuning
   - 适用场景: 少样本分类任务，需要灵活处理提示的场景
   - 优点: 比Prompt-Tuning更灵活，仍然高度参数高效
   - 可调参数: num_virtual_tokens, encoder_hidden_size
   - 推荐配置: num_virtual_tokens=10-20, encoder_hidden_size=128
   
选择建议:
- 数据<100样本: Prompt-Tuning或P-Tuning
- 数据100-1000样本: Prefix-Tuning或LoRA
- 数据>1000样本: LoRA或考虑全参数微调
- 计算资源非常有限: Prompt-Tuning (参数最少)
- 追求性能与效率平衡: LoRA (通常是最佳选择)
"""


if __name__ == "__main__":
    print(PEFT_TECHNIQUES_GUIDE)