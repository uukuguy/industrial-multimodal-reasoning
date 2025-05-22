# -*- coding: utf-8 -*-

"""
工业技术文档多模态推理问答系统

该包包含系统的所有核心模块，包括文档预处理、多模态编码、
融合模块、推理问答等组件。
"""

# 导出核心模块，方便使用 from model import X 的形式导入
from .pdf_processor import process_pdf
from .multimodal_encoder import MultimodalEncoder
from .multimodal_fusion import MultimodalFusion, HierarchicalMultimodalFusion
from .cross_modal_attention import CrossModalAttention, MultiModalAttentionHub
from .modality_reconstructor import ModalityReconstructor
from .uncertainty_estimator import UncertaintyEstimator
from .qa_module import ReasoningQAModule, EnhancedReasoningQAModule
from .model import EnhancedMultimodalModel
from .config import load_config, save_config, init_config

# 导出训练相关模块
from .dataset import MultimodalQuestionAnsweringDataset, DocumentTransforms, DataCollator
from .trainer import EnhancedMultiModalTrainer, create_trainer

__version__ = "1.0.0"