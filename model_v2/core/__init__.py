"""Core package for the multimodal model framework.

This package contains the core components of the multimodal model framework:
1. Base model implementation
2. Optimized model implementation
3. Device management
4. Model management
5. Uncertainty estimation
"""

from .model import OptimizedMultimodalModel
from .base.model import BaseModel, ModelOutput
from .base.device import DeviceManager
from .base.manager import ModelManager
from .base.uncertainty import UncertaintyEstimator

__all__ = [
    "OptimizedMultimodalModel",
    "BaseModel",
    "ModelOutput",
    "DeviceManager",
    "ModelManager",
    "UncertaintyEstimator"
] 