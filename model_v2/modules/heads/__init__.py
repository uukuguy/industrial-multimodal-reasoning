"""
Heads Package

This package contains the output head modules:
- Classification head
- Regression head
- Multi-label head
"""

from .base import BaseHead
from .implementations import ClassificationHead, RegressionHead, MultiLabelHead

__all__ = [
    "BaseHead",
    "ClassificationHead",
    "RegressionHead",
    "MultiLabelHead"
] 