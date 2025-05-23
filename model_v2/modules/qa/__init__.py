"""
QA Package

This package contains the question answering modules:
- Span-based QA
- Generative QA
- Multi-choice QA
"""

from .base import BaseQAModule
from .implementations import SpanQAModule

__all__ = [
    "BaseQAModule",
    "SpanQAModule"
] 