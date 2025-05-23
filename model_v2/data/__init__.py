"""数据包

此包包含数据加载和预处理模块：
- 数据集类 (MultimodalDataset)
- 数据转换 (transforms)
- 数据处理器 (DataProcessor)
- 数据加载器 (create_data_loader)
- 数据工具函数 (split_dataset等)
"""

from .dataset import MultimodalDataset, collate_fn
from .transforms import (
    get_image_transforms,
    get_layout_transforms,
    process_layout,
    process_text,
    process_label
)
from .data_utils import (
    DataProcessor,
    split_dataset,
    create_data_loader
)

__all__ = [
    # 数据集
    "MultimodalDataset",
    "collate_fn",
    
    # 数据转换
    "get_image_transforms",
    "get_layout_transforms",
    "process_layout",
    "process_text",
    "process_label",
    
    # 数据处理器
    "DataProcessor",
    "split_dataset",
    "create_data_loader"
] 