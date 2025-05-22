import os
import logging
import sys
from datetime import datetime
from typing import Optional

def setup_logging(
    log_dir: str,
    rank: int = 0,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """设置日志记录
    
    Args:
        log_dir: 日志目录
        rank: 进程排名，用于分布式训练
        log_level: 日志级别
        log_file: 日志文件名，如果为None则使用时间戳
    
    Returns:
        配置好的logger对象
    """
    # 只在主进程记录日志
    if rank != 0:
        return logging.getLogger()
    
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志文件名
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'train_{timestamp}.log'
    log_path = os.path.join(log_dir, log_file)
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(log_level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_config(logger: logging.Logger, config: dict):
    """记录配置信息
    
    Args:
        logger: logger对象
        config: 配置字典
    """
    logger.info('Training Configuration:')
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f'{key}:')
            for sub_key, sub_value in value.items():
                logger.info(f'  {sub_key}: {sub_value}')
        else:
            logger.info(f'{key}: {value}')

def log_metrics(
    logger: logging.Logger,
    metrics: dict,
    epoch: int,
    prefix: str = ''
):
    """记录评估指标
    
    Args:
        logger: logger对象
        metrics: 评估指标字典
        epoch: 当前epoch
        prefix: 日志前缀
    """
    logger.info(f'{prefix}Epoch {epoch} Metrics:')
    for key, value in metrics.items():
        logger.info(f'{prefix}{key}: {value:.4f}')

def log_class_metrics(
    logger: logging.Logger,
    class_metrics: dict,
    epoch: int,
    prefix: str = ''
):
    """记录每个类别的评估指标
    
    Args:
        logger: logger对象
        class_metrics: 类别指标字典
        epoch: 当前epoch
        prefix: 日志前缀
    """
    logger.info(f'{prefix}Epoch {epoch} Class Metrics:')
    for label, metrics in class_metrics.items():
        logger.info(f'{prefix}Class: {label}')
        for key, value in metrics.items():
            logger.info(f'{prefix}  {key}: {value:.4f}')

def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    step: int,
    total_steps: int,
    loss: float,
    lr: float,
    prefix: str = ''
):
    """记录训练进度
    
    Args:
        logger: logger对象
        epoch: 当前epoch
        step: 当前步数
        total_steps: 总步数
        loss: 当前损失值
        lr: 当前学习率
        prefix: 日志前缀
    """
    progress = step / total_steps * 100
    logger.info(
        f'{prefix}Epoch {epoch} [{step}/{total_steps} ({progress:.1f}%)] '
        f'Loss: {loss:.4f} LR: {lr:.2e}'
    )

def log_memory_usage(logger: logging.Logger):
    """记录GPU内存使用情况
    
    Args:
        logger: logger对象
    """
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
            logger.info(
                f'GPU {i} Memory: '
                f'Allocated: {memory_allocated:.1f}MB, '
                f'Reserved: {memory_reserved:.1f}MB'
            ) 