import os
import torch
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from ..config.model_config import ModelConfig

logger = logging.getLogger(__name__)

class CheckpointManager:
    """检查点管理器
    
    负责管理模型的检查点，包括：
    1. 保存检查点
    2. 加载检查点
    3. 检查点轮换
    4. 最佳模型保存
    """
    
    def __init__(
        self,
        save_dir: str,
        max_to_keep: int = 5,
        save_best_only: bool = True,
        metric_name: str = "accuracy",
        metric_mode: str = "max"
    ):
        """初始化检查点管理器
        
        Args:
            save_dir: 保存目录
            max_to_keep: 最大保存数量
            save_best_only: 是否只保存最佳模型
            metric_name: 评估指标名称
            metric_mode: 评估指标模式（max/min）
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_to_keep = max_to_keep
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        
        self.best_metric = float("-inf") if metric_mode == "max" else float("inf")
        self.checkpoints = []
        
        logger.info(f"Checkpoint manager initialized at {save_dir}")
        
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: ModelConfig,
        metrics: Dict[str, float],
        step: int,
        epoch: int
    ) -> str:
        """保存检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            config: 模型配置
            metrics: 评估指标
            step: 训练步数
            epoch: 训练轮数
            
        Returns:
            检查点路径
        """
        # 检查是否需要保存
        if self.save_best_only:
            current_metric = metrics.get(self.metric_name)
            if current_metric is None:
                logger.warning(f"Metric {self.metric_name} not found in metrics")
                return None
                
            if self.metric_mode == "max" and current_metric <= self.best_metric:
                return None
            if self.metric_mode == "min" and current_metric >= self.best_metric:
                return None
                
            self.best_metric = current_metric
            
        # 创建检查点目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = self.save_dir / f"checkpoint-{timestamp}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        model_path = checkpoint_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        # 保存优化器
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path)
        
        # 保存调度器
        if scheduler:
            scheduler_path = checkpoint_dir / "scheduler.pt"
            torch.save(scheduler.state_dict(), scheduler_path)
            
        # 保存配置
        config_path = checkpoint_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
            
        # 保存训练状态
        state = {
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "best_metric": self.best_metric
        }
        state_path = checkpoint_dir / "state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
            
        # 更新检查点列表
        self.checkpoints.append(checkpoint_dir)
        
        # 轮换检查点
        if len(self.checkpoints) > self.max_to_keep:
            old_checkpoint = self.checkpoints.pop(0)
            self._remove_checkpoint(old_checkpoint)
            
        logger.info(f"Checkpoint saved at {checkpoint_dir}")
        return str(checkpoint_dir)
        
    def load_checkpoint(
        self,
        checkpoint_dir: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Tuple[int, int, Dict[str, float]]:
        """加载检查点
        
        Args:
            checkpoint_dir: 检查点目录
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            
        Returns:
            训练步数、轮数和指标
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        # 加载模型
        model_path = checkpoint_dir / "model.pt"
        model.load_state_dict(torch.load(model_path))
        
        # 加载优化器
        if optimizer:
            optimizer_path = checkpoint_dir / "optimizer.pt"
            optimizer.load_state_dict(torch.load(optimizer_path))
            
        # 加载调度器
        if scheduler:
            scheduler_path = checkpoint_dir / "scheduler.pt"
            scheduler.load_state_dict(torch.load(scheduler_path))
            
        # 加载训练状态
        state_path = checkpoint_dir / "state.json"
        with open(state_path) as f:
            state = json.load(f)
            
        self.best_metric = state["best_metric"]
        
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")
        return state["step"], state["epoch"], state["metrics"]
        
    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新的检查点
        
        Returns:
            最新检查点路径
        """
        if not self.checkpoints:
            return None
        return str(self.checkpoints[-1])
        
    def get_best_checkpoint(self) -> Optional[str]:
        """获取最佳检查点
        
        Returns:
            最佳检查点路径
        """
        if not self.checkpoints:
            return None
            
        best_checkpoint = None
        best_metric = float("-inf") if self.metric_mode == "max" else float("inf")
        
        for checkpoint in self.checkpoints:
            state_path = checkpoint / "state.json"
            with open(state_path) as f:
                state = json.load(f)
                metric = state["metrics"].get(self.metric_name)
                if metric is None:
                    continue
                    
                if self.metric_mode == "max" and metric > best_metric:
                    best_metric = metric
                    best_checkpoint = checkpoint
                elif self.metric_mode == "min" and metric < best_metric:
                    best_metric = metric
                    best_checkpoint = checkpoint
                    
        return str(best_checkpoint) if best_checkpoint else None
        
    def _remove_checkpoint(self, checkpoint_dir: Path):
        """删除检查点
        
        Args:
            checkpoint_dir: 检查点目录
        """
        try:
            for file in checkpoint_dir.iterdir():
                file.unlink()
            checkpoint_dir.rmdir()
            logger.info(f"Removed checkpoint {checkpoint_dir}")
        except Exception as e:
            logger.error(f"Failed to remove checkpoint {checkpoint_dir}: {e}")
            
    def save_best_model(self, model: torch.nn.Module, config: ModelConfig):
        """保存最佳模型
        
        Args:
            model: 模型
            config: 模型配置
        """
        best_model_dir = self.save_dir / "best_model"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        model_path = best_model_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        # 保存配置
        config_path = best_model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
            
        # 保存指标
        metrics_path = best_model_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"best_metric": self.best_metric}, f, indent=2)
            
        logger.info(f"Best model saved at {best_model_dir}")
        
    def load_best_model(
        self,
        model: torch.nn.Module,
        config: Optional[ModelConfig] = None
    ) -> ModelConfig:
        """加载最佳模型
        
        Args:
            model: 模型
            config: 模型配置
            
        Returns:
            模型配置
        """
        best_model_dir = self.save_dir / "best_model"
        
        # 加载模型
        model_path = best_model_dir / "model.pt"
        model.load_state_dict(torch.load(model_path))
        
        # 加载配置
        if config is None:
            config_path = best_model_dir / "config.json"
            with open(config_path) as f:
                config = ModelConfig.from_dict(json.load(f))
                
        logger.info(f"Best model loaded from {best_model_dir}")
        return config 