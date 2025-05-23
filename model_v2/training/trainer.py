import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from tqdm import tqdm
import time
from pathlib import Path
import json
import argparse
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
    get_scheduler,
    set_seed
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    get_last_checkpoint,
    speed_metrics
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)

from ..config.model_config import ModelConfig
from ..core.model import BaseModel
from .checkpoint import CheckpointManager
from ..utils.metrics import calculate_metrics
from ..utils.optimization import (
    get_optimizer,
    get_scheduler,
    get_grad_scaler
)

logger = logging.getLogger(__name__)

class TrainerCallback:
    """训练回调基类"""
    
    def on_init_end(self, args, state, control, **kwargs):
        """训练初始化结束时调用"""
        pass
        
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时调用"""
        pass
        
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时调用"""
        pass
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """每个epoch开始时调用"""
        pass
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """每个epoch结束时调用"""
        pass
        
    def on_step_begin(self, args, state, control, **kwargs):
        """每个step开始时调用"""
        pass
        
    def on_step_end(self, args, state, control, **kwargs):
        """每个step结束时调用"""
        pass
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """评估时调用"""
        pass
        
    def on_save(self, args, state, control, **kwargs):
        """保存检查点时调用"""
        pass
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录日志时调用"""
        pass

class TrainerState:
    """训练状态"""
    
    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.max_steps = 0
        self.num_train_epochs = 0
        self.log_history = []
        self.best_metric = None
        self.best_model_checkpoint = None
        self.is_local_process_zero = True
        self.is_world_process_zero = True
        
    def save_to_json(self, json_path: str):
        """保存状态到JSON文件"""
        json_string = json.dumps(self.__dict__, indent=2, sort_keys=True)
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)
            
    def load_from_json(self, json_path: str):
        """从JSON文件加载状态"""
        with open(json_path, "r", encoding="utf-8") as f:
            json_string = f.read()
        self.__dict__.update(json.loads(json_string))

class BaseTrainer:
    """训练器基类
    
    实现了训练的核心功能，包括：
    1. 训练循环
    2. 验证循环
    3. 优化器和学习率调度
    4. 检查点管理
    5. 分布式训练支持
    6. 混合精度训练
    7. 回调系统
    8. 实验跟踪
    9. PEFT/LoRA 训练支持
    """
    
    def __init__(
        self,
        model: BaseModel,
        config: ModelConfig,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        local_rank: int = -1,
        callbacks: Optional[List[TrainerCallback]] = None
    ):
        """初始化训练器
        
        Args:
            model: 模型
            config: 模型配置
            train_dataset: 训练数据集
            eval_dataset: 验证数据集
            local_rank: 本地GPU编号
            callbacks: 回调列表
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.local_rank = local_rank
        self.callbacks = callbacks or []
        
        # 设置随机种子
        set_seed(self.config.training.seed)
        
        # 设置设备
        self._setup_device()
        
        # 应用 PEFT/LoRA
        if self.config.training.use_peft:
            self._setup_peft()
            
        # 初始化优化器
        self.optimizer = get_optimizer(
            model,
            config.optimization.optimizer_name,
            config.optimization.learning_rate,
            config.optimization.weight_decay
        )
        
        # 初始化学习率调度器
        self.scheduler = get_scheduler(
            self.optimizer,
            config.optimization.scheduler_name,
            config.optimization.warmup_steps,
            config.optimization.max_steps
        )
        
        # 初始化梯度缩放器
        self.scaler = get_grad_scaler() if config.optimization.use_amp else None
        
        # 初始化检查点管理器
        self.checkpoint_manager = CheckpointManager(
            save_dir=config.training.output_dir,
            max_to_keep=config.training.max_checkpoints,
            save_best_only=config.training.save_best_only,
            metric_name=config.training.metric_name,
            metric_mode=config.training.metric_mode
        )
        
        # 训练状态
        self.state = TrainerState()
        self.state.max_steps = config.training.max_steps
        self.state.num_train_epochs = config.training.num_epochs
        self.state.is_local_process_zero = self.local_rank in [-1, 0]
        self.state.is_world_process_zero = self.local_rank in [-1, 0]
        
        # 控制对象
        self.control = TrainerControl()
        
        # 调用回调
        for callback in self.callbacks:
            callback.on_init_end(self.config, self.state, self.control)
            
        logger.info("Trainer initialized")
        
    def _setup_device(self):
        """设置设备"""
        if self.config.training.distributed.enabled:
            if self.local_rank != -1:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
                self.model = self.model.to(self.device)
                
                # 同步BatchNorm
                if self.config.training.distributed.sync_bn:
                    self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                    
                # 创建DDP模型
                self.model = nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=self.config.training.distributed.find_unused_parameters
                )
            else:
                raise ValueError("Distributed training is enabled but local_rank is not set")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            
        logger.info(f"Using device: {self.device}")
        
    def _get_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """获取数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            shuffle: 是否打乱数据
            
        Returns:
            数据加载器
        """
        if self.config.training.distributed.enabled:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=shuffle,
                num_replicas=dist.get_world_size(),
                rank=self.local_rank
            )
            shuffle = False
        else:
            sampler = None
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            drop_last=self.config.training.distributed.enabled  # 在分布式训练中丢弃不完整的最后一个批次
        )
        
    def train(self):
        """训练模型"""
        # 准备数据加载器
        train_dataloader = self._get_dataloader(
            self.train_dataset,
            self.config.training.batch_size,
            shuffle=True
        )
        
        eval_dataloader = None
        if self.eval_dataset:
            eval_dataloader = self._get_dataloader(
                self.eval_dataset,
                self.config.evaluation.batch_size,
                shuffle=False
            )
            
        # 调用训练开始回调
        for callback in self.callbacks:
            callback.on_train_begin(self.config, self.state, self.control)
            
        # 训练循环
        for epoch in range(self.config.training.num_epochs):
            self.state.epoch = epoch
            
            # 在分布式训练中设置epoch
            if self.config.training.distributed.enabled:
                train_dataloader.sampler.set_epoch(epoch)
                
            # 调用epoch开始回调
            for callback in self.callbacks:
                callback.on_epoch_begin(self.config, self.state, self.control)
                
            # 训练一个轮次
            self._train_epoch(train_dataloader)
            
            # 验证
            if eval_dataloader:
                metrics = self._evaluate(eval_dataloader)
                
                # 调用评估回调
                for callback in self.callbacks:
                    callback.on_evaluate(self.config, self.state, self.control, metrics=metrics)
                    
                # 保存检查点
                if self.local_rank in [-1, 0]:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        config=self.config,
                        metrics=metrics,
                        step=self.state.global_step,
                        epoch=self.state.epoch
                    )
                    
                    # 保存最佳模型
                    if self._is_best_metric(metrics):
                        self.state.best_metric = metrics[self.config.training.metric_name]
                        self.state.best_model_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
                        self.checkpoint_manager.save_best_model(
                            model=self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model,
                            config=self.config
                        )
                        
            # 调用epoch结束回调
            for callback in self.callbacks:
                callback.on_epoch_end(self.config, self.state, self.control)
                
        # 调用训练结束回调
        for callback in self.callbacks:
            callback.on_train_end(self.config, self.state, self.control)
        
        # 清理分布式环境
        if self.config.training.distributed.enabled:
            self.cleanup_distributed()
        
    def _train_epoch(self, dataloader: DataLoader):
        """训练一个轮次
        
        Args:
            dataloader: 数据加载器
        """
        self.model.train()
        
        # 设置进度条
        if self.local_rank in [-1, 0]:
            pbar = tqdm(dataloader, desc=f"Epoch {self.state.epoch}")
        else:
            pbar = dataloader
            
        # 训练循环
        total_loss = 0
        num_batches = 0
        start_time = time.time()
        
        for batch in pbar:
            # 调用step开始回调
            for callback in self.callbacks:
                callback.on_step_begin(self.config, self.state, self.control)
                
            # 前向传播
            with autocast(enabled=self.config.optimization.use_amp):
                outputs = self.model(**batch)
                loss = outputs.loss
                
            # 反向传播
            if self.config.optimization.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # 梯度裁剪
            if self.config.optimization.max_grad_norm > 0:
                if self.config.optimization.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimization.max_grad_norm
                )
                
            # 优化器步进
            if self.config.optimization.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
                
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
                
            # 清空梯度
            self.optimizer.zero_grad()
            
            # 更新步数
            self.state.global_step += 1
            
            # 更新统计信息
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            if self.local_rank in [-1, 0]:
                pbar.set_postfix({
                    'loss': f"{total_loss / num_batches:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}" if self.scheduler else "N/A"
                })
                
            # 记录日志
            if self.state.global_step % self.config.training.logging_steps == 0:
                logs = {
                    "loss": total_loss / num_batches,
                    "learning_rate": self.scheduler.get_last_lr()[0] if self.scheduler else None,
                    "epoch": self.state.epoch,
                    "step": self.state.global_step
                }
                
                # 添加速度指标
                logs.update(speed_metrics(
                    "train",
                    start_time,
                    num_samples=num_batches * self.config.training.batch_size,
                    num_steps=self.state.global_step
                ))
                
                # 调用日志回调
                for callback in self.callbacks:
                    callback.on_log(self.config, self.state, self.control, logs=logs)
                    
                logger.info(
                    f"Step {self.state.global_step}: " +
                    ", ".join([f"{k} = {v:.4f}" if isinstance(v, float) else f"{k} = {v}" for k, v in logs.items()])
                )
                
            # 保存检查点
            if self.state.global_step % self.config.training.save_steps == 0:
                if self.local_rank in [-1, 0]:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        config=self.config,
                        metrics={"loss": total_loss / num_batches},
                        step=self.state.global_step,
                        epoch=self.state.epoch
                    )
                    
                    # 调用保存回调
                    for callback in self.callbacks:
                        callback.on_save(self.config, self.state, self.control)
                        
            # 调用step结束回调
            for callback in self.callbacks:
                callback.on_step_end(self.config, self.state, self.control)
                
    def _evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            评估指标
        """
        self.model.eval()
        
        # 设置进度条
        if self.local_rank in [-1, 0]:
            pbar = tqdm(dataloader, desc="Evaluating")
        else:
            pbar = dataloader
            
        # 评估循环
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in pbar:
                # 前向传播
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # 更新统计信息
                total_loss += loss.item()
                
                # 收集预测和标签
                predictions = outputs.predictions
                labels = batch["labels"]
                
                if self.local_rank != -1:
                    # 分布式评估：收集所有进程的预测和标签
                    predictions = self._gather_tensors(predictions)
                    labels = self._gather_tensors(labels)
                    
                if self.local_rank in [-1, 0]:
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                # 更新进度条
                if self.local_rank in [-1, 0]:
                    pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}"})
                    
        # 计算指标
        metrics = {}
        if self.local_rank in [-1, 0]:
            metrics = calculate_metrics(
                predictions=all_predictions,
                labels=all_labels,
                task_type=self.config.training.task_type
            )
            metrics["loss"] = total_loss / len(dataloader)
            
            # 记录日志
            logger.info(
                f"Evaluation results: " + 
                ", ".join([f"{k} = {v:.4f}" for k, v in metrics.items()])
            )
            
        # 同步所有进程的指标
        if self.local_rank != -1:
            metrics = self._broadcast_metrics(metrics)
            
        return metrics
        
    def _gather_tensors(self, tensor: torch.Tensor) -> torch.Tensor:
        """收集所有进程的张量
        
        Args:
            tensor: 输入张量
            
        Returns:
            收集后的张量
        """
        world_size = dist.get_world_size()
        if world_size == 1:
            return tensor
            
        # 获取张量大小
        local_size = torch.tensor([tensor.size(0)], device=tensor.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        
        # 准备接收缓冲区
        max_size = max(size.item() for size in all_sizes)
        tensor_list = [
            torch.zeros((max_size, *tensor.size()[1:]), device=tensor.device)
            for _ in range(world_size)
        ]
        
        # 填充本地张量
        tensor_list[self.local_rank] = tensor
        
        # 收集所有张量
        dist.all_gather(tensor_list, tensor_list[self.local_rank])
        
        # 合并张量
        gathered = []
        for i, size in enumerate(all_sizes):
            gathered.append(tensor_list[i][:size.item()])
            
        return torch.cat(gathered, dim=0)
        
    def _broadcast_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """广播指标到所有进程
        
        Args:
            metrics: 指标字典
            
        Returns:
            广播后的指标字典
        """
        if self.local_rank == 0:
            # 主进程：序列化指标
            metrics_str = json.dumps(metrics)
            metrics_tensor = torch.tensor(
                [ord(c) for c in metrics_str],
                dtype=torch.long,
                device=self.device
            )
            metrics_size = torch.tensor([len(metrics_tensor)], device=self.device)
        else:
            # 其他进程：准备接收缓冲区
            metrics_size = torch.zeros(1, dtype=torch.long, device=self.device)
            metrics_tensor = None
            
        # 广播大小
        dist.broadcast(metrics_size, src=0)
        
        if self.local_rank != 0:
            # 其他进程：创建接收缓冲区
            metrics_tensor = torch.zeros(
                metrics_size.item(),
                dtype=torch.long,
                device=self.device
            )
            
        # 广播指标
        dist.broadcast(metrics_tensor, src=0)
        
        # 反序列化指标
        metrics_str = "".join(chr(x.item()) for x in metrics_tensor)
        return json.loads(metrics_str)
        
    def _is_best_metric(self, metrics: Dict[str, float]) -> bool:
        """检查是否是最佳指标
        
        Args:
            metrics: 评估指标
            
        Returns:
            是否是最佳指标
        """
        current_metric = metrics.get(self.config.training.metric_name)
        if current_metric is None:
            return False
            
        if self.config.training.metric_mode == "max":
            return current_metric > self.state.best_metric
        else:
            return current_metric < self.state.best_metric
            
    def _setup_peft(self):
        """设置 PEFT/LoRA"""
        if not hasattr(self.config.training, "peft_config"):
            logger.warning("PEFT is enabled but no PEFT config is provided")
            return
            
        peft_config = self.config.training.peft_config
        
        # 准备模型
        if peft_config.use_8bit or peft_config.use_4bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=peft_config.gradient_checkpointing
            )
            
        # 创建 LoRA 配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM if peft_config.task_type == "causal_lm" else TaskType.SEQ_CLS,
            inference_mode=False,
            r=peft_config.lora_r,
            lora_alpha=peft_config.lora_alpha,
            lora_dropout=peft_config.lora_dropout,
            target_modules=peft_config.target_modules,
            bias=peft_config.bias,
            modules_to_save=peft_config.modules_to_save
        )
        
        # 应用 LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数
        self.model.print_trainable_parameters()
        
        logger.info("PEFT/LoRA setup completed")
        
    def save_model(self, save_path: str):
        """保存模型
        
        Args:
            save_path: 保存路径
        """
        if self.local_rank in [-1, 0]:
            if self.config.training.use_peft:
                # 保存 PEFT 模型
                self.model.save_pretrained(save_path)
            else:
                # 保存完整模型
                self.model.module.save_pretrained(save_path) if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model.save_pretrained(save_path)
                
    def load_model(self, load_path: str):
        """加载模型
        
        Args:
            load_path: 加载路径
        """
        self.model = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        
        if self.config.training.use_peft:
            # 加载 PEFT 模型
            self.model = self.model.from_pretrained(load_path)
        else:
            # 加载完整模型
            self.model = self.model.from_pretrained(load_path)
            
        self.model = self.model.to(self.device)
        
        if self.local_rank != -1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
            
    @classmethod
    def setup_distributed(cls, rank: int, world_size: int, config: ModelConfig):
        """设置分布式训练环境
        
        Args:
            rank: 当前进程的排名
            world_size: 总进程数
            config: 模型配置
        """
        os.environ['MASTER_ADDR'] = config.training.distributed.master_addr
        os.environ['MASTER_PORT'] = config.training.distributed.master_port
        
        # 初始化进程组
        dist.init_process_group(
            backend=config.training.distributed.backend,
            init_method=config.training.distributed.init_method,
            world_size=world_size,
            rank=rank
        )
        
        # 设置当前设备
        torch.cuda.set_device(rank)
        
    @classmethod
    def cleanup_distributed(cls):
        """清理分布式训练环境"""
        dist.destroy_process_group()
        
    @classmethod
    def run_distributed_training(
        cls,
        model: BaseModel,
        config: ModelConfig,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        num_gpus: Optional[int] = None
    ):
        """运行分布式训练
        
        Args:
            model: 模型
            config: 模型配置
            train_dataset: 训练数据集
            eval_dataset: 验证数据集
            num_gpus: GPU数量，如果为None则使用所有可用GPU
        """
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
            
        if num_gpus < 2:
            logger.warning("GPU数量小于2，将使用单GPU训练")
            trainer = cls(
                model=model,
                config=config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
            )
            trainer.train()
            return
            
        # 设置多进程启动方法
        mp.set_start_method('spawn', force=True)
        
        # 启动训练进程
        mp.spawn(
            cls._train_worker,
            args=(num_gpus, model, config, train_dataset, eval_dataset),
            nprocs=num_gpus,
            join=True
        )
        
    @classmethod
    def _train_worker(
        cls,
        rank: int,
        world_size: int,
        model: BaseModel,
        config: ModelConfig,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None
    ):
        """训练工作进程
        
        Args:
            rank: 当前进程的排名
            world_size: 总进程数
            model: 模型
            config: 模型配置
            train_dataset: 训练数据集
            eval_dataset: 验证数据集
        """
        # 设置分布式环境
        cls.setup_distributed(rank, world_size, config)
        
        try:
            # 创建训练器
            trainer = cls(
                model=model,
                config=config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                local_rank=rank
            )
            
            # 开始训练
            trainer.train()
            
        finally:
            # 清理分布式环境
            cls.cleanup_distributed()
            
    @classmethod
    def main(cls):
        """主函数"""
        parser = argparse.ArgumentParser(description="训练启动脚本")
        parser.add_argument("--config", type=str, help="配置文件路径")
        parser.add_argument("--num_gpus", type=int, help="使用的GPU数量")
        parser.add_argument("--local_rank", type=int, default=-1, help="本地GPU编号")
        parser.add_argument("--output_dir", type=str, help="输出目录")
        parser.add_argument("--model_name_or_path", type=str, help="模型名称或路径")
        parser.add_argument("--train_data_path", type=str, help="训练数据路径")
        parser.add_argument("--eval_data_path", type=str, help="验证数据路径")
        args = parser.parse_args()
        
        # 加载配置
        if args.config:
            config = ModelConfig.from_yaml(args.config)
        else:
            # 使用默认配置
            config = ModelConfig()
        
        # 更新命令行参数
        if args.output_dir:
            config.launch.output_dir = args.output_dir
        if args.model_name_or_path:
            config.launch.model_name_or_path = args.model_name_or_path
        if args.train_data_path:
            config.launch.train_data_path = args.train_data_path
        if args.eval_data_path:
            config.launch.eval_data_path = args.eval_data_path
        if args.num_gpus:
            config.launch.distributed.world_size = args.num_gpus
        
        # 设置日志
        logging.basicConfig(
            level=getattr(logging, config.launch.logging_level),
            format=config.launch.logging_format
        )
        
        if config.launch.log_to_file:
            log_dir = Path(config.launch.output_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / config.launch.log_file
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(config.launch.logging_format))
            logger.addHandler(file_handler)
        
        # 创建输出目录
        output_dir = Path(config.launch.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config_path = output_dir / "config.yaml"
        config.save(config_path)
        
        # 设置随机种子
        set_seed(config.launch.seed)
        
        # 创建模型
        model = BaseModel.from_config(config)
        
        # 创建数据集
        from ..data import create_dataset
        train_dataset = create_dataset(
            data_path=config.launch.train_data_path,
            config=config,
            cache_dir=config.launch.cache_dir,
            is_training=True
        )
        
        eval_dataset = None
        if config.launch.eval_data_path:
            eval_dataset = create_dataset(
                data_path=config.launch.eval_data_path,
                config=config,
                cache_dir=config.launch.cache_dir,
                is_training=False
            )
        
        # 运行训练
        if config.launch.distributed.enabled:
            cls.run_distributed_training(
                model=model,
                config=config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                num_gpus=config.launch.distributed.world_size
            )
        else:
            trainer = cls(
                model=model,
                config=config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
            )
            trainer.train()
            
if __name__ == "__main__":
    BaseTrainer.main() 