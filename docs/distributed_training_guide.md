# 分布式训练指南

本文档提供工业技术文档多模态推理问答系统的各种分布式训练配置指南，包括单机单卡、单机多卡和多机多卡场景。

## 1. 单机单卡训练 (基础配置)

这是最基本的训练设置，适用于小规模实验或资源有限的情况。

### 命令示例：

```bash
python scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --valid_questions data/raw_data/valid/questions.jsonl \
    --documents data/raw_data/train/documents \
    --output_dir outputs/single_gpu \
    --batch_size 4 \
    --epochs 10 \
    --fp16 \
    --use_peft \
    --use_data_augmentation
```

### 关键参数说明：

- **基本参数**：指定训练数据、验证数据和文档路径
- **批量大小**：单GPU训练时批量大小通常较小（4-8）
- **优化选项**：启用FP16混合精度训练，节省显存
- **增强选项**：启用PEFT和数据增强提高性能

### 性能估计：

- 训练时间：约4-6小时（取决于数据规模和GPU型号）
- 显存占用：约6-10GB
- 适用场景：快速原型验证，小规模实验

## 2. 单机多卡训练

利用单台机器上的多个GPU进行并行训练，显著提高训练速度。提供两种启动方式：

### 2.1 使用PyTorch分布式启动器

```bash
# 使用torch.distributed.launch (PyTorch原生方式)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env \
    scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --documents data/raw_data/train/documents \
    --output_dir outputs/multi_gpu \
    --batch_size 8 \
    --epochs 10 \
    --ddp \
    --fp16 \
    --gradient_accumulation_steps 2 \
    --use_peft \
    --use_data_augmentation \
    --use_optimized_loss
```

### 2.2 使用Accelerate启动器（更简洁）

```bash
# 首次使用需配置accelerate
accelerate config

# 启动训练
accelerate launch \
    scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --documents data/raw_data/train/documents \
    --output_dir outputs/multi_gpu_accelerate \
    --batch_size 8 \
    --fp16 \
    --use_peft \
    --use_data_augmentation
```

### 关键参数说明：

- **nproc_per_node**：使用的GPU数量（例如4表示使用4张GPU）
- **ddp**：启用DistributedDataParallel模式
- **batch_size**：这里指的是每个GPU的批量大小（总批量大小=batch_size×GPU数量）
- **gradient_accumulation_steps**：梯度累积步数，可以进一步增大有效批量大小

### 性能估计：

- 训练速度：接近线性加速（与GPU数量成正比）
- 显存占用：与单GPU相似（每GPU 6-10GB）
- 适用场景：标准训练任务，中等规模数据集

## 3. 多机多卡训练

适用于大规模模型训练，利用多台机器的多个GPU，实现最大化的训练并行性。

### 3.1 使用PyTorch分布式启动（节点1 - 主节点）

```bash
# 在主节点（MASTER_ADDR）上运行
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --documents data/raw_data/train/documents \
    --output_dir outputs/multi_node \
    --batch_size 8 \
    --ddp \
    --fp16 \
    --use_peft \
    --use_data_augmentation \
    --use_wandb \
    --wandb_project "industrial-multimodal-qa" \
    --wandb_name "multi-node-training"
```

### 3.2 节点2运行命令

```bash
# 在第二个节点上运行
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --documents data/raw_data/train/documents \
    --output_dir outputs/multi_node \
    --batch_size 8 \
    --ddp \
    --fp16 \
    --use_peft \
    --use_data_augmentation \
    --use_wandb \
    --wandb_project "industrial-multimodal-qa" \
    --wandb_name "multi-node-training"
```

### 3.3 使用Slurm集群管理系统（更简洁）

如果使用Slurm集群，可以更方便地启动多节点训练：

```bash
# 创建提交脚本 train.slurm
cat << EOF > train.slurm
#!/bin/bash
#SBATCH --job-name=multimodal_qa
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

# 加载必要的环境模块
module load cuda/11.8
module load anaconda3

# 激活环境
source activate multimodal_env

# 获取节点信息
export MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# 启动分布式训练
srun python scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --documents data/raw_data/train/documents \
    --output_dir outputs/slurm_training \
    --batch_size 8 \
    --ddp \
    --fp16 \
    --use_peft \
    --use_data_augmentation
EOF

# 提交作业
sbatch train.slurm
```

### 关键参数说明：

- **nnodes**：总节点数（例如2表示使用2台机器）
- **node_rank**：当前节点的排名（0为主节点）
- **master_addr**：主节点的IP地址
- **master_port**：通信端口
- **nproc_per_node**：每个节点上使用的GPU数量

### 性能估计：

- 训练速度：理论上可达到单机的N倍（N为节点数）
- 实际加速比：通常为0.7-0.9N（受网络带宽和延迟影响）
- 适用场景：大规模数据集，需要快速训练的生产环境

## 4. 高级配置与优化

### 4.1 内存优化

```bash
# 大模型训练内存优化
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --documents data/raw_data/train/documents \
    --output_dir outputs/memory_optimized \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --fp16 \
    --ddp \
    --use_peft \
    --optimizer_type "8bit-adam" \
    --cpu_offload
```

### 4.2 性能监控

```bash
# 启用性能分析
python scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --documents data/raw_data/train/documents \
    --output_dir outputs/profiling \
    --profile \
    --logging_steps 1 \
    --use_wandb \
    --wandb_project "industrial-multimodal-qa" \
    --wandb_name "profiling-run"
```

### 4.3 检查点管理

```bash
# 优化检查点管理
python scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --documents data/raw_data/train/documents \
    --output_dir outputs/checkpoints \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --metric_for_best_model "eval_accuracy"
```

## 5. 常见问题与解决方案

### 5.1 NCCL错误

如果遇到NCCL相关错误，可以尝试：

```bash
# 设置环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果不使用InfiniBand网络
export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口

# 然后启动训练
python -m torch.distributed.launch ...
```

### 5.2 显存不足

如果遇到OOM（内存溢出）错误：

```bash
# 降低批量大小，增加梯度累积步数
python scripts/train_model.py \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --fp16 \
    --use_peft
```

### 5.3 节点通信问题

确保所有节点之间可以通过指定端口互相访问：

```bash
# 在各节点测试连接性
nc -zv master_node_ip 29500
```

## 6. 性能对比

| 配置 | GPU数量 | 批量大小 | 训练时间 | 加速比 | 显存使用 |
|------|---------|----------|----------|--------|----------|
| 单机单卡 | 1 | 4 | 6小时 | 1x | 8GB |
| 单机4卡 | 4 | 8/GPU | 1.8小时 | 3.3x | 8GB/GPU |
| 双机8卡 | 16 | 8/GPU | 28分钟 | 12.8x | 8GB/GPU |

注意：实际性能会因数据规模、网络带宽和GPU型号而异。