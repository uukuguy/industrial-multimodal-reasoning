#!/bin/bash
# =============================================================================
# 单机多卡训练脚本
# 使用PyTorch的分布式数据并行(DDP)实现多GPU训练加速
# =============================================================================

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用4个GPU
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0

# 基本配置参数
TRAIN_DATA="data/raw_data/train/questions.jsonl"
# VALID_DATA="data/raw_data/valid/questions.jsonl"  # 注释掉，使用自动划分
DOC_DIR="data/raw_data/train/documents"
# OUTPUT_DIR="outputs/multi_gpu_$(date +%Y%m%d_%H%M%S)"  # 使用时间戳创建唯一输出目录
OUTPUT_DIR="outputs"

# 验证集划分参数
VALIDATION_SPLIT_RATIO=0.05  # 使用20%的数据作为验证集
VALIDATION_SPLIT_COUNT=20  # 指定样本数量（优先级高于比例）
VALIDATION_SPLIT_SEED=42  # 随机种子，确保可复现性

# GPU数量配置
NUM_GPUS=4  # 修改此参数以适应您的硬件配置

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 记录训练配置
echo "开始单机多卡训练 $(date)" | tee -a ${OUTPUT_DIR}/training.log
echo "使用GPU数量: ${NUM_GPUS}" | tee -a ${OUTPUT_DIR}/training.log
echo "训练数据: ${TRAIN_DATA}" | tee -a ${OUTPUT_DIR}/training.log
echo "验证集划分样本数: ${VALIDATION_SPLIT_COUNT} (优先)" | tee -a ${OUTPUT_DIR}/training.log
echo "验证集划分比例: ${VALIDATION_SPLIT_RATIO}" | tee -a ${OUTPUT_DIR}/training.log
echo "验证集划分随机种子: ${VALIDATION_SPLIT_SEED}" | tee -a ${OUTPUT_DIR}/training.log
echo "文档目录: ${DOC_DIR}" | tee -a ${OUTPUT_DIR}/training.log
echo "输出目录: ${OUTPUT_DIR}" | tee -a ${OUTPUT_DIR}/training.log

# 每GPU批量大小（注意：总批量大小 = BATCH_SIZE_PER_GPU * NUM_GPUS * GRAD_ACCUM_STEPS）
BATCH_SIZE_PER_GPU=6
GRAD_ACCUM_STEPS=2

# 训练超参数
LEARNING_RATE=1e-4
NUM_EPOCHS=10
WEIGHT_DECAY=0.01
GRADIENT_ACCUMULATION_STEPS=1
SAVE_STEPS=1000
LOGGING_STEPS=100
WARMUP_STEPS=1000
MAX_STEPS=100000

echo "每个GPU的批量大小: ${BATCH_SIZE_PER_GPU}" | tee -a ${OUTPUT_DIR}/training.log
echo "梯度累积步数: ${GRAD_ACCUM_STEPS}" | tee -a ${OUTPUT_DIR}/training.log
echo "有效总批量大小: $((BATCH_SIZE_PER_GPU * NUM_GPUS * GRAD_ACCUM_STEPS))" | tee -a ${OUTPUT_DIR}/training.log
echo "学习率: ${LEARNING_RATE}" | tee -a ${OUTPUT_DIR}/training.log
echo "训练轮数: ${NUM_EPOCHS}" | tee -a ${OUTPUT_DIR}/training.log
echo "权重衰减: ${WEIGHT_DECAY}" | tee -a ${OUTPUT_DIR}/training.log
echo "保存步数: ${SAVE_STEPS}" | tee -a ${OUTPUT_DIR}/training.log
echo "日志步数: ${LOGGING_STEPS}" | tee -a ${OUTPUT_DIR}/training.log
echo "预热步数: ${WARMUP_STEPS}" | tee -a ${OUTPUT_DIR}/training.log
echo "最大步数: ${MAX_STEPS}" | tee -a ${OUTPUT_DIR}/training.log

# 优化配置
USE_FLASH_ATTENTION=true
USE_MODEL_COMPILATION=true
USE_DYNAMIC_BATCHING=true
USE_MIXED_PRECISION=true
USE_GRADIENT_CHECKPOINTING=true
USE_QUANTIZATION=true
QUANTIZATION_BITS=8
USE_ATTENTION_CACHE=true
USE_SLIDING_WINDOW=true
WINDOW_SIZE=512
MAX_MEMORY_USAGE=0.9
CLEAR_CACHE_FREQUENCY=100

# 使用 torch.distributed.launch 启动分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --use_env \
    scripts/train_model.py \
    --train_questions ${TRAIN_DATA} \
    --validation_split_ratio ${VALIDATION_SPLIT_RATIO} \
    --validation_split_count ${VALIDATION_SPLIT_COUNT} \
    --validation_split_seed ${VALIDATION_SPLIT_SEED} \
    --documents ${DOC_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE_PER_GPU} \
    --epochs ${NUM_EPOCHS} \
    --fp16 \
    --ddp \
    --use_peft \
    --peft_technique lora \
    --use_data_augmentation \
    --use_optimized_loss \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio 0.1 \
    --logging_steps ${LOGGING_STEPS} \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --use_flash_attention ${USE_FLASH_ATTENTION} \
    --use_model_compilation ${USE_MODEL_COMPILATION} \
    --use_dynamic_batching ${USE_DYNAMIC_BATCHING} \
    --use_mixed_precision ${USE_MIXED_PRECISION} \
    --use_gradient_checkpointing ${USE_GRADIENT_CHECKPOINTING} \
    --use_quantization ${USE_QUANTIZATION} \
    --quantization_bits ${QUANTIZATION_BITS} \
    --use_attention_cache ${USE_ATTENTION_CACHE} \
    --use_sliding_window ${USE_SLIDING_WINDOW} \
    --window_size ${WINDOW_SIZE} \
    --max_memory_usage ${MAX_MEMORY_USAGE} \
    --clear_cache_frequency ${CLEAR_CACHE_FREQUENCY} \
    --max_steps ${MAX_STEPS} \
    --num_workers ${NUM_GPUS} \
    --local_rank $LOCAL_RANK | tee -a ${OUTPUT_DIR}/training.log

# 训练完成后评估性能 (评估只需要单进程)
echo "训练完成，开始评估..." | tee -a ${OUTPUT_DIR}/evaluation.log

# 如果有测试集，进行评估
TEST_DATA="data/raw_data/test/questions.jsonl"
if [ -f "$TEST_DATA" ]; then
    export CUDA_VISIBLE_DEVICES=0  # 使用单个GPU进行评估
    
    python scripts/train_model.py \
        --test_questions ${TEST_DATA} \
        --documents ${DOC_DIR} \
        --resume ${OUTPUT_DIR}/best_model \
        --output_dir ${OUTPUT_DIR}/evaluation \
        --output_file ${OUTPUT_DIR}/predictions.jsonl \
        --predict_only | tee -a ${OUTPUT_DIR}/evaluation.log
    
    echo "评估完成，结果保存至 ${OUTPUT_DIR}/predictions.jsonl" | tee -a ${OUTPUT_DIR}/evaluation.log
else
    echo "未找到测试数据，跳过评估步骤" | tee -a ${OUTPUT_DIR}/evaluation.log
fi

echo "全部流程完成 $(date)" | tee -a ${OUTPUT_DIR}/training.log

# 清理NCCL共享文件（可选，某些环境可能会遇到NCCL文件锁问题）
rm -rf /tmp/torch-ddp-*