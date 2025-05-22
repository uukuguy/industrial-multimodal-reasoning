#!/bin/bash
# =============================================================================
# 单机多卡训练脚本
# 使用PyTorch的分布式数据并行(DDP)实现多GPU训练加速
# =============================================================================

# 设置环境变量
export PYTHONPATH=$(pwd):$PYTHONPATH

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
LEARNING_RATE=3e-5  # 多GPU时可以适当增大学习率
NUM_EPOCHS=10

echo "每个GPU的批量大小: ${BATCH_SIZE_PER_GPU}" | tee -a ${OUTPUT_DIR}/training.log
echo "梯度累积步数: ${GRAD_ACCUM_STEPS}" | tee -a ${OUTPUT_DIR}/training.log
echo "有效总批量大小: $((BATCH_SIZE_PER_GPU * NUM_GPUS * GRAD_ACCUM_STEPS))" | tee -a ${OUTPUT_DIR}/training.log
echo "学习率: ${LEARNING_RATE}" | tee -a ${OUTPUT_DIR}/training.log
echo "训练轮数: ${NUM_EPOCHS}" | tee -a ${OUTPUT_DIR}/training.log

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
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --save_total_limit 3 \
    --load_best_model_at_end | tee -a ${OUTPUT_DIR}/training.log

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