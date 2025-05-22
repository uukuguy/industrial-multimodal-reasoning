#!/bin/bash
# =============================================================================
# 单机多卡训练脚本 (使用YAML配置文件)
# 使用PyTorch的分布式数据并行(DDP)实现多GPU训练加速
# =============================================================================

# 设置CUDA环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_GID_INDEX=3

# 训练参数
BATCH_SIZE=16
NUM_EPOCHS=15
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
GRADIENT_ACCUMULATION_STEPS=1
FP16=true
NUM_WORKERS=4
SEED=42

# 数据路径
TRAIN_DATA_PATH="data/train"
VAL_DATA_PATH="data/val"
CONFIG_PATH="config/training_config.yaml"

# 输出路径
OUTPUT_DIR="outputs"
CHECKPOINT_DIR="checkpoints"
LOG_DIR="logs"

# 创建必要的目录
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${LOG_DIR}

# 记录训练配置
echo "开始单机多卡训练 $(date)" | tee -a ${OUTPUT_DIR}/training.log
echo "使用GPU数量: ${CUDA_VISIBLE_DEVICES}" | tee -a ${OUTPUT_DIR}/training.log
echo "配置文件: ${CONFIG_PATH}" | tee -a ${OUTPUT_DIR}/training.log
echo "输出目录: ${OUTPUT_DIR}" | tee -a ${OUTPUT_DIR}/training.log

# 启动分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train.py \
    --train_data_path ${TRAIN_DATA_PATH} \
    --val_data_path ${VAL_DATA_PATH} \
    --config_path ${CONFIG_PATH} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio ${WARMUP_RATIO} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --fp16 ${FP16} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED} \
    --output_dir ${OUTPUT_DIR} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --log_dir ${LOG_DIR} \
    2>&1 | tee $LOG_DIR/train.log

# 训练完成后评估性能 (评估只需要单进程)
echo "训练完成，开始评估..." | tee -a ${OUTPUT_DIR}/evaluation.log

# 如果有测试集，进行评估
TEST_DATA="data/raw_data/test/questions.jsonl"
if [ -f "$TEST_DATA" ]; then
    export CUDA_VISIBLE_DEVICES=0  # 使用单个GPU进行评估
    
    python scripts/train_model.py \
        --config ${CONFIG_PATH} \
        --test_questions ${TEST_DATA} \
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