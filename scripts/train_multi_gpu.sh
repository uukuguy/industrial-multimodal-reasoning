#!/bin/bash
# =============================================================================
# 单机多卡训练脚本 (使用YAML配置文件)
# 使用PyTorch的分布式数据并行(DDP)实现多GPU训练加速
# =============================================================================

# 设置环境变量
export PYTHONPATH=$(pwd):$PYTHONPATH

# 配置文件路径
CONFIG_FILE="config/training_config.yaml"

# GPU数量配置
NUM_GPUS=4  # 修改此参数以适应您的硬件配置

# 输出目录
OUTPUT_DIR="outputs"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 记录训练配置
echo "开始单机多卡训练 $(date)" | tee -a ${OUTPUT_DIR}/training.log
echo "使用GPU数量: ${NUM_GPUS}" | tee -a ${OUTPUT_DIR}/training.log
echo "配置文件: ${CONFIG_FILE}" | tee -a ${OUTPUT_DIR}/training.log
echo "输出目录: ${OUTPUT_DIR}" | tee -a ${OUTPUT_DIR}/training.log

# 使用 torch.distributed.launch 启动分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --use_env \
    scripts/train_model.py \
    --config ${CONFIG_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --ddp | tee -a ${OUTPUT_DIR}/training.log

# 训练完成后评估性能 (评估只需要单进程)
echo "训练完成，开始评估..." | tee -a ${OUTPUT_DIR}/evaluation.log

# 如果有测试集，进行评估
TEST_DATA="data/raw_data/test/questions.jsonl"
if [ -f "$TEST_DATA" ]; then
    export CUDA_VISIBLE_DEVICES=0  # 使用单个GPU进行评估
    
    python scripts/train_model.py \
        --config ${CONFIG_FILE} \
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