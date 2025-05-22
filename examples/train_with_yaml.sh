#!/bin/bash
# =============================================================================
# 使用YAML配置文件训练示例
# =============================================================================

# 设置环境变量
export PYTHONPATH=$(pwd):$PYTHONPATH

# 配置文件路径
CONFIG_FILE="config/training_config.yaml"

# GPU数量配置
NUM_GPUS=1  # 修改此参数以适应您的硬件配置

# 输出目录
OUTPUT_DIR="outputs/yaml_example"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 记录训练配置
echo "开始使用YAML配置文件训练 $(date)" | tee -a ${OUTPUT_DIR}/training.log
echo "使用GPU数量: ${NUM_GPUS}" | tee -a ${OUTPUT_DIR}/training.log
echo "配置文件: ${CONFIG_FILE}" | tee -a ${OUTPUT_DIR}/training.log
echo "输出目录: ${OUTPUT_DIR}" | tee -a ${OUTPUT_DIR}/training.log

# 单GPU训练
if [ $NUM_GPUS -eq 1 ]; then
    python scripts/train_model.py \
        --config ${CONFIG_FILE} \
        --output_dir ${OUTPUT_DIR} | tee -a ${OUTPUT_DIR}/training.log
# 多GPU训练
else
    python -m torch.distributed.launch \
        --nproc_per_node=${NUM_GPUS} \
        --use_env \
        scripts/train_model.py \
        --config ${CONFIG_FILE} \
        --output_dir ${OUTPUT_DIR} \
        --ddp | tee -a ${OUTPUT_DIR}/training.log
fi

echo "训练完成 $(date)" | tee -a ${OUTPUT_DIR}/training.log