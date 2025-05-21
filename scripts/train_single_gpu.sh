#!/bin/bash
# =============================================================================
# 单机单卡训练脚本
# 适用于入门实验、调试和小规模训练
# =============================================================================

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 指定使用的GPU编号，单卡训练时使用GPU 0

# 基本配置参数
TRAIN_DATA="data/raw_data/train/questions.jsonl"
VALID_DATA="data/raw_data/valid/questions.jsonl"
DOC_DIR="data/raw_data/train/documents"
OUTPUT_DIR="outputs/single_gpu_$(date +%Y%m%d_%H%M%S)"  # 使用时间戳创建唯一输出目录

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 记录训练配置
echo "开始单机单卡训练 $(date)" | tee -a ${OUTPUT_DIR}/training.log
echo "训练数据: ${TRAIN_DATA}" | tee -a ${OUTPUT_DIR}/training.log
echo "验证数据: ${VALID_DATA}" | tee -a ${OUTPUT_DIR}/training.log
echo "文档目录: ${DOC_DIR}" | tee -a ${OUTPUT_DIR}/training.log
echo "输出目录: ${OUTPUT_DIR}" | tee -a ${OUTPUT_DIR}/training.log

# 启动训练
python scripts/train_model.py \
    --train_questions ${TRAIN_DATA} \
    --valid_questions ${VALID_DATA} \
    --documents ${DOC_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --epochs 15 \
    --fp16 \
    --use_peft \
    --peft_technique lora \
    --use_data_augmentation \
    --use_optimized_loss \
    --logging_steps 10 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --save_total_limit 3 \
    --load_best_model_at_end True | tee -a ${OUTPUT_DIR}/training.log

# 训练完成后评估性能
echo "训练完成，开始评估..." | tee -a ${OUTPUT_DIR}/training.log

# 如果有测试集，进行评估
TEST_DATA="data/raw_data/test/questions.jsonl"
if [ -f "$TEST_DATA" ]; then
    python scripts/train_model.py \
        --test_questions ${TEST_DATA} \
        --documents ${DOC_DIR} \
        --resume ${OUTPUT_DIR}/best_model \
        --output_dir ${OUTPUT_DIR}/evaluation \
        --output_file ${OUTPUT_DIR}/predictions.jsonl \
        --predict_only | tee -a ${OUTPUT_DIR}/evaluation.log
    
    echo "评估完成，结果保存至 ${OUTPUT_DIR}/predictions.jsonl" | tee -a ${OUTPUT_DIR}/evaluation.log
else
    echo "未找到测试数据，跳过评估步骤" | tee -a ${OUTPUT_DIR}/training.log
fi

echo "全部流程完成 $(date)" | tee -a ${OUTPUT_DIR}/training.log