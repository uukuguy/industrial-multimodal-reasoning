#!/bin/bash
# =============================================================================
# 多机多卡训练脚本
# 支持在多台机器上分布式训练，实现大规模加速
# =============================================================================

# 使用说明
function show_usage {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  --master_addr ADDR       主节点IP地址 (必需)"
    echo "  --master_port PORT       通信端口，默认为29500"
    echo "  --nnodes N               总节点数，默认为2"
    echo "  --node_rank RANK         当前节点排名(0为主节点)"
    echo "  --nproc_per_node NPROC   每个节点的GPU数量，默认为全部可用GPU"
    echo "  --train_data PATH        训练数据路径，默认为data/raw_data/train/questions.jsonl"
    echo "  --valid_data PATH        验证数据路径，默认为data/raw_data/valid/questions.jsonl"
    echo "  --doc_dir PATH           文档目录路径，默认为data/raw_data/train/documents"
    echo "  --output_dir PATH        输出目录，默认为outputs/multi_node_{时间戳}"
    echo "  --batch_size N           每个GPU的批量大小，默认为4"
    echo "  --epochs N               训练轮数，默认为8"
    echo "  --grad_accum_steps N     梯度累积步数，默认为4"
    echo "  --learning_rate RATE     学习率，默认为3e-5"
    echo "  --no_peft               禁用参数高效微调"
    echo "  --no_data_aug           禁用数据增强"
    echo "  --no_optim_loss         禁用优化损失函数"
    echo "  --use_wandb             启用Weights & Biases日志"
    echo "  --wandb_project NAME     W&B项目名称"
    echo "  --wandb_name NAME        W&B运行名称"
    echo "  --help                   显示此帮助信息"
    exit 1
}

# 解析命令行参数
MASTER_ADDR=""
MASTER_PORT=29500
NNODES=2
NODE_RANK=""
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # 默认使用所有可用GPU
TRAIN_DATA="data/raw_data/train/questions.jsonl"
VALID_DATA="data/raw_data/valid/questions.jsonl"
DOC_DIR="data/raw_data/train/documents"
OUTPUT_DIR="outputs/multi_node_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=4
EPOCHS=8
GRAD_ACCUM_STEPS=4
LEARNING_RATE=3e-5
USE_PEFT=true
USE_DATA_AUG=true
USE_OPTIM_LOSS=true
USE_WANDB=false
WANDB_PROJECT="industrial-multimodal-qa"
WANDB_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --nproc_per_node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --train_data)
            TRAIN_DATA="$2"
            shift 2
            ;;
        --valid_data)
            VALID_DATA="$2"
            shift 2
            ;;
        --doc_dir)
            DOC_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --grad_accum_steps)
            GRAD_ACCUM_STEPS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --no_peft)
            USE_PEFT=false
            shift
            ;;
        --no_data_aug)
            USE_DATA_AUG=false
            shift
            ;;
        --no_optim_loss)
            USE_OPTIM_LOSS=false
            shift
            ;;
        --use_wandb)
            USE_WANDB=true
            shift
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb_name)
            WANDB_NAME="$2"
            shift 2
            ;;
        --help)
            show_usage
            ;;
        *)
            echo "未知选项: $1"
            show_usage
            ;;
    esac
done

# 检查必需参数
if [[ -z "$MASTER_ADDR" ]]; then
    echo "错误: 必须指定主节点地址 (--master_addr)"
    show_usage
fi

if [[ -z "$NODE_RANK" ]]; then
    echo "错误: 必须指定节点排名 (--node_rank)"
    show_usage
fi

# 设置WANDB运行名称（如果未指定）
if [[ -z "$WANDB_NAME" ]]; then
    WANDB_NAME="multi-node-train-n${NNODES}-r${NODE_RANK}"
fi

# 设置环境变量
export PYTHONPATH=$(pwd):$PYTHONPATH
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
export RANK=$((NODE_RANK * NPROC_PER_NODE))

# 创建输出目录(只在主节点上)
if [[ "$NODE_RANK" == "0" ]]; then
    mkdir -p $OUTPUT_DIR
fi

# 等待主节点创建输出目录
sleep 2

# 所有节点创建自己的日志目录
mkdir -p ${OUTPUT_DIR}/logs
LOG_FILE="${OUTPUT_DIR}/logs/node_${NODE_RANK}.log"

# 记录训练配置
{
    echo "=========================================================="
    echo "多机多卡训练配置 (节点 ${NODE_RANK}/${NNODES-1})"
    echo "开始时间: $(date)"
    echo "=========================================================="
    echo "主节点地址: ${MASTER_ADDR}"
    echo "通信端口: ${MASTER_PORT}"
    echo "总节点数: ${NNODES}"
    echo "当前节点排名: ${NODE_RANK}"
    echo "每节点GPU数: ${NPROC_PER_NODE}"
    echo "总GPU数: $((NNODES * NPROC_PER_NODE))"
    echo "训练数据: ${TRAIN_DATA}"
    echo "验证数据: ${VALID_DATA}"
    echo "文档目录: ${DOC_DIR}"
    echo "输出目录: ${OUTPUT_DIR}"
    echo "每GPU批量大小: ${BATCH_SIZE}"
    echo "梯度累积步数: ${GRAD_ACCUM_STEPS}"
    echo "有效总批量大小: $((BATCH_SIZE * NNODES * NPROC_PER_NODE * GRAD_ACCUM_STEPS))"
    echo "训练轮数: ${EPOCHS}"
    echo "学习率: ${LEARNING_RATE}"
    echo "使用PEFT: ${USE_PEFT}"
    echo "使用数据增强: ${USE_DATA_AUG}"
    echo "使用优化损失函数: ${USE_OPTIM_LOSS}"
    echo "使用W&B: ${USE_WANDB}"
    if [[ "$USE_WANDB" == "true" ]]; then
        echo "W&B项目: ${WANDB_PROJECT}"
        echo "W&B运行名称: ${WANDB_NAME}"
    fi
    echo "主机名: $(hostname)"
    echo "可见GPU: $(nvidia-smi --list-gpus | wc -l)"
    echo "=========================================================="
} | tee -a $LOG_FILE

# 构建命令行参数
CMD_ARGS=""
if [[ "$USE_PEFT" == "true" ]]; then
    CMD_ARGS="$CMD_ARGS --use_peft"
fi
if [[ "$USE_DATA_AUG" == "true" ]]; then
    CMD_ARGS="$CMD_ARGS --use_data_augmentation"
fi
if [[ "$USE_OPTIM_LOSS" == "true" ]]; then
    CMD_ARGS="$CMD_ARGS --use_optimized_loss"
fi
if [[ "$USE_WANDB" == "true" ]]; then
    CMD_ARGS="$CMD_ARGS --use_wandb --wandb_project ${WANDB_PROJECT} --wandb_name ${WANDB_NAME}"
fi

# 启动分布式训练
echo "启动训练进程..." | tee -a $LOG_FILE

# 使用 torch.distributed.launch 启动分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    scripts/train_model.py \
    --train_questions ${TRAIN_DATA} \
    --valid_questions ${VALID_DATA} \
    --documents ${DOC_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --fp16 \
    --ddp \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    $CMD_ARGS 2>&1 | tee -a $LOG_FILE

TRAIN_STATUS=$?

# 检查训练状态
if [[ $TRAIN_STATUS -ne 0 ]]; then
    echo "训练进程异常退出，状态码: $TRAIN_STATUS" | tee -a $LOG_FILE
    exit $TRAIN_STATUS
fi

# 训练完成后评估性能 (仅在主节点上执行)
if [[ "$NODE_RANK" == "0" ]]; then
    echo "训练完成，开始评估..." | tee -a $LOG_FILE
    
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
            --predict_only 2>&1 | tee -a ${OUTPUT_DIR}/evaluation.log
        
        echo "评估完成，结果保存至 ${OUTPUT_DIR}/predictions.jsonl" | tee -a $LOG_FILE
    else
        echo "未找到测试数据，跳过评估步骤" | tee -a $LOG_FILE
    fi
else
    echo "训练完成，在工作节点上跳过评估步骤" | tee -a $LOG_FILE
fi

echo "全部流程完成 $(date)" | tee -a $LOG_FILE

# 清理NCCL共享文件
rm -rf /tmp/torch-ddp-*

# 退出
exit 0