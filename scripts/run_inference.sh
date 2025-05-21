#!/bin/bash
# =============================================================================
# 工业技术文档多模态推理问答系统高效推理运行脚本
# 提供多种预配置推理选项，适用于不同场景
# =============================================================================

# 设置默认参数
MODEL_PATH="outputs/best_model"
QUESTIONS_FILE="data/raw_data/test/questions.jsonl"
DOCUMENTS_DIR="data/raw_data/test/documents"
OUTPUT_FILE="results/predictions.jsonl"
PROCESSED_DATA_DIR="data/processed_data"
BATCH_SIZE=1
MAX_BATCH_SIZE=8
DEVICE="cuda"
MODE="default"  # 默认模式

# 帮助信息
function show_help {
    echo "用法: $0 [选项] --mode 模式"
    echo ""
    echo "工业技术文档多模态推理问答系统高效推理脚本"
    echo ""
    echo "选项:"
    echo "  --model PATH          模型路径 (默认: $MODEL_PATH)"
    echo "  --questions FILE      问题文件路径 (默认: $QUESTIONS_FILE)"
    echo "  --documents DIR       文档目录 (默认: $DOCUMENTS_DIR)"
    echo "  --output FILE         输出文件路径 (默认: $OUTPUT_FILE)"
    echo "  --processed_data DIR  预处理数据目录 (默认: $PROCESSED_DATA_DIR)"
    echo "  --batch_size N        批处理大小 (默认: $BATCH_SIZE)"
    echo "  --device DEVICE       推理设备 (默认: $DEVICE)"
    echo "  --mode MODE           推理模式 (见下文)"
    echo "  --help                显示此帮助信息"
    echo ""
    echo "支持的推理模式:"
    echo "  default       - 默认模式，平衡性能和质量"
    echo "  fast          - 快速推理模式，优先考虑速度"
    echo "  accurate      - 高精度模式，优先考虑准确性"
    echo "  lite          - 轻量级模式，适用于资源受限环境"
    echo "  onnx          - 使用ONNX运行时加速"
    echo "  tensorrt      - 使用TensorRT加速 (需要GPU)"
    echo "  batch         - 批量处理模式，适用于大量问题"
    echo "  cpu           - 仅CPU推理模式"
    echo ""
    echo "示例:"
    echo "  $0 --mode fast                       # 快速推理模式"
    echo "  $0 --mode accurate --batch_size 4    # 高精度批量推理"
    echo "  $0 --mode cpu --model outputs/quantized_model  # CPU推理"
    exit 1
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --questions)
            QUESTIONS_FILE="$2"
            shift 2
            ;;
        --documents)
            DOCUMENTS_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --processed_data)
            PROCESSED_DATA_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "未知选项: $1"
            show_help
            ;;
    esac
done

# 确保输出目录存在
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

# 根据模式配置参数
case $MODE in
    default)
        echo "使用默认推理模式 - 平衡性能和准确性"
        ADDITIONAL_ARGS="--fp16"
        ;;
    fast)
        echo "使用快速推理模式 - 优先考虑速度"
        if [[ "$DEVICE" == *"cuda"* ]]; then
            ADDITIONAL_ARGS="--fp16 --cuda_graph --max_batch_size 16"
            BATCH_SIZE=4
        else
            ADDITIONAL_ARGS="--int8"
        fi
        ;;
    accurate)
        echo "使用高精度推理模式 - 优先考虑准确性"
        ADDITIONAL_ARGS="--no_fp16"
        ;;
    lite)
        echo "使用轻量级推理模式 - 适用于资源受限环境"
        ADDITIONAL_ARGS="--int8 --workers 2"
        ;;
    onnx)
        echo "使用ONNX运行时加速推理"
        ADDITIONAL_ARGS="--use_onnx --fp16"
        ;;
    tensorrt)
        if [[ "$DEVICE" == *"cuda"* ]]; then
            echo "使用TensorRT加速推理"
            # 注意: 需要先将模型转换为TensorRT格式
            ADDITIONAL_ARGS="--use_onnx --fp16 --cuda_graph"
            BATCH_SIZE=8
        else
            echo "错误: TensorRT模式需要GPU设备"
            exit 1
        fi
        ;;
    batch)
        echo "使用批量处理模式 - 适用于大量问题"
        BATCH_SIZE=8
        MAX_BATCH_SIZE=32
        ADDITIONAL_ARGS="--fp16 --workers 4"
        ;;
    cpu)
        echo "使用CPU推理模式"
        DEVICE="cpu"
        ADDITIONAL_ARGS="--int8 --workers 8"
        ;;
    *)
        echo "未知模式: $MODE"
        show_help
        ;;
esac

# 日志文件
LOG_FILE="${OUTPUT_DIR}/inference_${MODE}_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================================="
echo "工业技术文档多模态推理问答系统高效推理"
echo "开始时间: $(date)"
echo "----------------------------------------"
echo "模型路径: $MODEL_PATH"
echo "问题文件: $QUESTIONS_FILE"
echo "文档目录: $DOCUMENTS_DIR"
echo "输出文件: $OUTPUT_FILE"
echo "预处理数据: $PROCESSED_DATA_DIR"
echo "批处理大小: $BATCH_SIZE (最大 $MAX_BATCH_SIZE)"
echo "推理设备: $DEVICE"
echo "推理模式: $MODE"
echo "日志文件: $LOG_FILE"
echo "=========================================================="

# 执行推理
python scripts/inference.py \
    --model "$MODEL_PATH" \
    --questions "$QUESTIONS_FILE" \
    --documents "$DOCUMENTS_DIR" \
    --output "$OUTPUT_FILE" \
    --processed_data "$PROCESSED_DATA_DIR" \
    --batch_size "$BATCH_SIZE" \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --device "$DEVICE" \
    --perf_stats \
    $ADDITIONAL_ARGS 2>&1 | tee -a "$LOG_FILE"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "推理完成，结果已保存至: $OUTPUT_FILE"
    echo "日志文件已保存至: $LOG_FILE"
    
    # 显示性能统计
    STATS_FILE="${OUTPUT_FILE%.*}_stats.json"
    if [ -f "$STATS_FILE" ]; then
        echo "----------------------------------------"
        echo "性能统计:"
        python -c "
import json
with open('$STATS_FILE', 'r') as f:
    stats = json.load(f)
    print(f\"处理问题数: {stats.get('processed_questions', 0)}问题\")
    print(f\"总耗时: {stats.get('total_time', 0):.2f}秒\")
    print(f\"每问题平均时间: {stats.get('avg_time_per_question', 0):.4f}秒\")
    print(f\"处理速度: {stats.get('questions_per_second', 0):.2f}问题/秒\")
    if 'gpu_memory_allocated' in stats:
        print(f\"GPU内存使用: {stats.get('gpu_memory_allocated', 0):.2f}MB\")
        print(f\"峰值GPU内存: {stats.get('gpu_max_memory_allocated', 0):.2f}MB\")
"
    fi
    
    # 输出示例结果
    echo "----------------------------------------"
    echo "结果样例 (前5条):"
    head -n 5 "$OUTPUT_FILE"
    echo "----------------------------------------"
    echo "全部推理完成!"
else
    echo "推理过程中出现错误，请查看日志: $LOG_FILE"
fi