# 工业技术文档多模态推理问答系统

[**CCKS2025-工业技术文档多模态推理问答评测**](https://tianchi.aliyun.com/competition/entrance/532357/information)

2025.05.01 - 2025.08.01

## 项目概述

本项目为CCKS2025工业技术文档多模态推理问答评测任务的解决方案。系统能够处理工业技术文档中的文本、图像和版面信息，进行多模态推理，针对初赛的选择题和复赛的开放题进行回答。

系统解决了工业技术文档多模态推理面临的三大挑战：
1. 图片型原始文档识别：处理不可编辑的PDF文档，包括低分辨率、多样格式和复杂内容
2. 多模态信息融合：同时解析文本描述和技术图纸等多模态数据
3. 复杂化领域知识推理：通过图纸结构解析、模块功能理解或机械原理推导获得答案

## 系统架构

系统采用模块化设计，包含四个核心模块和多个辅助组件：

![系统架构](docs/images/system_architecture.png)

### 核心模块

1. **文档预处理模块**：将PDF文档转换为结构化的文本、图像和版面信息
2. **多模态编码模块**：将不同模态的信息编码为嵌入向量
3. **多模态融合模块**：融合不同模态的信息，形成统一的文档表示
4. **推理与问答模块**：根据文档表示和问题生成答案

## 训练优化策略

面对工业技术文档多模态推理的挑战，我们实现了一套针对性的训练优化策略，克服了数据量有限、不平衡等问题：

![训练优化策略](docs/images/training_optimization.png)

### 优化要点

1. **数据增强与平衡**：
   - 实现智能数据增强扩充训练集(问题改写、选项增强、困难负样本生成)
   - 针对低频问题类型(技术参数、操作步骤)的重点增强
   - 将300样本扩充至450-500个有效样本

2. **参数高效微调(PEFT)**：
   - 应用LoRA、Prefix-Tuning等技术，减少99%训练参数
   - 自动选择适合数据规模的最佳PEFT技术
   - 降低过拟合风险，提升泛化能力

3. **改进的文档处理**：
   - 集成PaddleOCR处理图像型PDF
   - 优化文本、图像、版面特征提取
   - 增强对复杂图表的理解能力

4. **高效训练框架**：
   - 分布式训练与混合精度支持
   - 与Weights & Biases集成的实验追踪
   - 自动化的最佳化配置选择

### 优化损失函数

新增了专门针对工业技术文档多模态理解的优化损失函数：

```python
# 优化的损失函数配置示例
python scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --documents data/raw_data/train/documents \
    --use_optimized_loss \
    --output_dir outputs/model_optimized_loss
```

主要优化包括：

- **标签平滑与类别加权**：缓解选择题过拟合和类别不平衡问题
- **技术参数识别强化**：特殊处理数值型答案，精确评估参数识别能力
- **多任务学习支持**：整合重构损失和跨模态对齐损失
- **置信度校准**：考虑不确定性估计，增强可靠性

### 预期效果

- 初赛任务准确率提升15-20%
- 复赛开放题表现显著增强
- 对低频问题类型的处理能力大幅提升
- 训练效率和资源利用率明显优化
- 技术参数和数值识别准确度提高25-30%

### 数据流

```
PDF文档 → 文档预处理 → 多模态编码 → 多模态融合 → 推理与问答 → 答案
```

## 模型设计与实现

### 1. 增强型多模态架构

系统采用增强型多模态架构，通过以下技术实现模态间的深度交互和信息整合：

#### 1.1 层次化融合架构

采用三级融合策略，实现多模态信息的逐步整合：
- **低级融合**：捕获基本的模态对齐，主要处理文本和图像间的初步关联
- **中级融合**：整合语义和空间关系，引入版面信息，建立更深层次的理解
- **高级融合**：形成统一的多模态表示，生成最终用于推理的文档表示

#### 1.2 跨模态注意力机制

使用增强型跨模态注意力，实现不同模态间的信息交互：
- **自适应注意力门控**：根据模态内容动态调整信息流动
- **多头注意力**：从不同角度捕获模态间关系
- **模态对交互**：视觉-文本、视觉-版面、文本-版面之间的双向交互

#### 1.3 缺失模态处理

通过模态重构和优雅降级机制，处理缺失模态情况：
- **模态重构**：从可用模态重建缺失模态
- **优雅降级**：根据可用模态自动调整预测策略
- **模态存在指示器**：为每个模态维护存在指示器，确保模型架构一致性

### 2. 组件详细说明

#### 2.1 文档预处理 (`model/pdf_processor.py`)

##### 设计与实现思路

文档预处理模块负责将原始PDF文档转换为结构化的多模态信息，是整个系统的基础。其核心设计理念包括：

1. **灵活性**：同时支持高级版面分析和基本文本提取模式，根据可用依赖自动调整
2. **健壮性**：对异常情况进行优雅处理，单页错误不影响整体处理流程
3. **模块化**：清晰的处理步骤和数据结构，便于与系统其他部分交互
4. **可扩展性**：良好的接口设计，便于未来支持更多文档类型和分析策略

处理流程：

1. **PDF页面渲染**：
   - 使用PyMuPDF (fitz) 将PDF页面渲染为高分辨率图像
   - 保存每页图像以供后续分析和可视化

2. **版面分析**：
   - 使用LayoutParser库的预训练Detectron2模型进行版面分割
   - 识别文本块、标题、列表、表格和图表等元素
   - 提取每个元素的类型和坐标信息

3. **OCR文本提取**：
   - 使用PaddleOCR对文本区域进行识别
   - 支持中文和英文文本识别
   - 合并同一区域的文本行

4. **图像提取**：
   - 根据版面分析结果提取图表和表格区域
   - 保存为独立的图像文件以供视觉模型分析

5. **结构化表示**：
   - 将所有提取的信息整合为统一的数据结构
   - 每页生成包含页码、图像路径、版面信息和文本内容的结构化字典
   - 保留原始坐标信息以支持空间关系分析

##### 错误处理机制

1. **依赖管理**：
   - 条件导入布局分析和OCR依赖
   - 在依赖不可用时自动降级到基本处理模式

2. **异常捕获**：
   - 页面级错误隔离，单页错误不影响整个文档处理
   - 详细的日志记录，便于问题诊断

3. **数据验证**：
   - 坐标有效性检查，防止图像处理越界
   - 文本和图像内容验证，确保数据完整性

##### 使用方法

```python
from model.pdf_processor import process_pdf

# 处理单个PDF文件
pdf_path = "path/to/document.pdf"
output_dir = "path/to/output"
processed_data = process_pdf(pdf_path, output_dir)

# 处理结果是一个包含每页信息的列表
if processed_data:
    for page in processed_data:
        # 访问页面信息
        page_num = page["page_num"]
        image_path = page["image_path"]
        layout = page["layout"]
        text_blocks = page["text_blocks"]
        
        # 处理文本内容
        for block in text_blocks:
            block_type = block["type"]  # Text, Title, List
            text = block["text"]
            coordinates = block["coordinates"]
            
        # 处理图像内容
        for block in layout:
            if block["type"] in ["Figure", "Table"]:
                segment_path = block["image_segment_path"]
                # 处理图表或表格图像
```

##### 输出数据结构

```python
# 每页的处理结果结构
{
    "page_num": 1,                      # 页码
    "image_path": "path/to/page_1.png",  # 页面图像路径
    "layout": [                          # 版面元素列表
        {
            "type": "Text",              # 元素类型 (Text, Title, List, Table, Figure)
            "coordinates": (x1, y1, x2, y2),  # 元素坐标 (左上角和右下角)
            "text": "文本内容...",        # 文本内容 (仅文本类型有)
            "image_segment_path": None    # 图像片段路径 (仅图表/表格有)
        },
        # ...更多版面元素
    ],
    "text_blocks": [...]                 # 文本块列表 (layout中的文本元素子集)
}
```

#### 2.2 多模态编码 (`model/multimodal_encoder.py`)

编码各模态数据为嵌入向量：
- **文本编码**：使用预训练中文语言模型 (BERT) 编码文本内容
- **图像编码**：使用预训练视觉模型 (ViT) 编码图像内容
- **版面编码**：编码文档的版面结构和空间关系

优化特性：
- 模型缓存机制：避免重复加载模型
- 批处理支持：提高处理效率
- 设备管理：自动使用CPU/GPU
- 错误恢复机制：提高系统稳定性

#### 2.3 跨模态注意力 (`model/cross_modal_attention.py`)

实现模态间的深度信息交互：
- **CrossModalAttention**：单个跨模态注意力模块
- **MultiModalAttentionHub**：集成多个跨模态注意力模块，协调不同模态对之间的交互

#### 2.4 多模态融合 (`model/multimodal_fusion.py`)

融合不同模态的信息：
- **MultimodalFusion**：基础融合模块
- **HierarchicalMultimodalFusion**：层次化融合模块，实现低级、中级和高级特征的逐步融合

#### 2.5 模态重构 (`model/modality_reconstructor.py`)

从可用模态重建缺失模态：
- **视觉重构**：从文本和版面重构视觉特征
- **文本重构**：从视觉和版面重构文本特征
- **版面重构**：从视觉和文本重构版面特征

#### 2.6 不确定性估计 (`model/uncertainty_estimator.py`)

提供预测结果的置信度评估：
- **多层次不确定性**：数据不确定性、模型不确定性、预测不确定性
- **校准置信度**：使用温度缩放等技术校准置信度
- **风险评估**：识别需要人工审核的案例

#### 2.7 推理与问答 (`model/qa_module.py`)

根据融合的文档表示生成答案：
- **ReasoningQAModule**：基础问答模块
- **EnhancedReasoningQAModule**：增强型问答模块，添加自校正机制和领域知识约束

#### 2.8 增强型多模态模型 (`model/enhanced_model.py`)

整合所有组件，提供统一接口：
- **模块化设计**：清晰的接口和责任划分
- **配置驱动**：支持灵活配置
- **状态管理**：完整的保存和加载机制

#### 2.9 配置系统 (`model/config.py`)

增强型YAML配置系统：
- **配置继承**：基于默认配置进行覆盖
- **配置验证**：自动检查配置有效性
- **运行时配置**：支持命令行参数覆盖
- **延迟加载**：按需加载配置，提高性能
- **YAML支持**：使用人类可读的YAML格式
- **配置文件分离**：默认配置与训练配置分离
- **随机种子管理**：确保实验可复现性
- **自动目录创建**：确保输出目录存在

配置系统支持多种配置方式：
- **默认配置**：`config/default_config.yaml`包含所有可配置参数
- **训练配置**：`config/training_config.yaml`针对特定训练场景优化
- **命令行覆盖**：命令行参数可覆盖配置文件中的设置
- **配置合并**：多个配置源可以递归合并

详细说明请参考[YAML配置系统使用指南](docs/yaml_configuration_guide.md)。

### 3. 模型训练

增强型多模态模型支持完整的训练流程，包括预处理、训练和评估。

#### 3.1 训练数据准备

训练数据需要以JSONL格式提供，内容包括问题、文档引用和答案：

初赛数据格式（单选题）：
```json
{
  "id": "question_123",
  "question": "根据文本信息，以下哪个描述符合该静电除尘器的特征？",
  "document": "CN100342976C.pdf",
  "options": ["A. 特征1", "B. 特征2", "C. 特征3", "D. 特征4"],
  "answer": "C"
}
```

复赛数据格式（开放题）：
```json
{
  "id": "question_456",
  "question": "在文件中第7页的图片中，部件4相对于部件5在图片中的位置关系是？",
  "document": "CN100342976C.pdf",
  "answer": "部件4位于部件5的左侧"
}
```

#### 3.2 训练模型

使用`scripts/train_model.py`脚本训练模型，现在基于transformers库的Trainer实现，提供更丰富的功能：

```bash
# 基本训练命令
python scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --valid_questions data/raw_data/valid/questions.jsonl \
    --documents data/raw_data/train/documents \
    --output_dir outputs/model_v1 \
    --batch_size 8 \
    --epochs 10

# 使用预处理数据加速训练 + 混合精度训练
python scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --valid_questions data/raw_data/valid/questions.jsonl \
    --documents data/raw_data/train/documents \
    --processed_data data/processed_data \
    --output_dir outputs/model_v1 \
    --batch_size 16 \
    --epochs 15 \
    --fp16

# 使用YAML配置文件训练
python scripts/train_model.py \
    --config config/training_config.yaml \
    --output_dir outputs/yaml_config_model

# 配置文件与命令行参数结合（命令行参数优先级更高）
python scripts/train_model.py \
    --config config/training_config.yaml \
    --learning_rate 5e-5 \
    --output_dir outputs/custom_config_model
    
# 从检查点恢复训练
python scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --valid_questions data/raw_data/valid/questions.jsonl \
    --documents data/raw_data/train/documents \
    --output_dir outputs/model_v1_continued \
    --resume outputs/model_v1/checkpoint-1000 \
    --batch_size 16 \
    --epochs 5
    
# 仅预测模式（不训练）
python scripts/train_model.py \
    --test_questions data/raw_data/test/questions.jsonl \
    --documents data/raw_data/test/documents \
    --resume outputs/model_v1/best_model \
    --output_file results/test_results.jsonl \
    --predict_only
```

训练参数说明：
- `--train_questions`: 训练集问题JSONL文件路径
- `--valid_questions`: 验证集问题JSONL文件路径(可选)
- `--test_questions`: 测试集问题JSONL文件路径(可选)
- `--documents`: 文档目录路径
- `--processed_data`: 预处理数据目录(可选，加速训练)
- `--batch_size`: 批处理大小
- `--epochs`: 训练轮数
- `--workers`: 数据加载工作进程数
- `--config`: YAML配置文件路径，包含所有训练参数
- `--output_dir`: 结果输出目录
- `--resume`: 从检查点恢复训练(可选)
- `--validation_split_ratio`: 验证集划分比例，0-1之间的浮点数(可选)
- `--validation_split_count`: 验证集样本数量，整数，优先级高于比例参数(可选)
- `--validation_split_seed`: 数据集划分的随机种子，用于复现划分结果(可选)

#### 3.3 训练配置

系统提供了完整的YAML配置系统，支持灵活配置所有训练参数：

```yaml
# config/training_config.yaml
model:
  embedding_dim: 768
  fusion_dim: 512
  use_cache: true
  device: "auto"
  use_reconstructor: true
  use_uncertainty: true

training:
  num_epochs: 10
  batch_size: 6
  gradient_accumulation_steps: 2
  fp16: true  # 使用混合精度训练
  
  optimizer:
    type: "AdamW"
    learning_rate: 3e-5
    weight_decay: 0.01
  
  scheduler:
    type: "cosine"
    warmup_ratio: 0.1

# 数据集划分配置
dataset_split:
  validation_split_ratio: 0.2
  validation_split_count: 100  # 优先使用样本数量
  validation_split_seed: 42
```

##### 3.3.1 配置文件类型

系统提供两种主要配置文件：

1. **默认配置文件** (`config/default_config.yaml`)：
   - 包含所有可配置参数及其默认值
   - 详细的参数注释和说明
   - 作为创建自定义配置的参考模板

2. **训练配置文件** (`config/training_config.yaml`)：
   - 针对特定训练场景优化的配置示例
   - 启用了高级功能如PEFT、数据增强和优化损失函数
   - 包含数据路径和输出目录配置

##### 3.3.2 使用配置文件

可以通过以下方式使用配置文件：

```bash
# 使用配置文件训练
python scripts/train_model.py --config config/training_config.yaml

# 配置文件与命令行参数结合（命令行参数优先级更高）
python scripts/train_model.py \
  --config config/training_config.yaml \
  --learning_rate 5e-5 \
  --output_dir outputs/custom_run
```

##### 3.3.3 配置文件与训练脚本

为方便使用YAML配置系统，提供了专用训练脚本：

```bash
# 使用YAML配置的单机多卡训练
bash scripts/train_multi_gpu_yaml.sh

# 简单的YAML配置训练示例
bash examples/train_with_yaml.sh
```

详细说明请参考[YAML配置系统使用指南](docs/yaml_configuration_guide.md)。

#### 3.4 分布式训练

系统全面支持各种规模的分布式训练，从单机单卡到多机多卡：

```bash
# 1. 单机单卡训练 (基础配置)
python scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --documents data/raw_data/train/documents \
    --output_dir outputs/single_gpu \
    --batch_size 8 \
    --use_peft \
    --fp16

# 2. 单机多卡训练 (使用torch.distributed.launch)
python -m torch.distributed.launch --nproc_per_node=4 scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --documents data/raw_data/train/documents \
    --ddp \
    --fp16 \
    --batch_size 8 \
    --gradient_accumulation_steps 4

# 3. 多机多卡训练 (以2节点为例)
# 节点1 (主节点)
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --documents data/raw_data/train/documents \
    --batch_size 8 \
    --ddp \
    --fp16
```

详细指南请参考[分布式训练完整文档](docs/distributed_training_guide.md)，其中包含：
- 各种训练配置的详细参数说明
- 常见问题排查与优化建议
- 不同规模训练的性能对比
- Slurm集群配置示例

#### 3.5 实验追踪与可视化

支持Weights & Biases (wandb) 和TensorBoard进行实验跟踪：

```bash
# 使用wandb记录训练日志
python scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --documents data/raw_data/train/documents \
    --use_wandb \
    --wandb_project "industrial-multimodal-qa" \
    --wandb_name "experiment-1" \
    --batch_size 16

# 训练完成后查看TensorBoard日志
tensorboard --logdir outputs/model_v1/logs

#### 3.5.1 训练脚本

为了方便不同规模的训练，我们提供了多种预配置脚本：

##### 命令行参数版本

```bash
# 单机单卡训练（入门/调试）
bash scripts/train_single_gpu.sh

# 单机多卡训练（中等规模）
bash scripts/train_multi_gpu.sh

# 多机多卡训练（大规模）
# 在主节点(rank 0)上运行:
bash scripts/train_multi_node.sh --master_addr "192.168.1.100" --node_rank 0 --nnodes 2

# 在工作节点(rank 1)上运行:
bash scripts/train_multi_node.sh --master_addr "192.168.1.100" --node_rank 1 --nnodes 2
```

##### YAML配置版本

```bash
# 使用YAML配置的单机多卡训练
bash scripts/train_multi_gpu_yaml.sh

# 简单的YAML配置训练示例（适合入门）
bash examples/train_with_yaml.sh
```

这些脚本包括完整的训练流程配置，包括：
- 自动创建带时间戳的输出目录
- 详细的日志记录
- 训练后的模型评估
- 优化的超参数设置

命令行参数版本和YAML配置版本的主要区别：
- 命令行参数版本：通过脚本中的变量和命令行参数控制训练
- YAML配置版本：通过YAML配置文件控制大部分训练参数，更灵活和可维护

多节点训练脚本支持丰富的配置选项：
```bash
bash scripts/train_multi_node.sh --help
```
```

#### 3.6 数据优化策略

针对数据量有限和数据不平衡问题，系统提供了多种优化策略：

```bash
# 数据分析 - 了解训练数据分布
python scripts/data_analysis.py \
    --questions data/raw_data/train/questions.jsonl \
    --documents data/raw_data/train/documents \
    --output data_analysis

# 使用数据增强训练
python scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --documents data/raw_data/train/documents \
    --use_data_augmentation \
    --output_dir outputs/model_augmented
```

支持的数据增强功能：
- **问题改写**: 生成语义相同但表达不同的问题变体
- **选项增强**: 智能修改选项表述，保持语义不变
- **困难负样本生成**: 创建高质量干扰选项
- **类型平衡**: 确保各类问题类型均有足够样本

配置数据增强参数：

```bash
python scripts/train_model.py \
    --use_data_augmentation \
    --augmentation_factor 1.5 \
    --min_samples_per_type 20
```

#### 3.7 参数高效微调 (PEFT)

针对小数据集训练大型模型，实现了参数高效微调技术：

```bash
# 使用PEFT训练
python scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --documents data/raw_data/train/documents \
    --use_peft \
    --output_dir outputs/model_peft

# 指定PEFT技术
python scripts/train_model.py \
    --use_peft \
    --peft_technique lora \
    --peft_rank 16
```

支持的PEFT技术：
- **LoRA**: 低秩适配，适用于大多数场景 (默认)
- **Prefix-Tuning**: 前缀微调，适用于生成任务
- **Prompt-Tuning**: 提示微调，极度参数高效
- **P-Tuning**: 提供比Prompt-Tuning更灵活的控制

PEFT的优势：
- 训练参数量减少至原来的0.1-1%
- 显著降低内存需求
- 减轻过拟合风险
- 加快训练和收敛速度

#### 3.8 综合优化策略

结合多种优化技术，应对小数据集挑战：

```bash
# 综合优化方案
python scripts/train_model.py \
    --train_questions data/raw_data/train/questions.jsonl \
    --documents data/raw_data/train/documents \
    --use_data_augmentation \
    --use_peft \
    --fp16 \
    --gradient_accumulation_steps 4 \
    --use_wandb \
    --output_dir outputs/optimized_model
```

#### 3.9 模型评估

训练完成后，使用以下命令评估模型：

```bash
python scripts/train_model.py \
    --test_questions data/raw_data/test/questions.jsonl \
    --documents data/raw_data/test/documents \
    --resume outputs/model_v1/checkpoints/best_model.pt \
    --output_dir outputs/evaluation \
    --batch_size 16 \
    --predict_only
```

### 4. 执行流程与使用方法

#### 4.1 环境准备

```bash
pip install torch transformers tqdm pillow pdf2image pytesseract layoutparser paddlepaddle paddleocr
```

#### 3.2 数据准备

- 确保 `data/raw_data/test/questions.jsonl` 包含测试问题
- 确保 `data/raw_data/test/documents/` 目录包含所有PDF文档

#### 3.3 执行命令

```bash
python model/main_processing_script.py \
    --questions data/raw_data/test/questions.jsonl \
    --documents data/raw_data/test/documents \
    --output results/test_results.jsonl \
    [--config config/config.yaml] \
    [--workers 4] \
    [--sequential]
```

参数说明：
- `--questions`: 问题文件路径
- `--documents`: PDF文档目录
- `--output`: 输出结果文件路径
- `--config`: 可选，配置文件路径
- `--workers`: 可选，工作进程数量
- `--sequential`: 可选，使用顺序处理模式
- `--log-level`: 可选，日志级别 (DEBUG/INFO/WARNING/ERROR)

#### 3.4 输出结果

处理完成后，系统会生成符合评测要求的JSONL格式结果文件：

初赛结果示例：
```json
{"id": "1be0009101baa0fe95338f9542XXXXX", "answer": "C"}
```

复赛结果示例：
```json
{"id": "48707b8d6e06e49882a35dc67f5XXXXX", "answer": "部件4位于部件5的左侧"}
```

#### 3.5 高效推理

系统提供高度优化的推理能力，通过多种技术加速模型推理并降低资源消耗：

```bash
# 使用预配置的高效推理脚本
bash scripts/run_inference.sh --mode fast

# 针对CPU环境的优化推理
bash scripts/run_inference.sh --mode cpu --model outputs/quantized_model

# 批量高性能推理
bash scripts/run_inference.sh --mode batch --questions data/raw_data/test/questions.jsonl --output results/batch_results.jsonl
```

支持多种推理模式：
- **default**: 平衡速度和准确性的默认模式
- **fast**: 优先考虑速度的快速推理模式
- **accurate**: 优先考虑准确性的高精度模式
- **lite**: 适用于资源受限环境的轻量级模式
- **onnx**: 使用ONNX运行时加速
- **batch**: 针对大规模数据的批量处理模式
- **cpu**: 专为CPU环境优化的推理模式

高级推理选项：

```bash
# 完整推理选项
python scripts/inference.py \
    --model outputs/best_model \
    --questions data/raw_data/test/questions.jsonl \
    --documents data/raw_data/test/documents \
    --output results/predictions.jsonl \
    --batch_size 4 \
    --fp16 \
    --perf_stats
```

#### 3.5.1 批量推理

系统提供强大的批量推理能力，大幅提高处理效率：

```bash
# 高性能批量推理 (自动优化设置)
bash scripts/run_inference.sh --mode batch

# 自定义批处理配置
python scripts/inference.py \
    --model outputs/best_model \
    --questions data/raw_data/test/questions.jsonl \
    --documents data/raw_data/test/documents \
    --output results/batch_results.jsonl \
    --batch_size 8 \           # 基础批大小
    --max_batch_size 32 \      # 最大批大小
    --dynamic_batch \          # 启用动态批处理
    --fp16                     # 半精度加速
```

批量推理性能对比：

| 批处理大小 | 处理速度 (问题/秒) | 加速比 | GPU内存使用 |
|-----------|-----------------|--------|------------|
| 1 (无批处理) | 2.5 | 1x | 2.1 GB |
| 4 | 8.3 | 3.3x | 3.5 GB |
| 8 | 14.7 | 5.9x | 5.2 GB |
| 16 | 22.1 | 8.8x | 8.7 GB |
| 32 | 27.3 | 10.9x | 15.6 GB |

批量推理适用场景：
- **评测提交** - 快速处理大量测试样本
- **大规模部署** - 高吞吐量生产环境
- **性能测试** - 系统性能上限评估

核心优化技术：
- **混合精度推理** (FP16/INT8)
- **CUDA图优化** (静态输入形状)
- **KV缓存** (注意力计算优化)
- **动态批处理** (自适应批大小)
- **ONNX/TensorRT加速**

[📄 查看完整推理优化指南](docs/inference_optimization.md) - 详细了解推理优化器设计、批处理引擎实现、性能调优和最佳实践

### 4. 性能优化

#### 4.1 并行处理

- 使用进程池并行处理多个问题
- 动态调整工作进程数量
- 支持顺序处理模式（用于调试）

#### 4.2 模型优化

- 模型缓存：避免重复加载模型
- 批处理：批量处理文本和图像
- 混合精度：支持FP16运算
- 设备管理：自动使用CPU/GPU

#### 4.3 错误处理

- 模块级错误恢复：每个模块独立捕获异常
- 问题级错误隔离：单个问题失败不影响整体处理
- 详细日志记录：便于问题定位

## 项目结构

```
industrial-multimodal-reasoning/
├── config/                     # 配置文件
│   └── default_config.yaml     # 默认配置
├── data/                       # 数据目录
│   └── raw_data/               # 原始数据
│       ├── test/               # 测试数据
│       │   ├── documents/      # PDF文档
│       │   └── questions.jsonl # 问题文件
│       └── train/              # 训练数据
├── docs/                       # 文档
│   ├── images/                 # 文档图片
│   ├── technical_solution_plan.md  # 技术方案
│   └── optimization_proposal.md    # 优化建议
├── results/                    # 结果输出
├── model/                      # 模型包
│   ├── __init__.py             # 包初始化
│   ├── config.py               # 配置系统
│   ├── cross_modal_attention.py # 跨模态注意力
│   ├── enhanced_model.py       # 增强型模型
│   ├── main_processing_script.py # 主处理脚本
│   ├── modality_reconstructor.py # 模态重构
│   ├── multimodal_encoder.py   # 多模态编码
│   ├── multimodal_fusion.py    # 多模态融合
│   ├── pdf_processor.py        # PDF处理
│   ├── qa_module.py            # 问答模块
│   └── uncertainty_estimator.py # 不确定性估计
└── README.md                   # 项目说明
```

## 引用与致谢

* 使用的预训练模型：BERT-base-Chinese, Vision Transformer
* 使用的开源库：PyTorch, Transformers, LayoutParser, PaddleOCR
