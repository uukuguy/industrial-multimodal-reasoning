# YAML配置系统使用指南

本文档介绍如何使用YAML配置文件来管理训练参数，使训练过程更加灵活和可维护。

## 配置文件结构

YAML配置文件包含以下主要部分：

1. **模型配置** (`model`): 定义模型的基本参数
2. **编码器配置** (`encoders`): 配置文本、视觉和布局编码器
3. **融合模块配置** (`fusion`): 设置多模态融合策略
4. **问答模块配置** (`qa_module`): 配置问答生成模块
5. **训练配置** (`training`): 设置训练参数、优化器和学习率调度器
6. **PEFT配置** (`peft`): 参数高效微调设置
7. **数据增强配置** (`data_augmentation`): 数据增强策略
8. **数据集划分配置** (`dataset_split`): 训练集和验证集划分参数
9. **损失函数配置** (`loss`): 自定义损失函数设置
10. **系统配置** (`system`): 系统级参数如日志级别和随机种子
11. **数据路径配置** (`data`): 数据文件和目录路径，包括训练集、验证集、测试集文件路径和文档目录路径

## 示例配置文件

完整的配置文件示例位于 `config/training_config.yaml`，包含所有可配置参数及其默认值。

### 数据路径配置

数据路径配置在 `data` 部分中指定，包括：

```yaml
data:
  train_questions: "data/raw_data/train/questions.jsonl"  # 训练集问题文件路径
  valid_questions: "data/raw_data/valid/questions.jsonl"  # 验证集问题文件路径（可选）
  test_questions: "data/raw_data/test/questions.jsonl"    # 测试集问题文件路径（可选）
  documents_dir: "data/raw_data/train/documents"          # 文档目录路径（必需）
  processed_data_dir: "data/processed"                    # 预处理数据目录（可选）
  output_dir: "outputs"                                   # 输出目录
```

**注意**：当使用YAML配置文件时，必须在 `data` 部分中提供 `documents_dir` 参数，或者在命令行中使用 `--documents` 参数指定文档目录路径。

## 使用配置文件

### 命令行方式

使用 `--config` 参数指定配置文件路径：

```bash
python scripts/train_model.py --config config/training_config.yaml
```

### 使用脚本

可以使用提供的示例脚本：

```bash
bash examples/train_with_yaml.sh
```

或者使用多GPU训练脚本：

```bash
bash scripts/train_multi_gpu_yaml.sh
```

## 配置优先级

参数优先级从高到低为：

1. 命令行参数
2. YAML配置文件
3. 代码中的默认值

这意味着您可以在命令行中覆盖配置文件中的设置，例如：

```bash
python scripts/train_model.py --config config/training_config.yaml --learning_rate 5e-5
```

## 创建自定义配置

您可以基于默认配置创建自定义配置文件：

1. 复制默认配置文件：
   ```bash
   cp config/default_config.yaml config/my_custom_config.yaml
   ```

2. 编辑自定义配置文件，只需包含您想要修改的参数：
   ```yaml
   # 自定义训练配置
   training:
     num_epochs: 20
     batch_size: 16
     optimizer:
       learning_rate: 5e-5
   
   # 使用PEFT
   peft:
     use_peft: true
     peft_technique: lora
   ```

3. 使用自定义配置文件：
   ```bash
   python scripts/train_model.py --config config/my_custom_config.yaml
   ```

## 配置文件验证

系统会自动验证配置文件，确保必要的参数存在，并为缺失的参数提供默认值。如果配置文件中有无效的参数，系统会发出警告。

## 配置文件与命令行参数的结合

您可以结合使用配置文件和命令行参数，例如：

```bash
python scripts/train_model.py \
  --config config/training_config.yaml \
  --output_dir outputs/custom_run \
  --learning_rate 2e-5 \
  --use_peft
```

这将使用配置文件中的大部分设置，但覆盖输出目录和学习率，并启用PEFT。

## 数据集划分

配置文件支持自动划分训练集和验证集，可以通过以下参数控制：

```yaml
dataset_split:
  validation_split_ratio: 0.2  # 使用20%的数据作为验证集
  validation_split_count: 100  # 或者指定样本数量（优先）
  validation_split_seed: 42    # 随机种子，确保可复现性
```

## 注意事项

- 确保YAML文件格式正确，缩进一致
- 布尔值可以使用 `true/false` 或 `yes/no`
- 数值可以使用科学计数法，如 `3e-5`
- 字符串通常不需要引号，除非包含特殊字符