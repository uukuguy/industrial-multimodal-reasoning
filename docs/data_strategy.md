# 数据特征分析与训练策略

## 训练数据分析

根据对训练集的分析，我们发现以下关键特征：

### 1. 数据规模
- **总问题数量**：300个
- **唯一文档数**：51个
- **问题/文档比率**：平均每个文档约5.9个问题

### 2. 问题类型分布
- **结构组成类**：109个 (36.3%)
- **位置关系类**：87个 (29.0%)
- **功能描述类**：60个 (20.0%)
- **技术参数类**：5个 (1.7%)
- **操作步骤类**：4个 (1.3%)
- **其他类型**：35个 (11.7%)

### 3. 答案分布
- **B选项**：125个 (41.7%)
- **C选项**：101个 (33.7%)
- **A选项**：43个 (14.3%)
- **D选项**：31个 (10.3%)

## 数据限制

1. **数据量不足**：300个问题对于训练高性能的多模态模型明显不足，特别是对于需要理解复杂图文关系的工业文档理解任务。

2. **类型不平衡**：技术参数和操作步骤类问题严重不足，这将导致模型在这些类型的问题上表现较差。

3. **答案分布偏斜**：B和C选项占比过高，可能导致模型产生答案偏好。

4. **文档多样性不足**：51个文档样本对于理解多样化的工业技术文档不够全面。

## 优化训练策略

基于以上分析，我们提出以下数据和训练策略：

### 1. 数据增强技术

```python
# 在model/dataset.py中实现数据增强
class DataAugmentation:
    @staticmethod
    def question_paraphrase(question, document):
        """问题改写增强"""
        # 实现问题改写逻辑
        return augmented_questions
    
    @staticmethod
    def document_based_generation(document):
        """基于文档生成新问题"""
        # 使用文档内容生成新的问题-答案对
        return new_question_answer_pairs
```

### 2. 预训练模型选择

采用在工业/技术领域有良好表现的多模态预训练模型：

1. **视觉编码器**：
   - **CLIP ViT-L/14**：对视觉内容有强大的理解能力
   - **Donut**：专门针对文档图像设计

2. **文本编码器**：
   - **ChatGLM-6B**：对中文有良好支持
   - **BERTweet**：针对简短技术文本优化

### 3. 低资源微调技术

```python
# 在model/trainer.py中实现
class LowResourceTrainer(EnhancedMultiModalTrainer):
    def setup_peft(self, peft_config=None):
        """设置参数高效微调"""
        from peft import get_peft_model, LoraConfig
        
        if peft_config is None:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.05,
                bias="none"
            )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
```

### 4. 多阶段训练策略

1. **阶段1**：在大规模通用数据集上预训练
   ```bash
   python scripts/train_model.py \
       --train_questions external_data/general_qa.jsonl \
       --documents external_data/documents \
       --output_dir outputs/stage1_pretrain
   ```

2. **阶段2**：在增强后的工业文档数据集上微调
   ```bash
   python scripts/train_model.py \
       --train_questions data/augmented_train.jsonl \
       --documents data/raw_data/train/documents \
       --resume outputs/stage1_pretrain/best_model \
       --output_dir outputs/stage2_finetune
   ```

3. **阶段3**：针对特定问题类型的适应性微调
   ```bash
   # 针对位置关系问题微调
   python scripts/train_model.py \
       --train_questions data/position_questions.jsonl \
       --documents data/raw_data/train/documents \
       --resume outputs/stage2_finetune/best_model \
       --output_dir outputs/stage3_position_tuning
   ```

### 5. 正则化与平衡策略

1. **针对答案偏好的处理**：
   ```python
   # 修改损失函数以处理答案分布不平衡
   def compute_weighted_loss(outputs, labels, class_weights):
       # 计算带权重的损失
       return weighted_loss
   ```

2. **问题类型平衡采样**：
   ```python
   class BalancedSampler:
       """平衡采样器"""
       def __init__(self, dataset, question_types):
           self.dataset = dataset
           self.question_types = question_types
           
       def __iter__(self):
           # 实现平衡采样逻辑
           # 确保每种问题类型都被充分采样
   ```

### 6. 集成与不确定性估计

为了提高模型在不同问题类型上的鲁棒性，实现模型集成：

```python
class ModelEnsemble:
    """模型集成"""
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
        
    def predict(self, inputs):
        # 集成多个模型的预测结果
        # 同时计算预测的不确定性
        return predictions, uncertainty
```

## 结论

当前训练数据存在明显的规模和分布限制，难以直接训练出高性能模型。建议采用数据增强、预训练模型迁移、参数高效微调和多阶段训练策略来克服这些限制。特别是要重点关注技术参数和操作步骤类问题的性能提升，以及解决答案分布不平衡问题。