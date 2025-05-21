# 多模态推理系统优化建议

基于对 "可再生能源设施多模态模型" 的设计理念分析，以下是针对我们工业技术文档多模态推理问答系统的优化建议。

## 1. 架构优化

### 1.1 多层次融合架构
当前系统采用单一的融合机制，可以借鉴层次化融合架构：

```python
class HierarchicalMultimodalFusion(nn.Module):
    """
    层次化多模态融合模块，实现低级、中级和高级特征的逐步融合
    """
    def __init__(self, embedding_dim, hidden_dims=None):
        super().__init__()
        # 设置合理的融合层次结构
        self.hidden_dims = hidden_dims or [embedding_dim, embedding_dim//2, embedding_dim//4]
        
        # 低级特征融合层（基本对齐）
        self.low_level_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, self.hidden_dims[0]),
            nn.LayerNorm(self.hidden_dims[0]),
            nn.ReLU()
        )
        
        # 中级特征融合层（语义和空间关系）
        self.mid_level_fusion = nn.Sequential(
            nn.Linear(self.hidden_dims[0] * 2, self.hidden_dims[1]),
            nn.LayerNorm(self.hidden_dims[1]),
            nn.ReLU()
        )
        
        # 高级特征融合层（统一表示）
        self.high_level_fusion = nn.Sequential(
            nn.Linear(self.hidden_dims[1] * 2, self.hidden_dims[2]),
            nn.LayerNorm(self.hidden_dims[2]),
            nn.ReLU()
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(self.hidden_dims[2], embedding_dim)
        
    def forward(self, text_embeddings, visual_embeddings, layout_embeddings=None):
        # 低级融合：文本和视觉
        if isinstance(text_embeddings, list) and len(text_embeddings) > 0:
            text_emb = torch.cat(text_embeddings, dim=0).mean(dim=0, keepdim=True)
        else:
            text_emb = text_embeddings
            
        low_fusion = self.low_level_fusion(torch.cat([text_emb, visual_embeddings], dim=1))
        
        # 中级融合：基础融合结果和布局信息
        if layout_embeddings is not None:
            mid_fusion = self.mid_level_fusion(torch.cat([low_fusion, layout_embeddings], dim=1))
        else:
            mid_fusion = low_fusion
            
        # 高级融合：处理所有可用信息
        high_fusion = self.high_level_fusion(torch.cat([mid_fusion, low_fusion], dim=1))
        
        # 输出
        output = self.output_projection(high_fusion)
        
        return output
```

### 1.2 跨模态注意力机制
添加增强型跨模态注意力模块：

```python
class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块，用于增强模态间的信息交流
    """
    def __init__(self, embedding_dim, num_heads=8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            dropout=0.1
        )
        
        # 自适应门控机制
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        
        # 输出层标准化
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, query_modal, key_modal, value_modal=None):
        """
        query_modal: 查询模态的嵌入 [batch_size, seq_len_q, embedding_dim]
        key_modal: 键模态的嵌入 [batch_size, seq_len_k, embedding_dim]
        value_modal: 值模态的嵌入，默认等于key_modal
        """
        if value_modal is None:
            value_modal = key_modal
            
        # 调整维度顺序以适应多头注意力机制
        query_modal = query_modal.transpose(0, 1)  # [seq_len_q, batch_size, embedding_dim]
        key_modal = key_modal.transpose(0, 1)      # [seq_len_k, batch_size, embedding_dim]
        value_modal = value_modal.transpose(0, 1)  # [seq_len_k, batch_size, embedding_dim]
        
        # 计算跨模态注意力
        attn_output, attn_weights = self.attention(
            query=query_modal,
            key=key_modal,
            value=value_modal
        )
        
        # 转回原始维度顺序
        attn_output = attn_output.transpose(0, 1)  # [batch_size, seq_len_q, embedding_dim]
        
        # 计算自适应门控
        gate_value = self.gate(torch.cat([query_modal.transpose(0, 1), attn_output], dim=-1))
        
        # 应用门控机制
        output = gate_value * attn_output + (1 - gate_value) * query_modal.transpose(0, 1)
        
        # 应用层标准化
        output = self.layer_norm(output)
        
        return output, attn_weights
```

## 2. 模态缺失与重构

添加模态重构功能，提高系统鲁棒性：

```python
class ModalityReconstructor(nn.Module):
    """模态重构模块，用于从可用模态重建缺失模态"""
    
    def __init__(self, embedding_dim, hidden_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 视觉模态重构网络
        self.visual_reconstructor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 文本模态重构网络
        self.text_reconstructor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 布局模态重构网络
        self.layout_reconstructor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 重构质量评估
        self.quality_estimator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
    def reconstruct_visual(self, text_embedding, layout_embedding):
        """从文本和布局重构视觉特征"""
        combined = torch.cat([text_embedding, layout_embedding], dim=1)
        reconstructed = self.visual_reconstructor(combined)
        quality = self.quality_estimator(reconstructed)
        return reconstructed, quality
        
    def reconstruct_text(self, visual_embedding, layout_embedding):
        """从视觉和布局重构文本特征"""
        combined = torch.cat([visual_embedding, layout_embedding], dim=1)
        reconstructed = self.text_reconstructor(combined)
        quality = self.quality_estimator(reconstructed)
        return reconstructed, quality
        
    def reconstruct_layout(self, visual_embedding, text_embedding):
        """从视觉和文本重构布局特征"""
        combined = torch.cat([visual_embedding, text_embedding], dim=1)
        reconstructed = self.layout_reconstructor(combined)
        quality = self.quality_estimator(reconstructed)
        return reconstructed, quality
```

## 3. 置信度和不确定性估计

添加不确定性估计功能，提高系统可靠性：

```python
class UncertaintyEstimator(nn.Module):
    """不确定性估计模块"""
    
    def __init__(self, embedding_dim, num_classes=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # 多任务不确定性
        self.task_uncertainty = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim//2),
            nn.ReLU(),
            nn.Linear(embedding_dim//2, 1),
            nn.Softplus()  # 确保不确定性为正值
        )
        
        # 对于分类任务的不确定性（适用于初赛-单选题）
        if num_classes:
            self.classification_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim//2),
                nn.ReLU(),
                nn.Linear(embedding_dim//2, num_classes),
            )
            
            self.classification_uncertainty = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim//2),
                nn.ReLU(),
                nn.Linear(embedding_dim//2, num_classes),
                nn.Softplus()  # 确保不确定性为正值
            )
        
    def forward(self, embedding):
        # 任务级不确定性
        task_uncertainty = self.task_uncertainty(embedding)
        
        result = {
            'task_uncertainty': task_uncertainty
        }
        
        # 分类任务不确定性（如果适用）
        if self.num_classes:
            logits = self.classification_head(embedding)
            uncertainties = self.classification_uncertainty(embedding)
            
            # 温度缩放校准
            temperature = 1.5  # 可调参数
            calibrated_probs = F.softmax(logits / temperature, dim=1)
            
            result.update({
                'classification_logits': logits,
                'classification_probs': calibrated_probs,
                'classification_uncertainties': uncertainties
            })
            
        return result
```

## 4. 模块化与配置驱动

### 4.1 配置驱动系统

添加灵活的配置系统：

```python
# config.py
import os
import yaml
import json
from typing import Dict, Any, Optional

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
    _, ext = os.path.splitext(config_path)
    
    if ext.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    elif ext.lower() == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {ext}")
        
    return config

def validate_config(config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> bool:
    """验证配置有效性"""
    # 基本配置验证
    required_fields = [
        'model', 'encoders', 'fusion', 'qa_module'
    ]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"配置缺少必要字段: {field}")
    
    # 如果提供了schema，进行详细验证
    if schema:
        # 实现详细的schema验证
        pass
        
    return True

def save_config(config: Dict[str, Any], output_path: str) -> None:
    """保存配置到文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    _, ext = os.path.splitext(output_path)
    
    if ext.lower() in ['.yaml', '.yml']:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    elif ext.lower() == '.json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"不支持的配置文件格式: {ext}")
```

### 4.2 默认配置文件

创建默认配置文件 `config/default_config.yaml`：

```yaml
# 模型总体配置
model:
  name: "IndustrialDocQA"
  version: "1.0.0"
  embedding_dim: 768
  fusion_dim: 512
  use_cache: true
  device: "auto"  # "auto", "cpu", "cuda", "cuda:0"

# 编码器配置
encoders:
  text:
    model_name: "bert-base-chinese"
    pooling_strategy: "cls"
    max_length: 512
    batch_size: 16
  
  vision:
    model_name: "google/vit-base-patch16-224"
    image_size: 224
    batch_size: 8
    multiscale_fusion: true
  
  layout:
    feature_dim: 5
    use_positional_encoding: true

# 融合模块配置
fusion:
  strategy: "hierarchical"  # "simple", "attention", "hierarchical"
  num_attention_heads: 8
  dropout: 0.1
  use_layer_norm: true

# 问答模块配置
qa_module:
  model_name: "placeholder/lmm-model"
  temperature: 1.0
  max_answer_length: 100
  confidence_threshold: 0.5
  top_k: 1

# 系统配置
system:
  temp_dir: "temp_processed_data"
  max_workers: 4
  batch_size: 8
  cache_dir: ".cache"
  log_level: "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
```

## 5. 多设备支持与批处理优化

### 5.1 设备管理

增强设备管理功能：

```python
def get_optimal_device(device_preference="auto"):
    """获取最优的计算设备"""
    if device_preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"自动选择GPU设备: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("自动选择CPU设备")
    else:
        device = torch.device(device_preference)
        logger.info(f"使用指定设备: {device}")
        
    return device

def move_tensors_to_device(data, device):
    """将张量数据移动到指定设备"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_tensors_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_tensors_to_device(x, device) for x in data]
    elif isinstance(data, tuple):
        return tuple(move_tensors_to_device(x, device) for x in data)
    else:
        return data
```

### 5.2 批处理优化

添加动态批处理支持：

```python
def dynamic_batch_processing(items, process_fn, batch_size, device=None, desc=None):
    """
    动态批处理处理函数
    
    Args:
        items: 要处理的项目列表
        process_fn: 处理函数，接受一个批次并返回处理结果
        batch_size: 批次大小
        device: 计算设备
        desc: 进度条描述
    
    Returns:
        处理结果列表
    """
    results = []
    
    # 创建进度条
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    with tqdm(total=total_batches, desc=desc or "批处理") as pbar:
        for i in range(0, len(items), batch_size):
            # 获取当前批次
            batch = items[i:i+batch_size]
            
            # 移动到设备（如果需要）
            if device is not None:
                batch = move_tensors_to_device(batch, device)
            
            # 处理批次
            try:
                batch_results = process_fn(batch)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"批处理错误 (项目 {i} 到 {i+len(batch)-1}): {e}")
                # 回退到单项处理以隔离错误
                for j, item in enumerate(batch):
                    try:
                        single_result = process_fn([item])
                        results.extend(single_result)
                    except Exception as e_inner:
                        logger.error(f"单项处理错误 (项目 {i+j}): {e_inner}")
                        # 添加占位符结果
                        results.append(None)
            
            # 更新进度条
            pbar.update(1)
    
    return results
```

## 6. 保存与加载机制

增强模型保存和加载功能：

```python
def save_model(model, optimizer, epoch, config, metrics, path):
    """保存模型状态"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 保存模型状态
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'config': config,
        'metrics': metrics,
        'version': "1.0.0"  # 版本号
    }
    
    torch.save(state_dict, path)
    logger.info(f"模型已保存到: {path}")
    
def load_model(path, model_class, device=None):
    """加载模型状态"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")
    
    # 自动选择设备
    if device is None:
        device = get_optimal_device("auto")
    
    # 加载状态
    state_dict = torch.load(path, map_location=device)
    
    # 版本兼容性检查
    version = state_dict.get('version', "0.0.0")
    logger.info(f"加载模型版本: {version}")
    
    # 创建模型实例
    config = state_dict['config']
    model = model_class(**config['model'])
    
    # 加载模型权重
    model.load_state_dict(state_dict['model'])
    model = model.to(device)
    
    # 返回模型和附加信息
    return {
        'model': model,
        'config': config,
        'epoch': state_dict.get('epoch', 0),
        'metrics': state_dict.get('metrics', {}),
        'optimizer_state': state_dict.get('optimizer', None)
    }
```

## 7. 错误处理与日志

增强错误处理和日志系统：

```python
import logging
import traceback
from functools import wraps

# 创建日志器
logger = logging.getLogger("industrial_doc_qa")

def setup_logging(level=logging.INFO, log_file=None):
    """设置日志系统"""
    # 设置格式
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果提供）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 设置日志级别
    logger.setLevel(level)
    
    return logger

def error_handler(func):
    """错误处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"{func.__name__} 函数发生错误: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            
            # 由调用者决定如何处理错误
            raise
    
    return wrapper
```

## 8. 高性能推理服务

创建推理服务模块：

```python
# inference_server.py
import os
import json
import torch
import base64
from PIL import Image
from io import BytesIO
from datetime import datetime
from flask import Flask, request, jsonify

from config import load_config
from main_processing_script import process_single_question

app = Flask(__name__)

# 全局模型和配置
global_model = None
global_config = None

def init_model(model_path, device=None):
    """初始化模型"""
    global global_model, global_config
    
    # 加载模型
    model_data = load_model(model_path, EnhancedMultiModalModel, device)
    global_model = model_data['model']
    global_config = model_data['config']
    
    return global_model, global_config

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': global_model is not None,
        'config_loaded': global_config is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """单个预测端点"""
    try:
        # 获取请求数据
        data = request.json
        
        # 验证输入
        if not data:
            return jsonify({'error': '无效的请求数据'}), 400
        
        # 处理输入
        question_data = {
            'id': data.get('id', f"api_{int(datetime.now().timestamp())}"),
            'question': data.get('question', ''),
            'document': data.get('document', ''),
            'options': data.get('options', [])
        }
        
        # 处理图像（如果是Base64）
        if 'image_base64' in data:
            try:
                # 解码Base64图像
                image_data = base64.b64decode(data['image_base64'])
                img = Image.open(BytesIO(image_data))
                
                # 保存到临时目录
                temp_dir = os.path.join('temp_processed_data', question_data['id'])
                os.makedirs(temp_dir, exist_ok=True)
                img_path = os.path.join(temp_dir, 'document.png')
                img.save(img_path)
                
                # 更新文档路径
                question_data['document'] = img_path
            except Exception as e:
                return jsonify({'error': f'图像处理错误: {str(e)}'}), 400
        
        # 处理问题
        result = process_single_question(
            question_data,
            pdf_documents_dir='.',  # 使用相对路径
            modules={
                'encoder': global_model.encoder,
                'fusion': global_model.fusion,
                'qa': global_model.qa_module
            }
        )
        
        # 返回结果
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测端点"""
    try:
        # 获取请求数据
        data = request.json
        
        # 验证输入
        if not data or not isinstance(data.get('questions', []), list):
            return jsonify({'error': '无效的请求数据'}), 400
        
        questions = data['questions']
        batch_size = data.get('batch_size', 4)
        
        # 批量处理
        results = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            batch_results = []
            
            for q in batch:
                # 处理单个问题
                result = process_single_question(
                    q,
                    pdf_documents_dir='.',  # 使用相对路径
                    modules={
                        'encoder': global_model.encoder,
                        'fusion': global_model.fusion,
                        'qa': global_model.qa_module
                    }
                )
                batch_results.append(result)
            
            results.extend(batch_results)
        
        # 返回结果
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_server(model_path, host='0.0.0.0', port=8000, device=None):
    """运行推理服务器"""
    # 初始化模型
    init_model(model_path, device)
    
    # 启动服务器
    app.run(host=host, port=port)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='推理服务器')
    parser.add_argument('--model_path', required=True, help='模型路径')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机')
    parser.add_argument('--port', type=int, default=8000, help='服务器端口')
    parser.add_argument('--device', default=None, help='计算设备')
    
    args = parser.parse_args()
    
    run_server(args.model_path, args.host, args.port, args.device)
```

## 9. 可视化工具

添加可视化工具：

```python
# visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

class VisualizationManager:
    """可视化管理器"""
    
    def __init__(self, output_dir='visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_attention_weights(self, attention_weights, modality_names=None, save_path=None):
        """
        可视化注意力权重
        
        Args:
            attention_weights: 注意力权重矩阵
            modality_names: 模态名称列表
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # 如果未提供模态名称，使用默认名称
        if modality_names is None:
            modality_names = [f'Modality {i+1}' for i in range(attention_weights.shape[0])]
        
        # 绘制热力图
        sns.heatmap(
            attention_weights, 
            annot=True, 
            fmt='.2f', 
            cmap='viridis',
            xticklabels=modality_names,
            yticklabels=modality_names
        )
        
        plt.title('Cross-Modal Attention Weights')
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"注意力图已保存到: {full_path}")
        
        plt.close()
    
    def plot_embeddings(self, embeddings, labels=None, method='tsne', save_path=None):
        """
        可视化嵌入空间
        
        Args:
            embeddings: 嵌入向量
            labels: 标签
            method: 降维方法 ('tsne', 'pca')
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 10))
        
        # 降维
        if method.lower() == 'tsne':
            tsne = TSNE(n_components=2, random_state=42)
            reduced_data = tsne.fit_transform(embeddings)
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(embeddings)
        
        # 绘制散点图
        if labels is not None:
            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(
                    reduced_data[mask, 0],
                    reduced_data[mask, 1],
                    label=f'Class {label}',
                    alpha=0.7
                )
            plt.legend()
        else:
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7)
        
        plt.title(f'Embedding Space Visualization ({method.upper()})')
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"嵌入空间图已保存到: {full_path}")
        
        plt.close()
    
    def plot_uncertainty(self, predictions, uncertainties, save_path=None):
        """
        可视化不确定性
        
        Args:
            predictions: 预测值
            uncertainties: 不确定性估计
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        
        # 绘制预测与不确定性关系
        plt.errorbar(
            range(len(predictions)),
            predictions,
            yerr=uncertainties,
            fmt='o',
            ecolor='red',
            capsize=5
        )
        
        plt.title('Predictions with Uncertainty')
        plt.xlabel('Sample Index')
        plt.ylabel('Predicted Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"不确定性图已保存到: {full_path}")
        
        plt.close()
```

## 10. 实现细节优化

### 10.1 类增强型多模态模型

创建 `enhanced_model.py` 文件实现增强型多模态模型：

```python
# enhanced_model.py
import os
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Any, Optional, Union

from .config import load_config, validate_config
from .multimodal_encoder import MultimodalEncoder
from .multimodal_fusion import MultimodalFusion, HierarchicalMultimodalFusion
from .qa_module import ReasoningQAModule

logger = logging.getLogger(__name__)

class EnhancedMultiModalModel(nn.Module):
    """
    增强型多模态模型，集成了多模态编码、融合和问答能力
    """
    
    def __init__(self, config_path=None, **kwargs):
        super().__init__()
        
        # 加载配置
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = kwargs
        
        # 验证配置
        validate_config(self.config)
        
        # 设置设备
        self.device = self._get_device(self.config.get('device', 'auto'))
        
        # 初始化编码器
        self.encoder = self._init_encoder()
        
        # 初始化融合模块
        self.fusion = self._init_fusion()
        
        # 初始化问答模块
        self.qa_module = self._init_qa_module()
        
        # 初始化可选组件：模态重构器
        self.reconstructor = self._init_reconstructor()
        
        # 初始化可选组件：不确定性估计器
        self.uncertainty_estimator = self._init_uncertainty_estimator()
        
        # 移动到指定设备
        self.to(self.device)
        
        logger.info(f"增强型多模态模型初始化完成，使用设备: {self.device}")
    
    def _get_device(self, device_preference):
        """获取计算设备"""
        if device_preference == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"自动选择GPU设备: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                logger.info("自动选择CPU设备")
        else:
            device = torch.device(device_preference)
            logger.info(f"使用指定设备: {device}")
            
        return device
    
    def _init_encoder(self):
        """初始化编码器"""
        encoder_config = self.config.get('encoders', {})
        
        text_model = encoder_config.get('text', {}).get('model_name', 'bert-base-chinese')
        vision_model = encoder_config.get('vision', {}).get('model_name', 'google/vit-base-patch16-224')
        
        # 其他编码器参数
        use_cache = self.config.get('model', {}).get('use_cache', True)
        batch_size = encoder_config.get('text', {}).get('batch_size', 16)
        
        encoder = MultimodalEncoder(
            text_model_name=text_model,
            vision_model_name=vision_model,
            use_cache=use_cache,
            device=self.device,
            batch_size=batch_size
        )
        
        return encoder
    
    def _init_fusion(self):
        """初始化融合模块"""
        fusion_config = self.config.get('fusion', {})
        model_config = self.config.get('model', {})
        
        embedding_dim = model_config.get('embedding_dim', 768)
        fusion_dim = model_config.get('fusion_dim', 512)
        strategy = fusion_config.get('strategy', 'hierarchical')
        
        if strategy == 'hierarchical':
            fusion = HierarchicalMultimodalFusion(
                embedding_dim=embedding_dim,
                hidden_dims=[embedding_dim, embedding_dim//2, fusion_dim]
            )
        else:
            # 默认使用简单融合
            fusion = MultimodalFusion(
                embedding_dim=embedding_dim,
                fused_embedding_dim=fusion_dim
            )
            
        return fusion
    
    def _init_qa_module(self):
        """初始化问答模块"""
        qa_config = self.config.get('qa_module', {})
        model_config = self.config.get('model', {})
        
        lmm_model_name = qa_config.get('model_name', 'placeholder/lmm-model')
        fusion_dim = model_config.get('fusion_dim', 512)
        
        qa_module = ReasoningQAModule(
            lmm_model_name=lmm_model_name,
            embedding_dim=fusion_dim
        )
            
        return qa_module
    
    def _init_reconstructor(self):
        """初始化模态重构器（可选）"""
        if not self.config.get('model', {}).get('use_reconstructor', False):
            return None
            
        model_config = self.config.get('model', {})
        embedding_dim = model_config.get('embedding_dim', 768)
        
        reconstructor = ModalityReconstructor(
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim
        )
            
        return reconstructor
    
    def _init_uncertainty_estimator(self):
        """初始化不确定性估计器（可选）"""
        if not self.config.get('model', {}).get('use_uncertainty', False):
            return None
            
        model_config = self.config.get('model', {})
        fusion_dim = model_config.get('fusion_dim', 512)
        num_classes = 4  # 对于初赛的 A/B/C/D 选择题
        
        estimator = UncertaintyEstimator(
            embedding_dim=fusion_dim,
            num_classes=num_classes
        )
            
        return estimator
    
    def forward(self, imagery_features=None, text_features=None, coord_features=None):
        """
        前向传播
        
        Args:
            imagery_features: 图像特征
            text_features: 文本特征
            coord_features: 坐标特征（对应版面信息）
            
        Returns:
            包含预测结果的字典
        """
        results = {}
        
        # 记录模态存在情况
        modality_presence = {
            'imagery': imagery_features is not None,
            'text': text_features is not None,
            'coord': coord_features is not None
        }
        results['modality_presence'] = modality_presence
        
        # 检查是否所有模态都缺失
        if not any(modality_presence.values()):
            logger.warning("所有模态都缺失，无法生成预测")
            return {
                'modality_presence': modality_presence,
                'status': 'error',
                'error': '所有模态都缺失'
            }
        
        # 模态重构（如果启用）
        if self.reconstructor is not None:
            # 根据可用模态重构缺失模态
            reconstructed = {}
            
            # 重构图像特征
            if not modality_presence['imagery'] and modality_presence['text'] and modality_presence['coord']:
                reconstructed['imagery'], reconstructed['imagery_quality'] = self.reconstructor.reconstruct_visual(
                    text_features, coord_features
                )
            
            # 重构文本特征
            if not modality_presence['text'] and modality_presence['imagery']:
                reconstructed['text'], reconstructed['text_quality'] = self.reconstructor.reconstruct_text(
                    imagery_features, coord_features if modality_presence['coord'] else None
                )
            
            # 重构坐标特征
            if not modality_presence['coord'] and modality_presence['imagery'] and modality_presence['text']:
                reconstructed['coord'], reconstructed['coord_quality'] = self.reconstructor.reconstruct_layout(
                    imagery_features, text_features
                )
                
            # 使用重构特征
            if 'imagery' in reconstructed:
                imagery_features = reconstructed['imagery']
                modality_presence['imagery'] = True
                
            if 'text' in reconstructed:
                text_features = reconstructed['text']
                modality_presence['text'] = True
                
            if 'coord' in reconstructed:
                coord_features = reconstructed['coord']
                modality_presence['coord'] = True
                
            results['reconstructed'] = reconstructed
        
        # 进行多模态融合
        fused_embedding = self.fusion(
            text_embeddings=text_features,
            visual_embeddings=imagery_features,
            layout_embeddings=coord_features
        )
        
        results['fused_embedding'] = fused_embedding
        
        # 不确定性估计（如果启用）
        if self.uncertainty_estimator is not None:
            uncertainty = self.uncertainty_estimator(fused_embedding)
            results['uncertainty'] = uncertainty
        
        # 生成问答结果
        if hasattr(self.qa_module, 'answer_question'):
            answer = self.qa_module.answer_question(fused_embedding, "")  # 问题文本在处理阶段已经编码
            results['answer'] = answer
        
        return results
    
    def get_embeddings(self, imagery_features=None, text_features=None, coord_features=None):
        """
        获取统一嵌入向量
        
        Args:
            imagery_features: 图像特征
            text_features: 文本特征
            coord_features: 坐标特征
            
        Returns:
            统一的嵌入向量
        """
        # 前向传播，但只返回融合嵌入
        results = self.forward(imagery_features, text_features, coord_features)
        return results.get('fused_embedding')
    
    def reconstruct_modality(self, target_modality, **kwargs):
        """
        重构缺失模态
        
        Args:
            target_modality: 目标模态索引 (0=视觉, 1=文本, 2=坐标)
            **kwargs: 可用模态
            
        Returns:
            重构的特征和置信度
        """
        if self.reconstructor is None:
            raise ValueError("模态重构器未启用")
        
        imagery_features = kwargs.get('imagery_features')
        text_features = kwargs.get('text_features')
        coord_features = kwargs.get('coord_features')
        
        if target_modality == 0:  # 重构视觉
            if text_features is None or coord_features is None:
                raise ValueError("重构视觉特征需要文本和坐标特征")
            return self.reconstructor.reconstruct_visual(text_features, coord_features)
            
        elif target_modality == 1:  # 重构文本
            if imagery_features is None:
                raise ValueError("重构文本特征需要视觉特征")
            return self.reconstructor.reconstruct_text(
                imagery_features, 
                coord_features if coord_features is not None else None
            )
            
        elif target_modality == 2:  # 重构坐标
            if imagery_features is None or text_features is None:
                raise ValueError("重构坐标特征需要视觉和文本特征")
            return self.reconstructor.reconstruct_layout(imagery_features, text_features)
            
        else:
            raise ValueError(f"无效的目标模态索引: {target_modality}")
    
    @classmethod
    def load(cls, path, device=None):
        """
        加载模型
        
        Args:
            path: 模型路径
            device: 计算设备
            
        Returns:
            加载的模型
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        # 自动选择设备
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载状态
        state_dict = torch.load(path, map_location=device)
        
        # 版本兼容性检查
        version = state_dict.get('version', "0.0.0")
        logger.info(f"加载模型版本: {version}")
        
        # 创建模型实例
        config = state_dict['config']
        model = cls(**config)
        
        # 加载模型权重
        model.load_state_dict(state_dict['model'])
        model = model.to(device)
        
        return model
    
    def save(self, path, optimizer=None, epoch=0, metrics=None):
        """
        保存模型
        
        Args:
            path: 保存路径
            optimizer: 优化器（可选）
            epoch: 当前轮次
            metrics: 评估指标
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型状态
        state_dict = {
            'model': self.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'epoch': epoch,
            'config': self.config,
            'metrics': metrics or {},
            'version': "1.0.0"  # 版本号
        }
        
        torch.save(state_dict, path)
        logger.info(f"模型已保存到: {path}")
```

## 总结

通过借鉴 "可再生能源设施多模态模型" 的设计理念，我们可以将工业技术文档多模态推理问答系统升级为更先进、更健壮的架构。主要改进包括：

1. **多层次融合架构**：实现低级、中级和高级特征的逐步融合，提高不同模态间的信息整合质量。

2. **跨模态注意力机制**：增强模态间的信息交互，更有效地捕获文本、图像和版面信息之间的关系。

3. **模态缺失与重构**：添加模态重构功能，提高系统对不完整输入的鲁棒性。

4. **不确定性估计**：添加置信度评分机制，提供决策支持和结果可靠性评估。

5. **配置驱动系统**：实现灵活的配置系统，便于调整参数和适应不同环境。

6. **模块化架构**：改进系统的模块化程度，清晰界定各组件的职责和接口。

7. **高效批处理与设备管理**：优化批处理机制和设备利用，提高系统性能。

8. **完善的错误处理与日志**：增强系统的健壮性和可维护性。

9. **高性能推理服务**：添加RESTful API服务，支持模型部署和在线推理。

10. **可视化工具**：提供直观的结果展示和模型行为分析能力。

这些改进将使系统在处理工业技术文档的多模态推理任务时更加高效、可靠和灵活。