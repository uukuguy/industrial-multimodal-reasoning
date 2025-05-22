import torch
import torch.nn as nn
import numpy as np
from transformers import (
    # 文本模型
    BertModel, BertTokenizer,
    RobertaModel, RobertaTokenizer,
    XLMRobertaModel, XLMRobertaTokenizer,
    MT5Model, MT5Tokenizer,
    MBartModel, MBartTokenizer,
    DebertaModel, DebertaTokenizer,
    AutoModel, AutoTokenizer,
    # 视觉模型
    ViTModel, ViTImageProcessor,
    SwinModel, SwinImageProcessor,
    LayoutLMv2Model, LayoutLMv2Processor,
    CLIPModel, CLIPProcessor,
)
from sentence_transformers import SentenceTransformer

# ============= 基础类 =============

class BaseTextEncoder(nn.Module):
    """基础文本编码器"""
    
    def __init__(self, model_name: str, freeze_layers: list = None):
        super().__init__()
        self.model_name = model_name
        self.freeze_layers = freeze_layers or []
        
    def freeze_specified_layers(self):
        """冻结指定层"""
        for layer_idx in self.freeze_layers:
            for param in self.model.encoder.layer[layer_idx].parameters():
                param.requires_grad = False

class BaseVisualEncoder(nn.Module):
    """基础视觉编码器"""
    
    def __init__(self, model_name: str, image_size: int = 224):
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size

class BaseMultimodalEncoder(nn.Module):
    """基础多模态编码器"""
    
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

# ============= 工厂类 =============

class TextEncoderFactory:
    """文本编码器工厂类，支持多种预训练模型"""
    
    @staticmethod
    def create_encoder(model_type: str, model_name: str, **kwargs):
        """
        创建文本编码器
        
        Args:
            model_type: 模型类型 ('bert', 'roberta', 'xlm-roberta', 'mt5', 'mbart', 
                                'sentence-transformer', 'auto', 'deberta', 'nv-embed')
            model_name: 预训练模型名称
            **kwargs: 其他参数
        """
        if model_type == 'bert':
            return BertTextEncoder(model_name, **kwargs)
        elif model_type == 'roberta':
            return RobertaTextEncoder(model_name, **kwargs)
        elif model_type == 'xlm-roberta':
            return XLMRobertaTextEncoder(model_name, **kwargs)
        elif model_type == 'mt5':
            return MT5TextEncoder(model_name, **kwargs)
        elif model_type == 'mbart':
            return MBartTextEncoder(model_name, **kwargs)
        elif model_type == 'sentence-transformer':
            return SentenceTransformerEncoder(model_name, **kwargs)
        elif model_type == 'auto':
            return AutoTextEncoder(model_name, **kwargs)
        elif model_type == 'deberta':
            return DeBERTaTextEncoder(model_name, **kwargs)
        elif model_type == 'nv-embed':
            return NVEmbedTextEncoder(model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

class VisualEncoderFactory:
    """视觉编码器工厂类，支持多种预训练模型"""
    
    @staticmethod
    def create_encoder(model_type: str, model_name: str, **kwargs):
        """
        创建视觉编码器
        
        Args:
            model_type: 模型类型 ('vit', 'swin', 'layoutlm')
            model_name: 预训练模型名称
            **kwargs: 其他参数
        """
        if model_type == 'vit':
            return ViTVisualEncoder(model_name, **kwargs)
        elif model_type == 'swin':
            return SwinVisualEncoder(model_name, **kwargs)
        elif model_type == 'layoutlm':
            return LayoutLMVisualEncoder(model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

class MultimodalEncoderFactory:
    """多模态编码器工厂类，支持多种预训练模型"""
    
    @staticmethod
    def create_encoder(model_type: str, model_name: str, **kwargs):
        """
        创建多模态编码器
        
        Args:
            model_type: 模型类型 ('clip', 'layoutlmv2')
            model_name: 预训练模型名称
            **kwargs: 其他参数
        """
        if model_type == 'clip':
            return CLIPMultimodalEncoder(model_name, **kwargs)
        elif model_type == 'layoutlmv2':
            return LayoutLMv2MultimodalEncoder(model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

# ============= 文本编码器 =============

class BertTextEncoder(BaseTextEncoder):
    """BERT文本编码器"""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, kwargs.get('freeze_layers'))
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.freeze_specified_layers()
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

class RobertaTextEncoder(BaseTextEncoder):
    """RoBERTa文本编码器"""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, kwargs.get('freeze_layers'))
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.freeze_specified_layers()
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

class XLMRobertaTextEncoder(BaseTextEncoder):
    """XLM-RoBERTa文本编码器"""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, kwargs.get('freeze_layers'))
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.model = XLMRobertaModel.from_pretrained(model_name)
        self.freeze_specified_layers()
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

class MT5TextEncoder(BaseTextEncoder):
    """mT5文本编码器"""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, kwargs.get('freeze_layers'))
        self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
        self.model = MT5Model.from_pretrained(model_name)
        self.freeze_specified_layers()
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model.encoder(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

class MBartTextEncoder(BaseTextEncoder):
    """mBART文本编码器"""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, kwargs.get('freeze_layers'))
        self.tokenizer = MBartTokenizer.from_pretrained(model_name)
        self.model = MBartModel.from_pretrained(model_name)
        self.freeze_specified_layers()
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model.encoder(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

class DeBERTaTextEncoder(BaseTextEncoder):
    """DeBERTa文本编码器"""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, kwargs.get('freeze_layers'))
        self.tokenizer = DebertaTokenizer.from_pretrained(model_name)
        self.model = DebertaModel.from_pretrained(model_name)
        self.freeze_specified_layers()
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

class NVEmbedTextEncoder(BaseTextEncoder):
    """NV-Embed-v2文本编码器"""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, kwargs.get('freeze_layers'))
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.freeze_specified_layers()
        
    def forward(self, input_ids, attention_mask=None):
        # NV-Embed-v2 使用特殊的池化策略
        outputs = self.model(input_ids, attention_mask=attention_mask)
        
        # 使用平均池化获取句子表示
        if attention_mask is not None:
            # 计算每个序列的有效长度
            mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            return mean_embeddings
        else:
            # 如果没有attention mask，使用所有token的平均值
            return torch.mean(outputs.last_hidden_state, dim=1)

class SentenceTransformerEncoder(BaseTextEncoder):
    """SentenceTransformer文本编码器"""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, kwargs.get('freeze_layers'))
        self.model = SentenceTransformer(model_name)
        self.freeze_specified_layers()
        
    def forward(self, input_ids, attention_mask=None):
        # SentenceTransformer期望输入是文本而不是token ids
        if isinstance(input_ids, torch.Tensor):
            # 如果输入是token ids，需要先解码
            texts = self.model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        else:
            texts = input_ids
            
        # 获取句子嵌入
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings

class AutoTextEncoder(BaseTextEncoder):
    """自动选择文本编码器"""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, kwargs.get('freeze_layers'))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.freeze_specified_layers()
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

# ============= 视觉编码器 =============

class ViTVisualEncoder(BaseVisualEncoder):
    """ViT视觉编码器"""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, kwargs.get('image_size', 224))
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        
    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        return outputs.last_hidden_state

class SwinVisualEncoder(BaseVisualEncoder):
    """Swin视觉编码器"""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, kwargs.get('image_size', 224))
        self.processor = SwinImageProcessor.from_pretrained(model_name)
        self.model = SwinModel.from_pretrained(model_name)
        
    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        return outputs.last_hidden_state

class LayoutLMVisualEncoder(BaseVisualEncoder):
    """LayoutLM视觉编码器"""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, kwargs.get('image_size', 224))
        self.processor = LayoutLMv2Processor.from_pretrained(model_name)
        self.model = LayoutLMv2Model.from_pretrained(model_name)
        
    def forward(self, pixel_values, bbox=None):
        outputs = self.model(pixel_values, bbox=bbox)
        return outputs.last_hidden_state

# ============= 多模态编码器 =============

class CLIPMultimodalEncoder(BaseMultimodalEncoder):
    """CLIP多模态编码器"""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        
    def forward(self, input_ids, pixel_values, attention_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask
        )
        return {
            'text_features': outputs.text_embeds,
            'image_features': outputs.image_embeds
        }

class LayoutLMv2MultimodalEncoder(BaseMultimodalEncoder):
    """LayoutLMv2多模态编码器"""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        self.processor = LayoutLMv2Processor.from_pretrained(model_name)
        self.model = LayoutLMv2Model.from_pretrained(model_name)
        
    def forward(self, input_ids, pixel_values, bbox=None, attention_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            bbox=bbox,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state 