# -*- coding: utf-8 -*-

import os
import time
import torch
import logging
from typing import List, Dict, Any, Optional, Union
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
# import layoutparser as lp # 可能需要导入用于处理版面信息

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局模型缓存
MODEL_CACHE = {}

class MultimodalEncoder:
    """
    多模态编码器，用于将文本、图像和版面信息编码为嵌入向量。
    支持模型缓存、批处理和设备管理。
    """
    def __init__(self, text_model_name: str, vision_model_name: str,
                 use_cache: bool = True, device: Optional[str] = None,
                 batch_size: int = 8):
        """
        初始化多模态编码器。

        Args:
            text_model_name: 用于文本编码的预训练模型名称。
            vision_model_name: 用于图像编码的预训练模型名称。
            use_cache: 是否使用全局模型缓存。
            device: 运行模型的设备 ('cpu', 'cuda', 'cuda:0' 等)，None 表示自动选择。
            batch_size: 批处理大小，用于批量编码文本和图像。
        """
        # 设置设备
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 设置批处理大小
        self.batch_size = batch_size
        
        # 加载文本模型（使用缓存）
        if use_cache and text_model_name in MODEL_CACHE:
            logger.info(f"从缓存加载文本模型: {text_model_name}")
            self.tokenizer = MODEL_CACHE[f"{text_model_name}_tokenizer"]
            self.text_model = MODEL_CACHE[f"{text_model_name}_model"]
        else:
            logger.info(f"初始化文本编码器: {text_model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
                self.text_model = AutoModel.from_pretrained(text_model_name).to(self.device)
                
                # 缓存模型
                if use_cache:
                    MODEL_CACHE[f"{text_model_name}_tokenizer"] = self.tokenizer
                    MODEL_CACHE[f"{text_model_name}_model"] = self.text_model
                    logger.info(f"文本模型已缓存: {text_model_name}")
            except Exception as e:
                logger.error(f"加载文本模型失败: {e}")
                # 使用备用策略
                self.tokenizer = None
                self.text_model = None
        
        # 加载视觉模型（使用缓存）
        if use_cache and vision_model_name in MODEL_CACHE:
            logger.info(f"从缓存加载视觉模型: {vision_model_name}")
            self.feature_extractor = MODEL_CACHE[f"{vision_model_name}_extractor"]
            self.vision_model = MODEL_CACHE[f"{vision_model_name}_model"]
        else:
            logger.info(f"初始化图像编码器: {vision_model_name}")
            try:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(vision_model_name)
                self.vision_model = AutoModel.from_pretrained(vision_model_name).to(self.device)
                
                # 缓存模型
                if use_cache:
                    MODEL_CACHE[f"{vision_model_name}_extractor"] = self.feature_extractor
                    MODEL_CACHE[f"{vision_model_name}_model"] = self.vision_model
                    logger.info(f"视觉模型已缓存: {vision_model_name}")
            except Exception as e:
                logger.error(f"加载视觉模型失败: {e}")
                # 使用备用策略
                self.feature_extractor = None
                self.vision_model = None

    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        编码文本，支持单个文本或文本列表（批处理）。

        Args:
            text: 输入文本字符串或文本列表。

        Returns:
            文本的嵌入向量。对于单个文本，返回形状为 [1, embedding_dim]；
            对于文本列表，返回形状为 [batch_size, embedding_dim]。
        """
        if self.tokenizer is None or self.text_model is None:
            logger.warning("文本模型未初始化，返回零向量")
            # 返回零向量
            embedding_dim = 768  # 默认嵌入维度
            if isinstance(text, list):
                return torch.zeros(len(text), embedding_dim)
            else:
                return torch.zeros(1, embedding_dim)
        
        try:
            # 处理单个文本或文本列表
            is_batch = isinstance(text, list)
            texts = text if is_batch else [text]
            
            # 批处理
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i+self.batch_size]
                
                # 对批次进行编码
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                        truncation=True, max_length=512).to(self.device)
                
                with torch.no_grad():
                    outputs = self.text_model(**inputs)
                
                # 获取 [CLS] token 的输出作为文本表示
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
                all_embeddings.append(batch_embeddings)
            
            # 合并所有批次的结果
            embeddings = torch.cat(all_embeddings, dim=0)
            
            # 如果输入是单个文本，则返回单个嵌入
            return embeddings if is_batch else embeddings[0].unsqueeze(0)
            
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            # 返回零向量作为备用
            embedding_dim = 768  # 默认嵌入维度
            if isinstance(text, list):
                return torch.zeros(len(text), embedding_dim)
            else:
                return torch.zeros(1, embedding_dim)

    def encode_image(self, image_path: Union[str, List[str]]) -> torch.Tensor:
        """
        编码图像，支持单个图像路径或图像路径列表（批处理）。

        Args:
            image_path: 输入图像文件的路径或路径列表。

        Returns:
            图像的嵌入向量。对于单个图像，返回形状为 [1, embedding_dim]；
            对于图像列表，返回形状为 [batch_size, embedding_dim]。
        """
        if self.feature_extractor is None or self.vision_model is None:
            logger.warning("视觉模型未初始化，返回零向量")
            # 返回零向量
            embedding_dim = 768  # 默认嵌入维度
            if isinstance(image_path, list):
                return torch.zeros(len(image_path), embedding_dim)
            else:
                return torch.zeros(1, embedding_dim)
        
        try:
            from PIL import Image, UnidentifiedImageError
            
            # 处理单个图像路径或路径列表
            is_batch = isinstance(image_path, list)
            paths = image_path if is_batch else [image_path]
            
            # 加载图像
            images = []
            valid_indices = []
            
            for i, path in enumerate(paths):
                try:
                    if os.path.exists(path):
                        img = Image.open(path).convert("RGB")
                        images.append(img)
                        valid_indices.append(i)
                    else:
                        logger.warning(f"图像文件不存在: {path}")
                except (UnidentifiedImageError, OSError) as e:
                    logger.warning(f"无法加载图像 {path}: {e}")
            
            # 如果没有有效图像，返回零向量
            if not images:
                logger.warning("没有有效的图像可以编码")
                embedding_dim = 768  # 默认嵌入维度
                if is_batch:
                    return torch.zeros(len(paths), embedding_dim)
                else:
                    return torch.zeros(1, embedding_dim)
            
            # 批处理
            all_embeddings = []
            for i in range(0, len(images), self.batch_size):
                batch_images = images[i:i+self.batch_size]
                
                # 对批次进行编码
                inputs = self.feature_extractor(images=batch_images, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.vision_model(**inputs)
                
                # 使用平均池化作为图像表示
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
                all_embeddings.append(batch_embeddings)
            
            # 合并所有批次的结果
            embeddings = torch.cat(all_embeddings, dim=0)
            
            # 如果有无效图像，需要填充结果
            if is_batch and len(valid_indices) < len(paths):
                full_embeddings = torch.zeros(len(paths), embeddings.shape[1])
                for i, idx in enumerate(valid_indices):
                    full_embeddings[idx] = embeddings[i]
                embeddings = full_embeddings
            
            # 如果输入是单个图像，则返回单个嵌入
            return embeddings if is_batch else embeddings[0].unsqueeze(0)
            
        except Exception as e:
            logger.error(f"图像编码失败: {e}")
            # 返回零向量作为备用
            embedding_dim = 768  # 默认嵌入维度
            if isinstance(image_path, list):
                return torch.zeros(len(image_path), embedding_dim)
            else:
                return torch.zeros(1, embedding_dim)

    def encode_layout(self, layout_info: list, img_width: int = 1000, img_height: int = 1000) -> torch.Tensor:
        """
        编码版面信息，增强了错误处理和归一化。

        Args:
            layout_info: 从版面分析中提取的信息列表。
            img_width: 图像宽度，用于坐标归一化。
            img_height: 图像高度，用于坐标归一化。

        Returns:
            版面信息的嵌入向量，形状为 [1, feature_dim]。
        """
        try:
            # 如果版面信息为空，返回零向量
            if not layout_info:
                return torch.zeros(1, 5)  # 特征维度为 1 + 4 = 5
            
            # 类型到ID的映射
            type_map = {
                "Text": 0, "Title": 1, "List": 2, "Table": 3,
                "Figure": 4, "Caption": 5, "Header": 6, "Footer": 7, "Other": 8
            }
            
            # 收集每个块的特征
            layout_features = []
            
            for block in layout_info:
                try:
                    # 获取块类型
                    block_type = block.get("type", "Other")
                    block_type_id = type_map.get(block_type, 8)  # 默认为 "Other"
                    
                    # 获取坐标并进行归一化
                    coords = block.get("coordinates", (0, 0, 0, 0))
                    if len(coords) != 4:
                        logger.warning(f"无效的坐标格式: {coords}，使用默认值 (0,0,0,0)")
                        coords = (0, 0, 0, 0)
                    
                    # 确保坐标是有效的数值
                    coords = [max(0, min(c, max(img_width, img_height))) for c in coords]
                    
                    # 归一化坐标到 [0, 1] 范围
                    normalized_coords = [
                        coords[0] / img_width,
                        coords[1] / img_height,
                        coords[2] / img_width,
                        coords[3] / img_height
                    ]
                    
                    # 组合特征
                    feature = [block_type_id] + normalized_coords
                    layout_features.append(feature)
                    
                except Exception as e:
                    logger.warning(f"处理版面块时出错: {e}")
                    continue
            
            # 如果没有有效特征，返回零向量
            if not layout_features:
                return torch.zeros(1, 5)
            
            # 将特征列表转换为 Tensor
            layout_features_tensor = torch.tensor(layout_features, dtype=torch.float32)
            
            # 计算平均值作为整个版面的表示
            return layout_features_tensor.mean(dim=0).unsqueeze(0)
            
        except Exception as e:
            logger.error(f"版面编码失败: {e}")
            return torch.zeros(1, 5)  # 返回零向量作为备用


    def encode_document_page(self, page_data: dict) -> dict:
        """
        编码单个文档页面的多模态信息，增强了批处理和错误处理。

        Args:
            page_data: 从文档预处理模块获得的单个页面数据。

        Returns:
            包含编码后嵌入的字典。
        """
        start_time = time.time()
        page_num = page_data.get("page_num", "未知")
        logger.info(f"开始编码页面 {page_num}")
        
        encoded_data = {
            "page_num": page_num,
            "text_embeddings": [],
            "page_image_embedding": None,
            "layout_embedding": None,
            "image_segment_embeddings": []
        }
        
        try:
            # 1. 批量编码文本块
            text_blocks = page_data.get("text_blocks", [])
            if text_blocks:
                # 收集所有非空文本
                texts = []
                valid_indices = []
                
                for i, block in enumerate(text_blocks):
                    text = block.get("text", "").strip()
                    if text:
                        texts.append(text)
                        valid_indices.append(i)
                
                if texts:
                    # 批量编码所有文本
                    text_embeddings = self.encode_text(texts)
                    
                    # 将嵌入与原始文本块关联
                    for i, idx in enumerate(valid_indices):
                        encoded_data["text_embeddings"].append({
                            "text": text_blocks[idx].get("text", ""),
                            "coordinates": text_blocks[idx].get("coordinates", (0, 0, 0, 0)),
                            "type": text_blocks[idx].get("type", "Text"),
                            "embedding": text_embeddings[i].unsqueeze(0)  # 确保形状为 [1, embedding_dim]
                        })
            
            # 2. 编码页面图像
            page_image_path = page_data.get("image_path")
            if page_image_path and os.path.exists(page_image_path):
                encoded_data["page_image_embedding"] = self.encode_image(page_image_path)
            else:
                logger.warning(f"页面 {page_num} 的图像路径无效: {page_image_path}")
                encoded_data["page_image_embedding"] = torch.zeros(1, 768)  # 默认嵌入维度
            
            # 3. 编码版面信息
            # 获取图像尺寸（如果可用）
            img_width, img_height = 1000, 1000  # 默认值
            if page_image_path and os.path.exists(page_image_path):
                try:
                    from PIL import Image
                    with Image.open(page_image_path) as img:
                        img_width, img_height = img.size
                except Exception as e:
                    logger.warning(f"无法获取图像尺寸: {e}")
            
            encoded_data["layout_embedding"] = self.encode_layout(
                page_data.get("layout", []),
                img_width=img_width,
                img_height=img_height
            )
            
            # 4. 批量编码图像片段 (Figure/Table)
            segment_blocks = []
            segment_paths = []
            
            for block in page_data.get("layout", []):
                if block.get("type") in ["Figure", "Table", "Chart", "Diagram"]:
                    segment_path = block.get("image_segment_path")
                    if segment_path and os.path.exists(segment_path):
                        segment_blocks.append(block)
                        segment_paths.append(segment_path)
            
            if segment_paths:
                # 批量编码所有图像片段
                segment_embeddings = self.encode_image(segment_paths)
                
                # 将嵌入与原始块关联
                for i, block in enumerate(segment_blocks):
                    if i < len(segment_embeddings):
                        encoded_data["image_segment_embeddings"].append({
                            "type": block.get("type", "Unknown"),
                            "coordinates": block.get("coordinates", (0, 0, 0, 0)),
                            "embedding": segment_embeddings[i].unsqueeze(0)  # 确保形状为 [1, embedding_dim]
                        })
            
            logger.info(f"页面 {page_num} 编码完成，耗时: {time.time() - start_time:.2f}秒")
            return encoded_data
            
        except Exception as e:
            logger.error(f"编码页面 {page_num} 时发生错误: {e}")
            # 返回部分结果或默认值
            return encoded_data

if __name__ == "__main__":
    # 示例用法 (需要替换为实际的模型名称和处理后的页面数据)
    # 注意：运行此示例需要先运行 pdf_processor.py 生成处理后的数据和图像文件
    text_model = "bert-base-chinese"
    vision_model = "google/vit-base-patch16-224"
    # 检查模型是否可用，实际应用中需要更健壮的模型加载和检查
    try:
        AutoTokenizer.from_pretrained(text_model)
        AutoModel.from_pretrained(text_model)
        AutoFeatureExtractor.from_pretrained(vision_model)
        AutoModel.from_pretrained(vision_model)
        models_available = True
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保已安装 transformers 库并有网络连接以下载模型。")
        models_available = False


    if models_available:
        encoder = MultimodalEncoder(text_model_name=text_model, vision_model_name=vision_model)

        # 假设有一些处理后的页面数据 (需要替换为实际加载的数据)
        # 这里创建一个模拟的页面数据结构，包含文本块和图像片段
        example_page_data = {
            "page_num": 1,
            "image_path": "data/processed_data/CN1071381C/page_1.png", # 示例页面图像路径
            "layout": [
                {"type": "Text", "coordinates": (10, 10, 100, 50), "text": "这是一个示例文本块。"},
                {"type": "Figure", "coordinates": (150, 100, 400, 300), "image_segment_path": "data/processed_data/CN1071381C/page_1_Figure_150_100.png"}, # 示例图表片段路径
                {"type": "Text", "coordinates": (10, 60, 100, 100), "text": "另一个文本块。"},
                {"type": "Table", "coordinates": (500, 100, 800, 300), "image_segment_path": "data/processed_data/CN1071381C/page_1_Table_500_100.png"} # 示例表格片段路径
            ],
            "text_blocks": [
                 {"type": "Text", "coordinates": (10, 10, 100, 50), "text": "这是一个示例文本块。"},
                 {"type": "Text", "coordinates": (10, 60, 100, 100), "text": "另一个文本块。"}
            ]
        }

        # 检查示例图像文件是否存在
        page_image_exists = os.path.exists(example_page_data["image_path"])
        figure_segment_exists = os.path.exists(example_page_data["layout"][1]["image_segment_path"])
        table_segment_exists = os.path.exists(example_page_data["layout"][3]["image_segment_path"])


        if page_image_exists and figure_segment_exists and table_segment_exists:
             print(f"\n开始编码示例页面数据...")
             encoded_page = encoder.encode_document_page(example_page_data)
             print("\n示例页面编码结果：")
             print(f"文本嵌入数量: {len(encoded_page.get('text_embeddings', []))}")
             print(f"页面图像嵌入形状: {encoded_page['page_image_embedding'].shape if encoded_page.get('page_image_embedding') is not None else 'None'}")
             print(f"版面嵌入形状: {encoded_page['layout_embedding'].shape if encoded_page.get('layout_embedding') is not None else 'None'}")
             print(f"图像片段嵌入数量: {len(encoded_page.get('image_segment_embeddings', []))}")
             for i, segment_embed in enumerate(encoded_page.get('image_segment_embeddings', [])):
                 print(f"  片段 {i+1} (类型: {segment_embed['type']}) 嵌入形状: {segment_embed['embedding'].shape}")

        else:
             print("\n示例图像文件或图像片段文件不存在。请先运行 pdf_processor.py 生成这些文件。")
             print(f"页面图像存在: {page_image_exists}")
             print(f"图表片段存在: {figure_segment_exists}")
             print(f"表格片段存在: {table_segment_exists}")


    else:
        print("\n由于模型加载失败，跳过示例编码。")