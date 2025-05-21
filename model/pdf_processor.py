# -*- coding: utf-8 -*-

import os
import fitz  # PyMuPDF
import cv2
import logging
from typing import List, Dict, Any, Optional

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入可选依赖
try:
    import layoutparser as lp
    import numpy as np
    from paddleocr import PaddleOCR
    LAYOUT_ANALYSIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"布局分析依赖导入失败: {e}")
    logger.warning("将使用基本文本提取模式")
    LAYOUT_ANALYSIS_AVAILABLE = False

# 尝试导入嵌入模型
try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    from PIL import Image
    import torchvision.transforms as transforms
    EMBEDDING_AVAILABLE = True
except ImportError:
    logger.warning("嵌入计算依赖导入失败，将禁用嵌入计算功能")
    EMBEDDING_AVAILABLE = False


def convert_processed_data_to_tensors(processed_data, device=None):
    """
    将处理后的PDF数据转换为PyTorch张量格式
    
    Args:
        processed_data: 由process_pdf函数处理后的结果
        device: 张量所在设备，如'cpu'或'cuda:0'
    
    Returns:
        转换后的字典，包含页面编码和文档特征的张量
    """
    if not EMBEDDING_AVAILABLE:
        logger.warning("PyTorch未安装，无法转换为张量")
        return processed_data
        
    import torch
    
    # 确定设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    # 初始化结果字典
    result = {
        'page_encodings': [],
        'document_features': None,
        'metadata': processed_data.get('metadata', {}).copy()
    }
    
    # 处理页面数据
    pages_data = processed_data.get('pages', {})
    if isinstance(pages_data, dict):  # 如果是字典形式 {page_idx: page_data}
        pages_data = [pages_data[idx] for idx in sorted(pages_data.keys())]
    elif not isinstance(pages_data, list):
        pages_data = [pages_data]
    
    for page_data in pages_data:
        page_encoding = {}
        
        # 文本嵌入
        if 'text_embeddings' in page_data:
            embeddings = page_data['text_embeddings']
            # 如果是多个文本块的嵌入，合并它们
            if isinstance(embeddings, list) and all(isinstance(e, list) for e in embeddings):
                # 计算平均值或连接
                if len(embeddings) > 0:
                    if isinstance(embeddings[0], list):
                        embeddings = [sum(e) / len(e) if len(e) > 0 else [0] for e in embeddings]
            page_encoding['text_embeddings'] = torch.tensor(
                embeddings,
                dtype=torch.float32,
                device=device
            )
        
        # 图像嵌入
        if 'image_embedding' in page_data:
            page_encoding['page_image_embedding'] = torch.tensor(
                page_data['image_embedding'],
                dtype=torch.float32,
                device=device
            )
        elif 'page_image' in page_data:
            # 提供默认的图像嵌入
            page_encoding['page_image_embedding'] = torch.zeros(
                (768,),  # 默认维度
                dtype=torch.float32,
                device=device
            )
        
        # 版面嵌入
        if 'layout_embedding' in page_data:
            page_encoding['layout_embedding'] = torch.tensor(
                page_data['layout_embedding'],
                dtype=torch.float32,
                device=device
            )
        elif 'layout' in page_data:
            # 提供默认的版面嵌入
            page_encoding['layout_embedding'] = torch.zeros(
                (256,),  # 默认维度
                dtype=torch.float32,
                device=device
            )
        
        # 如果没有任何嵌入，添加默认值
        if not page_encoding:
            page_encoding = {
                'text_embeddings': torch.zeros((768,), dtype=torch.float32, device=device),
                'page_image_embedding': torch.zeros((768,), dtype=torch.float32, device=device),
                'layout_embedding': torch.zeros((256,), dtype=torch.float32, device=device)
            }
        
        result['page_encodings'].append(page_encoding)
    
    # 文档特征
    if 'document_features' in processed_data:
        result['document_features'] = torch.tensor(
            processed_data['document_features'],
            dtype=torch.float32,
            device=device
        )
    
    return result


def process_pdf(pdf_path: str, output_dir: Optional[str] = None, target_pages: Optional[List[int]] = None, 
               extract_text: bool = True, extract_images: bool = True, extract_layout: bool = True,
               compute_embeddings: bool = False, visualize: bool = False) -> Dict[str, Any]:
    """
    处理单个 PDF 文件，进行解析、渲染、版面分析和 OCR。
    
    Args:
        pdf_path: 输入 PDF 文件的路径。
        output_dir: 处理结果的输出目录，如果不提供则只返回结果不保存。
        target_pages: 要处理的特定页面列表，None表示处理所有页面。
        extract_text: 是否提取文本内容。
        extract_images: 是否提取图像内容。
        extract_layout: 是否进行版面分析。
        compute_embeddings: 是否计算嵌入向量。
        visualize: 是否生成可视化结果。
        
    Returns:
        包含处理结果的字典。
    """
    try:
        # 确保输入 PDF 文件存在
        if not os.path.exists(pdf_path):
            logger.error(f"PDF 文件不存在: {pdf_path}")
            return {"error": "PDF file not found"}
        
        # 获取 PDF 文件名（不带扩展名）作为输出目录名
        pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # 如果提供了输出目录，创建输出目录
        if output_dir:
            pdf_output_dir = os.path.join(output_dir, pdf_basename)
            os.makedirs(pdf_output_dir, exist_ok=True)
            
            # 图像输出目录
            images_dir = os.path.join(pdf_output_dir, "images")
            if extract_images:
                os.makedirs(images_dir, exist_ok=True)
            
            # 可视化结果目录
            if visualize:
                viz_dir = os.path.join(pdf_output_dir, "visualize")
                os.makedirs(viz_dir, exist_ok=True)
        
        # 打开 PDF 文件
        pdf_document = fitz.open(pdf_path)
        num_pages = len(pdf_document)
        
        # 确定要处理的页面
        if target_pages is not None:
            pages_to_process = [p for p in target_pages if 0 <= p < num_pages]
        else:
            pages_to_process = list(range(num_pages))
        
        if not pages_to_process:
            logger.warning(f"没有有效的页面可处理: {pdf_path}")
            return {"error": "No valid pages to process"}
        
        # 初始化结果字典
        result = {
            "filename": pdf_basename,
            "num_pages": num_pages,
            "pages": {}
        }
        
        # 初始化嵌入模型（如果需要）
        text_model = None
        image_model = None
        layout_model = None
        
        if compute_embeddings and EMBEDDING_AVAILABLE:
            try:
                # 文本嵌入模型
                text_model_name = "bert-base-chinese"  # 或其他适合中文的模型
                text_model = AutoModel.from_pretrained(text_model_name)
                text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
                
                # 图像嵌入模型（简化示例）
                image_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                
                logger.info("嵌入模型加载成功")
            except Exception as e:
                logger.error(f"嵌入模型加载失败: {e}")
                compute_embeddings = False
        
        # 处理每个页面
        for page_idx in pages_to_process:
            page = pdf_document[page_idx]
            page_result = {}
            
            # 1. 提取页面文本（如果需要）
            if extract_text:
                # 首先使用PyMuPDF直接提取文本
                page_text = page.get_text()
                page_result["text"] = page_text
                
                # 如果文本内容太少，可能是图像PDF，尝试使用OCR
                if len(page_text.strip()) < 100 and LAYOUT_ANALYSIS_AVAILABLE:
                    try:
                        # 创建PaddleOCR实例
                        ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False)
                        
                        # 获取当前页面图像用于OCR
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x分辨率
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                        
                        # 执行OCR
                        logger.info(f"原始文本内容不足，对页面 {page_idx+1} 执行OCR")
                        ocr_result = ocr.ocr(img_array, cls=True)
                        
                        # 提取OCR文本
                        ocr_text = ""
                        if ocr_result and len(ocr_result) > 0 and ocr_result[0]:
                            for line in ocr_result[0]:
                                if len(line) >= 2:  # 确保结果包含文本和置信度
                                    text, confidence = line[1]
                                    ocr_text += text + "\n"
                        
                        # 如果OCR文本比原始文本更丰富，则使用OCR文本
                        if len(ocr_text.strip()) > len(page_text.strip()):
                            logger.info(f"使用OCR文本替代原始文本，页面 {page_idx+1}")
                            page_text = ocr_text
                            page_result["text"] = ocr_text
                            page_result["ocr_used"] = True
                    except Exception as e:
                        logger.warning(f"OCR处理失败: {e}")
                        page_result["ocr_error"] = str(e)
                
                # 计算文本嵌入（如果需要）
                if compute_embeddings and text_model and page_text.strip():
                    try:
                        inputs = text_tokenizer(page_text, return_tensors="pt", truncation=True, max_length=512)
                        with torch.no_grad():
                            outputs = text_model(**inputs)
                        # 使用最后一层的[CLS]令牌表示作为文本嵌入
                        text_embedding = outputs.last_hidden_state[:, 0, :].numpy().tolist()[0]
                        page_result["text_embeddings"] = text_embedding
                    except Exception as e:
                        logger.error(f"计算文本嵌入失败: {e}")
                        page_result["text_embeddings"] = [0.0] * 768  # 假设模型输出是768维
            
            # 2. 渲染页面图像（如果需要）
            if extract_images:
                # 渲染页面为图像
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x分辨率
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_array = np.array(img)
                
                # 保存图像（如果需要）
                if output_dir:
                    img_path = os.path.join(images_dir, f"page_{page_idx+1}.png")
                    img.save(img_path)
                    page_result["image_path"] = img_path
                
                # 计算图像嵌入（如果需要）
                if compute_embeddings and image_model:
                    try:
                        img_tensor = image_transform(img).unsqueeze(0)
                        with torch.no_grad():
                            image_features = image_model(img_tensor)
                        img_embedding = image_features.numpy().tolist()[0]
                        page_result["image_embedding"] = img_embedding
                    except Exception as e:
                        logger.error(f"计算图像嵌入失败: {e}")
                        page_result["image_embedding"] = [0.0] * 768  # 假设输出是768维
            
            # 3. 进行版面分析（如果需要）
            if extract_layout and LAYOUT_ANALYSIS_AVAILABLE:
                try:
                    # 使用 LayoutParser 进行版面分析
                    model = lp.models.Detectron2LayoutModel(
                        config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
                    )
                    layout = model.detect(img_array)
                    
                    # 提取布局信息
                    layout_info = []
                    for block in layout:
                        x1, y1, x2, y2 = block.coordinates
                        layout_info.append({
                            "type": block.type,
                            "score": float(block.score),
                            "bbox": [float(x1), float(y1), float(x2), float(y2)]
                        })
                    
                    page_result["layout"] = layout_info
                    
                    # 生成可视化结果（如果需要）
                    if visualize and output_dir:
                        viz_img = lp.draw_box(img_array, layout, box_width=3)
                        viz_path = os.path.join(viz_dir, f"layout_page_{page_idx+1}.png")
                        cv2.imwrite(viz_path, cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR))
                        page_result["layout_visualization"] = viz_path
                    
                    # 计算布局嵌入（简化示例）
                    if compute_embeddings:
                        # 将布局信息转换为特征向量
                        layout_embedding = []
                        for block in layout_info:
                            # 简单示例：将边界框和类型ID编码为特征
                            block_type_id = ["Text", "Title", "List", "Table", "Figure"].index(block["type"])
                            block_features = block["bbox"] + [block_type_id, block["score"]]
                            layout_embedding.extend(block_features)
                        
                        # 填充或截断到固定长度
                        target_length = 256  # 假设我们想要256维的特征
                        if len(layout_embedding) > target_length:
                            layout_embedding = layout_embedding[:target_length]
                        else:
                            layout_embedding.extend([0.0] * (target_length - len(layout_embedding)))
                        
                        page_result["layout_embedding"] = layout_embedding
                except Exception as e:
                    logger.error(f"版面分析失败: {e}")
            
            # 添加页面结果到总结果
            result["pages"][page_idx] = page_result
        
        # 关闭 PDF 文件
        pdf_document.close()
        
        # 如果需要输出目录，保存元数据
        if output_dir:
            import json
            with open(os.path.join(pdf_output_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result
    
    except Exception as e:
        logger.error(f"处理 PDF 失败: {e}")
        return {"error": str(e)}