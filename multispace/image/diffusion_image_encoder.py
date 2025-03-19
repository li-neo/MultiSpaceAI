"""
Diffusion API 图像编码器模块
==========================

该模块包含使用Stable Diffusion API进行图像编码的类。
"""

import torch
import numpy as np
import requests
import json
import base64
import io
import os
from typing import Optional, Dict, Any, Union
from PIL import Image

from ..utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)


class DiffusionImageEncoder:
    """
    Diffusion API 图像编码器类，使用Stable Diffusion的API服务对图像进行编码。
    """
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        初始化Diffusion API图像编码器
        
        参数:
            api_key: Stable Diffusion API密钥
            api_url: Stable Diffusion API URL，如果为None则使用默认URL
        """
        self.api_key = api_key
        self.api_url = api_url or "https://api.stability.ai/v1/embeddings/image"
        
        # 检查API密钥
        if not self.api_key:
            logger.warning("未提供Stable Diffusion API密钥，可能导致API调用失败")
        
        logger.info("Diffusion API图像编码器初始化完成")
    
    def _load_image(self, image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """
        加载图像
        
        参数:
            image: 输入图像，可以是图像路径、numpy数组或PIL图像
            
        返回:
            PIL图像
        """
        # 如果是路径，加载图像
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"图像文件不存在: {image}")
            image = Image.open(image).convert("RGB")
        
        # 如果是numpy数组，转换为PIL图像
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(np.uint8(image))
        
        return image
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """
        将PIL图像转换为base64编码
        
        参数:
            image: PIL图像
            
        返回:
            base64编码的图像字符串
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def encode(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        使用Diffusion API编码图像
        
        参数:
            image: 输入图像，可以是图像路径、numpy数组或PIL图像
            
        返回:
            图像特征表示
        """
        # 记录日志
        logger.info("使用Diffusion API编码图像")
        
        # 加载图像
        img = self._load_image(image)
        
        # 调整图像大小为API要求的尺寸
        img = img.resize((512, 512))
        
        # 转换为base64
        image_base64 = self._image_to_base64(img)
        
        # 构建请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建请求体
        data = {
            "image_base64": image_base64,
            "model": "stable-diffusion-xl-1024-v1-0"  # 使用最新的SDXL模型
        }
        
        try:
            # 发送请求
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()  # 如果请求失败，抛出异常
            
            # 解析响应
            result = response.json()
            
            # 获取嵌入向量
            embedding = result.get("embedding") or result.get("data", {}).get("embedding")
            
            if not embedding:
                raise ValueError(f"API响应中未找到嵌入向量: {result}")
            
            # 转换为torch.Tensor
            embedding_tensor = torch.tensor(embedding, dtype=torch.float)
            
            return embedding_tensor
            
        except Exception as e:
            logger.error(f"Diffusion API调用失败: {str(e)}")
            
            # 返回一个全零的向量作为备用
            # 假设嵌入维度为1024
            return torch.zeros(1024, dtype=torch.float)
    
    def batch_encode(self, images: list) -> torch.Tensor:
        """
        批量编码图像
        
        参数:
            images: 图像列表
            
        返回:
            批量图像特征表示
        """
        # 记录日志
        logger.info(f"使用Diffusion API批量编码{len(images)}个图像")
        
        # 创建一个空列表来存储嵌入向量
        embeddings = []
        
        # 依次处理每个图像
        for image in images:
            embedding = self.encode(image)
            embeddings.append(embedding)
        
        # 堆叠所有嵌入向量
        if embeddings:
            return torch.stack(embeddings)
        else:
            # 如果没有嵌入向量，返回一个空的张量
            return torch.zeros((0, 1024), dtype=torch.float) 