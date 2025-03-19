"""
图像编码器模块
===========

该模块包含用于处理和编码图像输入的类。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Union
from PIL import Image
import os
from transformers import ViTFeatureExtractor, ViTModel

from ..config import ImageEncoderConfig
from ..utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)


class ImageEncoder(nn.Module):
    """
    图像编码器类，用于处理和编码图像输入。
    基于Hugging Face的Vision Transformer模型。
    """
    
    def __init__(self, config: ImageEncoderConfig, device: Optional[str] = None):
        """
        初始化图像编码器
        
        参数:
            config: 图像编码器配置
            device: 运行设备
        """
        super().__init__()
        self.config = config
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化特征提取器和模型
        logger.info(f"加载图像编码器预训练模型: {config.model_name}")
        
        # 加载特征提取器
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(config.model_name)
        
        # 加载预训练模型
        self.model = ViTModel.from_pretrained(config.model_name)
        
        # 输出维度适配层
        if self.model.config.hidden_size != config.hidden_dim:
            self.output_adapter = nn.Linear(self.model.config.hidden_size, config.hidden_dim)
        else:
            self.output_adapter = nn.Identity()
        
        # 移动模型到指定设备
        self.to(self.device)
        
        logger.info("图像编码器初始化完成")
    
    def preprocess(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, torch.Tensor]:
        """
        对图像进行预处理
        
        参数:
            image: 输入图像，可以是图像路径、numpy数组或PIL图像
            
        返回:
            预处理后的输入
        """
        # 如果是路径，加载图像
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"图像文件不存在: {image}")
            image = Image.open(image).convert("RGB")
        
        # 如果是numpy数组，转换为PIL图像
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(np.uint8(image))
        
        # 使用特征提取器处理图像
        inputs = self.feature_extractor(
            images=image,
            return_tensors="pt"
        )
        
        # 移动到指定设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            pixel_values: 像素值
            
        返回:
            图像特征表示
        """
        # 获取模型输出
        outputs = self.model(pixel_values=pixel_values)
        
        # 获取最后一层的隐藏状态
        last_hidden_state = outputs.last_hidden_state
        
        # 使用[CLS]标记的表示作为整个图像的表示
        # 对应位置为 last_hidden_state 的第一个标记
        pooled_output = last_hidden_state[:, 0, :]
        
        # 应用输出适配层
        features = self.output_adapter(pooled_output)
        
        return features
    
    def encode(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        编码图像
        
        参数:
            image: 输入图像，可以是图像路径、numpy数组或PIL图像
            
        返回:
            图像特征表示
        """
        self.eval()  # 设置为评估模式
        
        with torch.no_grad():
            # 预处理图像
            inputs = self.preprocess(image)
            
            # 前向传播
            features = self.forward(pixel_values=inputs["pixel_values"])
        
        return features 