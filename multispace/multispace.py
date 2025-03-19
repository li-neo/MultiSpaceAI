"""
MultiSpaceAI 主类模块
====================

该模块包含 MultiSpaceAI 类，它是整个系统的主要入口点。
该类协调各个模态的编码器、融合模块和解码器的工作。
"""

import os
import logging
from typing import Optional, Union, Dict, Any

import torch
import numpy as np

from .config import ModelConfig
from .text import TextEncoder, DeepSeekTextEncoder
from .image import ImageEncoder, DiffusionImageEncoder
from .audio import AudioEncoder, WhisperAudioEncoder
from .fusion import MultimodalFusionModule
from .decoder import DecoderModule
from .utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)

class MultiSpaceAI:
    """
    MultiSpaceAI 是一个多模态大语言模型系统，能够处理文本、图像和语音输入，
    实现复杂的跨模态理解与生成任务。
    
    该类是整个系统的主要入口点，负责协调各个模块的工作。
    
    属性:
        config (ModelConfig): 模型配置
        text_encoder: 文本编码器模块
        image_encoder: 图像编码器模块
        audio_encoder: 语音编码器模块
        fusion_module: 多模态融合模块
        decoder_module: 解码器模块
        device (torch.device): 模型运行的设备
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 text_encoder: str = "custom",
                 image_encoder: str = "custom",
                 audio_encoder: str = "custom",
                 device: Optional[str] = None):
        """
        初始化 MultiSpaceAI 实例
        
        参数:
            config_path: 配置文件路径，如果为None则使用默认配置
            text_encoder: 文本编码器类型，可选值为 "custom" 或 "deepseek-api"
            image_encoder: 图像编码器类型，可选值为 "custom" 或 "diffusion-api"
            audio_encoder: 语音编码器类型，可选值为 "custom" 或 "whisper-api"
            device: 模型运行的设备，如果为None则自动选择
        """
        # 加载配置
        self.config = ModelConfig(config_path)
        
        # 设置设备
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 初始化编码器
        self._init_encoders(text_encoder, image_encoder, audio_encoder)
        
        # 初始化融合模块
        self.fusion_module = MultimodalFusionModule(
            config=self.config.fusion_config,
            device=self.device
        )
        
        # 初始化解码器模块
        self.decoder_module = DecoderModule(
            config=self.config.decoder_config,
            device=self.device
        )
        
        logger.info("MultiSpaceAI 初始化完成")
    
    def _init_encoders(self, text_encoder_type, image_encoder_type, audio_encoder_type):
        """
        初始化各模态编码器
        
        参数:
            text_encoder_type: 文本编码器类型
            image_encoder_type: 图像编码器类型
            audio_encoder_type: 语音编码器类型
        """
        # 初始化文本编码器
        if text_encoder_type == "deepseek-api":
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                logger.warning("未找到 DEEPSEEK_API_KEY 环境变量，请确保已正确设置")
            self.text_encoder = DeepSeekTextEncoder(api_key=api_key)
            logger.info("使用 DeepSeek API 作为文本编码器")
        else:
            self.text_encoder = TextEncoder(
                config=self.config.text_encoder_config,
                device=self.device
            )
            logger.info("使用自定义模型作为文本编码器")
        
        # 初始化图像编码器
        if image_encoder_type == "diffusion-api":
            api_key = os.environ.get("STABLE_DIFFUSION_API_KEY")
            if not api_key:
                logger.warning("未找到 STABLE_DIFFUSION_API_KEY 环境变量，请确保已正确设置")
            self.image_encoder = DiffusionImageEncoder(api_key=api_key)
            logger.info("使用 Diffusion API 作为图像编码器")
        else:
            self.image_encoder = ImageEncoder(
                config=self.config.image_encoder_config,
                device=self.device
            )
            logger.info("使用自定义模型作为图像编码器")
        
        # 初始化语音编码器
        if audio_encoder_type == "whisper-api":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("未找到 OPENAI_API_KEY 环境变量，请确保已正确设置")
            self.audio_encoder = WhisperAudioEncoder(api_key=api_key)
            logger.info("使用 Whisper API 作为语音编码器")
        else:
            self.audio_encoder = AudioEncoder(
                config=self.config.audio_encoder_config,
                device=self.device
            )
            logger.info("使用自定义模型作为语音编码器")
    
    def process(self, 
                text: Optional[str] = None, 
                image: Optional[Union[str, np.ndarray]] = None,
                audio: Optional[Union[str, np.ndarray]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        处理多模态输入并生成输出
        
        参数:
            text: 文本输入，可以是字符串
            image: 图像输入，可以是图像路径或numpy数组
            audio: 语音输入，可以是音频文件路径或numpy数组
            **kwargs: 其他参数
            
        返回:
            包含处理结果的字典
        """
        # 检查是否至少有一种模态的输入
        if not any([text, image, audio]):
            raise ValueError("至少需要提供一种模态的输入（文本、图像或语音）")
        
        # 初始化各模态的特征
        text_features = None
        image_features = None
        audio_features = None
        
        # 处理文本输入
        if text:
            logger.info("处理文本输入")
            text_features = self.text_encoder.encode(text)
        
        # 处理图像输入
        if image:
            logger.info("处理图像输入")
            image_features = self.image_encoder.encode(image)
        
        # 处理语音输入
        if audio:
            logger.info("处理语音输入")
            audio_features = self.audio_encoder.encode(audio)
        
        # 多模态融合
        logger.info("执行多模态融合")
        fused_features = self.fusion_module.fuse(
            text_features=text_features,
            image_features=image_features,
            audio_features=audio_features
        )
        
        # 解码生成输出
        logger.info("生成输出")
        output = self.decoder_module.decode(fused_features, **kwargs)
        
        return output
    
    def save(self, path: str):
        """
        保存模型
        
        参数:
            path: 保存路径
        """
        logger.info(f"保存模型到 {path}")
        os.makedirs(path, exist_ok=True)
        
        # 保存配置
        self.config.save(os.path.join(path, "config.json"))
        
        # 保存模型权重
        model_state = {
            "text_encoder": self.text_encoder.state_dict() if hasattr(self.text_encoder, "state_dict") else None,
            "image_encoder": self.image_encoder.state_dict() if hasattr(self.image_encoder, "state_dict") else None,
            "audio_encoder": self.audio_encoder.state_dict() if hasattr(self.audio_encoder, "state_dict") else None,
            "fusion_module": self.fusion_module.state_dict(),
            "decoder_module": self.decoder_module.state_dict()
        }
        
        torch.save(model_state, os.path.join(path, "model.pt"))
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        """
        从保存的文件加载模型
        
        参数:
            path: 模型路径
            device: 模型运行的设备
            
        返回:
            MultiSpaceAI 实例
        """
        logger.info(f"从 {path} 加载模型")
        
        # 加载配置
        config_path = os.path.join(path, "config.json")
        instance = cls(config_path=config_path, device=device)
        
        # 加载模型权重
        model_path = os.path.join(path, "model.pt")
        model_state = torch.load(model_path, map_location=instance.device)
        
        # 加载各模块的权重
        if hasattr(instance.text_encoder, "load_state_dict") and model_state["text_encoder"]:
            instance.text_encoder.load_state_dict(model_state["text_encoder"])
            
        if hasattr(instance.image_encoder, "load_state_dict") and model_state["image_encoder"]:
            instance.image_encoder.load_state_dict(model_state["image_encoder"])
            
        if hasattr(instance.audio_encoder, "load_state_dict") and model_state["audio_encoder"]:
            instance.audio_encoder.load_state_dict(model_state["audio_encoder"])
            
        instance.fusion_module.load_state_dict(model_state["fusion_module"])
        instance.decoder_module.load_state_dict(model_state["decoder_module"])
        
        return instance 