"""
语音编码器模块
===========

该模块包含用于处理和编码语音输入的类。
"""

import torch
import torch.nn as nn
import numpy as np
import os
import librosa
from typing import Optional, Dict, Any, Union
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from ..config import AudioEncoderConfig
from ..utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)


class AudioEncoder(nn.Module):
    """
    语音编码器类，用于处理和编码语音输入。
    基于Hugging Face的Wav2Vec2模型。
    """
    
    def __init__(self, config: AudioEncoderConfig, device: Optional[str] = None):
        """
        初始化语音编码器
        
        参数:
            config: 语音编码器配置
            device: 运行设备
        """
        super().__init__()
        self.config = config
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化处理器和模型
        logger.info(f"加载语音编码器预训练模型: {config.model_name}")
        
        # 加载处理器
        self.processor = Wav2Vec2Processor.from_pretrained(config.model_name)
        
        # 加载预训练模型
        self.model = Wav2Vec2Model.from_pretrained(config.model_name)
        
        # 输出维度适配层
        if self.model.config.hidden_size != config.hidden_dim:
            self.output_adapter = nn.Linear(self.model.config.hidden_size, config.hidden_dim)
        else:
            self.output_adapter = nn.Identity()
        
        # 移动模型到指定设备
        self.to(self.device)
        
        logger.info("语音编码器初始化完成")
    
    def preprocess(self, audio: Union[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        对语音进行预处理
        
        参数:
            audio: 输入语音，可以是音频文件路径或numpy数组
            
        返回:
            预处理后的输入
        """
        # 如果是路径，加载音频
        if isinstance(audio, str):
            if not os.path.exists(audio):
                raise FileNotFoundError(f"音频文件不存在: {audio}")
            
            # 使用librosa加载音频，调整采样率
            waveform, sample_rate = librosa.load(audio, sr=self.config.sample_rate)
            
        # 如果已经是numpy数组，确保采样率正确
        else:
            waveform = audio
        
        # 限制音频长度
        max_length = self.config.max_audio_length * self.config.sample_rate
        if len(waveform) > max_length:
            logger.warning(f"音频长度超过{self.config.max_audio_length}秒，将被截断")
            waveform = waveform[:max_length]
        
        # 使用处理器处理音频
        inputs = self.processor(
            waveform,
            sampling_rate=self.config.sample_rate,
            return_tensors="pt"
        )
        
        # 移动到指定设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            input_values: 输入值
            attention_mask: 注意力掩码（可选）
            
        返回:
            语音特征表示
        """
        # 获取模型输出
        outputs = self.model(input_values=input_values, attention_mask=attention_mask)
        
        # 获取最后一层的隐藏状态
        last_hidden_state = outputs.last_hidden_state
        
        # 对时间维度进行平均池化，得到整个音频的表示
        if attention_mask is not None:
            # 使用注意力掩码进行池化
            pooled_output = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            # 简单的平均池化
            pooled_output = last_hidden_state.mean(dim=1)
        
        # 应用输出适配层
        features = self.output_adapter(pooled_output)
        
        return features
    
    def encode(self, audio: Union[str, np.ndarray]) -> torch.Tensor:
        """
        编码语音
        
        参数:
            audio: 输入语音，可以是音频文件路径或numpy数组
            
        返回:
            语音特征表示
        """
        self.eval()  # 设置为评估模式
        
        with torch.no_grad():
            # 预处理语音
            inputs = self.preprocess(audio)
            
            # 前向传播
            features = self.forward(
                input_values=inputs["input_values"],
                attention_mask=inputs.get("attention_mask")
            )
        
        return features 