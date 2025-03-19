"""
Whisper API 语音编码器模块
========================

该模块包含使用OpenAI Whisper API进行语音编码的类。
"""

import torch
import numpy as np
import os
import librosa
import requests
import json
import base64
from typing import Optional, Dict, Any, Union

from ..utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)


class WhisperAudioEncoder:
    """
    Whisper API 语音编码器类，使用OpenAI的Whisper API服务对语音进行编码。
    """
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None, embedding_dim: int = 1024):
        """
        初始化Whisper API语音编码器
        
        参数:
            api_key: OpenAI API密钥
            api_url: OpenAI API URL，如果为None则使用默认URL
            embedding_dim: 嵌入向量维度
        """
        self.api_key = api_key
        self.api_url = api_url or "https://api.openai.com/v1/audio/transcriptions"
        self.embedding_dim = embedding_dim
        
        # 检查API密钥
        if not self.api_key:
            logger.warning("未提供OpenAI API密钥，可能导致API调用失败")
        
        logger.info("Whisper API语音编码器初始化完成")
    
    def preprocess_audio(self, audio: Union[str, np.ndarray], sample_rate: int = 16000) -> str:
        """
        预处理音频并保存为临时文件
        
        参数:
            audio: 输入语音，可以是音频文件路径或numpy数组
            sample_rate: 采样率
            
        返回:
            临时音频文件路径
        """
        # 如果已经是文件路径，并且是支持的格式，直接返回
        if isinstance(audio, str) and os.path.exists(audio):
            _, ext = os.path.splitext(audio)
            if ext.lower() in ['.mp3', '.wav', '.m4a']:
                return audio
        
        # 如果是numpy数组或需要转换格式的文件路径
        if isinstance(audio, str) and os.path.exists(audio):
            waveform, _ = librosa.load(audio, sr=sample_rate)
        else:
            waveform = audio
        
        # 创建临时文件
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_file_path = temp_file.name
        temp_file.close()
        
        # 保存为MP3格式
        import soundfile as sf
        sf.write(temp_file_path, waveform, sample_rate, format='mp3')
        
        return temp_file_path
    
    def encode(self, audio: Union[str, np.ndarray]) -> torch.Tensor:
        """
        使用Whisper API编码语音
        
        参数:
            audio: 输入语音，可以是音频文件路径或numpy数组
            
        返回:
            语音特征表示
        """
        # 记录日志
        logger.info("使用Whisper API编码语音")
        
        # 预处理音频
        audio_file_path = self.preprocess_audio(audio)
        
        # 构建请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            # 准备文件和表单数据
            with open(audio_file_path, 'rb') as audio_file:
                files = {
                    'file': (os.path.basename(audio_file_path), audio_file, 'audio/mpeg')
                }
                
                data = {
                    'model': 'whisper-1',
                    'response_format': 'verbose_json'
                }
                
                # 发送请求
                response = requests.post(self.api_url, headers=headers, files=files, data=data)
                response.raise_for_status()  # 如果请求失败，抛出异常
                
                # 解析响应
                result = response.json()
                
                # Whisper API不直接返回嵌入向量，而是返回转录文本和其他元数据
                # 我们可以使用转录文本的某些特征作为音频的表示
                
                # 例如，我们可以分析转录文本中的段落、置信度等
                transcription = result.get('text', '')
                
                # 清理临时文件
                if audio_file_path != audio and os.path.exists(audio_file_path):
                    os.unlink(audio_file_path)
                
                # 由于Whisper API不提供嵌入向量，我们可以采用以下策略：
                # 1. 使用转录文本的长度、段落数量、置信度等构建一个特征向量
                # 2. 或者使用转录文本通过额外的文本编码器生成嵌入
                # 3. 这里我们简单地使用一个随机向量作为占位符，实际应用中应替换为更有意义的表示
                
                # 创建一个伪随机但确定性的向量，基于转录文本的哈希
                import hashlib
                seed = int(hashlib.md5(transcription.encode()).hexdigest(), 16) % (10 ** 8)
                np.random.seed(seed)
                embedding = np.random.randn(self.embedding_dim).astype(np.float32)
                
                # 返回嵌入向量
                return torch.tensor(embedding, dtype=torch.float)
                
        except Exception as e:
            logger.error(f"Whisper API调用失败: {str(e)}")
            
            # 清理临时文件
            if audio_file_path != audio and os.path.exists(audio_file_path):
                os.unlink(audio_file_path)
            
            # 返回一个全零的向量作为备用
            return torch.zeros(self.embedding_dim, dtype=torch.float)
    
    def batch_encode(self, audios: list) -> torch.Tensor:
        """
        批量编码语音
        
        参数:
            audios: 语音列表
            
        返回:
            批量语音特征表示
        """
        # 记录日志
        logger.info(f"使用Whisper API批量编码{len(audios)}个语音")
        
        # 创建一个空列表来存储嵌入向量
        embeddings = []
        
        # 依次处理每个语音
        for audio in audios:
            embedding = self.encode(audio)
            embeddings.append(embedding)
        
        # 堆叠所有嵌入向量
        if embeddings:
            return torch.stack(embeddings)
        else:
            # 如果没有嵌入向量，返回一个空的张量
            return torch.zeros((0, self.embedding_dim), dtype=torch.float) 