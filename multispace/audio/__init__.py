"""
语音处理模块
=========

该模块包含用于处理语音数据的编码器类。
"""

from .audio_encoder import AudioEncoder
from .whisper_audio_encoder import WhisperAudioEncoder

__all__ = [
    'AudioEncoder',
    'WhisperAudioEncoder'
] 