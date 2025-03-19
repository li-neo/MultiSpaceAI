"""
配置模块
=======

该模块包含模型配置相关的类和函数。
"""

from .model_config import ModelConfig, TextEncoderConfig, ImageEncoderConfig, AudioEncoderConfig, FusionConfig, DecoderConfig

__all__ = [
    'ModelConfig',
    'TextEncoderConfig',
    'ImageEncoderConfig',
    'AudioEncoderConfig',
    'FusionConfig',
    'DecoderConfig'
] 