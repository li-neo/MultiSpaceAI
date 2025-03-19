"""
模型配置类
========

该模块包含模型各个组件的配置类。
"""

import os
import json
from typing import Optional, Dict, Any, List


class BaseConfig:
    """所有配置类的基类"""
    
    def __init__(self, **kwargs):
        """使用提供的参数初始化配置"""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """从字典创建配置实例"""
        return cls(**config_dict)


class TextEncoderConfig(BaseConfig):
    """文本编码器配置"""
    
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        embedding_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        num_attention_heads: int = 8,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        初始化文本编码器配置
        
        参数:
            model_name: 预训练模型名称，用于加载Hugging Face模型
            embedding_dim: 词嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: Transformer层数
            num_attention_heads: 注意力头数
            max_seq_length: 最大序列长度
            dropout: Dropout比例
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        super().__init__(**kwargs)


class ImageEncoderConfig(BaseConfig):
    """图像编码器配置"""
    
    def __init__(
        self,
        model_name: str = "vit-base-patch16-224",
        image_size: int = 224,
        patch_size: int = 16,
        embedding_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        初始化图像编码器配置
        
        参数:
            model_name: 预训练模型名称
            image_size: 输入图像大小
            patch_size: 图像分块大小
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: Transformer层数
            num_attention_heads: 注意力头数
            dropout: Dropout比例
        """
        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        super().__init__(**kwargs)


class AudioEncoderConfig(BaseConfig):
    """语音编码器配置"""
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        sample_rate: int = 16000,
        embedding_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        max_audio_length: int = 30,  # 单位：秒
        dropout: float = 0.1,
        **kwargs
    ):
        """
        初始化语音编码器配置
        
        参数:
            model_name: 预训练模型名称
            sample_rate: 音频采样率
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: Transformer层数
            num_attention_heads: 注意力头数
            max_audio_length: 最大音频长度（秒）
            dropout: Dropout比例
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.max_audio_length = max_audio_length
        self.dropout = dropout
        super().__init__(**kwargs)


class FusionConfig(BaseConfig):
    """多模态融合模块配置"""
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_attention_heads: int = 16,
        fusion_type: str = "cross_attention",  # 可选：cross_attention, concat, sum
        use_modal_adapters: bool = True,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        初始化多模态融合模块配置
        
        参数:
            hidden_dim: 隐藏层维度
            num_layers: Transformer层数
            num_attention_heads: 注意力头数
            fusion_type: 融合类型，可选项：cross_attention, concat, sum
            use_modal_adapters: 是否使用模态适配器
            dropout: Dropout比例
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.fusion_type = fusion_type
        self.use_modal_adapters = use_modal_adapters
        self.dropout = dropout
        super().__init__(**kwargs)


class DecoderConfig(BaseConfig):
    """解码器模块配置"""
    
    def __init__(
        self,
        model_name: str = "fnlp/bart-base-chinese",
        embedding_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        num_attention_heads: int = 8,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        初始化解码器模块配置
        
        参数:
            model_name: 预训练模型名称
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: Transformer层数
            num_attention_heads: 注意力头数
            max_seq_length: 最大序列长度
            dropout: Dropout比例
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        super().__init__(**kwargs)


class ModelConfig:
    """整体模型配置类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化模型配置
        
        参数:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        # 使用默认配置初始化
        self.text_encoder_config = TextEncoderConfig()
        self.image_encoder_config = ImageEncoderConfig()
        self.audio_encoder_config = AudioEncoderConfig()
        self.fusion_config = FusionConfig()
        self.decoder_config = DecoderConfig()
        
        # 如果提供了配置文件路径，则从配置文件加载
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
    
    def _load_from_file(self, config_path: str):
        """从文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 加载各个模块的配置
        if 'text_encoder_config' in config_dict:
            self.text_encoder_config = TextEncoderConfig.from_dict(config_dict['text_encoder_config'])
        
        if 'image_encoder_config' in config_dict:
            self.image_encoder_config = ImageEncoderConfig.from_dict(config_dict['image_encoder_config'])
        
        if 'audio_encoder_config' in config_dict:
            self.audio_encoder_config = AudioEncoderConfig.from_dict(config_dict['audio_encoder_config'])
        
        if 'fusion_config' in config_dict:
            self.fusion_config = FusionConfig.from_dict(config_dict['fusion_config'])
        
        if 'decoder_config' in config_dict:
            self.decoder_config = DecoderConfig.from_dict(config_dict['decoder_config'])
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """将配置转换为字典"""
        return {
            'text_encoder_config': self.text_encoder_config.to_dict(),
            'image_encoder_config': self.image_encoder_config.to_dict(),
            'audio_encoder_config': self.audio_encoder_config.to_dict(),
            'fusion_config': self.fusion_config.to_dict(),
            'decoder_config': self.decoder_config.to_dict()
        }
    
    def save(self, config_path: str):
        """保存配置到文件"""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2) 