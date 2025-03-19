"""
文本编码器模块
===========

该模块包含用于处理和编码文本输入的类。
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from transformers import AutoTokenizer, AutoModel

from ..config import TextEncoderConfig
from ..utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)


class TextEncoder(nn.Module):
    """
    文本编码器类，用于处理和编码文本输入。
    基于Hugging Face的Transformer模型。
    """
    
    def __init__(self, config: TextEncoderConfig, device: Optional[str] = None):
        """
        初始化文本编码器
        
        参数:
            config: 文本编码器配置
            device: 运行设备
        """
        super().__init__()
        self.config = config
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化分词器和模型
        logger.info(f"加载文本编码器预训练模型: {config.model_name}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # 加载预训练模型
        self.model = AutoModel.from_pretrained(config.model_name)
        
        # 输出维度适配层
        if self.model.config.hidden_size != config.hidden_dim:
            self.output_adapter = nn.Linear(self.model.config.hidden_size, config.hidden_dim)
        else:
            self.output_adapter = nn.Identity()
        
        # 移动模型到指定设备
        self.to(self.device)
        
        logger.info("文本编码器初始化完成")
    
    def preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        """
        对文本进行预处理
        
        参数:
            text: 输入文本
            
        返回:
            预处理后的输入，包含输入ID和注意力掩码
        """
        # 使用分词器处理文本
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 移动到指定设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            
        返回:
            文本特征表示
        """
        # 获取模型输出
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 获取最后一层的隐藏状态
        last_hidden_state = outputs.last_hidden_state
        
        # 使用[CLS]标记的表示作为整个序列的表示
        # 对应位置为 last_hidden_state 的第一个标记
        pooled_output = last_hidden_state[:, 0, :]
        
        # 应用输出适配层
        features = self.output_adapter(pooled_output)
        
        return features
    
    def encode(self, text: Union[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        编码文本
        
        参数:
            text: 输入文本或已预处理的输入
            
        返回:
            文本特征表示
        """
        self.eval()  # 设置为评估模式
        
        with torch.no_grad():
            # 如果输入是文本字符串，进行预处理
            if isinstance(text, str):
                inputs = self.preprocess(text)
            else:
                inputs = text
            
            # 前向传播
            features = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        
        return features 