"""
解码器模块
========

该模块包含用于将融合特征解码生成输出的类。
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union
from transformers import BartForConditionalGeneration, BartTokenizer

from ..config import DecoderConfig
from ..utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)


class DecoderModule(nn.Module):
    """
    解码器模块，用于将融合的特征表示解码为文本输出。
    基于Hugging Face的BART模型。
    """
    
    def __init__(self, config: DecoderConfig, device: Optional[str] = None):
        """
        初始化解码器模块
        
        参数:
            config: 解码器配置
            device: 运行设备
        """
        super().__init__()
        self.config = config
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化分词器和模型
        logger.info(f"加载解码器预训练模型: {config.model_name}")
        
        # 加载分词器
        self.tokenizer = BartTokenizer.from_pretrained(config.model_name)
        
        # 加载预训练模型
        self.model = BartForConditionalGeneration.from_pretrained(config.model_name)
        
        # 特征投影层，将融合特征映射到解码器的隐藏维度
        self.feature_projector = nn.Linear(config.hidden_dim, self.model.config.d_model)
        
        # 自回归生成头
        self.lm_head = nn.Linear(self.model.config.d_model, len(self.tokenizer), bias=False)
        
        # 移动模型到指定设备
        self.to(self.device)
        
        logger.info("解码器模块初始化完成")
    
    def prepare_inputs_for_generation(self, 
                                     fused_features: torch.Tensor,
                                     decoder_input_ids: torch.Tensor = None,
                                     attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        准备用于生成的输入
        
        参数:
            fused_features: 融合特征
            decoder_input_ids: 解码器输入ID
            attention_mask: 注意力掩码
            
        返回:
            用于生成的输入字典
        """
        # 投影融合特征
        encoder_hidden_states = self.feature_projector(fused_features).unsqueeze(1)
        
        # 如果没有提供解码器输入ID，创建一个包含开始标记的序列
        if decoder_input_ids is None:
            decoder_input_ids = torch.tensor([[self.tokenizer.bos_token_id]]).to(self.device)
        
        # 准备输入字典
        model_inputs = {
            "encoder_outputs": [encoder_hidden_states],  # 使用融合特征作为编码器输出
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask if attention_mask is not None else torch.ones_like(decoder_input_ids),
            "use_cache": True
        }
        
        return model_inputs
    
    def forward(self, 
               encoder_hidden_states: torch.Tensor,
               decoder_input_ids: torch.Tensor,
               decoder_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            encoder_hidden_states: 编码器隐藏状态，即融合特征
            decoder_input_ids: 解码器输入ID
            decoder_attention_mask: 解码器注意力掩码
            
        返回:
            解码器输出
        """
        # 使用BART模型的解码器部分
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=decoder_attention_mask
        )
        
        # 获取解码器输出的隐藏状态
        hidden_states = decoder_outputs[0]
        
        # 应用语言模型头
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def generate(self, 
                 fused_features: torch.Tensor, 
                 max_length: int = 50,
                 num_beams: int = 4,
                 early_stopping: bool = True,
                 **kwargs) -> List[str]:
        """
        生成文本
        
        参数:
            fused_features: 融合特征
            max_length: 生成的最大长度
            num_beams: 束搜索的束数
            early_stopping: 是否提前停止生成
            **kwargs: 其他生成参数
            
        返回:
            生成的文本列表
        """
        # 确保模型处于评估模式
        self.eval()
        
        # 投影融合特征
        encoder_hidden_states = self.feature_projector(fused_features).unsqueeze(1)
        
        # 使用BART模型的生成方法
        with torch.no_grad():
            # 设置编码器输出
            encoder_outputs = [encoder_hidden_states]
            
            # 生成序列
            output_ids = self.model.generate(
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                **kwargs
            )
            
            # 解码生成的序列
            outputs = []
            for ids in output_ids:
                # 将ID转换为文本
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                outputs.append(text)
        
        return outputs
    
    def decode(self, fused_features: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        解码融合特征
        
        参数:
            fused_features: 融合特征
            **kwargs: 其他参数
            
        返回:
            包含生成结果的字典
        """
        # 生成文本
        generated_texts = self.generate(fused_features, **kwargs)
        
        # 返回结果
        result = {
            "generated_text": generated_texts[0] if generated_texts else "",  # 取第一个生成的文本
            "all_texts": generated_texts,                                    # 所有生成的文本
            "raw_features": fused_features.cpu().numpy().tolist()            # 原始特征
        }
        
        return result 