"""
多模态融合模块
===========

该模块包含用于融合多种模态特征的类，包括交叉注意力融合等方法。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple
import math

from ..config import FusionConfig
from ..utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)


class MultiheadCrossAttention(nn.Module):
    """
    多头交叉注意力模块，用于两种模态之间的特征融合。
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        """
        初始化多头交叉注意力模块
        
        参数:
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout比例
        """
        super().__init__()
        
        # 确保hidden_dim可以被num_heads整除
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 定义投影矩阵
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            query: 查询张量，形状为[batch_size, seq_len_q, hidden_dim]
            key: 键张量，形状为[batch_size, seq_len_k, hidden_dim]
            value: 值张量，形状为[batch_size, seq_len_v, hidden_dim]
            key_padding_mask: 键填充掩码，形状为[batch_size, seq_len_k]
            
        返回:
            注意力输出，形状为[batch_size, seq_len_q, hidden_dim]
        """
        batch_size, seq_len_q, _ = query.size()
        _, seq_len_k, _ = key.size()
        
        # 投影并变形
        # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_heads, head_dim]
        q = self.query_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = self.key_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v = self.value_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        
        # 调整维度顺序以便进行批量矩阵乘法
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        # [batch_size, num_heads, seq_len_q, head_dim] x [batch_size, num_heads, head_dim, seq_len_k]
        # -> [batch_size, num_heads, seq_len_q, seq_len_k]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用键填充掩码
        if key_padding_mask is not None:
            # [batch_size, seq_len_k] -> [batch_size, 1, 1, seq_len_k]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        # [batch_size, num_heads, seq_len_q, seq_len_k] x [batch_size, num_heads, seq_len_v, head_dim]
        # -> [batch_size, num_heads, seq_len_q, head_dim]
        output = torch.matmul(attn_weights, v)
        
        # 调整维度顺序并合并头
        # [batch_size, num_heads, seq_len_q, head_dim] -> [batch_size, seq_len_q, num_heads, head_dim]
        output = output.transpose(1, 2)
        
        # [batch_size, seq_len_q, num_heads, head_dim] -> [batch_size, seq_len_q, hidden_dim]
        output = output.reshape(batch_size, seq_len_q, self.hidden_dim)
        
        # 最终投影
        output = self.output_proj(output)
        
        return output


class ModalAdapter(nn.Module):
    """
    模态适配器，用于处理不同模态的特征表示。
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        """
        初始化模态适配器
        
        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            dropout: Dropout比例
        """
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入特征
            
        返回:
            适配后的特征
        """
        # 应用层归一化
        x_norm = self.layer_norm(x)
        
        # 两层前馈网络
        h = F.gelu(self.fc1(x_norm))
        h = self.dropout(h)
        h = self.fc2(h)
        
        return h


class MultimodalFusionLayer(nn.Module):
    """
    多模态融合层，用于多种模态之间的特征融合。
    """
    
    def __init__(self, config: FusionConfig):
        """
        初始化多模态融合层
        
        参数:
            config: 融合模块配置
        """
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # 模态适配器（如果使用）
        if config.use_modal_adapters:
            self.text_adapter = ModalAdapter(config.hidden_dim, config.hidden_dim, config.dropout)
            self.image_adapter = ModalAdapter(config.hidden_dim, config.hidden_dim, config.dropout)
            self.audio_adapter = ModalAdapter(config.hidden_dim, config.hidden_dim, config.dropout)
        
        # 交叉注意力模块
        if config.fusion_type == "cross_attention":
            # 文本-图像交叉注意力
            self.text_image_attn = MultiheadCrossAttention(
                config.hidden_dim, config.num_attention_heads, config.dropout
            )
            # 文本-语音交叉注意力
            self.text_audio_attn = MultiheadCrossAttention(
                config.hidden_dim, config.num_attention_heads, config.dropout
            )
            # 图像-语音交叉注意力
            self.image_audio_attn = MultiheadCrossAttention(
                config.hidden_dim, config.num_attention_heads, config.dropout
            )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(config.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(config.hidden_dim)
        
        # 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, 
                text_features: Optional[torch.Tensor] = None,
                image_features: Optional[torch.Tensor] = None,
                audio_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            text_features: 文本特征，形状为[batch_size, seq_len_text, hidden_dim]或[batch_size, hidden_dim]
            image_features: 图像特征，形状为[batch_size, seq_len_image, hidden_dim]或[batch_size, hidden_dim]
            audio_features: 语音特征，形状为[batch_size, seq_len_audio, hidden_dim]或[batch_size, hidden_dim]
            
        返回:
            融合后的特征，形状为[batch_size, hidden_dim]
        """
        # 检查至少有一种模态的特征
        if text_features is None and image_features is None and audio_features is None:
            raise ValueError("至少需要一种模态的特征")
        
        # 特征列表，用于记录有效的特征
        valid_features = []
        
        # 处理文本特征
        if text_features is not None:
            # 确保特征是3D的
            if text_features.dim() == 2:
                text_features = text_features.unsqueeze(1)  # [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim]
            
            # 应用模态适配器（如果使用）
            if self.config.use_modal_adapters:
                text_features = self.text_adapter(text_features)
            
            valid_features.append(text_features)
        
        # 处理图像特征
        if image_features is not None:
            # 确保特征是3D的
            if image_features.dim() == 2:
                image_features = image_features.unsqueeze(1)
            
            # 应用模态适配器（如果使用）
            if self.config.use_modal_adapters:
                image_features = self.image_adapter(image_features)
            
            valid_features.append(image_features)
        
        # 处理语音特征
        if audio_features is not None:
            # 确保特征是3D的
            if audio_features.dim() == 2:
                audio_features = audio_features.unsqueeze(1)
            
            # 应用模态适配器（如果使用）
            if self.config.use_modal_adapters:
                audio_features = self.audio_adapter(audio_features)
            
            valid_features.append(audio_features)
        
        # 根据融合类型进行处理
        if self.config.fusion_type == "cross_attention":
            # 如果只有一种模态，直接返回
            if len(valid_features) == 1:
                fused_features = valid_features[0]
            else:
                # 多模态交叉注意力
                fused_features = []
                
                # 获取每种模态的特征
                modal_features = {
                    "text": text_features,
                    "image": image_features,
                    "audio": audio_features
                }
                
                # 文本-图像融合
                if text_features is not None and image_features is not None:
                    text_image_fused = self.text_image_attn(text_features, image_features, image_features)
                    fused_features.append(text_image_fused)
                
                # 文本-语音融合
                if text_features is not None and audio_features is not None:
                    text_audio_fused = self.text_audio_attn(text_features, audio_features, audio_features)
                    fused_features.append(text_audio_fused)
                
                # 图像-语音融合
                if image_features is not None and audio_features is not None:
                    image_audio_fused = self.image_audio_attn(image_features, audio_features, audio_features)
                    fused_features.append(image_audio_fused)
                
                # 堆叠并平均所有融合特征
                fused_features = torch.stack(fused_features).mean(dim=0)
        
        elif self.config.fusion_type == "concat":
            # 将所有特征在序列维度上拼接
            fused_features = torch.cat(valid_features, dim=1)
            
        elif self.config.fusion_type == "sum":
            # 将所有特征加和
            # 首先确保所有特征的序列长度相同（取最短的）
            min_length = min(f.size(1) for f in valid_features)
            valid_features = [f[:, :min_length, :] for f in valid_features]
            
            # 加和特征
            fused_features = sum(valid_features)
        
        else:
            raise ValueError(f"不支持的融合类型: {self.config.fusion_type}")
        
        # 层归一化和残差连接
        fused_features = self.layer_norm1(fused_features)
        
        # 应用前馈神经网络
        ffn_output = self.ffn(fused_features)
        fused_features = self.layer_norm2(fused_features + ffn_output)
        
        # 对序列维度进行平均池化，得到固定维度的表示
        fused_features = fused_features.mean(dim=1)
        
        return fused_features


class MultimodalFusionModule(nn.Module):
    """
    多模态融合模块，用于融合多种模态的特征表示。
    """
    
    def __init__(self, config: FusionConfig, device: Optional[str] = None):
        """
        初始化多模态融合模块
        
        参数:
            config: 融合模块配置
            device: 运行设备
        """
        super().__init__()
        self.config = config
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 多层融合层
        self.fusion_layers = nn.ModuleList([
            MultimodalFusionLayer(config) for _ in range(config.num_layers)
        ])
        
        # 移动模型到指定设备
        self.to(self.device)
        
        logger.info("多模态融合模块初始化完成")
    
    def forward(self, 
                text_features: Optional[torch.Tensor] = None,
                image_features: Optional[torch.Tensor] = None,
                audio_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            text_features: 文本特征
            image_features: 图像特征
            audio_features: 语音特征
            
        返回:
            融合后的特征
        """
        # 检查至少有一种模态的特征
        if text_features is None and image_features is None and audio_features is None:
            raise ValueError("至少需要一种模态的特征")
        
        # 移动特征到设备
        if text_features is not None:
            text_features = text_features.to(self.device)
        if image_features is not None:
            image_features = image_features.to(self.device)
        if audio_features is not None:
            audio_features = audio_features.to(self.device)
        
        # 通过多层融合层
        fused_features = None
        for layer in self.fusion_layers:
            fused_features = layer(text_features, image_features, audio_features)
            
            # 更新每种模态的特征，以便下一层使用
            if text_features is not None:
                if text_features.dim() == 2:
                    text_features = fused_features.unsqueeze(1)
                else:
                    # 广播融合特征
                    text_features = text_features + fused_features.unsqueeze(1)
            
            if image_features is not None:
                if image_features.dim() == 2:
                    image_features = fused_features.unsqueeze(1)
                else:
                    image_features = image_features + fused_features.unsqueeze(1)
            
            if audio_features is not None:
                if audio_features.dim() == 2:
                    audio_features = fused_features.unsqueeze(1)
                else:
                    audio_features = audio_features + fused_features.unsqueeze(1)
        
        return fused_features
    
    def fuse(self,
             text_features: Optional[torch.Tensor] = None,
             image_features: Optional[torch.Tensor] = None,
             audio_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        融合多种模态的特征
        
        参数:
            text_features: 文本特征
            image_features: 图像特征
            audio_features: 语音特征
            
        返回:
            融合后的特征
        """
        self.eval()  # 设置为评估模式
        
        with torch.no_grad():
            fused_features = self.forward(text_features, image_features, audio_features)
        
        return fused_features 