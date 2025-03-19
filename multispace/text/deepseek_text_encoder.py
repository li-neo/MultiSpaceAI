"""
DeepSeek API 文本编码器模块
=========================

该模块包含使用DeepSeek API进行文本编码的类。
"""

import torch
import numpy as np
import requests
import json
from typing import Optional, Dict, Any, Union

from ..utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)


class DeepSeekTextEncoder:
    """
    DeepSeek API 文本编码器类，使用DeepSeek的API服务对文本进行编码。
    """
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        初始化DeepSeek API文本编码器
        
        参数:
            api_key: DeepSeek API密钥
            api_url: DeepSeek API URL，如果为None则使用默认URL
        """
        self.api_key = api_key
        self.api_url = api_url or "https://api.deepseek.com/v1/embeddings"
        
        # 检查API密钥
        if not self.api_key:
            logger.warning("未提供DeepSeek API密钥，可能导致API调用失败")
        
        logger.info("DeepSeek API文本编码器初始化完成")
    
    def encode(self, text: str) -> torch.Tensor:
        """
        使用DeepSeek API编码文本
        
        参数:
            text: 输入文本
            
        返回:
            文本特征表示
        """
        # 记录日志
        logger.info("使用DeepSeek API编码文本")
        
        # 构建请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建请求体
        data = {
            "input": text,
            "model": "deepseek-embedding"
        }
        
        try:
            # 发送请求
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()  # 如果请求失败，抛出异常
            
            # 解析响应
            result = response.json()
            
            # 获取嵌入向量
            embedding = result.get("embedding") or result.get("data", [{}])[0].get("embedding")
            
            if not embedding:
                raise ValueError(f"API响应中未找到嵌入向量: {result}")
            
            # 转换为torch.Tensor
            embedding_tensor = torch.tensor(embedding, dtype=torch.float)
            
            return embedding_tensor
            
        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {str(e)}")
            
            # 返回一个全零的向量作为备用
            # 假设嵌入维度为1024
            return torch.zeros(1024, dtype=torch.float)
    
    def batch_encode(self, texts: list) -> torch.Tensor:
        """
        批量编码文本
        
        参数:
            texts: 文本列表
            
        返回:
            批量文本特征表示
        """
        # 记录日志
        logger.info(f"使用DeepSeek API批量编码{len(texts)}个文本")
        
        # 构建请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建请求体
        data = {
            "input": texts,
            "model": "deepseek-embedding"
        }
        
        try:
            # 发送请求
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()  # 如果请求失败，抛出异常
            
            # 解析响应
            result = response.json()
            
            # 获取嵌入向量列表
            embeddings = []
            
            if "data" in result:
                # 新版API格式
                for item in result["data"]:
                    embeddings.append(item.get("embedding", []))
            else:
                # 旧版API格式或其他格式
                embeddings = result.get("embeddings", [])
            
            if not embeddings:
                raise ValueError(f"API响应中未找到嵌入向量: {result}")
            
            # 转换为torch.Tensor
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float)
            
            return embeddings_tensor
            
        except Exception as e:
            logger.error(f"DeepSeek API批量调用失败: {str(e)}")
            
            # 返回一个全零的向量作为备用
            # 假设嵌入维度为1024
            return torch.zeros((len(texts), 1024), dtype=torch.float) 