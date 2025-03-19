#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiSpaceAI LangChain 集成
=========================

将MultiSpaceAI与LangChain集成，实现更强大的多模态处理能力。
"""

import os
import json
from typing import Any, Dict, List, Optional, Union

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

# 引入项目根目录
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入MultiSpaceAI
from src.multispace import MultiSpaceAI


class MultiSpaceAITool(BaseTool):
    """将MultiSpaceAI封装为LangChain工具"""
    
    name = "multispace_ai"
    description = "通过MultiSpaceAI处理多模态输入，支持文本、图像和音频输入"
    
    multispace_model: MultiSpaceAI
    return_direct: bool = False
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        text_encoder: str = "custom",
        image_encoder: str = "custom",
        audio_encoder: str = "custom",
        device: Optional[str] = None,
        return_direct: bool = False
    ):
        """初始化MultiSpaceAI工具

        Args:
            config_path: 配置文件路径
            text_encoder: 文本编码器类型
            image_encoder: 图像编码器类型
            audio_encoder: 音频编码器类型
            device: 运行设备
            return_direct: 是否直接返回结果
        """
        super().__init__()
        
        # 初始化MultiSpaceAI模型
        self.multispace_model = MultiSpaceAI(
            config_path=config_path,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            audio_encoder=audio_encoder,
            device=device
        )
        
        self.return_direct = return_direct
    
    def _run(
        self, 
        query: str, 
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        max_length: int = 50,
        num_beams: int = 4,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """运行MultiSpaceAI处理

        Args:
            query: 查询
            text: 文本输入
            image: 图像文件路径
            audio: 音频文件路径
            max_length: 生成的最大长度
            num_beams: 束搜索的束数
            run_manager: 回调管理器

        Returns:
            处理结果
        """
        # 处理输入
        result = self.multispace_model.process(
            text=text or query,
            image=image,
            audio=audio,
            max_length=max_length,
            num_beams=num_beams
        )
        
        # 返回生成文本
        return result["generated_text"]
    
    async def _arun(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        max_length: int = 50,
        num_beams: int = 4,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """异步运行MultiSpaceAI处理

        Args:
            query: 查询
            text: 文本输入
            image: 图像文件路径
            audio: 音频文件路径
            max_length: 生成的最大长度
            num_beams: 束搜索的束数
            run_manager: 回调管理器

        Returns:
            处理结果
        """
        # 处理输入
        result = self.multispace_model.process(
            text=text or query,
            image=image,
            audio=audio,
            max_length=max_length,
            num_beams=num_beams
        )
        
        # 返回生成文本
        return result["generated_text"]


class MultiSpaceAIChain(LLMChain):
    """MultiSpaceAI LangChain 链"""
    
    multispace_model: MultiSpaceAI
    
    def __init__(
        self,
        llm: Any,
        prompt: PromptTemplate,
        config_path: Optional[str] = None,
        text_encoder: str = "custom",
        image_encoder: str = "custom",
        audio_encoder: str = "custom",
        device: Optional[str] = None,
        memory: Optional[ConversationBufferMemory] = None,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
    ):
        """初始化MultiSpaceAI链

        Args:
            llm: 大语言模型
            prompt: 提示模板
            config_path: 配置文件路径
            text_encoder: 文本编码器类型
            image_encoder: 图像编码器类型
            audio_encoder: 音频编码器类型
            device: 运行设备
            memory: 对话内存
            callbacks: 回调处理器列表
        """
        super().__init__(
            llm=llm,
            prompt=prompt,
            memory=memory,
            callbacks=callbacks
        )
        
        # 初始化MultiSpaceAI模型
        self.multispace_model = MultiSpaceAI(
            config_path=config_path,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            audio_encoder=audio_encoder,
            device=device
        )
    
    def process_multimodal(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        max_length: int = 50,
        num_beams: int = 4,
    ) -> Dict[str, Any]:
        """处理多模态输入

        Args:
            text: 文本输入
            image: 图像文件路径
            audio: 音频文件路径
            max_length: 生成的最大长度
            num_beams: 束搜索的束数

        Returns:
            处理结果
        """
        # 处理输入
        result = self.multispace_model.process(
            text=text,
            image=image,
            audio=audio,
            max_length=max_length,
            num_beams=num_beams
        )
        
        return result
    
    def __call__(
        self,
        inputs: Dict[str, Any],
        return_only_outputs: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """运行链

        Args:
            inputs: 输入参数
            return_only_outputs: 是否只返回输出
            **kwargs: 额外参数

        Returns:
            链运行结果
        """
        # 处理多模态输入
        if any(k in inputs for k in ["image", "audio"]):
            multimodal_result = self.process_multimodal(
                text=inputs.get("text", inputs.get("query", None)),
                image=inputs.get("image"),
                audio=inputs.get("audio"),
                max_length=inputs.get("max_length", 50),
                num_beams=inputs.get("num_beams", 4),
            )
            
            # 将多模态处理结果添加到输入
            inputs["multimodal_context"] = multimodal_result["generated_text"]
        
        # 调用父类的__call__方法
        return super().__call__(inputs, return_only_outputs, **kwargs)


class MultiSpaceAIAgent:
    """MultiSpaceAI LangChain 代理"""
    
    def __init__(
        self,
        llm: Any,
        config_path: Optional[str] = None,
        text_encoder: str = "custom",
        image_encoder: str = "custom",
        audio_encoder: str = "custom",
        device: Optional[str] = None,
        verbose: bool = False,
        memory: Optional[ConversationBufferMemory] = None,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
    ):
        """初始化MultiSpaceAI代理

        Args:
            llm: 大语言模型
            config_path: 配置文件路径
            text_encoder: 文本编码器类型
            image_encoder: 图像编码器类型
            audio_encoder: 音频编码器类型
            device: 运行设备
            verbose: 是否输出详细信息
            memory: 对话内存
            callbacks: 回调处理器列表
        """
        # 初始化MultiSpaceAI工具
        self.multispace_tool = MultiSpaceAITool(
            config_path=config_path,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            audio_encoder=audio_encoder,
            device=device
        )
        
        # 工具列表
        tools = [self.multispace_tool]
        
        # 初始化代理
        self.agent = initialize_agent(
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            tools=tools,
            llm=llm,
            verbose=verbose,
            memory=memory,
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="generate",
            callbacks=callbacks
        )
    
    def run(self, query: str, **kwargs) -> str:
        """运行代理

        Args:
            query: 查询
            **kwargs: 额外参数

        Returns:
            代理运行结果
        """
        return self.agent.run(query, **kwargs)
    
    async def arun(self, query: str, **kwargs) -> str:
        """异步运行代理

        Args:
            query: 查询
            **kwargs: 额外参数

        Returns:
            代理运行结果
        """
        return await self.agent.arun(query, **kwargs) 