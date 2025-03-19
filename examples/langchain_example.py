#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain 集成示例
================

展示如何使用MultiSpaceAI与LangChain集成，实现更强大的多模态处理能力。
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入LangChain相关模块
try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.memory import ConversationBufferMemory
except ImportError:
    print("请先安装LangChain相关依赖:")
    print("pip install langchain openai")
    sys.exit(1)

# 导入MultiSpaceAI的LangChain集成
from src.langchain_integration import (
    MultiSpaceAITool,
    MultiSpaceAIChain,
    MultiSpaceAIAgent,
    MultiSpaceAIRetrievalChain
)


def setup_llm(model_name: str = "gpt-3.5-turbo"):
    """设置LLM模型

    Args:
        model_name: 模型名称

    Returns:
        LLM模型实例
    """
    # 检查是否设置了OpenAI API密钥
    if not os.environ.get("OPENAI_API_KEY"):
        print("警告: 未设置OPENAI_API_KEY环境变量")
        print("您可以通过以下命令设置环境变量:")
        print("export OPENAI_API_KEY=your_api_key_here")
        print("或者在运行脚本时传入API密钥")
    
    # 初始化Chat模型
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=0.7,
        max_tokens=1000
    )
    
    return llm


def run_multispace_tool_example(args):
    """运行MultiSpaceAI工具示例

    Args:
        args: 命令行参数
    """
    print("\n===== 运行MultiSpaceAI工具示例 =====")
    
    # 设置LLM
    llm = setup_llm(args.model)
    
    # 初始化MultiSpaceAI工具
    multispace_tool = MultiSpaceAITool(
        config_path=args.config,
        text_encoder=args.text_encoder,
        image_encoder=args.image_encoder,
        audio_encoder=args.audio_encoder,
        device=args.device
    )
    
    # 运行MultiSpaceAI工具
    if args.image:
        print(f"\n处理图像: {args.image}")
        result = multispace_tool.run(
            query=args.query,
            image=args.image
        )
    elif args.audio:
        print(f"\n处理音频: {args.audio}")
        result = multispace_tool.run(
            query=args.query,
            audio=args.audio
        )
    else:
        print(f"\n处理文本: {args.query}")
        result = multispace_tool.run(
            query=args.query
        )
    
    print("\n生成结果:")
    print("-" * 50)
    print(result)
    print("-" * 50)


def run_multispace_chain_example(args):
    """运行MultiSpaceAI链示例

    Args:
        args: 命令行参数
    """
    print("\n===== 运行MultiSpaceAI链示例 =====")
    
    # 设置LLM
    llm = setup_llm(args.model)
    
    # 创建提示模板
    template = """你是一个多模态助手，能够处理文本、图像和音频输入。
    
{multimodal_context}

用户问题: {query}

请提供详细且有用的回答:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["multimodal_context", "query"]
    )
    
    # 初始化对话内存
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    # 初始化MultiSpaceAI链
    multispace_chain = MultiSpaceAIChain(
        llm=llm,
        prompt=prompt,
        config_path=args.config,
        text_encoder=args.text_encoder,
        image_encoder=args.image_encoder,
        audio_encoder=args.audio_encoder,
        device=args.device,
        memory=memory
    )
    
    # 准备输入
    inputs = {"query": args.query}
    
    if args.image:
        print(f"\n处理图像: {args.image}")
        inputs["image"] = args.image
    
    if args.audio:
        print(f"\n处理音频: {args.audio}")
        inputs["audio"] = args.audio
    
    # 运行MultiSpaceAI链
    result = multispace_chain(inputs)
    
    print("\n生成结果:")
    print("-" * 50)
    print(result["text"])
    print("-" * 50)


def run_multispace_agent_example(args):
    """运行MultiSpaceAI代理示例

    Args:
        args: 命令行参数
    """
    print("\n===== 运行MultiSpaceAI代理示例 =====")
    
    # 设置LLM
    llm = setup_llm(args.model)
    
    # 初始化对话内存
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # 初始化MultiSpaceAI代理
    multispace_agent = MultiSpaceAIAgent(
        llm=llm,
        config_path=args.config,
        text_encoder=args.text_encoder,
        image_encoder=args.image_encoder,
        audio_encoder=args.audio_encoder,
        device=args.device,
        verbose=True,
        memory=memory
    )
    
    # 构建查询
    if args.image:
        query = f"{args.query} 图像路径: {args.image}"
    elif args.audio:
        query = f"{args.query} 音频路径: {args.audio}"
    else:
        query = args.query
    
    # 运行MultiSpaceAI代理
    result = multispace_agent.run(query)
    
    print("\n代理结果:")
    print("-" * 50)
    print(result)
    print("-" * 50)


def run_multispace_retrieval_example(args):
    """运行MultiSpaceAI检索链示例

    Args:
        args: 命令行参数
    """
    print("\n===== 运行MultiSpaceAI检索链示例 =====")
    
    # 设置LLM
    llm = setup_llm(args.model)
    
    # 初始化MultiSpaceAI检索链
    retrieval_chain = MultiSpaceAIRetrievalChain(
        llm=llm,
        config_path=args.config,
        text_encoder=args.text_encoder,
        image_encoder=args.image_encoder,
        audio_encoder=args.audio_encoder,
        device=args.device,
        return_source_documents=True
    )
    
    # 如果指定了数据目录，则导入文档
    if args.data_dir:
        print(f"\n从目录导入文档: {args.data_dir}")
        retrieval_chain.ingest_from_directory(
            directory=args.data_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        print("文档导入完成")
    
    # 运行检索链
    result = retrieval_chain.run(
        query=args.query,
        image=args.image,
        audio=args.audio
    )
    
    print("\n检索结果:")
    print("-" * 50)
    print(result["result"])
    print("-" * 50)
    
    # 显示源文档
    if result.get("source_documents") and len(result["source_documents"]) > 0:
        print("\n源文档:")
        for i, doc in enumerate(result["source_documents"]):
            print(f"\n文档 {i+1}:")
            print(f"内容: {doc.page_content[:100]}...")
            print(f"来源: {doc.metadata.get('source', 'Unknown')}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MultiSpaceAI LangChain集成示例")
    
    # 基本参数
    parser.add_argument("--query", type=str, default="描述这个输入", help="查询文本")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--device", type=str, help="运行设备")
    
    # 输入参数
    parser.add_argument("--image", type=str, help="图像文件路径")
    parser.add_argument("--audio", type=str, help="音频文件路径")
    
    # LLM参数
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="LLM模型名称")
    parser.add_argument("--api-key", type=str, help="OpenAI API密钥")
    
    # 编码器参数
    parser.add_argument("--text-encoder", type=str, default="custom", choices=["custom", "deepseek-api"], help="文本编码器类型")
    parser.add_argument("--image-encoder", type=str, default="custom", choices=["custom", "diffusion-api"], help="图像编码器类型")
    parser.add_argument("--audio-encoder", type=str, default="custom", choices=["custom", "whisper-api"], help="音频编码器类型")
    
    # 检索链参数
    parser.add_argument("--data-dir", type=str, help="数据目录路径")
    parser.add_argument("--chunk-size", type=int, default=1000, help="文本块大小")
    parser.add_argument("--chunk-overlap", type=int, default=0, help="文本块重叠大小")
    
    # 示例类型
    parser.add_argument("--example", type=str, default="tool", 
                        choices=["tool", "chain", "agent", "retrieval", "all"],
                        help="要运行的示例类型")
    
    args = parser.parse_args()
    
    # 如果提供了API密钥，设置环境变量
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    # 根据示例类型运行相应示例
    if args.example == "tool" or args.example == "all":
        run_multispace_tool_example(args)
    
    if args.example == "chain" or args.example == "all":
        run_multispace_chain_example(args)
    
    if args.example == "agent" or args.example == "all":
        run_multispace_agent_example(args)
    
    if args.example == "retrieval" or args.example == "all":
        if not args.data_dir and args.example != "all":
            print("警告: 未提供数据目录，检索示例可能无法正常工作")
        run_multispace_retrieval_example(args)


if __name__ == "__main__":
    main() 