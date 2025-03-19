#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiSpaceAI LangChain 检索链
===========================

将MultiSpaceAI与LangChain的向量检索功能集成，实现多模态内容的检索和问答。
"""

import os
import json
from typing import Any, Dict, List, Optional, Union

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import Chroma, FAISS
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 引入项目根目录
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入MultiSpaceAI
from src.multispace import MultiSpaceAI
from src.langchain_integration.multispace_chain import MultiSpaceAIChain


class MultiSpaceAIEmbeddings(Embeddings):
    """MultiSpaceAI嵌入类，用于生成文本、图像和音频的嵌入向量"""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        text_encoder: str = "custom",
        image_encoder: str = "custom",
        audio_encoder: str = "custom",
        device: Optional[str] = None,
        embedding_dim: int = 768,
    ):
        """初始化MultiSpaceAI嵌入类

        Args:
            config_path: 配置文件路径
            text_encoder: 文本编码器类型
            image_encoder: 图像编码器类型
            audio_encoder: 音频编码器类型
            device: 运行设备
            embedding_dim: 嵌入维度
        """
        # 初始化MultiSpaceAI模型
        self.multispace_model = MultiSpaceAI(
            config_path=config_path,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            audio_encoder=audio_encoder,
            device=device
        )
        
        self.embedding_dim = embedding_dim
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """生成文档嵌入向量

        Args:
            texts: 文档文本列表

        Returns:
            嵌入向量列表
        """
        return [self.embed_query(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """生成查询嵌入向量

        Args:
            text: 查询文本

        Returns:
            嵌入向量
        """
        # 处理输入
        result = self.multispace_model.process(
            text=text,
            return_embeddings=True  # 假设MultiSpaceAI支持返回嵌入
        )
        
        # 返回嵌入向量
        embeddings = result.get("text_embeddings", None)
        
        # 如果返回的是张量，转换为列表
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        
        return embeddings


class MultiModalDocument(Document):
    """多模态文档类，支持文本、图像和音频内容"""
    
    def __init__(
        self,
        page_content: str,
        metadata: Dict[str, Any] = None,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ):
        """初始化多模态文档

        Args:
            page_content: 文档文本内容
            metadata: 文档元数据
            image_path: 图像文件路径
            audio_path: 音频文件路径
        """
        # 更新元数据
        metadata = metadata or {}
        if image_path:
            metadata["image_path"] = image_path
        if audio_path:
            metadata["audio_path"] = audio_path
        
        super().__init__(page_content=page_content, metadata=metadata)


class MultiSpaceAIRetrievalChain:
    """MultiSpaceAI检索链，用于多模态内容的检索和问答"""
    
    def __init__(
        self,
        llm: Any,
        config_path: Optional[str] = None,
        text_encoder: str = "custom",
        image_encoder: str = "custom",
        audio_encoder: str = "custom",
        device: Optional[str] = None,
        vectorstore: Optional[VectorStore] = None,
        embeddings: Optional[Embeddings] = None,
        memory: Optional[ConversationBufferMemory] = None,
        chain_type: str = "stuff",
        callbacks: Optional[List[BaseCallbackHandler]] = None,
        return_source_documents: bool = False,
    ):
        """初始化MultiSpaceAI检索链

        Args:
            llm: 大语言模型
            config_path: 配置文件路径
            text_encoder: 文本编码器类型
            image_encoder: 图像编码器类型
            audio_encoder: 音频编码器类型
            device: 运行设备
            vectorstore: 向量存储
            embeddings: 嵌入模型
            memory: 对话内存
            chain_type: 链类型
            callbacks: 回调处理器列表
            return_source_documents: 是否返回源文档
        """
        # 初始化MultiSpaceAI模型
        self.multispace_model = MultiSpaceAI(
            config_path=config_path,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            audio_encoder=audio_encoder,
            device=device
        )
        
        # 初始化嵌入模型
        if embeddings is None:
            embeddings = MultiSpaceAIEmbeddings(
                config_path=config_path,
                text_encoder=text_encoder,
                image_encoder=image_encoder,
                audio_encoder=audio_encoder,
                device=device
            )
        
        # 初始化向量存储
        if vectorstore is None:
            vectorstore = FAISS(embeddings=embeddings, index_name="multispace_index")
        
        # 初始化LLM压缩器
        compressor = LLMChainExtractor.from_llm(llm)
        
        # 初始化上下文压缩检索器
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectorstore.as_retriever()
        )
        
        # 创建多模态问答提示模板
        template = """使用以下检索到的上下文和多模态信息来回答问题。

检索到的上下文: {context}

多模态信息: {multimodal_context}

问题: {question}

答案:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "multimodal_context", "question"]
        )
        
        # 初始化检索QA链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=return_source_documents,
            chain_type_kwargs={"prompt": prompt}
        )
        
        # 初始化MultiSpaceAI链
        self.multispace_chain = MultiSpaceAIChain(
            llm=llm,
            prompt=prompt,
            config_path=config_path,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            audio_encoder=audio_encoder,
            device=device,
            memory=memory,
            callbacks=callbacks
        )
    
    def add_documents(self, documents: List[Union[Document, MultiModalDocument]]) -> None:
        """添加文档到向量存储

        Args:
            documents: 文档列表
        """
        # 获取检索器和向量存储
        retriever = self.qa_chain.retriever
        if isinstance(retriever, ContextualCompressionRetriever):
            vectorstore = retriever.base_retriever.vectorstore
        else:
            vectorstore = retriever.vectorstore
        
        # 添加文档到向量存储
        vectorstore.add_documents(documents)
    
    def ingest_from_directory(
        self,
        directory: str,
        text_exts: List[str] = [".txt", ".md", ".html"],
        image_exts: List[str] = [".jpg", ".jpeg", ".png"],
        audio_exts: List[str] = [".mp3", ".wav", ".ogg"],
        text_splitter = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 0
    ) -> None:
        """从目录导入文档

        Args:
            directory: 目录路径
            text_exts: 文本文件扩展名列表
            image_exts: 图像文件扩展名列表
            audio_exts: 音频文件扩展名列表
            text_splitter: 文本分割器
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        if text_splitter is None:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        documents = []
        
        # 遍历目录
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                # 处理文本文件
                if file_ext in text_exts:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    
                    # 分割文本
                    text_chunks = text_splitter.split_text(text)
                    
                    # 创建多模态文档
                    for chunk in text_chunks:
                        documents.append(
                            MultiModalDocument(
                                page_content=chunk,
                                metadata={"source": file_path}
                            )
                        )
                
                # 处理图像文件
                elif file_ext in image_exts:
                    # 使用MultiSpaceAI处理图像
                    result = self.multispace_model.process(
                        image=file_path
                    )
                    
                    # 创建多模态文档
                    documents.append(
                        MultiModalDocument(
                            page_content=result["generated_text"],
                            metadata={"source": file_path},
                            image_path=file_path
                        )
                    )
                
                # 处理音频文件
                elif file_ext in audio_exts:
                    # 使用MultiSpaceAI处理音频
                    result = self.multispace_model.process(
                        audio=file_path
                    )
                    
                    # 创建多模态文档
                    documents.append(
                        MultiModalDocument(
                            page_content=result["generated_text"],
                            metadata={"source": file_path},
                            audio_path=file_path
                        )
                    )
        
        # 添加文档到向量存储
        self.add_documents(documents)
    
    def run(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """运行检索链

        Args:
            query: 查询
            text: 文本输入
            image: 图像文件路径
            audio: 音频文件路径
            **kwargs: 额外参数

        Returns:
            检索链运行结果
        """
        # 处理多模态输入
        if any([image, audio]):
            multimodal_result = self.multispace_model.process(
                text=text or query,
                image=image,
                audio=audio
            )
            
            # 将多模态处理结果添加到输入
            multimodal_context = multimodal_result["generated_text"]
        else:
            multimodal_context = ""
        
        # 运行检索QA链
        return self.qa_chain(
            {"query": query, "multimodal_context": multimodal_context, "question": query},
            **kwargs
        ) 