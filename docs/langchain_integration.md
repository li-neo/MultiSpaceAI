# MultiSpaceAI LangChain 集成

本文档介绍如何使用 MultiSpaceAI 与 LangChain 集成，实现更强大的多模态处理能力。

## 概述

[LangChain](https://www.langchain.com/) 是一个用于构建基于大语言模型应用的框架，提供了一系列组件和工具，使开发人员能够创建复杂的基于 LLM 的应用程序。

MultiSpaceAI 与 LangChain 的集成允许您:

1. 将 MultiSpaceAI 作为 LangChain 工具使用
2. 创建包含多模态处理能力的 LangChain 链
3. 构建可以使用 MultiSpaceAI 处理多模态输入的代理
4. 实现基于向量数据库的多模态内容检索

## 安装依赖

要使用 MultiSpaceAI 的 LangChain 集成，需要安装以下依赖:

```bash
pip install langchain langchain-openai faiss-cpu chromadb openai tiktoken
```

或者直接使用项目的 requirements.txt:

```bash
pip install -r requirements.txt
```

## 基本用法

### 将 MultiSpaceAI 作为工具使用

```python
from langchain.chat_models import ChatOpenAI
from src.langchain_integration import MultiSpaceAITool

# 初始化 LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# 初始化 MultiSpaceAI 工具
multispace_tool = MultiSpaceAITool(
    config_path="path/to/config.json",
    text_encoder="custom",
    image_encoder="custom",
    audio_encoder="custom"
)

# 使用工具处理多模态输入
result = multispace_tool.run(
    query="描述这个图像",
    image="path/to/image.jpg"
)

print(result)
```

### 使用 MultiSpaceAI 链

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from src.langchain_integration import MultiSpaceAIChain

# 初始化 LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# 创建提示模板
template = """你是一个多模态助手，能够处理文本、图像和音频输入。
    
{multimodal_context}

用户问题: {query}

请提供详细且有用的回答:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["multimodal_context", "query"]
)

# 初始化 MultiSpaceAI 链
multispace_chain = MultiSpaceAIChain(
    llm=llm,
    prompt=prompt,
    config_path="path/to/config.json"
)

# 运行链
result = multispace_chain({
    "query": "这个图像中有什么?",
    "image": "path/to/image.jpg"
})

print(result["text"])
```

### 使用 MultiSpaceAI 代理

```python
from langchain.chat_models import ChatOpenAI
from src.langchain_integration import MultiSpaceAIAgent

# 初始化 LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# 初始化 MultiSpaceAI 代理
agent = MultiSpaceAIAgent(
    llm=llm,
    config_path="path/to/config.json",
    verbose=True
)

# 运行代理
result = agent.run("分析这个图像并告诉我其中的主要内容 图像路径: path/to/image.jpg")

print(result)
```

### 使用 MultiSpaceAI 检索链

```python
from langchain.chat_models import ChatOpenAI
from src.langchain_integration import MultiSpaceAIRetrievalChain

# 初始化 LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# 初始化检索链
retrieval_chain = MultiSpaceAIRetrievalChain(
    llm=llm,
    config_path="path/to/config.json",
    return_source_documents=True
)

# 导入文档
retrieval_chain.ingest_from_directory(
    directory="path/to/data",
    chunk_size=1000,
    chunk_overlap=0
)

# 运行检索链
result = retrieval_chain.run(
    query="关于这个图像，我们之前的数据库中有什么相关信息?",
    image="path/to/image.jpg"
)

print(result["result"])

# 打印源文档
for doc in result["source_documents"]:
    print(f"来源: {doc.metadata.get('source')}")
    print(f"内容: {doc.page_content[:100]}...")
```

## 组件详解

### MultiSpaceAITool

`MultiSpaceAITool` 类将 MultiSpaceAI 封装为 LangChain 工具，可以在代理中使用。

- 主要功能: 处理多模态输入并返回生成文本
- 支持异步调用 (`arun` 方法)
- 可作为代理的工具使用

### MultiSpaceAIChain

`MultiSpaceAIChain` 类扩展了 LangChain 的 `LLMChain`，添加了多模态处理能力。

- 主要功能: 处理多模态输入并将处理结果添加到链的上下文中
- 支持内存组件，可以保存对话历史
- 可以自定义提示模板

### MultiSpaceAIAgent

`MultiSpaceAIAgent` 类创建一个可以使用 MultiSpaceAI 处理多模态输入的代理。

- 主要功能: 创建一个可以规划和执行多模态任务的代理
- 基于 LangChain 的结构化代理实现
- 支持对话内存和回调

### MultiSpaceAIEmbeddings

`MultiSpaceAIEmbeddings` 类实现了 LangChain 的 `Embeddings` 接口，用于生成多模态内容的嵌入向量。

- 主要功能: 生成文本、图像和音频的嵌入向量
- 可以与 LangChain 的向量存储集成
- 支持文档和查询的嵌入生成

### MultiModalDocument

`MultiModalDocument` 类扩展了 LangChain 的 `Document` 类，添加了对图像和音频内容的支持。

- 主要功能: 表示包含多模态内容的文档
- 存储图像和音频文件路径到元数据中
- 与标准 LangChain 文档兼容

### MultiSpaceAIRetrievalChain

`MultiSpaceAIRetrievalChain` 类实现了一个基于向量数据库的多模态内容检索和问答系统。

- 主要功能: 根据多模态查询检索相关内容并回答问题
- 支持从目录导入文档
- 使用 LLM 进行上下文压缩和回答生成

## 示例

查看 `examples/langchain_example.py` 获取完整示例。

运行示例:

```bash
# 工具示例
python examples/langchain_example.py --example tool --image path/to/image.jpg --query "描述这个图像"

# 链示例
python examples/langchain_example.py --example chain --audio path/to/audio.mp3 --query "转录这段音频"

# 代理示例
python examples/langchain_example.py --example agent --query "分析这个图像" --image path/to/image.jpg

# 检索示例
python examples/langchain_example.py --example retrieval --data-dir path/to/data --query "查找相关内容"

# 运行所有示例
python examples/langchain_example.py --example all --image path/to/image.jpg --data-dir path/to/data
```

## 高级配置

### 自定义提示模板

您可以创建自定义提示模板来控制多模态内容的处理方式:

```python
from langchain.prompts import PromptTemplate

template = """系统: 你是一个专业的多模态内容分析专家。
多模态内容: {multimodal_context}
用户问题: {query}
分析结果:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["multimodal_context", "query"]
)

# 在链或检索链中使用自定义提示
```

### 集成外部向量存储

您可以使用外部向量存储进行多模态内容检索:

```python
from langchain.vectorstores import Chroma
from src.langchain_integration import MultiSpaceAIEmbeddings, MultiSpaceAIRetrievalChain

# 创建嵌入模型
embeddings = MultiSpaceAIEmbeddings(config_path="path/to/config.json")

# 创建向量存储
vectorstore = Chroma(embedding_function=embeddings, persist_directory="path/to/store")

# 创建检索链
retrieval_chain = MultiSpaceAIRetrievalChain(
    llm=llm,
    vectorstore=vectorstore
)
```

### 添加自定义回调

您可以添加自定义回调来监控处理流程:

```python
from langchain.callbacks import StdOutCallbackHandler
from src.langchain_integration import MultiSpaceAIAgent

# 创建回调处理器
callbacks = [StdOutCallbackHandler()]

# 初始化代理
agent = MultiSpaceAIAgent(
    llm=llm,
    callbacks=callbacks,
    verbose=True
)
```

## 故障排除

### 安装问题

如果遇到安装依赖的问题，可以尝试使用以下命令:

```bash
pip install --upgrade pip
pip install langchain langchain-openai faiss-cpu chromadb openai tiktoken
```

### API 密钥

确保设置了必要的 API 密钥环境变量:

```bash
export OPENAI_API_KEY=your_openai_api_key
```

### 内存问题

处理大型多模态数据集可能需要大量内存，可以通过以下方式减少内存使用:

1. 减小 `chunk_size` 参数
2. 使用 `Chroma` 而不是 `FAISS` 作为向量存储
3. 批量处理文档而不是一次性导入所有文档

## 更多资源

- [LangChain 文档](https://langchain.readthedocs.io/)
- [MultiSpaceAI 文档](../README.md)
- [示例代码](../examples/) 