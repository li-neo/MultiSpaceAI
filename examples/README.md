 # MultiSpaceAI 示例

本目录包含 MultiSpaceAI 的各种使用示例，帮助您快速上手多模态大语言模型。

## 示例列表

1. **基本示例** (`basic_example.py`)
   - 展示 MultiSpaceAI 的基本使用方法
   - 支持文本、图像和音频输入

2. **多模态处理示例** (`multimodal_example.py`)
   - 展示如何同时处理图像和文本输入
   - 可视化展示处理结果

3. **音频处理示例** (`audio_processing_example.py`)
   - 展示如何处理音频输入
   - 包含音频可视化功能

4. **配置示例** (`config_example.py`)
   - 展示如何自定义和保存模型配置
   - 包含默认配置和自定义配置示例

## 使用方法

### 基本示例

```bash
# 使用文本输入
python examples/basic_example.py --text "这是一段测试文本"

# 使用图像输入
python examples/basic_example.py --image path/to/image.jpg

# 使用音频输入
python examples/basic_example.py --audio path/to/audio.mp3

# 同时使用多种模态
python examples/basic_example.py --text "描述这个图像" --image path/to/image.jpg

# 指定输出文件
python examples/basic_example.py --text "测试文本" --output results.json

# 使用自定义配置文件
python examples/basic_example.py --text "测试文本" --config custom_config.json
```

### 多模态处理示例

```bash
# 处理图像和文本
python examples/multimodal_example.py --image path/to/image.jpg --text "描述这个图像"

# 显示输入图像
python examples/multimodal_example.py --image path/to/image.jpg --display
```

### 音频处理示例

```bash
# 处理音频
python examples/audio_processing_example.py --audio path/to/audio.mp3

# 可视化音频并处理
python examples/audio_processing_example.py --audio path/to/audio.mp3 --visualize

# 尝试播放音频（在支持的环境中）
python examples/audio_processing_example.py --audio path/to/audio.mp3 --play
```

### 配置示例

```bash
# 创建默认配置
python examples/config_example.py --type default --output default_config.json

# 创建自定义配置
python examples/config_example.py --type custom --output custom_config.json
```

## 环境变量

某些示例可能需要设置环境变量来使用API服务：

- `DEEPSEEK_API_KEY`: 用于DeepSeek文本编码器
- `STABLE_DIFFUSION_API_KEY`: 用于Diffusion图像编码器
- `OPENAI_API_KEY`: 用于Whisper语音编码器

```bash
# 设置环境变量示例（Linux/macOS）
export OPENAI_API_KEY=your_api_key_here

# 设置环境变量示例（Windows）
set OPENAI_API_KEY=your_api_key_here
```

## 额外资源

查看项目根目录的 README.md 获取更多信息。