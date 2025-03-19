 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频处理示例
==========

展示如何使用MultiSpaceAI处理音频输入。
"""

import os
import sys
import torch
import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.multispace import MultiSpaceAI


def display_waveform(audio_path):
    """显示音频波形"""
    # 加载音频文件
    waveform, sample_rate = librosa.load(audio_path, sr=None)
    
    # 计算时间轴
    time = np.arange(0, len(waveform)) / sample_rate
    
    # 绘制波形
    plt.figure(figsize=(12, 4))
    plt.plot(time, waveform)
    plt.title('音频波形')
    plt.ylabel('振幅')
    plt.xlabel('时间 (秒)')
    plt.tight_layout()
    plt.show()


def display_spectrogram(audio_path):
    """显示音频频谱图"""
    # 加载音频文件
    waveform, sample_rate = librosa.load(audio_path, sr=None)
    
    # 计算频谱图
    D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
    
    # 绘制频谱图
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('音频频谱图')
    plt.tight_layout()
    plt.show()


def play_audio(audio_path):
    """播放音频（如果环境支持）"""
    try:
        from IPython.display import Audio, display
        display(Audio(audio_path))
        print("音频播放中...")
    except ImportError:
        print("当前环境不支持直接播放音频。")


def main():
    """主函数"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="MultiSpaceAI音频处理示例")
    parser.add_argument("--audio", type=str, required=True, help="音频文件路径")
    parser.add_argument("--text", type=str, default="描述这段音频", help="提示文本")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--device", type=str, default=None, help="运行设备")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--visualize", action="store_true", help="可视化音频")
    parser.add_argument("--play", action="store_true", help="播放音频")
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查音频文件是否存在
    if not os.path.exists(args.audio):
        print(f"错误: 音频文件不存在: {args.audio}")
        sys.exit(1)
    
    # 显示音频可视化
    if args.visualize:
        try:
            import librosa.display
            display_waveform(args.audio)
            display_spectrogram(args.audio)
        except ImportError:
            print("警告: 无法导入librosa.display，音频可视化功能不可用。")
    
    # 播放音频
    if args.play:
        play_audio(args.audio)
    
    print(f"处理音频: {args.audio}")
    print(f"提示文本: {args.text}")
    
    # 初始化模型
    print("\n初始化MultiSpaceAI模型...")
    model = MultiSpaceAI(
        config_path=args.config,
        device=args.device,
        audio_encoder="whisper-api" if os.environ.get("OPENAI_API_KEY") else "custom"
    )
    
    # 处理输入
    print("处理输入中...")
    result = model.process(
        text=args.text,
        audio=args.audio
    )
    
    # 打印结果
    print("\n生成的文本:")
    print("-" * 50)
    print(result["generated_text"])
    print("-" * 50)
    
    # 如果提供了输出文件路径，将结果保存到文件
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            # 将Tensor转换为列表
            result_json = {k: v for k, v in result.items()}
            json.dump(result_json, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")
    
    print("\n处理完成!")


if __name__ == "__main__":
    main()