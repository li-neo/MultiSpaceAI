 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本示例
=======

使用MultiSpaceAI进行多模态处理的基本示例。
"""

import os
import sys
import argparse
from typing import Optional

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.multispace import MultiSpaceAI


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MultiSpaceAI基本示例")
    
    # 模型配置
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--device", type=str, default=None, help="运行设备")
    
    # 文本编码器配置
    parser.add_argument("--text-encoder", type=str, default="custom", 
                        choices=["custom", "deepseek-api"], help="文本编码器类型")
    
    # 图像编码器配置
    parser.add_argument("--image-encoder", type=str, default="custom",
                        choices=["custom", "diffusion-api"], help="图像编码器类型")
    
    # 语音编码器配置
    parser.add_argument("--audio-encoder", type=str, default="custom",
                        choices=["custom", "whisper-api"], help="语音编码器类型")
    
    # 输入数据
    parser.add_argument("--text", type=str, help="文本输入")
    parser.add_argument("--image", type=str, help="图像文件路径")
    parser.add_argument("--audio", type=str, help="音频文件路径")
    
    # 生成配置
    parser.add_argument("--max-length", type=int, default=50, help="生成的最大长度")
    parser.add_argument("--num-beams", type=int, default=4, help="束搜索的束数")
    
    # 输出配置
    parser.add_argument("--output", type=str, help="输出文件路径")
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 检查是否至少有一种模态的输入
    if not any([args.text, args.image, args.audio]):
        print("错误: 至少需要提供一种模态的输入（文本、图像或音频）")
        sys.exit(1)
    
    # 初始化模型
    model = MultiSpaceAI(
        config_path=args.config,
        text_encoder=args.text_encoder,
        image_encoder=args.image_encoder,
        audio_encoder=args.audio_encoder,
        device=args.device
    )
    
    # 处理输入
    result = model.process(
        text=args.text,
        image=args.image,
        audio=args.audio,
        max_length=args.max_length,
        num_beams=args.num_beams
    )
    
    # 打印结果
    print("\n生成的文本:")
    print(result["generated_text"])
    
    # 如果提供了输出文件路径，将结果保存到文件
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()