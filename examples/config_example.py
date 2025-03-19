 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置示例
=======

展示如何自定义和保存MultiSpaceAI的配置。
"""

import os
import sys
import json
import argparse

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.multispace.config import (
    ModelConfig,
    TextEncoderConfig,
    ImageEncoderConfig,
    AudioEncoderConfig,
    FusionConfig,
    DecoderConfig
)


def create_default_config(output_path):
    """创建默认配置并保存"""
    # 创建默认配置
    config = ModelConfig()
    
    # 保存配置
    config.save(output_path)
    print(f"默认配置已保存到: {output_path}")
    
    # 显示配置内容
    print("\n配置内容:")
    with open(output_path, 'r', encoding='utf-8') as f:
        print(json.dumps(json.load(f), ensure_ascii=False, indent=2))


def create_custom_config(output_path):
    """创建自定义配置并保存"""
    # 创建文本编码器配置
    text_config = TextEncoderConfig(
        model_name="bert-base-chinese",
        embedding_dim=768,
        hidden_dim=1024,
        num_layers=8,
        num_attention_heads=12,
        max_seq_length=512,
        dropout=0.1
    )
    
    # 创建图像编码器配置
    image_config = ImageEncoderConfig(
        model_name="google/vit-base-patch16-224",
        image_size=224,
        patch_size=16,
        embedding_dim=768,
        hidden_dim=1024,
        num_layers=12,
        num_attention_heads=12,
        dropout=0.1
    )
    
    # 创建语音编码器配置
    audio_config = AudioEncoderConfig(
        model_name="facebook/wav2vec2-large-960h-lv60-self",
        sample_rate=16000,
        embedding_dim=1024,
        hidden_dim=1024,
        num_layers=12,
        num_attention_heads=16,
        max_audio_length=60,
        dropout=0.1
    )
    
    # 创建融合模块配置
    fusion_config = FusionConfig(
        hidden_dim=1024,
        num_layers=6,
        num_attention_heads=16,
        fusion_type="cross_attention",
        use_modal_adapters=True,
        dropout=0.1
    )
    
    # 创建解码器模块配置
    decoder_config = DecoderConfig(
        model_name="fnlp/bart-large-chinese",
        embedding_dim=1024,
        hidden_dim=1024,
        num_layers=12,
        num_attention_heads=16,
        max_seq_length=512,
        dropout=0.1
    )
    
    # 创建模型配置
    config = ModelConfig()
    config.text_encoder_config = text_config
    config.image_encoder_config = image_config
    config.audio_encoder_config = audio_config
    config.fusion_config = fusion_config
    config.decoder_config = decoder_config
    
    # 保存配置
    config.save(output_path)
    print(f"自定义配置已保存到: {output_path}")
    
    # 显示配置内容
    print("\n配置内容:")
    with open(output_path, 'r', encoding='utf-8') as f:
        print(json.dumps(json.load(f), ensure_ascii=False, indent=2))


def main():
    """主函数"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="MultiSpaceAI配置示例")
    parser.add_argument("--type", type=str, choices=["default", "custom"], default="default",
                      help="配置类型: default (默认配置) 或 custom (自定义配置)")
    parser.add_argument("--output", type=str, default="config.json", help="输出配置文件路径")
    
    # 解析参数
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # 根据类型创建和保存配置
    if args.type == "default":
        create_default_config(args.output)
    else:
        create_custom_config(args.output)
    
    print(f"\n配置文件已创建: {args.output}")
    print("您可以在初始化MultiSpaceAI时使用此配置文件:")
    print(f"model = MultiSpaceAI(config_path='{args.output}')")


if __name__ == "__main__":
    main()