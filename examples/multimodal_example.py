 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态处理示例
===========

展示如何使用MultiSpaceAI同时处理图像和文本输入。
"""

import os
import sys
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.multispace import MultiSpaceAI


def display_image(image_path):
    """显示图像"""
    img = Image.open(image_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title("输入图像")
    plt.show()


def main():
    """主函数"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="MultiSpaceAI多模态处理示例")
    parser.add_argument("--image", type=str, required=True, help="图像文件路径")
    parser.add_argument("--text", type=str, default="描述这个图像", help="提示文本")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--device", type=str, default=None, help="运行设备")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--display", action="store_true", help="显示输入图像")
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查图像文件是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        sys.exit(1)
    
    # 显示图像
    if args.display:
        display_image(args.image)
    
    print(f"处理图像: {args.image}")
    print(f"提示文本: {args.text}")
    
    # 初始化模型
    print("\n初始化MultiSpaceAI模型...")
    model = MultiSpaceAI(
        config_path=args.config,
        device=args.device
    )
    
    # 处理输入
    print("处理输入中...")
    result = model.process(
        text=args.text,
        image=args.image
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