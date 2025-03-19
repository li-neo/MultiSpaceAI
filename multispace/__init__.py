"""
MultiSpaceAI - 多模态大语言模型
==================================

这个包实现了一个能够处理文本、图像和语音输入的多模态大语言模型系统。

主要功能:
- 多模态数据处理和编码
- 跨模态特征融合
- 基于融合特征的生成任务
"""

__version__ = '0.1.0'

from .multispace import MultiSpaceAI

__all__ = ['MultiSpaceAI'] 