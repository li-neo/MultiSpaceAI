"""
日志工具模块
=========

该模块包含用于设置和管理日志的函数。
"""

import logging
import os
import sys
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    设置日志记录器
    
    参数:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径，如果为None则仅输出到控制台
        
    返回:
        配置好的日志记录器
    """
    # 获取日志记录器
    logger = logging.getLogger(name)
    
    # 如果日志记录器已经有处理器，说明已经配置过，直接返回
    if logger.handlers:
        return logger
    
    # 设置日志级别
    logger.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果提供了日志文件路径，创建文件处理器
    if log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 阻止日志传播到父记录器
    logger.propagate = False
    
    return logger 