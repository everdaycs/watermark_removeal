"""
图像处理工具函数
"""
import os
from pathlib import Path


def ensure_dirs(*paths):
    """
    创建目录（如果不存在）
    
    Args:
        *paths: 一个或多个目录路径
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_image_files(directory, extensions=None):
    """
    递归获取目录中的所有图像文件
    
    Args:
        directory: 要搜索的目录
        extensions: 支持的文件扩展名集合，默认为 {'.jpg', '.jpeg', '.png', '.bmp'}
    
    Returns:
        list: 图像文件路径列表
    """
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))
    return files


def get_image_size(image_path):
    """
    获取图像尺寸
    
    Args:
        image_path: 图像文件路径
    
    Returns:
        tuple: (width, height)
    """
    from PIL import Image
    with Image.open(image_path) as img:
        return img.size
