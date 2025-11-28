"""
验证工具函数
"""
import numpy as np
from PIL import Image


def validate_watermark_visibility(watermark_pil, min_alpha=0.30, min_coverage=0.02):
    """
    验证水印是否可见
    
    Args:
        watermark_pil: PIL 图像对象（必须有 alpha 通道）
        min_alpha: 最小平均透明度（0-1）
        min_coverage: 最小可见覆盖率（0-1）
    
    Returns:
        bool: 水印是否满足可见性要求
    """
    if watermark_pil.mode != 'RGBA':
        return False
    
    # 转换为 numpy 数组
    watermark_array = np.array(watermark_pil)
    alpha_channel = watermark_array[:, :, 3]
    
    # 计算可见像素
    visible_pixels = alpha_channel > 0
    visible_count = np.sum(visible_pixels)
    
    # 检查是否有可见像素
    if visible_count == 0:
        return False
    
    # 计算可见区域的平均 alpha
    visible_alpha_values = alpha_channel[visible_pixels]
    avg_alpha = np.mean(visible_alpha_values) / 255.0
    
    # 计算可见覆盖率
    total_pixels = watermark_array.shape[0] * watermark_array.shape[1]
    coverage = visible_count / total_pixels
    
    # 验证约束
    return avg_alpha >= min_alpha and coverage >= min_coverage


def check_image_valid(image_path):
    """
    检查图像文件是否有效
    
    Args:
        image_path: 图像文件路径
    
    Returns:
        bool: 图像是否有效
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False
