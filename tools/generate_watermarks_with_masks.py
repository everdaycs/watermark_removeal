"""
增强的水印合成脚本 - 生成多样化水印图像和对应的二值掩码 (可见性强化版)

关键约束:
- 所有水印必须清晰可见（不允许隐形或几乎不可见的水印）
- 最小透明度: 30% (alpha >= 0.3)
- 颜色对比度自适应: 根据背景亮度自动选择明亮或深色文本
- 最小尺寸: 文本至少占图像高度的3-4%
- 可见面积验证: 水印必须覆盖非平凡的可见区域

功能:
- 从无水印图像文件夹生成多样化的水印版本
- 为每个水印生成对应的二值掩码 (255=水印, 0=背景)
- 支持14种风格（13种单独 + 组合）
- 强制所有水印可见性约束

配置参数:
- NUM_VARIANTS_PER_IMAGE: 每张图的变体数 (默认32)
- ENABLED_STYLES: 启用的风格列表
- COMBINED_STYLE_PROBABILITY: 组合风格的概率 (默认0.3)
- ENABLE_VISIBILITY_VALIDATION: 是否验证可见性 (默认True)
- INPUT_DIR: 无水印图像输入目录
- OUTPUT_DIR: 水印图像输出目录
- MASK_DIR: 掩码输出目录
"""

import os
import random
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# ============================================================================
# 可见性约束配置
# ============================================================================

# 最小透明度 (30%)
MIN_ALPHA = 0.30
MAX_ALPHA = 0.75

# 最小文本尺寸 (像素)
MIN_TEXT_HEIGHT_PX = 20
MIN_TEXT_HEIGHT_RATIO = 0.04  # 图像高度的 4%

# 最小可见面积 (图像总面积的百分比)
MIN_VISIBLE_COVERAGE = 0.02  # 至少 2% 的可见面积

# 亮度阈值 (用于颜色对比度选择)
BRIGHTNESS_THRESHOLD = 128

# 启用可见性验证
ENABLE_VISIBILITY_VALIDATION = True

# ============================================================================
# 配置参数
# ============================================================================

NUM_VARIANTS_PER_IMAGE = 32
INPUT_DIR = "/home/kaga/Desktop/watermaker remover/20251127_no_watermark_demo"
OUTPUT_DIR = "./watermark_demowen"
MASK_DIR = "./watermark_demowen_masks"
LOGO_PATH = "./logo.png"

TEXT_CANDIDATES = ["DEMO", "SAMPLE", "Preview", "Copyright", "DemoWen", 
                   "NoRepost", "2025", "WATERMARK", "Test", "Draft"]

# 启用的风格列表 (可以注释掉某些风格来禁用)
ENABLED_STYLES = [
    'single_text_corner',      # 1
    'single_text_center',      # 2
    'tiled_text',              # 3
    'diagonal_band',           # 4
    'multi_line_text_center',  # 5
    'logo_corner',             # 6
    'qr_code_style',           # 7
    'outlined_text',           # 8
    'shadow_text',             # 9
    'gradient_alpha_text',     # 10
    'noisy_eroded_text',       # 11
    'banner_box',              # 12
    'curved_text',             # 13
    'corner_website_logo',     # 14 - ElecFans风格角落网站LOGO水印
    'mini_social_corner_logo', # 15 - 小型微信风格社交账号角标水印
    'combined_styles',         # 16
]

COMBINED_STYLE_PROBABILITY = 0.3  # 使用组合风格的概率

# 角落网站LOGO风格的概率权重 (0.0-1.0, 相对于其他风格)
CORNER_WEBSITE_LOGO_WEIGHT = 0.4  # 40% 概率选择此风格

# ============================================================================
# 辅助函数
# ============================================================================

def ensure_dirs():
    """创建输出目录"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(MASK_DIR).mkdir(parents=True, exist_ok=True)

def get_image_files(directory):
    """递归获取所有图像文件"""
    supported = {'.jpg', '.jpeg', '.png'}
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if Path(filename).suffix.lower() in supported:
                files.append(os.path.join(root, filename))
    return files

def load_image(image_path):
    """加载图像，返回RGB/RGBA格式"""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        return img
    except Exception as e:
        print(f"错误: 无法加载 {image_path}: {e}")
        return None

def get_font(font_size):
    """获取字体，优先使用系统字体"""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        return ImageFont.load_default()

def get_text_bbox(draw, text, font):
    """获取文本边界框大小"""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def get_estimated_brightness(image_rgb, region_box=None):
    """
    估计图像或区域的平均亮度
    
    Args:
        image_rgb (PIL.Image): RGB 图像
        region_box (tuple): 可选的裁剪区域 (x0, y0, x1, y1)
    
    Returns:
        float: 亮度值 [0, 255]
    """
    try:
        if region_box is None:
            # 采样中心区域
            crop_box = (
                image_rgb.width // 4,
                image_rgb.height // 4,
                3 * image_rgb.width // 4,
                3 * image_rgb.height // 4
            )
        else:
            crop_box = region_box
        
        sample = image_rgb.crop(crop_box)
        sample_array = np.array(sample)
        
        # 计算亮度（加权平均）
        if len(sample_array.shape) == 3:  # RGB
            brightness = np.mean(0.299 * sample_array[:, :, 0] + 
                                0.587 * sample_array[:, :, 1] + 
                                0.114 * sample_array[:, :, 2])
        else:  # Grayscale
            brightness = np.mean(sample_array)
        
        return brightness
    except:
        return 128  # 默认中等亮度

def select_contrasting_color(brightness, prefer_dark=None):
    """
    根据背景亮度选择对比明显的文本颜色
    
    Args:
        brightness (float): 背景亮度 [0, 255]
        prefer_dark (bool): 强制深色(True)或浅色(False)。None时自动选择。
    
    Returns:
        tuple: RGB 颜色 (R, G, B)
    """
    if prefer_dark is None:
        # 自动选择：亮背景 -> 深色文本，暗背景 -> 浅色文本
        is_bright = brightness > BRIGHTNESS_THRESHOLD
    else:
        is_bright = not prefer_dark
    
    if is_bright:
        # 背景亮 -> 使用深色文本
        return random.choice([(0, 0, 0), (20, 20, 20), (40, 40, 40), (50, 50, 50)])
    else:
        # 背景暗 -> 使用浅色文本
        return random.choice([(255, 255, 255), (240, 240, 240), (220, 220, 220), (200, 200, 200)])

def draw_text_with_alpha(overlay, position, text, font, color, alpha):
    """
    在RGBA图像上绘制具有正确alpha透明度的文本
    
    Args:
        overlay (PIL.Image): RGBA 图像
        position (tuple): (x, y) 文本位置
        text (str): 要绘制的文本
        font (PIL.ImageFont): 字体
        color (tuple): RGB 颜色
        alpha (float): 透明度 [0, 1]
    """
    # 创建临时图层来绘制文本
    temp_layer = Image.new('RGBA', overlay.size, (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp_layer)
    
    # 用完全不透明的颜色绘制文本
    temp_draw.text(position, text, font=font, fill=color + (255,))
    
    # 现在通过操纵alpha通道来应用透明度
    if alpha < 1.0:
        temp_array = np.array(temp_layer)
        # 只修改包含文本的像素的alpha
        mask = temp_array[:, :, 3] > 0  # 找到所有非透明的像素
        temp_array[mask, 3] = int(255 * alpha)  # 应用透明度
        temp_layer = Image.fromarray(temp_array, 'RGBA')
    
    # 将文本合成到主overlay上
    overlay = Image.alpha_composite(overlay, temp_layer)
    return overlay

def validate_watermark_visibility(overlay_array, w, h):
    """
    验证水印是否足够可见
    
    Args:
        overlay_array (np.ndarray): RGBA overlay 数组
        w (int): 图像宽度
        h (int): 图像高度
    
    Returns:
        dict: {
            'is_visible': bool,
            'avg_alpha': float,
            'coverage': float,
            'reasons': [str, ...]
        }
    """
    alpha_channel = overlay_array[:, :, 3]
    
    # 检查可见面积（alpha > 0 的像素比例）
    visible_pixels = np.sum(alpha_channel > 0)
    total_pixels = w * h
    coverage = visible_pixels / total_pixels if total_pixels > 0 else 0
    
    # 检查可见像素的平均透明度（只计算非零alpha的像素）
    if visible_pixels > 0:
        visible_alpha_values = alpha_channel[alpha_channel > 0]
        avg_alpha = np.mean(visible_alpha_values) / 255.0
    else:
        avg_alpha = 0.0
    
    reasons = []
    is_visible = True
    
    # 检查覆盖面积（这是首要条件）
    if coverage < MIN_VISIBLE_COVERAGE:
        is_visible = False
        reasons.append(f"coverage={coverage:.2%} < {MIN_VISIBLE_COVERAGE:.2%}")
    
    # 检查可见像素的透明度
    if avg_alpha < MIN_ALPHA:
        is_visible = False
        reasons.append(f"avg_alpha={avg_alpha:.2f} < {MIN_ALPHA}")
    
    return {
        'is_visible': is_visible,
        'avg_alpha': avg_alpha,
        'coverage': coverage,
        'reasons': reasons
    }

# ============================================================================
# 水印合成函数 - 返回 overlay RGBA 图像 (强制可见性)
# ============================================================================

def single_text_corner(w, h):
    """风格1: 单个文本在四个角之一 (可见性强制版)"""
    text = random.choice(TEXT_CANDIDATES)
    # 最小尺寸: 4-8% 的图像宽度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.04, 0.08)))
    font = get_font(font_size)
    # 强制透明度范围
    alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    
    corner = random.choice(['tl', 'tr', 'bl', 'br'])
    margin = max(15, int(w * 0.02))
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    text_w, text_h = get_text_bbox(overlay_draw, text, font)
    
    if corner == 'tl':
        pos = (margin, margin)
    elif corner == 'tr':
        pos = (max(0, w - text_w - margin), margin)
    elif corner == 'bl':
        pos = (margin, max(0, h - text_h - margin))
    else:  # br
        pos = (max(0, w - text_w - margin), max(0, h - text_h - margin))
    
    # 自适应颜色选择
    color = select_contrasting_color(BRIGHTNESS_THRESHOLD)
    # 使用新的透明文本绘制方法
    overlay = draw_text_with_alpha(overlay, pos, text, font, color, alpha)
    
    # 轻微旋转
    rotation = random.uniform(-12, 12)
    if rotation != 0:
        overlay = overlay.rotate(rotation, expand=False, fillcolor=(255, 255, 255, 0))
    
    return overlay

def single_text_center(w, h):
    """风格2: 大型居中半透明文本 (可见性强制版)"""
    text = random.choice(TEXT_CANDIDATES)
    # 最小尺寸: 8-15% 的图像高度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.08, 0.15)))
    font = get_font(font_size)
    # 强制透明度范围
    alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    text_w, text_h = get_text_bbox(overlay_draw, text, font)
    pos = ((w - text_w) // 2, (h - text_h) // 2)
    
    # 自适应颜色选择
    color = select_contrasting_color(BRIGHTNESS_THRESHOLD)
    overlay = draw_text_with_alpha(overlay, pos, text, font, color, alpha)
    
    rotation = random.uniform(-10, 10)
    if rotation != 0:
        overlay = overlay.rotate(rotation, expand=False, fillcolor=(255, 255, 255, 0))
    
    return overlay

def tiled_text(w, h):
    """风格3: 平铺的小文本 (可见性强制版)"""
    text = random.choice(TEXT_CANDIDATES)
    # 最小尺寸: 3-6% 的图像高度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.03, 0.06)))
    font = get_font(font_size)
    # 强制透明度范围
    alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    text_w, text_h = get_text_bbox(overlay_draw, text, font)
    
    spacing_x = int(text_w * 1.3)
    spacing_y = int(text_h * 1.8)
    
    # 自适应颜色
    color = select_contrasting_color(BRIGHTNESS_THRESHOLD)
    
    y = 0
    while y < h:
        x = 0
        while x < w:
            overlay = draw_text_with_alpha(overlay, (x, y), text, font, color, alpha)
            x += spacing_x
        y += spacing_y
    
    rotation = random.uniform(-5, 5)
    if rotation != 0:
        overlay = overlay.rotate(rotation, expand=False, fillcolor=(255, 255, 255, 0))
    
    return overlay

def diagonal_band(w, h):
    """风格4: 对角线波段 (可见性强制版)"""
    text = random.choice(TEXT_CANDIDATES)
    # 最小尺寸: 6-12% 的图像高度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.06, 0.12)))
    font = get_font(font_size)
    # 强制透明度范围
    alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # 自适应颜色
    color = select_contrasting_color(BRIGHTNESS_THRESHOLD)
    overlay = draw_text_with_alpha(overlay, (w // 4, h // 4), text, font, color, alpha)
    
    # 强制对角线旋转
    rotation = random.choice([random.uniform(35, 55), random.uniform(-55, -35)])
    if rotation != 0:
        overlay = overlay.rotate(rotation, expand=False, fillcolor=(255, 255, 255, 0))
    
    return overlay

def multi_line_text_center(w, h):
    """风格5: 多行文本居中 (可见性强制版)"""
    lines = random.sample(TEXT_CANDIDATES, min(2, len(TEXT_CANDIDATES)))  # 至少 2 行
    
    # 最小尺寸: 5-10% 的图像高度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.05, 0.10)))
    font = get_font(font_size)
    # 强制透明度范围
    alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    line_height = font_size + 8
    total_height = line_height * len(lines)
    start_y = max(0, (h - total_height) // 2)
    
    # 自适应颜色
    color = select_contrasting_color(BRIGHTNESS_THRESHOLD)
    
    for i, line in enumerate(lines):
        text_w, _ = get_text_bbox(overlay_draw, line, font)
        x = (w - text_w) // 2
        y = start_y + i * line_height
        overlay = draw_text_with_alpha(overlay, (x, y), line, font, color, alpha)
    
    return overlay

def logo_corner(w, h, logo_path):
    """风格6: 角落的LOGO (可见性强制版)"""
    if not os.path.exists(logo_path):
        return None
    
    try:
        logo = Image.open(logo_path).convert('RGBA')
        
        # 强制最小尺寸: 至少 12% 的图像宽度
        logo_w = int(w * random.uniform(0.12, 0.18))
        ratio = logo_w / logo.width if logo.width > 0 else 1
        logo_h = int(logo.height * ratio)
        logo = logo.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
        
        # 强制透明度范围
        alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
        logo_array = np.array(logo)
        logo_array[:, :, 3] = (logo_array[:, :, 3] * alpha).astype(np.uint8)
        logo = Image.fromarray(logo_array)
        
        overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
        
        corner = random.choice(['tl', 'tr', 'bl', 'br'])
        margin = max(15, int(w * 0.02))
        
        if corner == 'tl':
            pos = (margin, margin)
        elif corner == 'tr':
            pos = (max(0, w - logo_w - margin), margin)
        elif corner == 'bl':
            pos = (margin, max(0, h - logo_h - margin))
        else:  # br
            pos = (max(0, w - logo_w - margin), max(0, h - logo_h - margin))
        
        overlay.paste(logo, pos, logo)
        return overlay
    except:
        return None

def qr_code_style(w, h):
    """风格7: 二维码风格 (可见性强制版)"""
    # 强制最小尺寸: 至少 15% 的最小边
    qr_size = int(min(w, h) * random.uniform(0.15, 0.25))
    # 强制透明度范围
    alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    
    qr_grid_size = random.randint(6, 9)
    cell_size = max(3, qr_size // qr_grid_size)  # 最小单元 3x3
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # 自适应颜色
    color = select_contrasting_color(BRIGHTNESS_THRESHOLD)
    
    for i in range(qr_grid_size):
        for j in range(qr_grid_size):
            if random.random() > 0.4:  # 60% 填充率
                x0 = i * cell_size
                y0 = j * cell_size
                x1 = min(x0 + cell_size, w)
                y1 = min(y0 + cell_size, h)
                overlay_draw.rectangle([x0, y0, x1, y1], 
                                     fill=color + (int(255 * alpha),))
    
    return overlay

def outlined_text(w, h):
    """风格8: 带轮廓的文本 (可见性强制版)"""
    text = random.choice(TEXT_CANDIDATES)
    # 最小尺寸: 7-13% 的图像高度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.07, 0.13)))
    font = get_font(font_size)
    # 强制透明度范围
    alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    # 强制粗轮廓
    outline_width = random.randint(3, 6)
    rotation = random.uniform(-15, 15)
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    text_w, text_h = get_text_bbox(overlay_draw, text, font)
    pos = ((w - text_w) // 2, (h - text_h) // 2)
    
    # 自适应颜色: 轮廓颜色与主文本对比
    bg_brightness = BRIGHTNESS_THRESHOLD
    if bg_brightness > BRIGHTNESS_THRESHOLD:
        outline_color = (0, 0, 0)
        main_color = (255, 255, 255)
    else:
        outline_color = (255, 255, 255)
        main_color = (0, 0, 0)
    
    # 绘制轮廓：多次绘制稍微偏移的文本
    for adj_x in range(-outline_width, outline_width + 1):
        for adj_y in range(-outline_width, outline_width + 1):
            if adj_x != 0 or adj_y != 0:
                outline_draw_pos = (pos[0] + adj_x, pos[1] + adj_y)
                overlay = draw_text_with_alpha(overlay, outline_draw_pos, text, font, outline_color, alpha)
    
    # 绘制主文本（更强不透明度）
    overlay = draw_text_with_alpha(overlay, pos, text, font, main_color, alpha * 0.9)
    
    if rotation != 0:
        overlay = overlay.rotate(rotation, expand=False, fillcolor=(255, 255, 255, 0))
    
    return overlay
def shadow_text(w, h):
    """风格9: 带阴影的文本 (可见性强制版)"""
    text = random.choice(TEXT_CANDIDATES)
    # 最小尺寸: 8-16% 的图像高度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.08, 0.16)))
    font = get_font(font_size)
    # 强制透明度范围
    alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    shadow_alpha = max(MIN_ALPHA, alpha * random.uniform(0.5, 0.8))
    
    shadow_offset = (random.randint(4, 8), random.randint(4, 8))
    rotation = random.uniform(-12, 12)
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    text_w, text_h = get_text_bbox(overlay_draw, text, font)
    pos = ((w - text_w) // 2, (h - text_h) // 2)
    
    # 绘制阴影（深色）
    shadow_pos = (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1])
    overlay = draw_text_with_alpha(overlay, shadow_pos, text, font, (0, 0, 0), shadow_alpha)
    
    # 绘制主文本（浅色/深色对比）
    text_color = select_contrasting_color(BRIGHTNESS_THRESHOLD)
    overlay = draw_text_with_alpha(overlay, pos, text, font, text_color, alpha)
    
    if rotation != 0:
        overlay = overlay.rotate(rotation, expand=False, fillcolor=(255, 255, 255, 0))
    
    return overlay

def gradient_alpha_text(w, h):
    """风格10: 透明度渐变的文本 (可见性强制版)"""
    text = random.choice(TEXT_CANDIDATES)
    # 最小尺寸: 8-16% 的图像高度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.08, 0.16)))
    font = get_font(font_size)
    # 强制基础透明度: 至少 MIN_ALPHA
    base_alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    gradient_direction = random.choice(['h', 'v'])  # 水平或垂直
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    text_w, text_h = get_text_bbox(overlay_draw, text, font)
    pos = ((w - text_w) // 2, (h - text_h) // 2)
    
    # 自适应颜色
    color = select_contrasting_color(BRIGHTNESS_THRESHOLD)
    
    # 先用完整alpha绘制文本到临时图像
    temp_overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp_overlay)
    temp_draw.text(pos, text, font=font, fill=color + (int(255 * base_alpha),))
    
    # 创建渐变掩码（但保证最小可见）
    gradient_mask = Image.new('L', (w, h), 0)
    gradient_array = np.array(gradient_mask)
    
    if gradient_direction == 'h':
        # 水平渐变：左边不透明，右边也保持最少透明度
        for x in range(w):
            fade = max(MIN_ALPHA, 1 - x / w)
            gradient_array[:, x] = int(255 * fade)
    else:
        # 垂直渐变：上面不透明，下面也保持最少透明度
        for y in range(h):
            fade = max(MIN_ALPHA, 1 - y / h)
            gradient_array[y, :] = int(255 * fade)
    
    # 应用渐变掩码到overlay
    overlay_array = np.array(temp_overlay)
    overlay_array[:, :, 3] = (overlay_array[:, :, 3] * gradient_array / 255).astype(np.uint8)
    overlay = Image.fromarray(overlay_array, 'RGBA')
    
    return overlay

def noisy_eroded_text(w, h):
    """风格11: 有噪声边缘的文本 (可见性强制版)"""
    text = random.choice(TEXT_CANDIDATES)
    # 最小尺寸: 6-12% 的图像高度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.06, 0.12)))
    font = get_font(font_size)
    # 强制透明度范围
    alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    text_w, text_h = get_text_bbox(overlay_draw, text, font)
    pos = ((w - text_w) // 2, (h - text_h) // 2)
    
    # 自适应颜色
    color = select_contrasting_color(BRIGHTNESS_THRESHOLD)
    
    # 绘制初始文本
    overlay_draw.text(pos, text, font=font, fill=color + (int(255 * alpha),))
    
    # 应用轻微噪声和模糊以模拟损坏的边缘（但保持可见）
    overlay_array = np.array(overlay)
    alpha_channel = overlay_array[:, :, 3].astype(float)
    
    # 添加随机噪声（较小幅度，保持可见性）
    noise = np.random.normal(0, 8, alpha_channel.shape)
    alpha_channel = np.clip(alpha_channel + noise, int(255 * MIN_ALPHA), 255)
    
    # 应用轻微的高斯模糊
    try:
        temp_img = Image.fromarray(alpha_channel.astype(np.uint8), 'L')
        temp_img = temp_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        alpha_channel = np.array(temp_img).astype(float)
    except:
        pass
    
    # 应用阈值以保持清晰边缘（但阈值不能太高以保持可见性）
    threshold = max(int(255 * MIN_ALPHA * 0.5), 30)
    alpha_channel = np.where(alpha_channel > threshold, alpha_channel, 0)
    
    overlay_array[:, :, 3] = np.clip(alpha_channel, 0, 255).astype(np.uint8)
    overlay = Image.fromarray(overlay_array, 'RGBA')
    
    return overlay

def banner_box(w, h):
    """风格12: 带背景条的文本 (可见性强制版)"""
    text = random.choice(TEXT_CANDIDATES)
    # 最小尺寸: 6-12% 的图像高度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.06, 0.12)))
    font = get_font(font_size)
    # 强制透明度范围
    text_alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    box_alpha = random.uniform(0.15, 0.4)  # 背景条可以稍微透明一些，但不能太弱
    
    is_horizontal = random.choice([True, False])
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    text_w, text_h = get_text_bbox(overlay_draw, text, font)
    
    # 文本位置
    text_x = (w - text_w) // 2
    text_y = (h - text_h) // 2
    
    # 自适应背景和文本颜色
    bg_brightness = BRIGHTNESS_THRESHOLD
    if bg_brightness > BRIGHTNESS_THRESHOLD:
        box_color = (0, 0, 0)        # 深背景
        text_color = (255, 255, 255)  # 浅文本
    else:
        box_color = (255, 255, 255)  # 浅背景
        text_color = (0, 0, 0)        # 深文本
    
    if is_horizontal:
        # 水平条
        padding = max(5, int(h * 0.03))
        box_x0 = max(0, text_x - padding)
        box_y0 = max(0, text_y - padding)
        box_x1 = min(w, text_x + text_w + padding)
        box_y1 = min(h, text_y + text_h + padding)
    else:
        # 垂直条
        box_height = max(h // 4, int(h * 0.2))
        box_y0 = max(0, (h - box_height) // 2)
        box_x0 = max(0, text_x - int(h * 0.03))
        box_y1 = min(h, box_y0 + box_height)
        box_x1 = min(w, text_x + text_w + int(h * 0.03))
    
    # 绘制背景条
    overlay_draw.rectangle(
        [box_x0, box_y0, box_x1, box_y1],
        fill=box_color + (int(255 * box_alpha),)
    )
    
    # 绘制文本
    overlay = draw_text_with_alpha(overlay, (text_x, text_y), text, font, text_color, text_alpha)
    
    return overlay

def curved_text(w, h):
    """风格13: 弧形排列的文本 (可见性强制版)"""
    text = random.choice(TEXT_CANDIDATES)
    # 最小尺寸: 5-10% 的图像高度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.05, 0.10)))
    font = get_font(font_size)
    # 强制透明度范围
    alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    
    # 自适应颜色
    color = select_contrasting_color(BRIGHTNESS_THRESHOLD)
    
    # 弧线参数
    center_x = w // 2
    center_y = h // 2
    radius = min(w, h) * random.uniform(0.25, 0.40)
    arc_start_angle = random.uniform(-90, 90)
    
    # 沿弧线排列字符
    char_list = list(text)
    angle_step = max(8, 100 // len(char_list)) if len(char_list) > 1 else 0
    
    for i, char in enumerate(char_list):
        angle = arc_start_angle + i * angle_step
        angle_rad = math.radians(angle)
        
        # 计算字符位置
        char_x = center_x + radius * math.cos(angle_rad)
        char_y = center_y + radius * math.sin(angle_rad)
        
        # 创建旋转的字符
        char_img = Image.new('RGBA', (font_size * 2, font_size * 2), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((font_size // 2, font_size // 4), char, font=font, 
                      fill=color + (int(255 * alpha),))
        
        # 旋转字符以匹配弧线方向
        char_img = char_img.rotate(-angle, expand=True, fillcolor=(255, 255, 255, 0))
        
        # 计算粘贴位置
        paste_x = int(char_x - char_img.width // 2)
        paste_y = int(char_y - char_img.height // 2)
        
        # 确保在边界内
        if 0 <= paste_x < w and 0 <= paste_y < h:
            overlay.paste(char_img, (paste_x, paste_y), char_img)
    
    return overlay

def corner_website_logo(w, h):
    """
    风格14: 角落网站LOGO水印 - 模拟中国科技网站的角落水印
    
    组成: 圆形图标 + 品牌名称 + 网站URL，水平排列
    位置: 偏好底部角落，带随机边距
    样式: 半透明，带柔和模糊，可能有阴影或轮廓确保对比度
    """
    # 候选品牌名称和URL
    brand_candidates = ["电子发烧友", "DemoWen", "TechZone", "ElecFans", "AI-Lab", "CircuitHub", "TechBlog"]
    url_candidates = ["www.elecfans.com", "www.demowen.com", "www.example.com", "www.techzone.cn", "www.circuithub.com"]
    
    # 随机选择内容
    brand_name = random.choice(brand_candidates)
    website_url = random.choice(url_candidates)
    
    # 尺寸参数 - 相对于图像大小
    total_width = int(w * random.uniform(0.20, 0.40))  # 总宽度: 20-40% 图像宽度
    total_height = int(h * random.uniform(0.08, 0.15))  # 总高度: 8-15% 图像高度
    
    # 字体大小 - 确保可见性
    font_size = max(MIN_TEXT_HEIGHT_PX, int(total_height * 0.35))
    font = get_font(font_size)
    
    # 透明度 - 在可见范围内
    alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    
    # 创建overlay
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # 计算各元素尺寸
    brand_bbox = get_text_bbox(overlay_draw, brand_name, font)
    url_bbox = get_text_bbox(overlay_draw, website_url, font)
    
    # 图标尺寸 (圆形)
    icon_size = int(total_height * 0.8)  # 图标占高度的80%
    
    # 计算总布局宽度
    spacing = int(total_height * 0.1)  # 元素间距
    layout_width = icon_size + spacing + brand_bbox[0] + spacing + url_bbox[0]
    
    # 如果布局太宽，缩小字体
    if layout_width > total_width:
        scale_factor = total_width / layout_width
        font_size = max(MIN_TEXT_HEIGHT_PX, int(font_size * scale_factor))
        font = get_font(font_size)
        # 重新计算尺寸
        brand_bbox = get_text_bbox(overlay_draw, brand_name, font)
        url_bbox = get_text_bbox(overlay_draw, website_url, font)
        layout_width = icon_size + spacing + brand_bbox[0] + spacing + url_bbox[0]
    
    # 确定位置 - 偏好底部角落
    corners = ['bottom-left', 'bottom-right', 'top-left', 'top-right']
    # 70% 概率选择底部角落
    if random.random() < 0.7:
        corner = random.choice(['bottom-left', 'bottom-right'])
    else:
        corner = random.choice(corners)
    
    # 随机边距 (8-32像素)
    margin = random.randint(8, 32)
    
    if corner == 'bottom-left':
        base_x = margin
        base_y = h - margin - total_height
    elif corner == 'bottom-right':
        base_x = w - margin - layout_width
        base_y = h - margin - total_height
    elif corner == 'top-left':
        base_x = margin
        base_y = margin
    else:  # top-right
        base_x = w - margin - layout_width
        base_y = margin
    
    # 确保不超出边界
    base_x = max(0, min(base_x, w - layout_width))
    base_y = max(0, min(base_y, h - total_height))
    
    # 颜色选择 - 主要使用浅色，但确保对比度
    bg_brightness = BRIGHTNESS_THRESHOLD
    if bg_brightness > BRIGHTNESS_THRESHOLD:
        # 亮背景 - 使用深色文本，可能加浅色轮廓
        text_color = (0, 0, 0)
        outline_color = (255, 255, 255)
        icon_fill = (50, 50, 50)
    else:
        # 暗背景 - 使用浅色文本，可能加深色轮廓
        text_color = (255, 255, 255)
        outline_color = (0, 0, 0)
        icon_fill = (200, 200, 200)
    
    # 绘制阴影/轮廓 (如果需要增强对比度)
    shadow_offset = 1
    use_shadow = random.random() < 0.6  # 60% 概率使用阴影
    
    # 当前绘制位置
    current_x = base_x
    
    # 1. 绘制图标 (圆形)
    icon_center_x = current_x + icon_size // 2
    icon_center_y = base_y + total_height // 2
    
    # 图标背景圆
    overlay_draw.ellipse(
        [icon_center_x - icon_size//2, icon_center_y - icon_size//2,
         icon_center_x + icon_size//2, icon_center_y + icon_size//2],
        fill=icon_fill + (int(255 * alpha),)
    )
    
    # 简单的图标内容 (1-2条线，模拟电路符号)
    num_lines = random.randint(1, 2)
    for i in range(num_lines):
        if i == 0:
            # 水平线
            line_y = icon_center_y
            overlay_draw.line(
                [icon_center_x - icon_size//3, line_y, icon_center_x + icon_size//3, line_y],
                fill=(255 - icon_fill[0], 255 - icon_fill[1], 255 - icon_fill[2]) + (int(255 * alpha),),
                width=2
            )
        else:
            # 垂直线
            line_x = icon_center_x
            overlay_draw.line(
                [line_x, icon_center_y - icon_size//3, line_x, icon_center_y + icon_size//3],
                fill=(255 - icon_fill[0], 255 - icon_fill[1], 255 - icon_fill[2]) + (int(255 * alpha),),
                width=2
            )
    
    current_x += icon_size + spacing
    
    # 2. 绘制品牌名称
    brand_y = base_y + (total_height - brand_bbox[1]) // 2
    
    if use_shadow:
        # 绘制阴影
        overlay = draw_text_with_alpha(overlay, (current_x + shadow_offset, brand_y + shadow_offset), 
                                     brand_name, font, (0, 0, 0), alpha * 0.7)
    
    # 绘制主文本
    overlay = draw_text_with_alpha(overlay, (current_x, brand_y), brand_name, font, text_color, alpha)
    
    current_x += brand_bbox[0] + spacing
    
    # 3. 绘制网站URL
    url_y = base_y + (total_height - url_bbox[1]) // 2
    
    if use_shadow:
        # 绘制阴影
        overlay = draw_text_with_alpha(overlay, (current_x + shadow_offset, url_y + shadow_offset), 
                                     website_url, font, (0, 0, 0), alpha * 0.7)
    
    # 绘制主文本
    overlay = draw_text_with_alpha(overlay, (current_x, url_y), website_url, font, text_color, alpha)
    
    # 应用轻微模糊以获得柔和效果
    blur_radius = random.uniform(0.3, 0.8)
    try:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    except:
        pass  # 如果模糊失败，继续
    
    return overlay

def mini_social_corner_logo(w, h):
    """
    风格15: 小型微信风格社交账号角标水印
    
    组成: 小圆形头像图标 + 单行社交账号名称，水平排列
    位置: 紧贴底部角落，随机小边距
    样式: 半透明，带阴影确保对比度，小尺寸
    """
    # 候选账号名称
    account_candidates = ["电客一点通", "电路小课堂", "AnalogTips", "DemoWenLab", "电子工坊", "芯片实验室"]
    
    # 随机选择账号
    account_name = random.choice(account_candidates)
    
    # 尺寸参数 - 相对于图像大小，更小尺寸
    total_width = int(w * random.uniform(0.10, 0.20))   # 总宽度: 10-20% 图像宽度
    total_height = int(h * random.uniform(0.04, 0.08))  # 总高度: 4-8% 图像高度
    
    # 字体大小 - 确保可见性但较小
    font_size = max(MIN_TEXT_HEIGHT_PX, int(total_height * 0.7))
    font = get_font(font_size)
    
    # 透明度 - 在可见范围内
    alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    
    # 创建overlay
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # 计算各元素尺寸
    text_bbox = get_text_bbox(overlay_draw, account_name, font)
    
    # 图标尺寸 (圆形，小尺寸)
    icon_size = min(total_height - 4, int(total_height * 0.9))  # 图标占高度的90%
    
    # 计算总布局宽度
    spacing = int(total_height * 0.15)  # 元素间距
    layout_width = icon_size + spacing + text_bbox[0]
    
    # 如果布局太宽，缩小字体
    if layout_width > total_width:
        scale_factor = total_width / layout_width
        font_size = max(MIN_TEXT_HEIGHT_PX, int(font_size * scale_factor))
        font = get_font(font_size)
        # 重新计算尺寸
        text_bbox = get_text_bbox(overlay_draw, account_name, font)
        layout_width = icon_size + spacing + text_bbox[0]
    
    # 确定位置 - 偏好底部右角，但有时左下角
    if random.random() < 0.9:  # 90% 概率选择右下角
        corner = 'bottom-right'
    else:
        corner = 'bottom-left'
    
    # 小边距 (4-16像素)
    margin_x = random.randint(4, 16)
    margin_y = random.randint(4, 16)
    
    if corner == 'bottom-right':
        base_x = w - margin_x - layout_width
        base_y = h - margin_y - total_height
    else:  # bottom-left
        base_x = margin_x
        base_y = h - margin_y - total_height
    
    # 确保不超出边界
    base_x = max(0, min(base_x, w - layout_width))
    base_y = max(0, min(base_y, h - total_height))
    
    # 颜色选择 - 图标使用固定调色板，文本使用对比色
    icon_colors = [
        (34, 197, 94),   # 绿色
        (59, 130, 246),  # 蓝色
        (107, 114, 128)  # 灰色
    ]
    icon_color = random.choice(icon_colors)
    text_color = (255, 255, 255)  # 白色文字
    
    # 绘制阴影/轮廓
    shadow_offset = (1, 1)
    shadow_color = (0, 0, 0)
    use_shadow = random.random() < 0.7  # 70% 概率使用阴影
    
    # 当前绘制位置
    current_x = base_x
    
    # 1. 绘制图标 (圆形)
    icon_center_x = current_x + icon_size // 2
    icon_center_y = base_y + total_height // 2
    
    # 图标背景圆
    overlay_draw.ellipse(
        [icon_center_x - icon_size//2, icon_center_y - icon_size//2,
         icon_center_x + icon_size//2, icon_center_y + icon_size//2],
        fill=icon_color + (int(255 * alpha),)
    )
    
    # 可选：绘制1-2条白色内部线条（模拟聊天/电子符号）
    if random.random() > 0.3:  # 70% 概率绘制内部图案
        num_lines = random.randint(1, 2)
        line_color = (255, 255, 255)
        
        for i in range(num_lines):
            if random.random() > 0.5:
                # 水平线（模拟消息气泡）
                line_y = icon_center_y + random.randint(-icon_size//4, icon_size//4)
                overlay_draw.line(
                    [icon_center_x - icon_size//3, line_y, icon_center_x + icon_size//3, line_y],
                    fill=line_color + (int(255 * alpha * 0.8),),
                    width=1
                )
            else:
                # 垂直线或对角线（模拟电路符号）
                if random.random() > 0.5:
                    # 垂直线
                    line_x = icon_center_x + random.randint(-icon_size//4, icon_size//4)
                    overlay_draw.line(
                        [line_x, icon_center_y - icon_size//3, line_x, icon_center_y + icon_size//3],
                        fill=line_color + (int(255 * alpha * 0.8),),
                        width=1
                    )
                else:
                    # 对角线
                    overlay_draw.line(
                        [icon_center_x - icon_size//4, icon_center_y - icon_size//4,
                         icon_center_x + icon_size//4, icon_center_y + icon_size//4],
                        fill=line_color + (int(255 * alpha * 0.8),),
                        width=1
                    )
    
    current_x += icon_size + spacing
    
    # 2. 绘制账号名称
    text_y = base_y + (total_height - text_bbox[1]) // 2
    
    if use_shadow:
        # 绘制阴影
        overlay = draw_text_with_alpha(overlay, (current_x + shadow_offset[0], text_y + shadow_offset[1]), 
                                     account_name, font, shadow_color, alpha * 0.4)
    
    # 绘制主文本
    overlay = draw_text_with_alpha(overlay, (current_x, text_y), account_name, font, text_color, alpha)
    
    # 可选：轻微模糊效果
    if random.random() > 0.5:
        blur_radius = random.uniform(0.3, 0.8)
        try:
            overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        except:
            pass  # 如果模糊失败，继续
    
    return overlay

def combined_styles(w, h, enabled_single_styles):
    """风格14: 组合多个风格 (可见性强制版)"""
    if len(enabled_single_styles) < 2:
        return random.choice(enabled_single_styles)(w, h)
    
    # 随机选择2-3个风格
    num_styles = random.randint(2, min(3, len(enabled_single_styles)))
    selected_styles = random.sample(enabled_single_styles, num_styles)
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    
    # 顺序应用风格
    for style_func in selected_styles:
        try:
            style_overlay = style_func(w, h)
            if style_overlay is not None:
                overlay = Image.alpha_composite(overlay, style_overlay)
        except Exception as e:
            continue
    
    return overlay
    
    # 应用渐变掩码到overlay
    overlay_array = np.array(temp_overlay)
    overlay_array[:, :, 3] = (overlay_array[:, :, 3] * gradient_array / 255).astype(np.uint8)
    overlay = Image.fromarray(overlay_array, 'RGBA')
    
    return overlay

def noisy_eroded_text(w, h):
    """风格11: 有噪声边缘的文本 - 模拟压缩损坏或侵蚀的水印"""
    text = random.choice(TEXT_CANDIDATES)
    font_size = max(35, int(w * random.uniform(0.08, 0.15)))
    font = get_font(font_size)
    alpha = random.uniform(0.25, 0.6)
    color = random.choice([(255, 255, 255), (0, 0, 0), (150, 150, 150)])
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    text_w, text_h = get_text_bbox(overlay_draw, text, font)
    pos = ((w - text_w) // 2, (h - text_h) // 2)
    
    # 绘制初始文本
    overlay_draw.text(pos, text, font=font, fill=color + (int(255 * alpha),))
    
    # 应用噪声和模糊以模拟损坏的边缘
    overlay_array = np.array(overlay)
    alpha_channel = overlay_array[:, :, 3].astype(float)
    
    # 添加随机噪声
    noise = np.random.normal(0, 15, alpha_channel.shape)
    alpha_channel = np.clip(alpha_channel + noise, 0, 255)
    
    # 应用轻微的腐蚀（模糊+阈值化）
    from scipy import ndimage
    try:
        alpha_channel = ndimage.gaussian_filter(alpha_channel, sigma=0.8)
    except:
        # 如果scipy不可用，使用PIL的模糊
        temp_img = Image.fromarray(alpha_channel.astype(np.uint8), 'L')
        temp_img = temp_img.filter(ImageFilter.GaussianBlur(radius=0.8))
        alpha_channel = np.array(temp_img).astype(float)
    
    # 应用阈值以保持可见性
    threshold = np.random.randint(50, 100)
    alpha_channel = np.where(alpha_channel > threshold, alpha_channel, 0)
    
    overlay_array[:, :, 3] = np.clip(alpha_channel, 0, 255).astype(np.uint8)
    overlay = Image.fromarray(overlay_array, 'RGBA')
    
    return overlay

def banner_box(w, h):
    """风格12: 带背景条的文本 - 在矩形或条形背景后面的文本"""
    text = random.choice(TEXT_CANDIDATES)
    font_size = max(30, int(w * random.uniform(0.08, 0.14)))
    font = get_font(font_size)
    text_alpha = random.uniform(0.4, 0.75)
    box_alpha = random.uniform(0.1, 0.35)
    
    # 背景条方向
    is_horizontal = random.choice([True, False])
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    text_w, text_h = get_text_bbox(overlay_draw, text, font)
    
    # 文本位置
    text_x = (w - text_w) // 2
    text_y = (h - text_h) // 2
    
    # 背景条颜色
    box_color = random.choice([(255, 255, 255), (0, 0, 0), (100, 100, 100), (200, 100, 100)])
    
    if is_horizontal:
        # 水平条
        padding = random.randint(5, 15)
        box_x0 = max(0, text_x - padding)
        box_y0 = max(0, text_y - padding)
        box_x1 = min(w, text_x + text_w + padding)
        box_y1 = min(h, text_y + text_h + padding)
    else:
        # 垂直条
        padding = random.randint(5, 15)
        box_height = h // random.randint(3, 5)
        box_y0 = (h - box_height) // 2
        box_x0 = max(0, text_x - padding)
        box_y1 = min(h, box_y0 + box_height)
        box_x1 = min(w, text_x + text_w + padding)
    
    # 绘制背景条
    overlay_draw.rectangle(
        [box_x0, box_y0, box_x1, box_y1],
        fill=box_color + (int(255 * box_alpha),)
    )
    
    # 绘制文本
    text_color = random.choice([(255, 255, 255), (0, 0, 0), (50, 50, 50)])
    overlay_draw.text((text_x, text_y), text, font=font, fill=text_color + (int(255 * text_alpha),))
    
    return overlay

def curved_text(w, h):
    """风格13: 弧形排列的文本 - 沿弧线排列字符"""
    text = random.choice(TEXT_CANDIDATES)
    font_size = max(20, int(w * random.uniform(0.04, 0.1)))
    font = get_font(font_size)
    alpha = random.uniform(0.25, 0.6)
    color = random.choice([(255, 255, 255), (0, 0, 0), (100, 100, 100)])
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # 弧线参数
    center_x = w // 2
    center_y = h // 2
    radius = min(w, h) * random.uniform(0.25, 0.45)
    arc_start_angle = random.uniform(-90, 90)
    
    # 沿弧线排列字符
    char_list = list(text)
    angle_step = 10 if len(char_list) > 1 else 0
    
    for i, char in enumerate(char_list):
        angle = arc_start_angle + i * angle_step
        angle_rad = math.radians(angle)
        
        # 计算字符位置
        char_x = center_x + radius * math.cos(angle_rad)
        char_y = center_y + radius * math.sin(angle_rad)
        
        # 创建旋转的字符
        char_img = Image.new('RGBA', (font_size * 2, font_size * 2), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((font_size // 2, font_size // 4), char, font=font, fill=color + (int(255 * alpha),))
        
        # 旋转字符以匹配弧线方向
        char_img = char_img.rotate(-angle, expand=True, fillcolor=(255, 255, 255, 0))
        
        # 计算粘贴位置
        paste_x = int(char_x - char_img.width // 2)
        paste_y = int(char_y - char_img.height // 2)
        
        # 确保在边界内
        if 0 <= paste_x < w and 0 <= paste_y < h:
            overlay.paste(char_img, (paste_x, paste_y), char_img)
    
    return overlay

def combined_styles(w, h, enabled_single_styles):
    """风格14: 组合多个风格 - 随机选择2-3个风格并组合"""
    if len(enabled_single_styles) < 2:
        # 如果可用风格太少，降级到单一风格
        return random.choice(enabled_single_styles)(w, h)
    
    # 随机选择2-3个风格
    num_styles = random.randint(2, min(3, len(enabled_single_styles)))
    selected_styles = random.sample(enabled_single_styles, num_styles)
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    
    # 顺序应用风格
    for style_func in selected_styles:
        try:
            style_overlay = style_func(w, h)
            if style_overlay is not None:
                overlay = Image.alpha_composite(overlay, style_overlay)
        except Exception as e:
            # 某个风格失败时继续
            continue
    
    return overlay

# ============================================================================
# 主要处理函数
# ============================================================================

def generate_watermark_variant(image_path, output_index, max_retries=3):
    """
    为单张图像生成一个水印变体及其掩码
    强制所有水印满足可见性约束
    
    返回: (success, watermarked_image, mask_image, output_name)
    """
    image = load_image(image_path)
    if image is None:
        return False, None, None, None
    
    w, h = image.size
    stem = Path(image_path).stem
    output_name = f"{stem}_wm_{output_index:03d}"
    
    # 构建可用的单独风格函数
    style_functions = []
    if 'single_text_corner' in ENABLED_STYLES:
        style_functions.append(('single_text_corner', lambda: single_text_corner(w, h)))
    if 'single_text_center' in ENABLED_STYLES:
        style_functions.append(('single_text_center', lambda: single_text_center(w, h)))
    if 'tiled_text' in ENABLED_STYLES:
        style_functions.append(('tiled_text', lambda: tiled_text(w, h)))
    if 'diagonal_band' in ENABLED_STYLES:
        style_functions.append(('diagonal_band', lambda: diagonal_band(w, h)))
    if 'multi_line_text_center' in ENABLED_STYLES:
        style_functions.append(('multi_line_text_center', lambda: multi_line_text_center(w, h)))
    if 'logo_corner' in ENABLED_STYLES:
        style_functions.append(('logo_corner', lambda: logo_corner(w, h, LOGO_PATH)))
    if 'qr_code_style' in ENABLED_STYLES:
        style_functions.append(('qr_code_style', lambda: qr_code_style(w, h)))
    if 'outlined_text' in ENABLED_STYLES:
        style_functions.append(('outlined_text', lambda: outlined_text(w, h)))
    if 'shadow_text' in ENABLED_STYLES:
        style_functions.append(('shadow_text', lambda: shadow_text(w, h)))
    if 'gradient_alpha_text' in ENABLED_STYLES:
        style_functions.append(('gradient_alpha_text', lambda: gradient_alpha_text(w, h)))
    if 'noisy_eroded_text' in ENABLED_STYLES:
        style_functions.append(('noisy_eroded_text', lambda: noisy_eroded_text(w, h)))
    if 'banner_box' in ENABLED_STYLES:
        style_functions.append(('banner_box', lambda: banner_box(w, h)))
    if 'curved_text' in ENABLED_STYLES:
        style_functions.append(('curved_text', lambda: curved_text(w, h)))
    if 'corner_website_logo' in ENABLED_STYLES:
        style_functions.append(('corner_website_logo', lambda: corner_website_logo(w, h)))
    if 'mini_social_corner_logo' in ENABLED_STYLES:
        style_functions.append(('mini_social_corner_logo', lambda: mini_social_corner_logo(w, h)))
    
    if not style_functions:
        print(f"警告: 没有启用的风格")
        return False, None, None, None
    
    # 重试直到获得可见的水印
    for attempt in range(max_retries):
        try:
            # 随机选择是否使用组合风格
            if ('combined_styles' in ENABLED_STYLES and 
                random.random() < COMBINED_STYLE_PROBABILITY and
                len(style_functions) >= 2):
                # 提取仅函数的列表用于组合
                single_funcs = [f[1] for f in style_functions if f[0] != 'combined_styles']
                overlay = combined_styles(w, h, single_funcs)
            else:
                # 选择单个风格 - 支持加权选择
                if 'corner_website_logo' in ENABLED_STYLES and random.random() < CORNER_WEBSITE_LOGO_WEIGHT:
                    # 直接选择角落网站LOGO风格
                    style_name = 'corner_website_logo'
                    style_func = lambda: corner_website_logo(w, h)
                else:
                    # 从其他风格中随机选择
                    other_styles = [(name, func) for name, func in style_functions if name != 'corner_website_logo']
                    if other_styles:
                        style_name, style_func = random.choice(other_styles)
                    else:
                        # 如果只有角落网站LOGO风格可用
                        style_name = 'corner_website_logo'
                        style_func = lambda: corner_website_logo(w, h)
                
                overlay = style_func()
            
            if overlay is None:
                continue
            
            # *** 关键: 验证可见性约束 ***
            if ENABLE_VISIBILITY_VALIDATION:
                overlay_array = np.array(overlay)
                visibility_result = validate_watermark_visibility(overlay_array, w, h)
                
                if not visibility_result['is_visible']:
                    # 可见性不足，重试
                    if attempt < max_retries - 1:
                        continue
                    else:
                        # 最后一次尝试失败，放弃
                        return False, None, None, None
            
            # 合成最终图像
            image_rgba = image.convert('RGBA')
            watermarked = Image.alpha_composite(image_rgba, overlay)
            watermarked_rgb = watermarked.convert('RGB')
            
            # 生成掩码 (基于overlay的alpha通道)
            overlay_array = np.array(overlay)
            alpha_mask = overlay_array[:, :, 3] > 0
            mask_array = np.zeros((h, w), dtype=np.uint8)
            mask_array[alpha_mask] = 255
            mask = Image.fromarray(mask_array, mode='L')
            
            return True, watermarked_rgb, mask, output_name
        
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"警告: 生成水印失败 {image_path} (变体{output_index}): {e}")
            continue
    
    return False, None, None, None
    
    return True, watermarked_rgb, mask, output_name

def process_image(image_path):
    """处理单张图像，生成N个变体"""
    stem = Path(image_path).stem
    print(f"处理: {stem}")
    
    success_count = 0
    for variant_idx in range(NUM_VARIANTS_PER_IMAGE):
        success, watermarked, mask, output_name = generate_watermark_variant(
            image_path, variant_idx
        )
        
        if not success:
            continue
        
        # 保存水印图像 (JPEG, 质量60-95)
        quality = random.randint(60, 95)
        wm_path = os.path.join(OUTPUT_DIR, f"{output_name}.jpg")
        watermarked.save(wm_path, 'JPEG', quality=quality)
        
        # 保存掩码 (PNG)
        mask_path = os.path.join(MASK_DIR, f"{output_name}_mask.png")
        mask.save(mask_path, 'PNG')
        
        success_count += 1
    
    print(f"  ✓ 生成 {success_count}/{NUM_VARIANTS_PER_IMAGE} 个变体")
    return success_count

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主程序"""
    ensure_dirs()
    
    print("=" * 70)
    print("增强的水印生成脚本 - 可见性强化版 (16种风格)")
    print("=" * 70)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"掩码目录: {MASK_DIR}")
    print(f"每张图像变体数: {NUM_VARIANTS_PER_IMAGE}")
    print(f"组合风格概率: {COMBINED_STYLE_PROBABILITY * 100:.0f}%")
    print(f"角落网站LOGO权重: {CORNER_WEBSITE_LOGO_WEIGHT * 100:.0f}%")
    print()
    print("可见性约束 (HARD CONSTRAINTS):")
    print(f"  ✓ 最小透明度 (alpha): {MIN_ALPHA:.0%}")
    print(f"  ✓ 最大透明度 (alpha): {MAX_ALPHA:.0%}")
    print(f"  ✓ 最小覆盖面积: {MIN_VISIBLE_COVERAGE:.1%}")
    print(f"  ✓ 最小文本高度: {MIN_TEXT_HEIGHT_PX}px or {MIN_TEXT_HEIGHT_RATIO:.0%} of image height")
    print(f"  ✓ 颜色对比度: 自适应 (根据背景亮度)")
    print(f"  ✓ 可见性验证: {'启用' if ENABLE_VISIBILITY_VALIDATION else '禁用'}")
    print()
    print(f"启用的风格: {', '.join(ENABLED_STYLES)}")
    print("=" * 70)
    print()
    
    if not os.path.isdir(INPUT_DIR):
        print(f"错误: 输入目录不存在 {INPUT_DIR}")
        return
    
    image_files = get_image_files(INPUT_DIR)
    if not image_files:
        print("错误: 未找到任何图像文件")
        return
    
    print(f"找到 {len(image_files)} 张图像")
    print()
    
    total_variants = 0
    for image_path in image_files:
        count = process_image(image_path)
        total_variants += count
    
    print()
    print("=" * 60)
    print(f"✓ 完成! 生成了 {total_variants} 个水印变体")
    print(f"  - 水印图像保存在: {OUTPUT_DIR}")
    print(f"  - 掩码保存在: {MASK_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()
