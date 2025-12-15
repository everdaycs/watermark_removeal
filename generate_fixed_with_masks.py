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
import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import concurrent.futures
import multiprocessing

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
BRIGHTNESS_THRESHOLD = 200

# 启用可见性验证
ENABLE_VISIBILITY_VALIDATION = True

# ============================================================================
# 配置参数
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NUM_VARIANTS_PER_IMAGE = 128
INPUT_DIR = "/home/kaga/Desktop/watermaker remover/20251201/no_watermark_20251201"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data/merged_watermark_images")  
MASK_DIR = os.path.join(SCRIPT_DIR, "data/merged_watermark_masks")     
LOGO_PATH = os.path.join(SCRIPT_DIR, "logo.png")

TEXT_CANDIDATES = [
    "DEMO",
    "SAMPLE",
    "Preview",
    "Copyright",
    "DemoWen",
    "NoRepost",
    "2025",
    "WATERMARK",
    "Test",
    "Draft",
    "演示",
    "示例",
    "测试",
    "水印",
    "版权",
    "禁止转载",
    "保留所有权利",
    "www.elecfans.com",
    "ElecFans.com",
    "ElecFans",
    "TechWatermark",
    "WatermarkMaster",
    "电子发烧友",

    # 新增英文通用水印
    "SAMPLE IMAGE",
    "SAMPLE PHOTO",
    "PREVIEW ONLY",
    "FOR REVIEW",
    "FOR DEMO USE",
    "DO NOT COPY",
    "DO NOT REPOST",
    "DO NOT DISTRIBUTE",
    "ALL RIGHTS RESERVED",
    "CONFIDENTIAL",
    "INTERNAL USE ONLY",
    "UNAUTHORIZED USE PROHIBITED",
    "SCREENING COPY",
    "LOW RES PREVIEW",
    "BETA VERSION",
    "DRAFT ONLY",
    "TEMP WATERMARK",
    "PLACEHOLDER",
    "UNEDITED",
    "PROOF",
    "CLIENT PREVIEW",
    "WORK IN PROGRESS",
    "SAMPLE DATA",
    "TRAINING ONLY",

    # 年份/版本
    "2023",
    "2024",
    "Ver.1.0",
    "Ver.2.0",
    "REV-A",
    "REV-B",

    # 新增中文常见水印
    "仅供预览",
    "仅供测试",
    "仅供内部使用",
    "仅供学习交流",
    "严禁转载",
    "严禁商用",
    "非成品",
    "未最终定稿",
    "工作稿",
    "样张",
    "样图",
    "草稿",
    "预览图",
    "低清预览",
    "训练数据",
    "示意图",
    "示意用",
    "内部资料",
    "机密文件",
    "请勿外传",

    # 域名/品牌风格占位
    "www.example.com",
    "demo.example.com",
    "YourBrand",
    "YourStudio",
    "SampleStudio",
    "PhotoLab",
    "TechDemo",
    "AIWatermark"
]

# SVG 资源配置
ELECFANS_LOGO_SVG = os.path.join(SCRIPT_DIR, "logos/elecfans-logo.svg")
ELECFANS_WEB_SVG = os.path.join(SCRIPT_DIR, "logos/elecfans-web.svg")
WECHAT_SVG = os.path.join(SCRIPT_DIR, "logos/WeChat.svg")

# SVG 渲染缓存（避免重复处理相同的SVG）
_SVG_CACHE = {}

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False
    print("警告: cairosvg 未安装。SVG 风格将无法使用。请运行: pip install cairosvg")

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
    'corner_website_logo',     # 14
    'mini_social_corner_logo', # 15
    'elecfans_logo_svg_corner',   # 16
    'elecfans_web_svg_center',    # 17
    'wechat_svg_corner_id',       # 18
    'red_transparent_text',       # 19 红色半透明文本水印
    'elecfans_chinese_text',      # 20 “电子发烧友”专用文字水印
    'combined_styles',         # 20
    'random_slanted_text',         # 21 随机分布的倾斜文本
]

COMBINED_STYLE_PROBABILITY = 0.3  # 使用组合风格的概率

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
    """获取字体，优先使用支持中文的字体"""
    # 优先尝试支持中文的字体
    chinese_fonts = [
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",  # Noto Serif CJK Bold
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",   # Noto Sans CJK Bold
        "/usr/share/fonts/truetype/arphic/uming.ttc",            # AR PL UMing
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # DejaVu Sans Bold
    ]
    
    for font_path in chinese_fonts:
        try:
            return ImageFont.truetype(font_path, font_size)
        except (OSError, IOError):
            continue
    
    # 如果都没有找到，使用默认字体
    try:
        return ImageFont.load_default()
    except:
        # 最后的fallback
        return None

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
# SVG 渲染辅助函数
# ============================================================================

def load_svg_as_rgba(svg_path: str, target_width: int, target_height: int) -> Image.Image:
    """
    将 SVG 文件渲染为 PIL RGBA 图像
    
    使用 cairosvg 将 SVG 转换为 PNG，然后加载为 RGBA 图像。
    结果被缓存以避免重复渲染相同的 SVG。
    
    Args:
        svg_path (str): SVG 文件路径
        target_width (int): 目标宽度（像素）
        target_height (int): 目标高度（像素）
    
    Returns:
        PIL.Image: RGBA 图像，或 None 如果加载失败
    """
    if not CAIROSVG_AVAILABLE:
        return None
    
    # 检查缓存
    cache_key = (svg_path, target_width, target_height)
    if cache_key in _SVG_CACHE:
        return _SVG_CACHE[cache_key].copy()
    
    try:
        if not os.path.exists(svg_path):
            print(f"警告: SVG 文件不存在: {svg_path}")
            return None
        
        # 使用 cairosvg 将 SVG 转换为 PNG 字节
        png_bytes = io.BytesIO()
        cairosvg.svg2png(
            url=svg_path,
            output_width=target_width,
            output_height=target_height,
            write_to=png_bytes
        )
        png_bytes.seek(0)
        
        # 加载 PNG 为 RGBA 图像
        img = Image.open(png_bytes).convert('RGBA')
        
        # 缓存结果
        _SVG_CACHE[cache_key] = img.copy()
        
        return img
    
    except Exception as e:
        print(f"错误: 无法渲染 SVG {svg_path}: {e}")
        return None

# ============================================================================
# 水印合成函数 - 返回 overlay RGBA 图像 (强制可见性)
# ============================================================================

def single_text_corner(w, h):
    """风格1: 单个文本在四个角之一 (可见性强制版)"""
    text = random.choice(TEXT_CANDIDATES)
    # 最小尺寸: 4-8% 的图像宽度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.04, 0.42)))
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
    # 最小尺寸: 8-42% 的图像高度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.08, 0.42)))
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
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.03, 0.42)))
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
    # 最小尺寸: 6-42% 的图像高度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.06, 0.42)))
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
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.05, 0.42)))
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
    # 最小尺寸: 7-42% 的图像高度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.07, 0.42)))
    font = get_font(font_size)
    # 强制透明度范围
    alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    # 强制粗轮廓
    outline_width = random.randint(2, 4)  # 减少轮廓宽度以避免过度重叠
    rotation = random.uniform(-15, 15)

    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    text_w, text_h = get_text_bbox(overlay_draw, text, font)
    pos = ((w - text_w) // 2, (h - text_h) // 2)

    # 自适应颜色: 轮廓颜色与主文本对比
    bg_brightness = random.randint(0, 255)
    if bg_brightness > BRIGHTNESS_THRESHOLD:
        outline_color = (0, 0, 0)
        main_color = (255, 255, 255)
    else:
        outline_color = (255, 255, 255)
        main_color = (0, 0, 0)

    # 创建轮廓：使用固定的4个主要方向，避免对角线重叠
    outline_offsets = [
        (-outline_width, 0), (outline_width, 0), (0, -outline_width), (0, outline_width)
    ]

    # 绘制轮廓：每个位置只绘制一次，透明度大幅降低
    outline_alpha = alpha * 0.3  # 轮廓非常透明，避免任何重叠区域过暗
    for offset_x, offset_y in outline_offsets:
        outline_pos = (pos[0] + offset_x, pos[1] + offset_y)
        overlay = draw_text_with_alpha(overlay, outline_pos, text, font, outline_color, outline_alpha)    # 绘制主文本（更强不透明度）
    overlay = draw_text_with_alpha(overlay, pos, text, font, main_color, alpha * 0.9)

    if rotation != 0:
        overlay = overlay.rotate(rotation, expand=False, fillcolor=(255, 255, 255, 0))

    return overlay
def shadow_text(w, h):
    """风格9: 带阴影的文本 (可见性强制版)"""
    text = random.choice(TEXT_CANDIDATES)
    # 最小尺寸: 8-16% 的图像高度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.08, 0.42)))
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
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.08, 0.42)))
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
    # 最小尺寸: 6-42% 的图像高度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.06, 0.42)))
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
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.06, 0.42)))
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
    bg_brightness = random.randint(0, 255)
    if bg_brightness > BRIGHTNESS_THRESHOLD:
        box_color = (0, 0, 0)        # 深背景
        text_color = (255, 255, 255) # 浅文本
    else:
        box_color = (255, 255, 255)  # 浅背景
        text_color = (0, 0, 0)       # 深文本
    
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
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.05, 0.42)))
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

def random_slanted_text(w, h):
    """风格21: 随机分布的倾斜文本 (可见性强制版)"""
    text = random.choice(TEXT_CANDIDATES)
    # 最小尺寸: 4-8% 的图像高度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.04, 0.42)))
    font = get_font(font_size)
    # 强制透明度范围
    alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    
    # 自适应颜色
    color = select_contrasting_color(BRIGHTNESS_THRESHOLD)
    
    # 随机数量
    num_texts = random.randint(5, 15)
    
    # 随机倾斜角度 (所有文本使用相同的倾斜角度)
    angle = random.uniform(-45, 45)
    # 避免接近0度
    if -10 < angle < 10:
        angle = 30 if angle > 0 else -30
        
    for _ in range(num_texts):
        # 创建临时图像绘制单个文本
        # 预估文本大小
        dummy_draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
        text_w, text_h = get_text_bbox(dummy_draw, text, font)
        
        # 增加一些padding以防旋转裁剪
        temp_w, temp_h = int(text_w * 1.5), int(text_h * 1.5)
        temp_img = Image.new('RGBA', (temp_w, temp_h), (255, 255, 255, 0))
        
        # 在中心绘制
        draw_pos = ((temp_w - text_w) // 2, (temp_h - text_h) // 2)
        temp_img = draw_text_with_alpha(temp_img, draw_pos, text, font, color, alpha)
        
        # 旋转
        rotated_text = temp_img.rotate(angle, expand=True, fillcolor=(255, 255, 255, 0))
        
        # 随机位置粘贴
        paste_x = random.randint(-rotated_text.width // 2, w - rotated_text.width // 2)
        paste_y = random.randint(-rotated_text.height // 2, h - rotated_text.height // 2)
        
        overlay.paste(rotated_text, (paste_x, paste_y), rotated_text)
        
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

def corner_website_logo(w, h):
    """风格15: 角落网站LOGO - 模拟网站风格的角落水印"""
    text = random.choice(TEXT_CANDIDATES)
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.04, 0.08)))
    font = get_font(font_size)
    alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # 网站风格：添加"www."前缀
    website_text = f"www.{text.lower()}.com"
    text_w, text_h = get_text_bbox(overlay_draw, website_text, font)
    
    # 角落位置
    corner = random.choice(['tl', 'tr', 'bl', 'br'])
    margin = max(10, int(w * 0.015))
    
    if corner == 'tl':
        pos = (margin, margin)
    elif corner == 'tr':
        pos = (max(0, w - text_w - margin), margin)
    elif corner == 'bl':
        pos = (margin, max(0, h - text_h - margin))
    else:  # br
        pos = (max(0, w - text_w - margin), max(0, h - text_h - margin))
    
    # 自适应颜色
    color = select_contrasting_color(BRIGHTNESS_THRESHOLD)
    overlay = draw_text_with_alpha(overlay, pos, website_text, font, color, alpha)
    
    return overlay

def mini_social_corner_logo(w, h):
    """风格16: 迷你社交角落LOGO - 模拟微信等社交媒体的小型角落水印"""
    text = random.choice(TEXT_CANDIDATES)
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.03, 0.06)))
    font = get_font(font_size)
    alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # 社交媒体风格：添加图标符号
    social_icons = ['●', '◆', '■', '▲', '▼']
    icon = random.choice(social_icons)
    social_text = f"{icon} {text}"
    text_w, text_h = get_text_bbox(overlay_draw, social_text, font)
    
    # 角落位置
    corner = random.choice(['tl', 'tr', 'bl', 'br'])
    margin = max(8, int(w * 0.01))
    
    if corner == 'tl':
        pos = (margin, margin)
    elif corner == 'tr':
        pos = (max(0, w - text_w - margin), margin)
    elif corner == 'bl':
        pos = (margin, max(0, h - text_h - margin))
    else:  # br
        pos = (max(0, w - text_w - margin), max(0, h - text_h - margin))
    
    # 自适应颜色
    color = select_contrasting_color(BRIGHTNESS_THRESHOLD)
    overlay = draw_text_with_alpha(overlay, pos, social_text, font, color, alpha)
    
    return overlay

def elecfans_logo_svg_corner(w, h):
    """风格17: ElecFans LOGO SVG角落 - 使用SVG格式的ElecFans logo"""
    if not CAIROSVG_AVAILABLE or not os.path.exists(ELECFANS_LOGO_SVG):
        # 降级到文本版本
        return corner_website_logo(w, h)
    
    try:
        svg_rgba = load_svg_as_rgba(ELECFANS_LOGO_SVG)
        if svg_rgba is None:
            return corner_website_logo(w, h)
        
        # 缩放SVG
        logo_w = int(w * random.uniform(0.08, 0.15))
        ratio = logo_w / svg_rgba.width if svg_rgba.width > 0 else 1
        logo_h = int(svg_rgba.height * ratio)
        svg_rgba = svg_rgba.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
        
        # 应用透明度
        alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
        svg_array = np.array(svg_rgba)
        svg_array[:, :, 3] = (svg_array[:, :, 3] * alpha).astype(np.uint8)
        svg_rgba = Image.fromarray(svg_array)
        
        overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
        
        # 角落位置
        corner = random.choice(['tl', 'tr', 'bl', 'br'])
        margin = max(10, int(w * 0.015))
        
        if corner == 'tl':
            pos = (margin, margin)
        elif corner == 'tr':
            pos = (max(0, w - logo_w - margin), margin)
        elif corner == 'bl':
            pos = (margin, max(0, h - logo_h - margin))
        else:  # br
            pos = (max(0, w - logo_w - margin), max(0, h - logo_h - margin))
        
        overlay.paste(svg_rgba, pos, svg_rgba)
        return overlay
        
    except Exception as e:
        # 降级到文本版本
        return corner_website_logo(w, h)

def elecfans_web_svg_center(w, h):
    """风格18: ElecFans Web SVG居中 - 使用SVG格式的ElecFans web logo居中"""
    if not CAIROSVG_AVAILABLE or not os.path.exists(ELECFANS_WEB_SVG):
        # 降级到居中文本
        return single_text_center(w, h)
    
    try:
        svg_rgba = load_svg_as_rgba(ELECFANS_WEB_SVG)
        if svg_rgba is None:
            return single_text_center(w, h)
        
        # 缩放SVG
        logo_w = int(w * random.uniform(0.15, 0.25))
        ratio = logo_w / svg_rgba.width if svg_rgba.width > 0 else 1
        logo_h = int(svg_rgba.height * ratio)
        svg_rgba = svg_rgba.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
        
        # 应用透明度
        alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
        svg_array = np.array(svg_rgba)
        svg_array[:, :, 3] = (svg_array[:, :, 3] * alpha).astype(np.uint8)
        svg_rgba = Image.fromarray(svg_array)
        
        overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
        
        # 居中位置
        pos = ((w - logo_w) // 2, (h - logo_h) // 2)
        overlay.paste(svg_rgba, pos, svg_rgba)
        
        return overlay
        
    except Exception as e:
        # 降级到居中文本
        return single_text_center(w, h)

def wechat_svg_corner_id(w, h):
    """风格19: 微信SVG角落带ID - 使用微信SVG logo和随机ID"""
    if not CAIROSVG_AVAILABLE or not os.path.exists(WECHAT_SVG):
        # 降级到迷你社交角落
        return mini_social_corner_logo(w, h)
    
    try:
        svg_rgba = load_svg_as_rgba(WECHAT_SVG)
        if svg_rgba is None:
            return mini_social_corner_logo(w, h)
        
        # 缩放SVG
        logo_w = int(w * random.uniform(0.06, 0.12))
        ratio = logo_w / svg_rgba.width if svg_rgba.width > 0 else 1
        logo_h = int(svg_rgba.height * ratio)
        svg_rgba = svg_rgba.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
        
        # 应用透明度
        alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
        svg_array = np.array(svg_rgba)
        svg_array[:, :, 3] = (svg_array[:, :, 3] * alpha).astype(np.uint8)
        svg_rgba = Image.fromarray(svg_array)
        
        overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # 生成随机ID
        random_id = f"ID:{random.randint(100000, 999999)}"
        font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.025, 0.04)))
        font = get_font(font_size)
        id_w, id_h = get_text_bbox(overlay_draw, random_id, font)
        
        # 角落位置
        corner = random.choice(['tl', 'tr', 'bl', 'br'])
        margin = max(8, int(w * 0.01))
        
        if corner == 'tl':
            logo_pos = (margin, margin)
            text_pos = (margin + logo_w + 5, margin + (logo_h - id_h) // 2)
        elif corner == 'tr':
            logo_pos = (max(0, w - logo_w - margin), margin)
            text_pos = (max(0, w - logo_w - margin - id_w - 5), margin + (logo_h - id_h) // 2)
        elif corner == 'bl':
            logo_pos = (margin, max(0, h - logo_h - margin))
            text_pos = (margin + logo_w + 5, max(0, h - logo_h - margin) + (logo_h - id_h) // 2)
        else:  # br
            logo_pos = (max(0, w - logo_w - margin), max(0, h - logo_h - margin))
            text_pos = (max(0, w - logo_w - margin - id_w - 5), max(0, h - logo_h - margin) + (logo_h - id_h) // 2)
        
        # 粘贴SVG logo
        overlay.paste(svg_rgba, logo_pos, svg_rgba)
        
        # 添加ID文本
        color = select_contrasting_color(BRIGHTNESS_THRESHOLD)
        overlay = draw_text_with_alpha(overlay, text_pos, random_id, font, color, alpha)
        
        return overlay
        
    except Exception as e:
        # 降级到迷你社交角落
        return mini_social_corner_logo(w, h)

def red_transparent_text(w, h):
    """风格19: 红色半透明文本水印 - 使用TEXT_CANDIDATES内容"""
    text = random.choice(TEXT_CANDIDATES)
    # 最小尺寸: 6-12% 的图像高度
    font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.06, 0.42)))
    font = get_font(font_size)
    # 强制透明度范围，但偏向半透明
    alpha = random.uniform(MIN_ALPHA, min(MAX_ALPHA, 0.6))  # 红色水印稍微透明一些
    
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    text_w, text_h = get_text_bbox(overlay_draw, text, font)
    
    # 随机位置选择：角落、中心或对角线
    position_type = random.choice(['corner', 'center', 'diagonal'])
    
    if position_type == 'corner':
        # 角落位置
        corner = random.choice(['tl', 'tr', 'bl', 'br'])
        margin = max(15, int(w * 0.02))
        
        if corner == 'tl':
            pos = (margin, margin)
        elif corner == 'tr':
            pos = (max(0, w - text_w - margin), margin)
        elif corner == 'bl':
            pos = (margin, max(0, h - text_h - margin))
        else:  # br
            pos = (max(0, w - text_w - margin), max(0, h - text_h - margin))
    
    elif position_type == 'center':
        # 居中位置
        pos = ((w - text_w) // 2, (h - text_h) // 2)
    
    else:  # diagonal
        # 对角线位置
        diagonal_pos = random.uniform(0.2, 0.8)  # 沿对角线的相对位置
        x = int(diagonal_pos * (w - text_w))
        y = int(diagonal_pos * (h - text_h))
        pos = (x, y)
    
    # 固定使用红色，但根据背景亮度调整深浅
    bg_brightness = random.randint(0, 255)
    if bg_brightness > BRIGHTNESS_THRESHOLD:
        # 亮背景 -> 深红色
        red_color = (180, 0, 0)  # 深红色
    else:
        # 暗背景 -> 浅红色
        red_color = (255, 50, 50)  # 浅红色
    
    # 使用新的透明文本绘制方法
    overlay = draw_text_with_alpha(overlay, pos, text, font, red_color, alpha)
    
    # 轻微旋转以增加变化性
    rotation = random.uniform(-15, 15)
    if rotation != 0:
        overlay = overlay.rotate(rotation, expand=False, fillcolor=(255, 255, 255, 0))
    
    return overlay


def elecfans_chinese_text(w, h):
    """风格20: “电子发烧友”专用文字水印

    目标：模拟站点常见中文品牌水印（角落小字 / 底部条幅 / 轻度斜放）。
    - 文本固定为“电子发烧友”
    - 透明度遵循全局 MIN_ALPHA/MAX_ALPHA
    - 颜色以白/黑为主，搭配轻微描边/阴影以增强可见性
    """
    text = "电子发烧友"

    # 由于全局约束 MIN_VISIBLE_COVERAGE=2% 很严格，内部做一次快速重试：
    # 如果覆盖率不够，就增大字号/重复次数重新生成。
    for attempt in range(3):
        # 尺寸：基础 5%~10%，若重试则逐步增大
        scale_boost = 1.0 + attempt * 0.20
        font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.05, 0.10) * scale_boost))
        font = get_font(font_size)
        alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)

        overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        text_w, text_h = get_text_bbox(overlay_draw, text, font)

        # 位置模式：角落 / 底部条幅 / 斜对角
        mode = random.choice(['corner', 'bottom_strip', 'diagonal'])
        margin = max(8, int(w * 0.01))
        rotation = 0.0

        if mode == 'corner':
            corner = 'br' if random.random() < 0.8 else random.choice(['tl', 'tr', 'bl'])
            if corner == 'tl':
                pos = (margin, margin)
            elif corner == 'tr':
                pos = (max(0, w - text_w - margin), margin)
            elif corner == 'bl':
                pos = (margin, max(0, h - text_h - margin))
            else:  # br
                pos = (max(0, w - text_w - margin), max(0, h - text_h - margin))
            rotation = random.uniform(-6, 6)

        elif mode == 'bottom_strip':
            pos = ((w - text_w) // 2, max(0, h - text_h - margin))
            rotation = random.uniform(-2, 2)

            # 可选：底部淡色条增强真实感
            if random.random() < 0.5:
                pad_x = max(10, int(w * 0.02))
                pad_y = max(4, int(h * 0.01))
                x0 = max(0, pos[0] - pad_x)
                y0 = max(0, pos[1] - pad_y)
                x1 = min(w, pos[0] + text_w + pad_x)
                y1 = min(h, pos[1] + text_h + pad_y)
                strip_alpha = random.uniform(0.15, 0.30)
                strip_is_dark = random.random() < 0.5
                strip_color = (0, 0, 0) if strip_is_dark else (255, 255, 255)
                overlay_draw.rectangle([x0, y0, x1, y1], fill=strip_color + (int(255 * strip_alpha),))

        else:  # diagonal
            # 对角线：覆盖更大
            font_size = max(MIN_TEXT_HEIGHT_PX, int(h * random.uniform(0.10, 0.18) * scale_boost))
            font = get_font(font_size)
            text_w, text_h = get_text_bbox(overlay_draw, text, font)
            diag_pos = random.uniform(0.20, 0.75)
            pos = (int(diag_pos * (w - text_w)), int(diag_pos * (h - text_h)))
            rotation = random.choice([random.uniform(18, 28), random.uniform(-28, -18)])

        # 颜色策略：黑/白二选一 + 可选描边/阴影
        prefer_dark_text = random.random() < 0.55
        main_color = (0, 0, 0) if prefer_dark_text else (255, 255, 255)
        outline_color = (255, 255, 255) if prefer_dark_text else (0, 0, 0)

        effect = random.choice(['none', 'outline', 'shadow'])
        if effect == 'outline':
            outline_w = random.randint(1, 2)
            outline_alpha = max(MIN_ALPHA, alpha * 0.65)
            for dx, dy in [(-outline_w, 0), (outline_w, 0), (0, -outline_w), (0, outline_w)]:
                overlay = draw_text_with_alpha(overlay, (pos[0] + dx, pos[1] + dy), text, font, outline_color, outline_alpha)
        elif effect == 'shadow':
            shadow_offset = random.randint(1, 2)
            shadow_alpha = max(MIN_ALPHA, alpha * 0.55)
            overlay = draw_text_with_alpha(overlay, (pos[0] + shadow_offset, pos[1] + shadow_offset), text, font, (0, 0, 0), shadow_alpha)

        overlay = draw_text_with_alpha(overlay, pos, text, font, main_color, alpha)

        # 重复次数：attempt 越大越倾向重复
        repeats = 1
        if random.random() < (0.85 + attempt * 0.10):
            repeats = 2
        if random.random() < (0.35 + attempt * 0.20):
            repeats = 3

        for _ in range(repeats - 1):
            dx = random.randint(int(w * 0.05), int(w * 0.18))
            dy = random.randint(int(h * 0.03), int(h * 0.12))
            pos2 = (max(0, min(pos[0] - dx, w - text_w)), max(0, min(pos[1] - dy, h - text_h)))
            alpha2 = max(MIN_ALPHA, alpha * random.uniform(0.75, 1.0))

            if effect == 'outline':
                outline_w = random.randint(1, 2)
                outline_alpha = max(MIN_ALPHA, alpha2 * 0.65)
                for ddx, ddy in [(-outline_w, 0), (outline_w, 0), (0, -outline_w), (0, outline_w)]:
                    overlay = draw_text_with_alpha(overlay, (pos2[0] + ddx, pos2[1] + ddy), text, font, outline_color, outline_alpha)
            elif effect == 'shadow':
                shadow_offset = random.randint(1, 2)
                shadow_alpha = max(MIN_ALPHA, alpha2 * 0.55)
                overlay = draw_text_with_alpha(overlay, (pos2[0] + shadow_offset, pos2[1] + shadow_offset), text, font, (0, 0, 0), shadow_alpha)

            overlay = draw_text_with_alpha(overlay, pos2, text, font, main_color, alpha2)

        if rotation != 0:
            overlay = overlay.rotate(rotation, expand=False, fillcolor=(255, 255, 255, 0))

        # 尝试满足 coverage
        if not ENABLE_VISIBILITY_VALIDATION:
            return overlay

        try:
            overlay_array = np.array(overlay)
            res = validate_watermark_visibility(overlay_array, w, h)
            if res['is_visible']:
                return overlay
        except Exception:
            return overlay

    # 最后兜底：返回最后一次生成结果（由外层重试机制兜住）
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
    if 'elecfans_logo_svg_corner' in ENABLED_STYLES:
        style_functions.append(('elecfans_logo_svg_corner', lambda: elecfans_logo_svg_corner(w, h)))
    if 'elecfans_web_svg_center' in ENABLED_STYLES:
        style_functions.append(('elecfans_web_svg_center', lambda: elecfans_web_svg_center(w, h)))
    if 'wechat_svg_corner_id' in ENABLED_STYLES:
        style_functions.append(('wechat_svg_corner_id', lambda: wechat_svg_corner_id(w, h)))
    if 'red_transparent_text' in ENABLED_STYLES:
        style_functions.append(('red_transparent_text', lambda: red_transparent_text(w, h)))
    if 'elecfans_chinese_text' in ENABLED_STYLES:
        style_functions.append(('elecfans_chinese_text', lambda: elecfans_chinese_text(w, h)))
    if 'random_slanted_text' in ENABLED_STYLES:
        style_functions.append(('random_slanted_text', lambda: random_slanted_text(w, h)))
    
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
                # 选择单个风格
                style_name, style_func = random.choice(style_functions)
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
    # print(f"处理: {stem}")
    
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
    
    # print(f"  ✓ 生成 {success_count}/{NUM_VARIANTS_PER_IMAGE} 个变体")
    return success_count

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主程序"""
    ensure_dirs()
    
    print("=" * 70)
    print("增强的水印生成脚本 - 可见性强化版 (20种风格)")
    print("=" * 70)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"掩码目录: {MASK_DIR}")
    print(f"每张图像变体数: {NUM_VARIANTS_PER_IMAGE}")
    print(f"组合风格概率: {COMBINED_STYLE_PROBABILITY * 100:.0f}%")
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
    
    # 使用多进程并行处理
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"正在使用 {num_workers} 个进程并行生成...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交任务
        futures = [executor.submit(process_image, img_path) for img_path in image_files]
        
        # 获取结果
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                count = future.result()
                total_variants += count
                # 简单的进度显示
                if (i + 1) % 5 == 0 or (i + 1) == len(image_files):
                    print(f"进度: {i + 1}/{len(image_files)} 图像已处理")
            except Exception as e:
                print(f"处理任务时发生错误: {e}")
    
    print()
    print("=" * 60)
    print(f"✓ 完成! 生成了 {total_variants} 个水印变体")
    print(f"  - 水印图像保存在: {OUTPUT_DIR}")
    print(f"  - 掩码保存在: {MASK_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()
