"""
SVG 水印生成脚本 - 专门生成基于SVG的水印图像

功能:
- 使用SVG格式的水印资源生成多样化的水印
- 支持ElecFans和微信SVG logo
- 生成对应的二值掩码
- 输出保存在watermark_demo文件夹

配置参数:
- NUM_VARIANTS_PER_IMAGE: 每张图的变体数 (默认16)
- ENABLED_SVG_STYLES: 启用的SVG风格列表
- INPUT_DIR: 无水印图像输入目录
- OUTPUT_DIR: 水印图像输出目录 (watermark_demo)
- MASK_DIR: 掩码输出目录 (watermark_demo_masks)
"""

import os
import random
import math
import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# ============================================================================
# 可见性约束配置
# ============================================================================

# 最小透明度 (lowered from 30% to allow fainter watermarks)
MIN_ALPHA = 0.20
MAX_ALPHA = 0.75

# 最小可见面积 (图像总面积的百分比)
# lowered from 0.002 (0.2%) to 0.0005 (0.05%) to allow very small corner marks
MIN_VISIBLE_COVERAGE = 0.0005  # allow very small corner marks (~0.05%)

# 亮度阈值 (用于颜色对比度选择)
BRIGHTNESS_THRESHOLD = 128

# 启用可见性验证
ENABLE_VISIBILITY_VALIDATION = True

# ============================================================================
# 配置参数
# ============================================================================

NUM_VARIANTS_PER_IMAGE = 16
INPUT_DIR = "/home/kaga/Desktop/watermaker remover/20251127_no_watermark_demo"
OUTPUT_DIR = "./merged_watermark_images"
MASK_DIR = "./merged_watermark_masks"

# SVG 资源配置
ELECFANS_LOGO_SVG = "./logos/elecfans-logo.svg"
ELECFANS_WEB_SVG = "./logos/elecfans-web.svg"
WECHAT_SVG = "./logos/WeChat.svg"

# 微信ID候选列表
WECHAT_ID_CANDIDATES = [
    "电客一点通",
    "电路小课堂", 
    "模拟笔记本",
    "某某公众号",
    "DemoWenLab",
]

# SVG 渲染缓存（避免重复处理相同的SVG）
_SVG_CACHE = {}

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False
    print("警告: cairosvg 未安装。SVG 风格将无法使用。请运行: pip install cairosvg")

# 启用的SVG风格列表
ENABLED_SVG_STYLES = [
    'elecfans_logo_svg_corner',     # ElecFans LOGO SVG角落
    'elecfans_web_svg_center',      # ElecFans Web SVG居中
    'wechat_svg_corner_id',         # 微信SVG角落带ID
    'elecfans_logo_svg_center',     # ElecFans LOGO SVG居中
    'elecfans_web_svg_corner',      # ElecFans Web SVG角落
    # 'wechat_svg_center',          # 禁用：真实微信水印通常是角落小图标
    'combined_svg_styles',          # 组合SVG风格
]

COMBINED_STYLE_PROBABILITY = 0.4  # 使用组合风格的概率

# ElecFans web corner watermark tuning
ELECFANS_WEB_CORNER_MIN_WIDTH = 0.15   # 15% of image width
ELECFANS_WEB_CORNER_MAX_WIDTH = 0.25   # 25% of image width

# opacity range (real watermark is quite visible but not 100% solid)
# Increase corner alpha to make corner watermarks clearer for training
ELECFANS_WEB_CORNER_ALPHA_MIN = 0.75
ELECFANS_WEB_CORNER_ALPHA_MAX = 0.95

# ElecFans logo corner (separate tuning from web horizontal logo)
ELECFANS_LOGO_CORNER_ALPHA_MIN = 0.75
ELECFANS_LOGO_CORNER_ALPHA_MAX = 0.95

# optional blur to imitate screenshot / compression
ELECFANS_WEB_CORNER_BLUR_PROB = 0.4    # 40% chance to blur
ELECFANS_WEB_CORNER_BLUR_RADIUS_RANGE = (0.3, 0.8)

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
        reasons.append(".2%")

    # 检查可见像素的透明度
    if avg_alpha < MIN_ALPHA:
        is_visible = False
        reasons.append(".2f")

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
# SVG 水印合成函数
# ============================================================================

def elecfans_logo_svg_corner(w, h):
    """ElecFans LOGO SVG角落"""
    if not CAIROSVG_AVAILABLE or not os.path.exists(ELECFANS_LOGO_SVG):
        return None

    try:
        svg_rgba = load_svg_as_rgba(ELECFANS_LOGO_SVG, int(w * 0.12), int(h * 0.12))
        if svg_rgba is None:
            return None

        # 缩放SVG
        logo_w = int(w * random.uniform(0.08, 0.15))
        ratio = logo_w / svg_rgba.width if svg_rgba.width > 0 else 1
        logo_h = int(svg_rgba.height * ratio)
        svg_rgba = svg_rgba.resize((logo_w, logo_h), Image.Resampling.LANCZOS)

        # 应用透明度（角落logo使用更高的不透明度以保证可见性）
        alpha = random.uniform(ELECFANS_LOGO_CORNER_ALPHA_MIN, ELECFANS_LOGO_CORNER_ALPHA_MAX)
        svg_array = np.array(svg_rgba)
        mask = svg_array[:, :, 3] > 0
        svg_array[mask, 3] = (svg_array[mask, 3] * alpha).astype(np.uint8)
        svg_rgba = Image.fromarray(svg_array, 'RGBA')

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
        return None

def elecfans_web_svg_center(w, h):
    """ElecFans Web SVG居中"""
    if not CAIROSVG_AVAILABLE or not os.path.exists(ELECFANS_WEB_SVG):
        return None

    try:
        svg_rgba = load_svg_as_rgba(ELECFANS_WEB_SVG, int(w * 0.2), int(h * 0.2))
        if svg_rgba is None:
            return None

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
        return None

def wechat_svg_corner_id(w, h):
    """微信SVG角落带ID - 模拟真实微信角落水印"""
    if not CAIROSVG_AVAILABLE or not os.path.exists(WECHAT_SVG):
        return None

    try:
        # 选择微信ID
        wechat_id = random.choice(WECHAT_ID_CANDIDATES)

        # 计算图标尺寸：3.5-6.0% 的图像高度，至少18px（略大以提高清晰度）
        icon_height = max(18, int(h * random.uniform(0.035, 0.06)))
        icon_width = icon_height  # 保持正方形

        # 加载并缩放SVG图标
        svg_rgba = load_svg_as_rgba(WECHAT_SVG, icon_width, icon_height)
        if svg_rgba is None:
            return None

        # 将图标改为纯黑（保留 alpha），提高在各种背景下的可见性
        try:
            svg_arr = np.array(svg_rgba)
            alpha_mask_icon = svg_arr[:, :, 3] > 0
            # 将所有非透明像素设为黑色
            svg_arr[alpha_mask_icon, 0] = 0
            svg_arr[alpha_mask_icon, 1] = 0
            svg_arr[alpha_mask_icon, 2] = 0
            svg_rgba = Image.fromarray(svg_arr, 'RGBA')
        except Exception:
            # 如果数组转换失败，则忽略该步骤，继续使用原始图标
            pass

        # 计算字体尺寸：约0.8倍图标高度，至少12px（更大以提高可读性）
        font_size = max(12, int(icon_height * 0.8))
        font = get_font(font_size)
        if font is None:
            return None

        # 测量文本尺寸
        text_w, text_h = get_text_bbox(ImageDraw.Draw(Image.new('RGBA', (1, 1))), wechat_id, font)

        # 计算间隙：图标尺寸的20%，至少4px
        gap = max(4, int(icon_width * 0.2))

        # 计算组合尺寸
        group_width = icon_width + gap + text_w
        group_height = max(icon_height, text_h)

        # 如果组合宽度超过图像宽度的15%，按比例缩小
        max_group_width = int(w * 0.15)
        if group_width > max_group_width:
            scale_factor = max_group_width / group_width
            icon_width = int(icon_width * scale_factor)
            icon_height = int(icon_height * scale_factor)
            font_size = max(12, int(font_size * scale_factor))
            gap = max(4, int(gap * scale_factor))

            # 重新加载图标和字体
            svg_rgba = load_svg_as_rgba(WECHAT_SVG, icon_width, icon_height)
            if svg_rgba is None:
                return None
            font = get_font(font_size)
            if font is None:
                return None

            # 重新测量文本
            text_w, text_h = get_text_bbox(ImageDraw.Draw(Image.new('RGBA', (1, 1))), wechat_id, font)
            group_width = icon_width + gap + text_w
            group_height = max(icon_height, text_h)

        # 创建组合图像
        group_img = Image.new('RGBA', (group_width, group_height), (255, 255, 255, 0))
        group_draw = ImageDraw.Draw(group_img)

        # 计算垂直居中位置
        icon_y = (group_height - icon_height) // 2
        text_y = (group_height - text_h) // 2

        # 粘贴微信图标
        group_img.paste(svg_rgba, (0, icon_y), svg_rgba)

        # 绘制文字阴影 (加深以提高可见性)
        shadow_offset = (1, 1)
        shadow_pos = (icon_width + gap + shadow_offset[0], text_y + shadow_offset[1])
        group_draw.text(shadow_pos, wechat_id, font=font, fill=(0, 0, 0, 110))

        # 绘制黑色文字以在大多数背景下都清晰可见
        text_pos = (icon_width + gap, text_y)
        group_draw.text(text_pos, wechat_id, font=font, fill=(0, 0, 0, 255))

        # 可选：5% 概率应用非常小的模糊（大多数情况下不模糊以保证清晰）
        if random.random() < 0.05:
            radius = random.uniform(0.1, 0.3)
            group_img = group_img.filter(ImageFilter.GaussianBlur(radius))

        # 应用全局透明度 (明显提高，使角落水印清晰可见)
        alpha = random.uniform(0.80, 0.95)
        group_array = np.array(group_img)
        alpha_mask = group_array[:, :, 3] > 0
        group_array[alpha_mask, 3] = (group_array[alpha_mask, 3] * alpha).astype(np.uint8)
        group_img = Image.fromarray(group_array)

        # 创建主图像overlay
        overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))

        # 总是右下角，更小的边距
        margin_x = max(int(w * 0.005), 3)
        margin_y = max(int(h * 0.005), 3)

        x = w - group_width - margin_x
        y = h - group_height - margin_y + random.randint(-1, 1)  # 小随机抖动

        # 粘贴组合图像到overlay
        overlay.paste(group_img, (x, y), group_img)

        return overlay

    except Exception as e:
        return None

def elecfans_logo_svg_center(w, h):
    """ElecFans LOGO SVG居中"""
    if not CAIROSVG_AVAILABLE or not os.path.exists(ELECFANS_LOGO_SVG):
        return None

    try:
        svg_rgba = load_svg_as_rgba(ELECFANS_LOGO_SVG, int(w * 0.18), int(h * 0.18))
        if svg_rgba is None:
            return None

        # 缩放SVG
        logo_w = int(w * random.uniform(0.12, 0.20))
        ratio = logo_w / svg_rgba.width if svg_rgba.width > 0 else 1
        logo_h = int(svg_rgba.height * ratio)
        svg_rgba = svg_rgba.resize((logo_w, logo_h), Image.Resampling.LANCZOS)

        # 应用透明度和轻微旋转
        alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
        rotation = random.uniform(-15, 15)

        svg_array = np.array(svg_rgba)
        svg_array[:, :, 3] = (svg_array[:, :, 3] * alpha).astype(np.uint8)
        svg_rgba = Image.fromarray(svg_array)

        if rotation != 0:
            svg_rgba = svg_rgba.rotate(rotation, expand=True, fillcolor=(255, 255, 255, 0))

        overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))

        # 居中位置
        pos = ((w - svg_rgba.width) // 2, (h - svg_rgba.height) // 2)
        overlay.paste(svg_rgba, pos, svg_rgba)

        return overlay

    except Exception as e:
        return None

def elecfans_web_svg_corner(w, h):
    """ElecFans Web SVG角落"""
    if not CAIROSVG_AVAILABLE or not os.path.exists(ELECFANS_WEB_SVG):
        return None

    try:
        # pick target width
        target_w = int(
            w * random.uniform(
                ELECFANS_WEB_CORNER_MIN_WIDTH,
                ELECFANS_WEB_CORNER_MAX_WIDTH
            )
        )
        svg_rgba = load_svg_as_rgba(ELECFANS_WEB_SVG, target_w, target_w)
        if svg_rgba is None:
            return None

        # resize with correct aspect
        ratio = target_w / svg_rgba.width
        target_h = int(svg_rgba.height * ratio)
        svg_rgba = svg_rgba.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # apply alpha in dedicated range
        alpha = random.uniform(ELECFANS_WEB_CORNER_ALPHA_MIN,
                               ELECFANS_WEB_CORNER_ALPHA_MAX)
        svg_array = np.array(svg_rgba)
        mask = svg_array[:, :, 3] > 0
        svg_array[mask, 3] = (svg_array[mask, 3] * alpha).astype(np.uint8)
        svg_rgba = Image.fromarray(svg_array, 'RGBA')

        # optional slight blur
        if random.random() < ELECFANS_WEB_CORNER_BLUR_PROB:
            r_min, r_max = ELECFANS_WEB_CORNER_BLUR_RADIUS_RANGE
            radius = random.uniform(r_min, r_max)
            svg_rgba = svg_rgba.filter(ImageFilter.GaussianBlur(radius))

        overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))

        margin_x = max(int(w * 0.01), 6)
        margin_y = max(int(h * 0.01), 6)

        # choose corner: mostly bottom-right
        if random.random() < 0.9:
            x = w - target_w - margin_x
            y = h - target_h - margin_y
        else:
            x = margin_x
            y = h - target_h - margin_y

        overlay.paste(svg_rgba, (x, y), svg_rgba)
        return overlay

    except Exception as e:
        return None

def combined_svg_styles(w, h):
    """组合多个SVG风格"""
    available_styles = [
        elecfans_logo_svg_corner,
        elecfans_web_svg_center,
        wechat_svg_corner_id,
        elecfans_logo_svg_center,
        elecfans_web_svg_corner,
        # wechat_svg_center  # 移除：不使用居中的微信水印
    ]

    if len(available_styles) < 2:
        return random.choice(available_styles)(w, h)

    # 随机选择2个风格组合
    selected_styles = random.sample(available_styles, 2)

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

# ============================================================================
# 主要处理函数
# ============================================================================

def generate_svg_watermark_variant(image_path, output_index, max_retries=3):
    """
    为单张图像生成一个SVG水印变体及其掩码

    返回: (success, watermarked_image, mask_image, output_name)
    """
    image = load_image(image_path)
    if image is None:
        return False, None, None, None

    w, h = image.size
    stem = Path(image_path).stem
    output_name = f"{stem}_svg_wm_{output_index:03d}"

    # 构建可用的SVG风格函数
    style_functions = []
    if 'elecfans_logo_svg_corner' in ENABLED_SVG_STYLES:
        style_functions.append(('elecfans_logo_svg_corner', lambda: elecfans_logo_svg_corner(w, h)))
    if 'elecfans_web_svg_center' in ENABLED_SVG_STYLES:
        style_functions.append(('elecfans_web_svg_center', lambda: elecfans_web_svg_center(w, h)))
    if 'wechat_svg_corner_id' in ENABLED_SVG_STYLES:
        style_functions.append(('wechat_svg_corner_id', lambda: wechat_svg_corner_id(w, h)))
    if 'elecfans_logo_svg_center' in ENABLED_SVG_STYLES:
        style_functions.append(('elecfans_logo_svg_center', lambda: elecfans_logo_svg_center(w, h)))
    if 'elecfans_web_svg_corner' in ENABLED_SVG_STYLES:
        style_functions.append(('elecfans_web_svg_corner', lambda: elecfans_web_svg_corner(w, h)))
    # wechat_svg_center 已禁用

    if not style_functions:
        print(f"警告: 没有启用的SVG风格")
        return False, None, None, None

    # 重试直到获得可见的水印
    for attempt in range(max_retries):
        try:
            # 随机选择是否使用组合风格
            if ('combined_svg_styles' in ENABLED_SVG_STYLES and
                random.random() < COMBINED_STYLE_PROBABILITY and
                len(style_functions) >= 2):
                overlay = combined_svg_styles(w, h)
            else:
                # 选择单个风格
                style_name, style_func = random.choice(style_functions)
                overlay = style_func()

            if overlay is None:
                continue

            # 验证可见性约束
            if ENABLE_VISIBILITY_VALIDATION:
                overlay_array = np.array(overlay)
                visibility_result = validate_watermark_visibility(overlay_array, w, h)

                if not visibility_result['is_visible']:
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return False, None, None, None

            # 合成最终图像
            image_rgba = image.convert('RGBA')
            watermarked = Image.alpha_composite(image_rgba, overlay)
            watermarked_rgb = watermarked.convert('RGB')

            # 生成掩码
            overlay_array = np.array(overlay)
            alpha_mask = overlay_array[:, :, 3] > 0
            mask_array = np.zeros((h, w), dtype=np.uint8)
            mask_array[alpha_mask] = 255
            mask = Image.fromarray(mask_array, mode='L')

            return True, watermarked_rgb, mask, output_name

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"警告: 生成SVG水印失败 {image_path} (变体{output_index}): {e}")
            continue

    return False, None, None, None

def process_image(image_path):
    """处理单张图像，生成SVG水印变体"""
    stem = Path(image_path).stem
    print(f"处理: {stem}")

    success_count = 0
    for variant_idx in range(NUM_VARIANTS_PER_IMAGE):
        success, watermarked, mask, output_name = generate_svg_watermark_variant(
            image_path, variant_idx
        )

        if not success:
            continue

        # 保存水印图像
        quality = random.randint(70, 95)
        wm_path = os.path.join(OUTPUT_DIR, f"{output_name}.jpg")
        watermarked.save(wm_path, 'JPEG', quality=quality)

        # 保存掩码
        mask_path = os.path.join(MASK_DIR, f"{output_name}_mask.png")
        mask.save(mask_path, 'PNG')

        success_count += 1

    print(f"  ✓ 生成 {success_count}/{NUM_VARIANTS_PER_IMAGE} 个SVG水印变体")
    return success_count

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主程序"""
    ensure_dirs()

    print("=" * 70)
    print("SVG 水印生成脚本 - 专门生成基于SVG的水印图像")
    print("=" * 70)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"掩码目录: {MASK_DIR}")
    print(f"每张图像变体数: {NUM_VARIANTS_PER_IMAGE}")
    print(f"组合风格概率: {COMBINED_STYLE_PROBABILITY * 100:.0f}%")
    print()
    print("可见性约束:")
    print(f"  ✓ 最小透明度: {MIN_ALPHA:.0%}")
    print(f"  ✓ 最大透明度: {MAX_ALPHA:.0%}")
    print(f"  ✓ 最小覆盖面积: {MIN_VISIBLE_COVERAGE:.1%}")
    print(f"  ✓ 可见性验证: {'启用' if ENABLE_VISIBILITY_VALIDATION else '禁用'}")
    print()
    print(f"启用的SVG风格: {', '.join(ENABLED_SVG_STYLES)}")
    print()

    if not CAIROSVG_AVAILABLE:
        print("错误: cairosvg 未安装，无法生成SVG水印")
        print("请运行: pip install cairosvg")
        return

    # 检查SVG文件是否存在
    svg_files = [ELECFANS_LOGO_SVG, ELECFANS_WEB_SVG, WECHAT_SVG]
    available_svgs = [f for f in svg_files if os.path.exists(f)]
    if not available_svgs:
        print("错误: 没有找到任何SVG文件")
        print(f"请确保以下文件存在: {', '.join(svg_files)}")
        return

    print(f"找到SVG文件: {', '.join(available_svgs)}")
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
    print(f"✓ 完成! 生成了 {total_variants} 个SVG水印变体")
    print(f"  - 水印图像保存在: {OUTPUT_DIR}")
    print(f"  - 掩码保存在: {MASK_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()
