"""
高级水印生成器 - 支持多样化水印效果（已增强版）

支持的水印类型（14种）：
1. transparent_url - 透明网址
2. transparent_logo_text - 透明LOGO+文字
3. transparent_chinese - 透明中文
4. semi_transparent_url - 半透明网址
5. semi_transparent_chinese - 半透明中文
6. watermark_stamp - 水印章
7. confidential_stamp - 机密章
8. company_logo - 公司LOGO
9. qr_code_style - 二维码风格
10. gradient_text - 渐变文字
11. diagonal_stripe - 对角线纹样
12. mixed_watermark - 混合水印
13. corner_website_logo - ElecFans风格角标网站水印
14. mini_social_corner_logo - 小型微信风格社交账号角标水印

水印位置（10种）：
- top_left: 左上角
- top_right: 右上角
- bottom_left: 左下角
- bottom_right: 右下角
- top_center: 顶部中心
- bottom_center: 底部中心
- left_center: 左边中心
- right_center: 右边中心
- center: 正中心
- full: 占满全图
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import os
import math


class WatermarkGenerator:
    def __init__(self, font_path=None):
        """
        初始化水印生成器
        
        Args:
            font_path: 中文字体路径，如果None则使用默认字体
        """
        self.font_path = font_path
        self._load_font()
        
        # 14种水印类型
        self.watermark_types = [
            'transparent_url',
            'transparent_logo_text',
            'transparent_chinese',
            'semi_transparent_url',
            'semi_transparent_chinese',
            'watermark_stamp',
            'confidential_stamp',
            'company_logo',
            'qr_code_style',
            'gradient_text',
            'diagonal_stripe',
            'mixed_watermark',
            'corner_website_logo',
            'mini_social_corner_logo'
        ]
        
        # 10种水印位置
        self.positions = [
            'top_left',
            'top_right',
            'bottom_left',
            'bottom_right',
            'top_center',
            'bottom_center',
            'left_center',
            'right_center',
            'center',
            'full'
        ]
        
    def _load_font(self):
        """加载中文字体"""
        if self.font_path and os.path.exists(self.font_path):
            try:
                self.pil_font_large = ImageFont.truetype(self.font_path, 60)
                self.pil_font_medium = ImageFont.truetype(self.font_path, 40)
                self.pil_font_small = ImageFont.truetype(self.font_path, 30)
            except Exception as e:
                print(f"字体加载失败: {e}")
                self.pil_font_large = ImageFont.load_default()
                self.pil_font_medium = ImageFont.load_default()
                self.pil_font_small = ImageFont.load_default()
        else:
            self.pil_font_large = ImageFont.load_default()
            self.pil_font_medium = ImageFont.load_default()
            self.pil_font_small = ImageFont.load_default()
    
    def generate(self, image, watermark_type=None, position=None, 
                 alpha=None, scale=None):
        """
        生成带水印的图像，自动适应不同尺寸的原始图像
        
        Args:
            image: numpy数组或图像路径 (BGR格式)
            watermark_type: 水印类型，None则随机选择
            position: 位置，None则随机选择
            alpha: 透明度 (0-1)，None则随机
            scale: 水印缩放比例 (0-1)，None则随机
            
        Returns:
            带水印的图像 (numpy数组，BGR格式)
        """
        # 加载图像
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
            
        if img is None:
            raise ValueError("无法加载图像")
        
        h, w = img.shape[:2]
        
        # 随机选择水印类型和位置（如果未指定）
        if watermark_type is None:
            watermark_type = random.choice(self.watermark_types)
        if position is None:
            position = random.choice(self.positions)
        
        # 根据图像大小自适应计算参数
        if alpha is None:
            alpha = random.uniform(0.1, 0.5)  # 透明度
        if scale is None:
            # 根据图像大小自适应缩放比例
            if h < 256 or w < 256:
                scale = random.uniform(0.08, 0.25)  # 小图片
            elif h < 512 or w < 512:
                scale = random.uniform(0.1, 0.35)   # 中等图片
            else:
                scale = random.uniform(0.15, 0.4)   # 大图片
        
        # 根据类型生成水印
        if watermark_type == 'transparent_url':
            watermark = self._create_url_watermark(w, h, scale, alpha)
        elif watermark_type == 'transparent_logo_text':
            watermark = self._create_logo_text_watermark(w, h, scale, alpha)
        elif watermark_type == 'transparent_chinese':
            watermark = self._create_chinese_watermark(w, h, scale, alpha)
        elif watermark_type == 'semi_transparent_url':
            watermark = self._create_url_watermark(w, h, scale, alpha, semi_transparent=True)
        elif watermark_type == 'semi_transparent_chinese':
            watermark = self._create_chinese_watermark(w, h, scale, alpha, semi_transparent=True)
        elif watermark_type == 'watermark_stamp':
            watermark = self._create_watermark_stamp(w, h, scale, alpha)
        elif watermark_type == 'confidential_stamp':
            watermark = self._create_confidential_stamp(w, h, scale, alpha)
        elif watermark_type == 'company_logo':
            watermark = self._create_company_logo(w, h, scale, alpha)
        elif watermark_type == 'qr_code_style':
            watermark = self._create_qr_code_style(w, h, scale, alpha)
        elif watermark_type == 'gradient_text':
            watermark = self._create_gradient_text(w, h, scale, alpha)
        elif watermark_type == 'diagonal_stripe':
            watermark = self._create_diagonal_stripe(w, h, scale, alpha)
        elif watermark_type == 'mixed_watermark':
            watermark = self._create_mixed_watermark(w, h, scale, alpha)
        elif watermark_type == 'corner_website_logo':
            watermark = self._create_corner_website_logo_watermark(w, h, alpha)
        elif watermark_type == 'mini_social_corner_logo':
            watermark = self._create_mini_social_corner_logo_watermark(w, h, alpha)
        else:
            raise ValueError(f"未知的水印类型: {watermark_type}")
        
        # 应用水印到不同位置
        result = self._apply_watermark(img, watermark, position)
        
        return result
    
    def _create_url_watermark(self, img_w, img_h, scale, alpha, semi_transparent=False):
        """创建透明网址水印"""
        urls = [
            'www.example.com',
            'watermark.com',
            'www.company.net',
            'http://site.org'
        ]
        url = random.choice(urls)
        
        # 计算水印大小
        watermark_w = int(img_w * scale)
        watermark_h = int(watermark_w * 0.3)
        
        # 创建水印图像
        watermark = np.zeros((watermark_h, watermark_w, 4), dtype=np.uint8)
        watermark_pil = Image.fromarray(watermark, 'RGBA')
        draw = ImageDraw.Draw(watermark_pil)
        
        # 绘制文字
        text_color = (255, 255, 255, int(255 * alpha))
        font = ImageFont.load_default()
        draw.text((10, 5), url, fill=text_color, font=font)
        
        # 绘制边框
        border_color = (200, 200, 200, int(255 * alpha * 0.8))
        draw.rectangle([0, 0, watermark_w-1, watermark_h-1], outline=border_color)
        
        return np.array(watermark_pil)
    
    def _create_logo_text_watermark(self, img_w, img_h, scale, alpha):
        """创建LOGO+文字水印"""
        watermark_w = int(img_w * scale)
        watermark_h = int(watermark_w * 0.4)
        
        watermark = np.zeros((watermark_h, watermark_w, 4), dtype=np.uint8)
        watermark_pil = Image.fromarray(watermark, 'RGBA')
        draw = ImageDraw.Draw(watermark_pil)
        
        # 绘制圆形logo框
        logo_size = watermark_h - 10
        draw.ellipse(
            [5, 5, 5+logo_size, 5+logo_size],
            outline=(255, 255, 255, int(255 * alpha))
        )
        
        # 绘制文字
        text = 'LOGO'
        text_color = (255, 255, 255, int(255 * alpha))
        draw.text((logo_size + 15, 10), text, fill=text_color, font=ImageFont.load_default())
        
        return np.array(watermark_pil)
    
    def _create_chinese_watermark(self, img_w, img_h, scale, alpha, semi_transparent=False):
        """创建中文水印"""
        chinese_texts = [
            '机密',
            '样品',
            '草稿',
            '内部',
            '版权所有',
            '禁止复制',
            '水印'
        ]
        text = random.choice(chinese_texts)
        
        # 计算水印大小
        watermark_w = int(img_w * scale)
        watermark_h = int(watermark_w * 0.6)
        
        # 创建水印图像
        watermark = np.zeros((watermark_h, watermark_w, 4), dtype=np.uint8)
        watermark_pil = Image.fromarray(watermark, 'RGBA')
        draw = ImageDraw.Draw(watermark_pil)
        
        # 绘制文字
        text_color = (255, 255, 255, int(255 * alpha))
        
        # 使用可用的字体
        font = self.pil_font_medium if self.pil_font_medium else ImageFont.load_default()
        
        # 计算居中位置
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (watermark_w - text_w) // 2
        y = (watermark_h - text_h) // 2
        
        draw.text((x, y), text, fill=text_color, font=font)
        
        return np.array(watermark_pil)
    
    def _create_watermark_stamp(self, img_w, img_h, scale, alpha):
        """创建水印章"""
        watermark_w = int(img_w * scale)
        watermark_h = int(watermark_w)  # 正方形
        
        watermark = np.zeros((watermark_h, watermark_w, 4), dtype=np.uint8)
        watermark_pil = Image.fromarray(watermark, 'RGBA')
        draw = ImageDraw.Draw(watermark_pil)
        
        # 绘制圆形边框
        color = (255, 100, 100, int(255 * alpha))
        draw.ellipse(
            [5, 5, watermark_w-5, watermark_h-5],
            outline=color,
            width=3
        )
        
        # 绘制文字
        text = "WATERMARK"
        text_color = (255, 100, 100, int(255 * alpha))
        font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (watermark_w - text_w) // 2
        y = (watermark_h - text_h) // 2
        draw.text((x, y), text, fill=text_color, font=font)
        
        return np.array(watermark_pil)
    
    def _create_confidential_stamp(self, img_w, img_h, scale, alpha):
        """创建机密章"""
        watermark_w = int(img_w * scale)
        watermark_h = int(watermark_w)
        
        watermark = np.zeros((watermark_h, watermark_w, 4), dtype=np.uint8)
        watermark_pil = Image.fromarray(watermark, 'RGBA')
        draw = ImageDraw.Draw(watermark_pil)
        
        # 绘制倾斜文字效果的矩形边框
        color = (255, 50, 50, int(255 * alpha))
        draw.rectangle(
            [10, 10, watermark_w-10, watermark_h-10],
            outline=color,
            width=2
        )
        
        # 绘制对角线
        draw.line([(10, 10), (watermark_w-10, watermark_h-10)], fill=color, width=2)
        draw.line([(watermark_w-10, 10), (10, watermark_h-10)], fill=color, width=2)
        
        text = "机密"
        text_color = (255, 50, 50, int(255 * alpha))
        font = self.pil_font_medium
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (watermark_w - text_w) // 2
        y = (watermark_h - text_h) // 2
        draw.text((x, y), text, fill=text_color, font=font)
        
        return np.array(watermark_pil)
    
    def _create_company_logo(self, img_w, img_h, scale, alpha):
        """创建公司LOGO风格"""
        watermark_w = int(img_w * scale)
        watermark_h = int(watermark_w * 0.5)
        
        watermark = np.zeros((watermark_h, watermark_w, 4), dtype=np.uint8)
        watermark_pil = Image.fromarray(watermark, 'RGBA')
        draw = ImageDraw.Draw(watermark_pil)
        
        # 绘制矩形背景
        color = (100, 150, 255, int(255 * alpha * 0.8))
        draw.rectangle([0, 0, watermark_w-1, watermark_h-1], fill=color)
        
        # 绘制内部图案
        text_color = (255, 255, 255, int(255 * alpha))
        text = "© Company"
        font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (watermark_w - text_w) // 2
        y = (watermark_h - text_h) // 2
        draw.text((x, y), text, fill=text_color, font=font)
        
        return np.array(watermark_pil)
    
    def _create_qr_code_style(self, img_w, img_h, scale, alpha):
        """创建二维码风格水印"""
        watermark_w = int(img_w * scale)
        watermark_h = int(watermark_w)
        
        watermark = np.zeros((watermark_h, watermark_w, 4), dtype=np.uint8)
        watermark_pil = Image.fromarray(watermark, 'RGBA')
        draw = ImageDraw.Draw(watermark_pil)
        
        # 绘制网格图案
        color = (150, 150, 150, int(255 * alpha))
        cell_size = watermark_w // 8
        
        for i in range(0, watermark_w, cell_size):
            # 随机决定是否画这个格子
            if random.random() > 0.5:
                draw.rectangle([i, 0, i+cell_size, watermark_h], outline=color)
        
        for j in range(0, watermark_h, cell_size):
            if random.random() > 0.5:
                draw.rectangle([0, j, watermark_w, j+cell_size], outline=color)
        
        # 绘制角标记
        for corner_x, corner_y in [(5, 5), (watermark_w-15, 5), (5, watermark_h-15)]:
            draw.rectangle(
                [corner_x, corner_y, corner_x+10, corner_y+10],
                fill=color
            )
        
        return np.array(watermark_pil)
    
    def _create_gradient_text(self, img_w, img_h, scale, alpha):
        """创建渐变文字水印"""
        watermark_w = int(img_w * scale)
        watermark_h = int(watermark_w * 0.5)
        
        watermark = np.zeros((watermark_h, watermark_w, 4), dtype=np.uint8)
        watermark_pil = Image.fromarray(watermark, 'RGBA')
        
        # 绘制渐变效果（通过不同透明度的条纹）
        for i in range(watermark_w):
            # 根据位置改变透明度
            gradient_alpha = int(255 * alpha * (1 - abs(i - watermark_w/2) / (watermark_w/2)))
            # 绘制垂直条纹
            for j in range(watermark_h):
                watermark[j, i] = (200, 200, 255, gradient_alpha)
        
        watermark_pil = Image.fromarray(watermark, 'RGBA')
        return np.array(watermark_pil)
    
    def _create_diagonal_stripe(self, img_w, img_h, scale, alpha):
        """创建对角线纹样水印"""
        watermark_w = int(img_w * scale)
        watermark_h = int(watermark_w * 0.6)
        
        watermark = np.zeros((watermark_h, watermark_w, 4), dtype=np.uint8)
        watermark_pil = Image.fromarray(watermark, 'RGBA')
        draw = ImageDraw.Draw(watermark_pil)
        
        # 绘制对角线纹样
        color = (200, 200, 200, int(255 * alpha))
        stripe_spacing = 15
        
        # 从左上到右下的对角线
        for i in range(-watermark_h, watermark_w, stripe_spacing):
            draw.line([(i, 0), (i + watermark_h, watermark_h)], fill=color, width=2)
        
        return np.array(watermark_pil)
    
    def _create_mixed_watermark(self, img_w, img_h, scale, alpha):
        """创建混合水印（组合多个元素）"""
        watermark_w = int(img_w * scale)
        watermark_h = int(watermark_w * 0.6)
        
        watermark = np.zeros((watermark_h, watermark_w, 4), dtype=np.uint8)
        watermark_pil = Image.fromarray(watermark, 'RGBA')
        draw = ImageDraw.Draw(watermark_pil)
        
        # 绘制背景
        bg_color = (100, 100, 100, int(255 * alpha * 0.5))
        draw.rectangle([0, 0, watermark_w-1, watermark_h-1], fill=bg_color)
        
        # 绘制边框
        border_color = (200, 200, 200, int(255 * alpha))
        draw.rectangle([2, 2, watermark_w-3, watermark_h-3], outline=border_color, width=2)
        
        # 绘制文字
        text = "© WATERMARK"
        text_color = (255, 255, 255, int(255 * alpha))
        font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (watermark_w - text_w) // 2
        y = int((watermark_h - text_h) * 0.6)
        draw.text((x, y), text, fill=text_color, font=font)
        
        # 绘制小图标
        icon_color = (100, 200, 100, int(255 * alpha))
        draw.ellipse([watermark_w//4, int(watermark_h*0.1), 
                      watermark_w//4+10, int(watermark_h*0.1)+10], 
                     fill=icon_color)
        draw.ellipse([watermark_w*3//4-10, int(watermark_h*0.1), 
                      watermark_w*3//4, int(watermark_h*0.1)+10], 
                     fill=icon_color)
        
        return np.array(watermark_pil)
    
    def _apply_watermark(self, image, watermark, position):
        """应用水印到图像，支持10种位置"""
        img = image.copy()
        h, w = img.shape[:2]
        wm_h, wm_w = watermark.shape[:2]
        
        # 计算边距（根据图像大小自适应）
        margin_x = max(10, int(w * 0.02))
        margin_y = max(10, int(h * 0.02))
        
        # 确定水印位置
        if position == 'top_left':
            x, y = margin_x, margin_y
        elif position == 'top_right':
            x, y = w - wm_w - margin_x, margin_y
        elif position == 'bottom_left':
            x, y = margin_x, h - wm_h - margin_y
        elif position == 'bottom_right':
            x, y = w - wm_w - margin_x, h - wm_h - margin_y
        elif position == 'top_center':
            x, y = (w - wm_w) // 2, margin_y
        elif position == 'bottom_center':
            x, y = (w - wm_w) // 2, h - wm_h - margin_y
        elif position == 'left_center':
            x, y = margin_x, (h - wm_h) // 2
        elif position == 'right_center':
            x, y = w - wm_w - margin_x, (h - wm_h) // 2
        elif position == 'center':
            x, y = (w - wm_w) // 2, (h - wm_h) // 2
        elif position == 'full':
            # 占满全图 - 平铺水印
            y = 0
            while y < h:
                x = 0
                while x < w:
                    img = self._overlay_watermark(img, watermark, x, y)
                    x += wm_w + 20
                y += wm_h + 20
            return img
        else:
            x, y = (w - wm_w) // 2, (h - wm_h) // 2
        
        img = self._overlay_watermark(img, watermark, x, y)
        return img
    
    def _overlay_watermark(self, image, watermark, x, y):
        """将水印叠加到图像"""
        img = image.copy()
        h, w = img.shape[:2]
        wm_h, wm_w = watermark.shape[:2]
        
        # 确保坐标在范围内
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        # 计算有效的水印区域
        x_end = min(x + wm_w, w)
        y_end = min(y + wm_h, h)
        wm_x_end = x_end - x
        wm_y_end = y_end - y
        
        # 提取透明通道作为alpha
        if watermark.shape[2] == 4:
            alpha = watermark[0:wm_y_end, 0:wm_x_end, 3].astype(float) / 255.0
            wm_rgb = watermark[0:wm_y_end, 0:wm_x_end, :3]
        else:
            alpha = np.ones((wm_y_end, wm_x_end), dtype=float)
            wm_rgb = watermark[0:wm_y_end, 0:wm_x_end]
        
        # 混合图像
        img_region = img[y:y_end, x:x_end].astype(float)
        
        if len(alpha.shape) == 2:
            alpha = np.stack([alpha] * 3, axis=2)
        
        blended = (wm_rgb.astype(float) * alpha + 
                  img_region * (1 - alpha))
        
        img[y:y_end, x:x_end] = blended.astype(np.uint8)
        
        return img
    
    def apply_corner_website_logo(self, image, alpha=None, position_bias=0.85):
        """
        应用ElecFans风格的角标网站水印
        
        Args:
            image: 输入图像 (numpy数组，BGR格式)
            alpha: 透明度 (0.35-0.7)，None则随机
            position_bias: 底部右角位置偏好度 (0-1)，越高越倾向右下角
            
        Returns:
            带水印的图像 (numpy数组，BGR格式)
        """
        img = image.copy()
        h, w = img.shape[:2]
        
        # 随机透明度
        if alpha is None:
            alpha = random.uniform(0.35, 0.7)
        
        # 决定位置（85%概率选择右下角，15%概率选择左下角）
        if random.random() < position_bias:
            position = 'bottom_right'
        else:
            position = 'bottom_left'
        
        # 创建水印
        watermark = self._create_corner_website_logo_watermark(w, h, alpha)
        
        # 应用水印
        result = self._apply_watermark(img, watermark, position)
        
        return result
    
    def _create_corner_website_logo_watermark(self, img_w, img_h, alpha):
        """创建ElecFans风格的角标网站水印"""
        # 计算水印尺寸（占图像的15-30%宽度，7-12%高度）
        watermark_w = int(img_w * random.uniform(0.15, 0.30))
        watermark_h = int(img_h * random.uniform(0.07, 0.12))
        
        # 确保最小尺寸
        min_text_height = max(16, int(img_h * 0.02))
        watermark_h = max(watermark_h, min_text_height * 2 + 10)  # 两行文字加间距
        
        # 创建水印图像（RGBA）
        watermark = np.zeros((watermark_h, watermark_w, 4), dtype=np.uint8)
        watermark_pil = Image.fromarray(watermark, 'RGBA')
        draw = ImageDraw.Draw(watermark_pil)
        
        # 随机选择品牌和URL
        brand_candidates = ["电子发烧友", "ElecFans", "DemoWen", "TechZone", "AI-Lab", "CircuitHub", "TechBlog"]
        url_candidates = ["www.elecfans.com", "www.demowen.com", "www.example.com", "www.techzone.cn", "www.circuithub.com"]
        
        brand = random.choice(brand_candidates)
        url = random.choice(url_candidates)
        
        # 计算布局参数
        icon_size = min(watermark_h - 8, int(watermark_w * 0.25))  # 图标尺寸
        text_start_x = icon_size + 8  # 文字起始X坐标
        text_width = watermark_w - text_start_x - 4  # 文字区域宽度
        
        # 绘制圆形图标
        icon_center_x = icon_size // 2 + 4
        icon_center_y = watermark_h // 2
        
        # 图标背景（浅色圆形）
        icon_bg_color = (240, 240, 240, int(255 * alpha))
        draw.ellipse(
            [icon_center_x - icon_size//2, icon_center_y - icon_size//2,
             icon_center_x + icon_size//2, icon_center_y + icon_size//2],
            fill=icon_bg_color
        )
        
        # 图标内部线条（1-2条简单线条）
        icon_line_color = (100, 100, 100, int(255 * alpha))
        num_lines = random.randint(1, 2)
        
        for i in range(num_lines):
            if random.random() > 0.5:
                # 水平线
                line_y = icon_center_y + random.randint(-icon_size//4, icon_size//4)
                draw.line(
                    [icon_center_x - icon_size//3, line_y, icon_center_x + icon_size//3, line_y],
                    fill=icon_line_color, width=2
                )
            else:
                # 垂直线
                line_x = icon_center_x + random.randint(-icon_size//4, icon_size//4)
                draw.line(
                    [line_x, icon_center_y - icon_size//3, line_x, icon_center_y + icon_size//3],
                    fill=icon_line_color, width=2
                )
        
        # 绘制文字
        text_color = (255, 255, 255, int(255 * alpha))
        
        # 第一行：品牌名称
        font_size_brand = max(12, min(int(watermark_h * 0.35), 24))
        try:
            font_brand = ImageFont.truetype(self.font_path, font_size_brand) if self.font_path else ImageFont.load_default()
        except:
            font_brand = ImageFont.load_default()
        
        # 计算品牌文字位置
        bbox_brand = draw.textbbox((0, 0), brand, font=font_brand)
        brand_w = bbox_brand[2] - bbox_brand[0]
        brand_h = bbox_brand[3] - bbox_brand[1]
        
        # 如果品牌文字太宽，缩小字体
        if brand_w > text_width:
            font_size_brand = max(10, int(font_size_brand * text_width / brand_w))
            try:
                font_brand = ImageFont.truetype(self.font_path, font_size_brand) if self.font_path else ImageFont.load_default()
            except:
                font_brand = ImageFont.load_default()
            bbox_brand = draw.textbbox((0, 0), brand, font=font_brand)
            brand_w = bbox_brand[2] - bbox_brand[0]
            brand_h = bbox_brand[3] - bbox_brand[1]
        
        brand_x = text_start_x
        brand_y = (watermark_h // 2) - brand_h - 2  # 上半部分
        
        # 第二行：URL
        font_size_url = max(10, min(int(watermark_h * 0.25), 18))
        try:
            font_url = ImageFont.truetype(self.font_path, font_size_url) if self.font_path else ImageFont.load_default()
        except:
            font_url = ImageFont.load_default()
        
        bbox_url = draw.textbbox((0, 0), url, font=font_url)
        url_w = bbox_url[2] - bbox_url[0]
        url_h = bbox_url[3] - bbox_url[1]
        
        # 如果URL太宽，缩小字体
        if url_w > text_width:
            font_size_url = max(8, int(font_size_url * text_width / url_w))
            try:
                font_url = ImageFont.truetype(self.font_path, font_size_url) if self.font_path else ImageFont.load_default()
            except:
                font_url = ImageFont.load_default()
            bbox_url = draw.textbbox((0, 0), url, font=font_url)
            url_w = bbox_url[2] - bbox_url[0]
            url_h = bbox_url[3] - bbox_url[1]
        
        url_x = text_start_x
        url_y = (watermark_h // 2) + 2  # 下半部分
        
        # 添加阴影/轮廓效果
        shadow_offset = (1, 1)
        shadow_color = (0, 0, 0, int(255 * alpha * 0.3))  # 浅黑色阴影
        
        # 品牌文字阴影
        draw.text((brand_x + shadow_offset[0], brand_y + shadow_offset[1]), 
                 brand, fill=shadow_color, font=font_brand)
        # URL文字阴影
        draw.text((url_x + shadow_offset[0], url_y + shadow_offset[1]), 
                 url, fill=shadow_color, font=font_url)
        
        # 绘制主要文字
        draw.text((brand_x, brand_y), brand, fill=text_color, font=font_brand)
        draw.text((url_x, url_y), url, fill=text_color, font=font_url)
        
        # 应用轻微高斯模糊
        blur_radius = random.uniform(0.5, 1.0)
        watermark_pil = watermark_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return np.array(watermark_pil)


    def apply_mini_social_corner_logo(self, image, alpha=None, position_bias=0.9):
        """
        应用小型微信风格社交账号角标水印
        
        Args:
            image: 输入图像 (numpy数组，BGR格式)
            alpha: 透明度 (0.35-0.7)，None则随机
            position_bias: 底部右角位置偏好度 (0-1)，越高越倾向右下角
            
        Returns:
            带水印的图像 (numpy数组，BGR格式)
        """
        img = image.copy()
        h, w = img.shape[:2]
        
        # 随机透明度（确保最小0.35）
        if alpha is None:
            alpha = random.uniform(0.35, 0.7)
        else:
            alpha = max(0.35, min(0.7, alpha))  # 确保在范围内
        
        # 决定位置（90%概率选择右下角，10%概率选择左下角）
        if random.random() < position_bias:
            position = 'bottom_right'
        else:
            position = 'bottom_left'
        
        # 创建水印
        watermark = self._create_mini_social_corner_logo_watermark(w, h, alpha)
        
        # 应用水印（使用自定义位置计算）
        result = self._apply_mini_social_watermark(img, watermark, position)
        
        return result


    def _create_mini_social_corner_logo_watermark(self, img_w, img_h, alpha):
        """创建小型微信风格社交账号角标水印"""
        # 计算水印尺寸（占图像的10-20%宽度，4-8%高度）
        watermark_w = int(img_w * random.uniform(0.10, 0.20))
        watermark_h = int(img_h * random.uniform(0.04, 0.08))
        
        # 确保最小文字高度（14-20像素）
        min_text_height = max(14, int(img_h * 0.015))
        max_text_height = max(20, int(img_h * 0.025))
        watermark_h = max(watermark_h, min_text_height + 8)  # 文字高度加边距
        
        # 创建水印图像（RGBA）
        watermark = np.zeros((watermark_h, watermark_w, 4), dtype=np.uint8)
        watermark_pil = Image.fromarray(watermark, 'RGBA')
        draw = ImageDraw.Draw(watermark_pil)
        
        # 随机选择账号名称
        account_candidates = ["电客一点通", "电路小课堂", "AnalogTips", "DemoWenLab", "电子工坊", "芯片实验室"]
        account_name = random.choice(account_candidates)
        
        # 计算布局参数
        icon_size = min(watermark_h - 6, int(watermark_h * 0.8))  # 图标尺寸
        text_start_x = icon_size + 6  # 文字起始X坐标
        text_width = watermark_w - text_start_x - 4  # 文字区域宽度
        
        # 绘制圆形图标
        icon_center_x = icon_size // 2 + 3
        icon_center_y = watermark_h // 2
        
        # 图标颜色（绿色、蓝色或灰色）
        icon_colors = [
            (34, 197, 94),   # 绿色
            (59, 130, 246),  # 蓝色
            (107, 114, 128)  # 灰色
        ]
        icon_color = random.choice(icon_colors) + (int(255 * alpha),)
        
        # 绘制圆形图标背景
        draw.ellipse(
            [icon_center_x - icon_size//2, icon_center_y - icon_size//2,
             icon_center_x + icon_size//2, icon_center_y + icon_size//2],
            fill=icon_color
        )
        
        # 可选：绘制1-2条白色内部线条（模拟聊天/电子符号）
        if random.random() > 0.3:  # 70%概率绘制内部图案
            num_lines = random.randint(1, 2)
            line_color = (255, 255, 255, int(255 * alpha * 0.8))
            
            for i in range(num_lines):
                if random.random() > 0.5:
                    # 水平线（模拟消息气泡）
                    line_y = icon_center_y + random.randint(-icon_size//4, icon_size//4)
                    draw.line(
                        [icon_center_x - icon_size//3, line_y, icon_center_x + icon_size//3, line_y],
                        fill=line_color, width=1
                    )
                else:
                    # 垂直线或对角线（模拟电路符号）
                    if random.random() > 0.5:
                        # 垂直线
                        line_x = icon_center_x + random.randint(-icon_size//4, icon_size//4)
                        draw.line(
                            [line_x, icon_center_y - icon_size//3, line_x, icon_center_y + icon_size//3],
                            fill=line_color, width=1
                        )
                    else:
                        # 对角线
                        draw.line(
                            [icon_center_x - icon_size//4, icon_center_y - icon_size//4,
                             icon_center_x + icon_size//4, icon_center_y + icon_size//4],
                            fill=line_color, width=1
                        )
        
        # 绘制文字
        text_color = (255, 255, 255, int(255 * alpha))  # 白色文字
        
        # 计算合适字体大小
        font_size = max(min_text_height, min(max_text_height, int(watermark_h * 0.7)))
        try:
            font = ImageFont.truetype(self.font_path, font_size) if self.font_path else ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 计算文字尺寸
        bbox = draw.textbbox((0, 0), account_name, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # 如果文字太宽，缩小字体
        if text_w > text_width:
            font_size = max(min_text_height, int(font_size * text_width / text_w))
            try:
                font = ImageFont.truetype(self.font_path, font_size) if self.font_path else ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), account_name, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        
        # 文字位置（垂直居中）
        text_x = text_start_x
        text_y = (watermark_h - text_h) // 2
        
        # 添加阴影/轮廓效果
        shadow_offset = (1, 1)
        shadow_color = (0, 0, 0, int(255 * alpha * 0.4))  # 深色阴影
        
        # 文字阴影
        draw.text((text_x + shadow_offset[0], text_y + shadow_offset[1]), 
                 account_name, fill=shadow_color, font=font)
        
        # 绘制主要文字
        draw.text((text_x, text_y), account_name, fill=text_color, font=font)
        
        # 可选：轻微模糊效果
        if random.random() > 0.5:
            blur_radius = random.uniform(0.3, 0.8)
            watermark_pil = watermark_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return np.array(watermark_pil)


    def _apply_mini_social_watermark(self, image, watermark, position):
        """应用小型社交水印到图像（自定义位置计算，确保靠近角落）"""
        img = image.copy()
        h, w = img.shape[:2]
        wm_h, wm_w = watermark.shape[:2]
        
        # 小边距（4-16像素，根据图像大小自适应）
        margin_x = random.randint(4, min(16, int(w * 0.02)))
        margin_y = random.randint(4, min(16, int(h * 0.02)))
        
        # 确定水印位置（紧贴角落）
        if position == 'bottom_right':
            x, y = w - wm_w - margin_x, h - wm_h - margin_y
        elif position == 'bottom_left':
            x, y = margin_x, h - wm_h - margin_y
        else:
            # 默认右下角
            x, y = w - wm_w - margin_x, h - wm_h - margin_y
        
        # 确保坐标在范围内
        x = max(0, min(x, w - wm_w))
        y = max(0, min(y, h - wm_h))
        
        img = self._overlay_watermark(img, watermark, x, y)
        return img


def batch_generate_watermarks(image_path, output_dir, num_variations=10, 
                             watermark_types=None, positions=None):
    """
    为单张图像生成多个水印变体
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        num_variations: 每种组合生成的变体数
        watermark_types: 水印类型列表，None则使用全部
        positions: 位置列表，None则使用全部
        
    Returns:
        生成的文件数
    """
    if watermark_types is None:
        watermark_types = [
            'transparent_url',
            'transparent_logo_text',
            'transparent_chinese',
            'semi_transparent_url',
            'semi_transparent_chinese',
            'corner_website_logo',
            'mini_social_corner_logo'
        ]
    
    if positions is None:
        positions = ['corner', 'center', 'diagonal', 'full']
    
    os.makedirs(output_dir, exist_ok=True)
    
    generator = WatermarkGenerator()
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    
    file_count = 0
    for wm_type in watermark_types:
        for position in positions:
            for i in range(num_variations):
                try:
                    result = generator.generate(image, wm_type, position)
                    
                    # 生成文件名
                    filename = f"{wm_type}_{position}_{i:03d}.jpg"
                    output_path = os.path.join(output_dir, filename)
                    
                    cv2.imwrite(output_path, result)
                    file_count += 1
                    
                except Exception as e:
                    print(f"生成水印失败 ({wm_type}, {position}, {i}): {e}")
    
    return file_count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='水印生成器')
    parser.add_argument('image', help='输入图像路径')
    parser.add_argument('--output', '-o', default='watermarked_output', 
                       help='输出目录')
    parser.add_argument('--num', '-n', type=int, default=5,
                       help='每种组合生成的变体数')
    
    args = parser.parse_args()
    
    count = batch_generate_watermarks(args.image, args.output, args.num)
    print(f"成功生成 {count} 张水印图像")
