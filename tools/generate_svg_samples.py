#!/usr/bin/env python3
"""
快速生成SVG水印样本脚本

这个脚本用于快速生成三种新SVG水印风格的样本图像和对应的掩码
用于演示和测试目的。

三种新风格:
1. elecfans_logo_svg_corner - ElecFans Logo SVG 角落水印
2. elecfans_web_svg_center - ElecFans Web SVG 中心水印
3. wechat_svg_corner_id - WeChat SVG 角落 + ID 文字水印
"""

import os
import sys
import random
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# 添加tools目录到路径
sys.path.insert(0, os.path.dirname(__file__))

# 从主脚本导入所需函数
from generate_watermarks_with_masks import (
    elecfans_logo_svg_corner,
    elecfans_web_svg_center,
    wechat_svg_corner_id,
    validate_watermark_visibility,
    draw_text_with_alpha,
    CAIROSVG_AVAILABLE,
)

def create_sample_image(width=800, height=600, style_name="sample"):
    """创建一个示例图像（渐变背景）"""
    img = Image.new('RGB', (width, height), color='white')
    pixels = img.load()
    
    # 创建渐变背景
    for y in range(height):
        for x in range(width):
            r = int(100 + (x / width) * 155)  # 100-255
            g = int(150 + (y / height) * 105)  # 150-255
            b = int(200 - (x / width) * 50)    # 200-150
            pixels[x, y] = (r, g, b)
    
    return img

def apply_watermark_and_save(base_image, watermark_func, output_path, mask_path, style_name):
    """应用水印并保存图像和掩码"""
    width, height = base_image.size
    
    try:
        # 调用水印函数 - 返回RGBA Image对象
        watermark_overlay = watermark_func(width, height)
        
        if watermark_overlay is None:
            print(f"  ✗ {style_name}: 失败（返回None）")
            return False
        
        # 转换基础图像为RGBA
        result = base_image.convert('RGBA')
        
        # 应用水印overlay
        result = Image.alpha_composite(result, watermark_overlay)
        
        # 转换回RGB保存
        result_rgb = result.convert('RGB')
        
        # 验证可见性 - 返回字典
        overlay_array = np.array(watermark_overlay)
        visibility_result = validate_watermark_visibility(overlay_array, width, height)
        if not visibility_result['is_visible']:
            reasons = ', '.join(visibility_result.get('reasons', ['未知']))
            print(f"  ! {style_name}: 生成成功但可见性验证失败 - {reasons}")
        
        # 保存图像
        result_rgb.save(output_path)
        
        # 从overlay的alpha通道生成掩码
        alpha_channel = np.array(watermark_overlay.split()[3])
        binary_mask = (alpha_channel > 0).astype(np.uint8) * 255
        mask_img = Image.fromarray(binary_mask, mode='L')
        mask_img.save(mask_path)
        
        print(f"  ✓ {style_name}")
        print(f"    └─ 图像: {output_path}")
        print(f"    └─ 掩码: {mask_path}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ {style_name}: 错误 - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 80)
    print("SVG 水印风格样本生成器")
    print("=" * 80)
    print()
    
    # 检查cairosvg
    print(f"cairosvg 状态: {'✓ 可用' if CAIROSVG_AVAILABLE else '✗ 不可用'}")
    if not CAIROSVG_AVAILABLE:
        print("错误: cairosvg 未安装。请运行: pip install cairosvg")
        return
    print()
    
    # 创建输出目录
    output_dir = Path("./svg_samples")
    output_dir.mkdir(exist_ok=True)
    
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    
    print(f"输出目录: {output_dir.absolute()}")
    print()
    
    # 生成多个样本图像
    samples = [
        {"name": "sample_1_blue", "width": 800, "height": 600},
        {"name": "sample_2_large", "width": 1280, "height": 720},
        {"name": "sample_3_small", "width": 400, "height": 300},
    ]
    
    styles = [
        {
            "func": elecfans_logo_svg_corner,
            "name": "elecfans_logo_svg_corner",
            "desc": "ElecFans Logo SVG 角落水印 (6-10% 高度, 10-18% 宽度)"
        },
        {
            "func": elecfans_web_svg_center,
            "name": "elecfans_web_svg_center",
            "desc": "ElecFans Web SVG 中心水印 (30-50% 宽度, 10-18% 高度)"
        },
        {
            "func": wechat_svg_corner_id,
            "name": "wechat_svg_corner_id",
            "desc": "WeChat SVG 角落 + ID 文字水印 (3-6% 高度, 8-15% 宽度)"
        },
    ]
    
    # 对每个样本图像生成所有风格
    for sample_info in samples:
        print(f"样本: {sample_info['name']}")
        print(f"  尺寸: {sample_info['width']}x{sample_info['height']}")
        print()
        
        # 创建基础图像
        base_image = create_sample_image(
            width=sample_info['width'],
            height=sample_info['height'],
            style_name=sample_info['name']
        )
        
        # 对每个风格生成水印
        for style_info in styles:
            print(f"  生成: {style_info['name']}")
            print(f"         {style_info['desc']}")
            
            # 文件路径
            base_name = f"{sample_info['name']}_{style_info['name']}"
            output_path = output_dir / f"{base_name}.png"
            mask_path = mask_dir / f"{base_name}_mask.png"
            
            # 应用水印
            success = apply_watermark_and_save(
                base_image,
                style_info['func'],
                str(output_path),
                str(mask_path),
                style_info['name']
            )
            print()
        
        print("-" * 80)
        print()
    
    # 生成总结
    print("=" * 80)
    print("生成完成！")
    print("=" * 80)
    print()
    print("生成的文件:")
    print(f"  图像: {len(list(output_dir.glob('*.png')))} 个")
    print(f"  掩码: {len(list(mask_dir.glob('*.png')))} 个")
    print()
    print("文件位置:")
    print(f"  {output_dir.absolute()}/")
    print()
    
    # 列出生成的文件
    print("图像文件:")
    for img_file in sorted(output_dir.glob("sample_*.png")):
        print(f"  • {img_file.name}")
    print()
    
    print("掩码文件:")
    for mask_file in sorted(mask_dir.glob("*.png")):
        print(f"  • {mask_file.name}")
    print()
    
    # 打开文件浏览器（如果在图形环境中）
    print("提示: 你可以在文件浏览器中查看生成的样本:")
    print(f"  nautilus {output_dir.absolute()}")
    print()

if __name__ == "__main__":
    main()
