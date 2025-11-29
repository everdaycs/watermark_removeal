#!/usr/bin/env python3
"""
SVG 水印样本展示总结

这个脚本生成一个对比总结，展示三种新 SVG 水印风格的特点
"""

import os
from pathlib import Path
from PIL import Image

def generate_summary():
    """生成样本总结"""
    
    svg_dir = Path("./svg_samples")
    if not svg_dir.exists():
        print("❌ svg_samples 目录不存在！请先运行 generate_svg_samples.py")
        return
    
    # 获取所有样本文件
    images = sorted(svg_dir.glob("*.png"))
    masks = sorted((svg_dir / "masks").glob("*.png"))
    
    print("\n" + "=" * 90)
    print("SVG 水印风格样本展示总结".center(90))
    print("=" * 90 + "\n")
    
    # 统计信息
    print("📊 样本统计:")
    print(f"  • 水印样本: {len(images)} 个")
    print(f"  • 掩码文件: {len(masks)} 个")
    print(f"  • 总计: {len(images) + len(masks)} 个文件")
    print()
    
    # 按风格分组
    styles = {
        'elecfans_logo_svg_corner': {
            'name': 'ElecFans Logo SVG 角落水印',
            'desc': '品牌 Logo，放在底部角落',
            'position': '底部角落 (90% 右下，10% 左下)',
            'size': '高度 6-10%，宽度 10-18%',
            'alpha': '30-50%',
            'use_case': '品牌标识、版权保护',
        },
        'elecfans_web_svg_center': {
            'name': 'ElecFans Web SVG 中心水印',
            'desc': '网站横幅，放在中心',
            'position': '水平居中，垂直中间 (±5% 偏移)',
            'size': '宽度 30-50%，高度 10-18%',
            'alpha': '30-50%',
            'use_case': '网站保护、宣传内容',
        },
        'wechat_svg_corner_id': {
            'name': 'WeChat SVG 角落 + ID 水印',
            'desc': '社交图标 + 文字，放在角落',
            'position': '底部角落 (90% 右下，10% 左下)',
            'size': '高度 3-6%，宽度 8-15%',
            'alpha': '30-50%',
            'use_case': '社交验证、账户标识',
        }
    }
    
    # 显示每种风格的详细信息
    print("🎨 三种 SVG 水印风格详解:")
    print()
    
    for i, (style_key, info) in enumerate(styles.items(), 1):
        print(f"{'─' * 90}")
        print(f"风格 {i}️⃣: {info['name']}")
        print(f"{'─' * 90}")
        print(f"  📝 描述:      {info['desc']}")
        print(f"  📍 位置:      {info['position']}")
        print(f"  📏 尺寸:      {info['size']}")
        print(f"  🔍 透明度:    {info['alpha']}")
        print(f"  🎯 适用场景:  {info['use_case']}")
        print()
        
        # 列出该风格的样本
        style_samples = [img for img in images if style_key in img.name]
        if style_samples:
            print(f"  样本文件 ({len(style_samples)} 个):")
            for sample in style_samples:
                file_size = sample.stat().st_size / 1024  # KB
                img = Image.open(sample)
                print(f"    • {sample.name:<50} ({img.size[0]}×{img.size[1]}, {file_size:.1f}KB)")
        
        # 列出对应的掩码
        style_masks = [m for m in masks if style_key in m.name]
        if style_masks:
            print(f"  对应掩码 ({len(style_masks)} 个):")
            for mask in style_masks:
                file_size = mask.stat().st_size / 1024  # KB
                mask_img = Image.open(mask)
                print(f"    • {mask.name:<50} ({mask_img.size[0]}×{mask_img.size[1]}, {file_size:.1f}KB)")
        
        print()
    
    # 样本尺寸对比
    print("=" * 90)
    print("📐 样本尺寸对比:")
    print("=" * 90)
    print()
    
    size_groups = {
        'sample_1_blue': {'desc': '标准尺寸', 'expected': '800×600'},
        'sample_2_large': {'desc': '高分辨率', 'expected': '1280×720'},
        'sample_3_small': {'desc': '缩略图', 'expected': '400×300'},
    }
    
    print(f"{'样本名称':<20} {'期望尺寸':<15} {'3 个风格样本':<90}")
    print("─" * 90)
    
    for sample_key, info in size_groups.items():
        samples = [img for img in images if sample_key in img.name]
        if samples:
            size_str = samples[0].name.split('_')
            img = Image.open(samples[0])
            actual_size = f"{img.size[0]}×{img.size[1]}"
            style_names = [img.name.split(f'{sample_key}_')[1].split('.')[0] for img in samples]
            styles_str = ' + '.join([s.replace('_', ' ') for s in style_names])
            print(f"{sample_key:<20} {info['expected']:<15} {styles_str:<90}")
    
    print()
    
    # 文件统计
    print("=" * 90)
    print("📁 文件统计:")
    print("=" * 90)
    print()
    
    total_size = sum(img.stat().st_size for img in images) + sum(m.stat().st_size for m in masks)
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"  水印样本总大小: {sum(img.stat().st_size for img in images) / 1024:.1f} KB")
    print(f"  掩码文件总大小: {sum(m.stat().st_size for m in masks) / 1024:.1f} KB")
    print(f"  总体大小:       {total_size_mb:.2f} MB")
    print()
    
    # 使用建议
    print("=" * 90)
    print("💡 使用建议:")
    print("=" * 90)
    print()
    print("1. 查看样本效果:")
    print("   • 用图像查看器打开 svg_samples 目录")
    print("   • 对比不同尺寸下的水印效果")
    print()
    print("2. 验证掩码正确性:")
    print("   • 打开 masks 目录中的掩码文件")
    print("   • 确认白色区域对应水印位置")
    print()
    print("3. 测试水印去除:")
    print("   • 用训练好的模型测试样本图像")
    print("   • 验证掩码与去除效果的一致性")
    print()
    print("4. 调整参数:")
    print("   • 根据需要修改水印大小、透明度、位置")
    print("   • 重新运行 generate_svg_samples.py 生成新样本")
    print()
    print("5. 生成完整数据集:")
    print("   • 运行 python3 tools/generate_watermarks_with_masks.py")
    print("   • 将生成 430×32 = 13,760 个水印变体")
    print()
    
    # 技术信息
    print("=" * 90)
    print("🔧 技术信息:")
    print("=" * 90)
    print()
    print("  • SVG 渲染: cairosvg 库")
    print("  • 缓存机制: 按 (路径, 宽度, 高度) 缓存，性能优化")
    print("  • 透明度处理: RGBA 模式，30-50% alpha")
    print("  • 掩码生成: 从 alpha 通道 (alpha > 0 → 255)")
    print("  • 可见性验证: 自动检查覆盖面积和对比度")
    print("  • 生成脚本: tools/generate_svg_samples.py")
    print()
    
    # 相关文档
    print("=" * 90)
    print("📚 相关文档:")
    print("=" * 90)
    print()
    print("  • SVG_SAMPLES_README.md          - 详细样本说明")
    print("  • SVG_STYLES_QUICK_START.md      - 快速开始指南")
    print("  • docs/SVG_STYLES_GUIDE.md       - 完整参考文档")
    print("  • docs/SVG_API_EXAMPLES.md       - 代码示例")
    print("  • SVG_IMPLEMENTATION_SUMMARY.md  - 实现细节")
    print()
    
    print("=" * 90)
    print("✅ 样本展示完成！".center(90))
    print("=" * 90 + "\n")

if __name__ == "__main__":
    generate_summary()
