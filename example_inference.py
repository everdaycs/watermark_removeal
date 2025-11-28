#!/usr/bin/env python3
"""
使用示例：如何使用训练好的模型处理数据
"""

import os
import sys
import cv2
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.quick_inference import WatermarkRemover


def example_1_single_image():
    """示例 1: 处理单张图像"""
    print("=" * 60)
    print("示例 1: 处理单张图像")
    print("=" * 60)
    
    # 初始化去水印工具
    remover = WatermarkRemover(
        checkpoint_path='checkpoints/unet_watermark_removal_best.pth',
        device='cuda'
    )
    
    # 处理图像（返回numpy数组，不保存文件）
    try:
        result = remover.process_file('watermark_demowen/17686_2_wm_000.jpg')
        
        print(f"✓ 处理成功")
        print(f"  结果形状: {result.shape}")
        print(f"  数据类型: {result.dtype}")
        print(f"  像素值范围: [{result.min()}, {result.max()}]")
        
        return result
    except FileNotFoundError:
        print("✗ 示例文件不存在，跳过此例子")
        return None


def example_2_batch_process():
    """示例 2: 批量处理图像"""
    print("\n" + "=" * 60)
    print("示例 2: 批量处理图像")
    print("=" * 60)
    
    remover = WatermarkRemover(
        checkpoint_path='checkpoints/unet_watermark_removal_best.pth',
        device='cuda'
    )
    
    # 检查目录是否存在
    if not os.path.exists('watermark_demowen'):
        print("✗ watermark_demowen 目录不存在")
        return
    
    # 批量处理（生成器模式，内存高效）
    results = {}
    count = 0
    for filename, processed_image in remover.batch_process('watermark_demowen'):
        results[filename] = processed_image
        count += 1
        if count % 100 == 0:
            print(f"  已处理 {count} 张...")
        if count >= 10:  # 只处理前10张作为演示
            break
    
    print(f"✓ 批量处理完成")
    print(f"  共处理: {len(results)} 张图像")
    print(f"  示例: {list(results.keys())[:3]}")
    
    return results


def example_3_in_memory_processing():
    """示例 3: 纯内存处理"""
    print("\n" + "=" * 60)
    print("示例 3: 纯内存处理")
    print("=" * 60)
    
    remover = WatermarkRemover(
        checkpoint_path='checkpoints/unet_watermark_removal_best.pth',
        device='cuda'
    )
    
    # 创建虚拟图像用于演示
    print("创建测试图像...")
    test_image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    
    # 处理（不需要保存中间文件）
    result = remover.process(test_image)
    
    print(f"✓ 处理完成")
    print(f"  输入形状: {test_image.shape}")
    print(f"  输出形状: {result.shape}")
    print(f"  值范围: [{result.min()}, {result.max()}]")
    
    return result


def example_4_comparison():
    """示例 4: 对比处理前后"""
    print("\n" + "=" * 60)
    print("示例 4: 处理前后对比（内存处理，无文件保存）")
    print("=" * 60)
    
    remover = WatermarkRemover(
        checkpoint_path='checkpoints/unet_watermark_removal_best.pth',
        device='cuda'
    )
    
    try:
        # 读取带水印图像
        watermarked = remover.process_file('watermark_demowen/17686_2_wm_000.jpg')
        
        # 在内存中计算统计信息
        print(f"处理结果统计:")
        print(f"  最小像素值: {watermarked.min()}")
        print(f"  最大像素值: {watermarked.max()}")
        print(f"  平均像素值: {watermarked.mean():.2f}")
        print(f"  标准差: {watermarked.std():.2f}")
        
        # 如果需要查看处理效果，可以手动保存用于对比
        print(f"\n如需保存对比图像:")
        print(f"  cv2.imwrite('result.jpg', watermarked)")
        
        return watermarked
    except FileNotFoundError:
        print("✗ 示例文件不存在，跳过此例子")
        return None


def example_5_api_usage():
    """示例 5: 在代码中集成使用"""
    print("\n" + "=" * 60)
    print("示例 5: 代码集成示例")
    print("=" * 60)
    
    print("""
# 在你的代码中集成：

from tools.quick_inference import WatermarkRemover
import cv2

# 初始化一次
remover = WatermarkRemover('checkpoints/best.pth')

# 处理单张
image = cv2.imread('input.jpg')
clean = remover.process(image)

# 或从文件处理
clean = remover.process_file('input.jpg')

# 或批量处理
for filename, result in remover.batch_process('input_folder'):
    # 在内存中处理 result
    process_result(result)
    """)


if __name__ == '__main__':
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     快速推理工具使用示例 (内存处理，无文件生成)          ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # 运行示例
    example_1_single_image()
    example_2_batch_process()
    example_3_in_memory_processing()
    example_4_comparison()
    example_5_api_usage()
    
    print("\n" + "=" * 60)
    print("✓ 所有示例演示完成！")
    print("=" * 60)
    print("\n更多信息请查看: INFERENCE_IN_MEMORY.md")
    print("\n命令行使用:")
    print("  python3 scripts/quick_inference.py --mode single --input test.jpg")
    print("  python3 scripts/quick_inference.py --mode batch --input input_folder/")
