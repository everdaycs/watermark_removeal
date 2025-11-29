#!/usr/bin/env python3
"""
批量水印去除推理脚本

功能：
- 自动加载训练好的最佳模型
- 批量处理输入文件夹中的所有图像
- 输出去水印后的图像到指定文件夹

使用方法：
    python3 batch_inference.py

输入：/media/kaga/本地磁盘/20251121_watermark_demo/data_trans/transparent or background wartermark/
输出：results/ 文件夹

支持格式：.jpg, .jpeg, .png, .bmp
"""

import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.unet import UNet

def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    model = UNet(n_channels=3, n_classes=3, bilinear=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ 模型已加载: {checkpoint_path}")
    print(f"✓ 训练轮数: {checkpoint['epoch']}")

    return model

def preprocess_image(image, target_size=(256, 256)):
    """预处理图像"""
    # 保存原始尺寸
    original_size = (image.shape[1], image.shape[0])  # (width, height)

    # 调整大小
    image_resized = cv2.resize(image, target_size)

    # BGR to RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # 归一化到[-1, 1]
    image_normalized = (image_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5

    # 转换为tensor并添加batch维度
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)

    return image_tensor, original_size

def postprocess_image(output_tensor, original_size):
    """后处理输出"""
    # 移除batch维度
    output = output_tensor.squeeze(0).cpu().numpy()

    # 从[-1, 1]转换回[0, 255]
    output = ((output * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

    # CHW to HWC
    output = output.transpose(1, 2, 0)

    # RGB to BGR
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # 恢复原始尺寸
    output = cv2.resize(output, original_size)

    return output

def remove_watermark(model, image_path, output_path, device):
    """去除单张图像的水印"""
    # 读取图像
    image = cv2.imread(image_path)

    if image is None:
        print(f"✗ 错误: 无法读取图像 {image_path}")
        return False

    # 预处理
    input_tensor, original_size = preprocess_image(image)
    input_tensor = input_tensor.to(device)

    # 推理
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 后处理
    output_image = postprocess_image(output_tensor, original_size)

    # 保存结果（高质量）
    if output_path.lower().endswith(('.jpg', '.jpeg')):
        cv2.imwrite(output_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(output_path, output_image)

    return True

def batch_remove_watermarks(model, input_dir, output_dir, device):
    """批量去除水印"""
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_dir)
                   if any(f.lower().endswith(fmt) for fmt in supported_formats)]

    print(f"✓ 找到 {len(image_files)} 张图像待处理")

    # 处理每张图像
    success_count = 0
    for filename in tqdm(image_files, desc="处理进度"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if remove_watermark(model, input_path, output_path, device):
            success_count += 1

    print(f"\n✓ 完成! 成功处理 {success_count}/{len(image_files)} 张图像")
    print(f"✓ 结果保存在: {output_dir}")

    return success_count, len(image_files)

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 批量水印去除推理脚本")
    print("   输入: transparent or background wartermark 文件夹")
    print("   输出: results 文件夹")
    print("=" * 60)

    # 配置路径
    config = Config()

    # 硬编码路径（根据用户需求）
    INPUT_DIR = "/media/kaga/本地磁盘/20251121_watermark_demo/data_trans/transparent or background wartermark/"
    OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "results")
    CHECKPOINT_PATH = os.path.join(config.PROJECT_ROOT, "checkpoints", "unet_watermark_removal_best.pth")

    print(f"📁 输入目录: {INPUT_DIR}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"🧠 模型路径: {CHECKPOINT_PATH}")
    print()

    # 检查输入目录是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误: 输入目录不存在")
        print(f"   路径: {INPUT_DIR}")
        print("   请检查路径是否正确")
        return

    # 检查模型文件是否存在
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ 错误: 模型文件不存在")
        print(f"   路径: {CHECKPOINT_PATH}")
        print("   请先训练模型或下载预训练模型")
        return

    # 设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"⚡ 使用设备: {device}")

    # 加载模型
    try:
        model = load_model(CHECKPOINT_PATH, device)
    except Exception as e:
        print(f"❌ 错误: 模型加载失败 - {e}")
        return

    # 批量处理
    try:
        success_count, total_count = batch_remove_watermarks(model, INPUT_DIR, OUTPUT_DIR, device)
    except Exception as e:
        print(f"❌ 错误: 批量处理失败 - {e}")
        return

    print("\n" + "=" * 60)
    if success_count == total_count:
        print("🎉 所有图像处理完成！")
        print(f"   共处理: {success_count} 张图像")
    else:
        print(f"⚠️  处理完成，但有 {total_count - success_count} 张图像处理失败")
        print(f"   成功: {success_count}/{total_count} 张图像")
    print(f"   结果位置: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()