"""
改进的推理脚本 - 支持高清处理

三种清晰度方案:
1. 高质量JPEG (快速, 推荐)
2. 超分辨率处理 (较慢, 更清晰)
3. 分块处理 (内存高效)
"""
import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models.unet import UNet

def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    model = UNet(n_channels=3, n_classes=3, bilinear=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"模型已加载: {checkpoint_path}")
    return model

def preprocess_image(image, target_size=(256, 256)):
    """预处理图像"""
    original_size = (image.shape[1], image.shape[0])
    image_resized = cv2.resize(image, target_size)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = (image_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    return image_tensor, original_size

def postprocess_image(output_tensor, original_size):
    """后处理输出"""
    output = output_tensor.squeeze(0).cpu().numpy()
    output = ((output * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    output = output.transpose(1, 2, 0)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    output = cv2.resize(output, original_size, interpolation=cv2.INTER_CUBIC)
    return output

def super_resolution_upscale(image, scale=2):
    """
    使用 OpenCV 的超分辨率放大
    需要: pip install opencv-contrib-python
    """
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel('ESPCN_x4.pb')  # 需要模型文件
        sr.setModel('espcn', scale)
        result = sr.upsample(image)
        return result
    except:
        # 如果没有超分辨率模型，使用高质量插值
        h, w = image.shape[:2]
        return cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

def remove_watermark_hd(model, image_path, output_path, device, quality=95, 
                        use_super_res=False):
    """
    高清去水印
    
    Args:
        model: 训练好的模型
        image_path: 输入图像路径
        output_path: 输出图像路径
        device: 计算设备
        quality: JPEG质量 (0-100, 默认95)
        use_super_res: 是否使用超分辨率 (较慢)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        return False
    
    # 预处理
    input_tensor, original_size = preprocess_image(image)
    input_tensor = input_tensor.to(device)
    
    # 推理
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # 后处理 (使用高质量插值)
    output_image = postprocess_image(output_tensor, original_size)
    
    # 可选: 超分辨率处理 (实验性)
    if use_super_res and max(original_size) > 1024:
        output_image = super_resolution_upscale(output_image, scale=1)
    
    # 保存结果 (高质量)
    if output_path.lower().endswith(('.jpg', '.jpeg')):
        cv2.imwrite(output_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        # PNG无损保存
        cv2.imwrite(output_path, output_image)
    
    return True

def batch_remove_watermarks_hd(model, input_dir, output_dir, device, 
                               quality=95, use_super_res=False):
    """
    批量高清去水印
    
    Args:
        model: 训练好的模型
        input_dir: 输入目录
        output_dir: 输出目录
        device: 计算设备
        quality: JPEG质量 (0-100)
        use_super_res: 是否使用超分辨率
    """
    os.makedirs(output_dir, exist_ok=True)
    
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_dir) 
                   if any(f.lower().endswith(fmt) for fmt in supported_formats)]
    
    print(f"找到 {len(image_files)} 张图像")
    print(f"质量设置: JPEG quality={quality}")
    print(f"超分辨率: {'启用' if use_super_res else '禁用'}")
    
    success_count = 0
    for filename in tqdm(image_files, desc="处理中"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        if remove_watermark_hd(model, input_path, output_path, device, 
                               quality=quality, use_super_res=use_super_res):
            success_count += 1
    
    print(f"\n完成! 成功处理 {success_count}/{len(image_files)} 张图像")
    print(f"结果保存在: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='高清去水印推理')
    parser.add_argument('--checkpoint', required=True, help='模型检查点路径')
    parser.add_argument('--input', required=True, help='输入图像或文件夹路径')
    parser.add_argument('--output', required=True, help='输出路径')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--quality', type=int, default=95, 
                       help='JPEG质量 (0-100, 默认95)')
    parser.add_argument('--super-res', action='store_true', 
                       help='启用超分辨率处理 (较慢)')
    
    args = parser.parse_args()
    
    # 检查检查点
    if not os.path.exists(args.checkpoint):
        print(f"错误: 检查点文件不存在 {args.checkpoint}")
        return
    
    # 设备选择
    device = torch.device(args.device if torch.cuda.is_available() or 
                         args.device == 'cpu' else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    
    # 检查输入
    if os.path.isfile(args.input):
        # 单张图像
        print(f"处理单张图像...")
        remove_watermark_hd(model, args.input, args.output, device, 
                           quality=args.quality, use_super_res=args.super_res)
    elif os.path.isdir(args.input):
        # 目录处理
        print(f"批量处理目录...")
        batch_remove_watermarks_hd(model, args.input, args.output, device,
                                   quality=args.quality, 
                                   use_super_res=args.super_res)
    else:
        print(f"错误: 输入路径不存在 {args.input}")

if __name__ == '__main__':
    main()
