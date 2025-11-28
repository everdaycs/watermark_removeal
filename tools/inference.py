"""
推理脚本 - 对新图像进行去水印处理
"""
import os
import cv2
import torch
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

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
    print(f"训练轮数: {checkpoint['epoch']}")
    
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
        print(f"错误: 无法读取图像 {image_path}")
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
    # 对于 JPEG: cv2.IMWRITE_JPEG_QUALITY=95 (0-100, 越高越清晰)
    # 对于 PNG: 保存为无损格式
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
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 处理每张图像
    success_count = 0
    for filename in tqdm(image_files, desc="处理中"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        if remove_watermark(model, input_path, output_path, device):
            success_count += 1
    
    print(f"\n完成! 成功处理 {success_count}/{len(image_files)} 张图像")
    print(f"结果保存在: {output_dir}")

def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """计算SSIM (简化版本)"""
    from skimage.metrics import structural_similarity as ssim
    
    # 确保图像是灰度或RGB
    if len(img1.shape) == 3:
        return ssim(img1, img2, multichannel=True, channel_axis=2)
    else:
        return ssim(img1, img2)

def evaluate_results(clean_dir, watermarked_dir, restored_dir):
    """评估去水印效果"""
    psnr_scores = []
    ssim_scores = []
    
    for filename in os.listdir(clean_dir):
        clean_path = os.path.join(clean_dir, filename)
        watermarked_path = os.path.join(watermarked_dir, filename)
        restored_path = os.path.join(restored_dir, filename)
        
        if not os.path.exists(restored_path):
            continue
        
        clean_img = cv2.imread(clean_path)
        watermarked_img = cv2.imread(watermarked_path)
        restored_img = cv2.imread(restored_path)
        
        if clean_img is None or restored_img is None:
            continue
        
        # 确保尺寸一致
        if clean_img.shape != restored_img.shape:
            restored_img = cv2.resize(restored_img, (clean_img.shape[1], clean_img.shape[0]))
        
        # 计算指标
        psnr = calculate_psnr(clean_img, restored_img)
        try:
            ssim_score = calculate_ssim(clean_img, restored_img)
            ssim_scores.append(ssim_score)
        except:
            pass
        
        psnr_scores.append(psnr)
    
    if psnr_scores:
        print(f"\n评估结果:")
        print(f"平均PSNR: {np.mean(psnr_scores):.2f} dB")
        if ssim_scores:
            print(f"平均SSIM: {np.mean(ssim_scores):.4f}")
        print(f"处理图像数: {len(psnr_scores)}")

def main():
    parser = argparse.ArgumentParser(description='水印去除推理脚本')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像/目录')
    parser.add_argument('--output', type=str, required=True, help='输出图像/目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--evaluate', action='store_true', help='是否评估结果')
    parser.add_argument('--clean_dir', type=str, help='干净图像目录(用于评估)')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    
    # 处理
    if os.path.isdir(args.input):
        # 批量处理
        batch_remove_watermarks(model, args.input, args.output, device)
        
        # 评估
        if args.evaluate and args.clean_dir:
            evaluate_results(args.clean_dir, args.input, args.output)
    else:
        # 单张图像
        success = remove_watermark(model, args.input, args.output, device)
        if success:
            print(f"结果已保存: {args.output}")

if __name__ == "__main__":
    # 如果没有命令行参数，使用默认配置
    import sys
    if len(sys.argv) == 1:
        print("使用示例:")
        print("python inference.py --checkpoint checkpoints/unet_watermark_removal_best.pth --input input.jpg --output output.jpg")
        print("python inference.py --checkpoint checkpoints/unet_watermark_removal_best.pth --input input_dir/ --output output_dir/")
    else:
        main()
