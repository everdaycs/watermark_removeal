#!/usr/bin/env python3
"""
分块推理脚本 - 保持高分辨率图像细节

特点：
- 使用滑动窗口处理原始分辨率图像
- 避免整体降采样带来的信息丢失
- 支持重叠区域的平滑混合
- 适合处理高分辨率电路图

使用方法：
    python3 batch_inference_tiled.py --input <输入目录> --output <输出目录>
"""

import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.unet import UNet


class TiledInference:
    """分块推理器 - 保持高分辨率细节"""
    
    def __init__(self, model, device, tile_size=256, overlap=32):
        """
        Args:
            model: 训练好的模型
            device: 计算设备
            tile_size: 每个块的大小（应与训练尺寸一致）
            overlap: 块之间的重叠像素数（用于平滑拼接）
        """
        self.model = model
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        
    def _create_weight_mask(self, size):
        """创建用于混合的权重掩码（中心权重高，边缘权重低）"""
        # 创建线性渐变
        ramp = np.linspace(0, 1, self.overlap)
        
        # 创建2D权重
        weight = np.ones((size, size), dtype=np.float32)
        
        # 边缘渐变
        if self.overlap > 0:
            # 上边缘
            weight[:self.overlap, :] *= ramp[:, np.newaxis]
            # 下边缘
            weight[-self.overlap:, :] *= ramp[::-1, np.newaxis]
            # 左边缘
            weight[:, :self.overlap] *= ramp[np.newaxis, :]
            # 右边缘
            weight[:, -self.overlap:] *= ramp[::-1][np.newaxis, :]
        
        return weight
    
    def _preprocess_tile(self, tile):
        """预处理单个块"""
        # BGR to RGB
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        # 归一化到[-1, 1]
        tile_normalized = (tile_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
        # 转换为tensor
        tile_tensor = torch.from_numpy(tile_normalized).permute(2, 0, 1).unsqueeze(0)
        return tile_tensor
    
    def _postprocess_tile(self, output_tensor):
        """后处理单个块的输出"""
        output = output_tensor.squeeze(0).cpu().numpy()
        # 从[-1, 1]转换回[0, 255]
        output = ((output * 0.5 + 0.5) * 255).clip(0, 255).astype(np.float32)
        # CHW to HWC
        output = output.transpose(1, 2, 0)
        # RGB to BGR
        output = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32)
        return output
    
    def process_image(self, image):
        """
        处理完整图像（分块推理）
        
        Args:
            image: BGR格式的输入图像
            
        Returns:
            处理后的BGR图像
        """
        h, w = image.shape[:2]
        
        # 如果图像小于tile_size，直接处理
        if h <= self.tile_size and w <= self.tile_size:
            return self._process_small_image(image)
        
        # 计算需要的块数
        n_tiles_h = max(1, int(np.ceil((h - self.overlap) / self.stride)))
        n_tiles_w = max(1, int(np.ceil((w - self.overlap) / self.stride)))
        
        # 计算填充后的尺寸
        padded_h = self.stride * n_tiles_h + self.overlap
        padded_w = self.stride * n_tiles_w + self.overlap
        
        # 填充图像（镜像填充以避免边缘伪影）
        pad_h = padded_h - h
        pad_w = padded_w - w
        
        padded_image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w,
            cv2.BORDER_REFLECT_101
        )
        
        # 创建输出和权重累积器
        output_sum = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
        weight_sum = np.zeros((padded_h, padded_w, 1), dtype=np.float32)
        
        # 权重掩码
        weight_mask = self._create_weight_mask(self.tile_size)
        weight_mask = weight_mask[:, :, np.newaxis]  # 扩展到3通道
        
        # 分块处理
        self.model.eval()
        with torch.no_grad():
            for i in range(n_tiles_h):
                for j in range(n_tiles_w):
                    # 计算块的位置
                    y_start = i * self.stride
                    x_start = j * self.stride
                    y_end = y_start + self.tile_size
                    x_end = x_start + self.tile_size
                    
                    # 提取块
                    tile = padded_image[y_start:y_end, x_start:x_end]
                    
                    # 确保块大小正确
                    if tile.shape[0] != self.tile_size or tile.shape[1] != self.tile_size:
                        tile = cv2.resize(tile, (self.tile_size, self.tile_size))
                    
                    # 预处理
                    tile_tensor = self._preprocess_tile(tile).to(self.device)
                    
                    # 推理
                    output_tensor = self.model(tile_tensor)
                    
                    # 后处理
                    output_tile = self._postprocess_tile(output_tensor)
                    
                    # 加权累积
                    output_sum[y_start:y_end, x_start:x_end] += output_tile * weight_mask
                    weight_sum[y_start:y_end, x_start:x_end] += weight_mask
        
        # 归一化
        weight_sum = np.maximum(weight_sum, 1e-8)  # 避免除零
        output = output_sum / weight_sum
        
        # 裁剪到原始尺寸
        output = output[:h, :w]
        
        return output.astype(np.uint8)
    
    def _process_small_image(self, image):
        """处理小于tile_size的图像"""
        h, w = image.shape[:2]
        
        # 填充到tile_size
        padded = cv2.copyMakeBorder(
            image,
            0, self.tile_size - h,
            0, self.tile_size - w,
            cv2.BORDER_REFLECT_101
        )
        
        # 处理
        self.model.eval()
        with torch.no_grad():
            tile_tensor = self._preprocess_tile(padded).to(self.device)
            output_tensor = self.model(tile_tensor)
            output = self._postprocess_tile(output_tensor)
        
        # 裁剪
        return output[:h, :w].astype(np.uint8)


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    model = UNet(n_channels=3, n_classes=3, bilinear=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ 模型已加载: {checkpoint_path}")
    print(f"✓ 训练轮数: {checkpoint.get('epoch', 'N/A')}")
    
    return model


def find_best_checkpoint(checkpoint_dir):
    """查找最佳检查点"""
    best_path = os.path.join(checkpoint_dir, "checkpoints","unet_watermark_removal_best.pth")
    if os.path.exists(best_path):
        return best_path
    
    # 查找其他检查点
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if checkpoints:
        return os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
    
    return None


def batch_process(model, input_dir, output_dir, device, tile_size=256, overlap=32):
    """批量处理"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建分块推理器
    inferencer = TiledInference(model, device, tile_size=tile_size, overlap=overlap)
    
    # 获取图像文件
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_dir)
                   if any(f.lower().endswith(fmt) for fmt in supported_formats)]
    
    print(f"✓ 找到 {len(image_files)} 张图像待处理")
    print(f"✓ 分块大小: {tile_size}x{tile_size}, 重叠: {overlap}")
    
    success_count = 0
    for filename in tqdm(image_files, desc="处理进度"):
        try:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # 读取图像
            image = cv2.imread(input_path)
            if image is None:
                print(f"\n✗ 无法读取: {filename}")
                continue
            
            # 分块推理
            output = inferencer.process_image(image)
            
            # 保存
            if output_path.lower().endswith(('.jpg', '.jpeg')):
                cv2.imwrite(output_path, output, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(output_path, output)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n✗ 处理失败 {filename}: {e}")
    
    print(f"\n✓ 完成! 成功处理 {success_count}/{len(image_files)} 张图像")
    print(f"✓ 结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='分块推理 - 保持高分辨率细节')
    parser.add_argument('--input', '-i', type=str, 
                        default="/home/kaga/Desktop/watermaker remover/20251201/transparent_watermark_20251201",
                        #default="/home/kaga/Desktop/watermaker remover/20251201/red_watermark_20251201",
                        help='输入图像目录')
    parser.add_argument('--output', '-o', type=str,
                        default=None,
                        help='输出目录（默认为 results_tiled）')
    parser.add_argument('--tile-size', type=int, default=256,
                        help='分块大小（默认256，应与训练尺寸一致）')
    parser.add_argument('--overlap', type=int, default=64,
                        help='重叠像素数（默认64，越大拼接越平滑但速度越慢）')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='模型检查点路径')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 分块推理 - 保持高分辨率图像细节")
    print("=" * 60)
    
    config = Config()
    
    # 设置输出目录
    if args.output is None:
        args.output = os.path.join(config.PROJECT_ROOT, "results_tiled_enhanced_")
    
    # 查找检查点
    if args.checkpoint is None:
        args.checkpoint = find_best_checkpoint(config.CHECKPOINT_PATH)
    
    if args.checkpoint is None or not os.path.exists(args.checkpoint):
        print("✗ 错误: 未找到模型检查点")
        print(f"  请确保检查点存在于: {config.CHECKPOINT_PATH}")
        sys.exit(1)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ 使用设备: {device}")
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    
    # 批量处理
    batch_process(model, args.input, args.output, device,
                  tile_size=args.tile_size, overlap=args.overlap)


if __name__ == "__main__":
    main()
