"""
大规模数据集生成脚本

对同一张图片生成多种水印变体（多位置、多类型、多透明度、多大小）
生成庞大的训练数据集

使用方法:
    python generate_large_dataset.py /path/to/clean/images --output data/synthetic --num 20

参数说明:
    image_dir: 干净图像目录路径
    --output: 输出目录（默认: data/synthetic）
    --num: 每张图每种组合生成的变体数（默认: 20）
    --train-ratio: 训练集比例（默认: 0.85）
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
import random
from tqdm import tqdm

# 导入水印生成器
from watermark_generator import WatermarkGenerator


class DatasetGenerator:
    def __init__(self, output_dir, train_ratio=0.85):
        """
        初始化数据集生成器
        
        Args:
            output_dir: 输出目录
            train_ratio: 训练集比例
        """
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        
        # 创建输出目录结构
        self.train_clean_dir = os.path.join(output_dir, 'train_clean')
        self.train_watermarked_dir = os.path.join(output_dir, 'train_watermarked')
        self.val_clean_dir = os.path.join(output_dir, 'val_clean')
        self.val_watermarked_dir = os.path.join(output_dir, 'val_watermarked')
        
        os.makedirs(self.train_clean_dir, exist_ok=True)
        os.makedirs(self.train_watermarked_dir, exist_ok=True)
        os.makedirs(self.val_clean_dir, exist_ok=True)
        os.makedirs(self.val_watermarked_dir, exist_ok=True)
        
        self.watermark_gen = WatermarkGenerator()
        
        # 12种水印类型
        self.watermark_types = self.watermark_gen.watermark_types
        
        # 10种水印位置
        self.positions = self.watermark_gen.positions
    
    def get_image_files(self, image_dir):
        """获取图像文件列表"""
        image_dir = Path(image_dir)
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
        
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.glob(f'**/*{ext}'))
        
        return sorted([str(f) for f in image_files])
    
    def process_image(self, image_path, image_id, num_watermarks=200):
        """
        处理单张图像，随机生成200种水印变体
        
        Args:
            image_path: 图像路径
            image_id: 图像ID（用于文件命名）
            num_watermarks: 生成的水印数量（默认200种随机组合）
            
        Returns:
            生成的图像对数 (干净图像, 水印图像)
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告: 无法加载图像 {image_path}")
            return 0
        
        # 确定是否为训练集还是验证集
        is_train = random.random() < self.train_ratio
        
        if is_train:
            clean_dir = self.train_clean_dir
            watermarked_dir = self.train_watermarked_dir
        else:
            clean_dir = self.val_clean_dir
            watermarked_dir = self.val_watermarked_dir
        
        # 调整图像大小为统一尺寸
        target_size = (256, 256)
        image_resized = cv2.resize(image, target_size)
        
        pair_count = 0
        
        # 保存干净图像（只保存一次）
        clean_path = os.path.join(clean_dir, f"{image_id}_clean.jpg")
        cv2.imwrite(clean_path, image_resized)
        
        # 随机生成200种水印组合
        for wm_idx in range(num_watermarks):
            try:
                # 随机选择水印类型和位置
                wm_type = random.choice(self.watermark_types)
                position = random.choice(self.positions)
                
                # 生成水印
                watermarked = self.watermark_gen.generate(
                    image_resized,
                    watermark_type=wm_type,
                    position=position
                )
                
                # 生成文件名：{image_id}_{index:03d}.jpg
                base_name = f"{image_id}_wm_{wm_idx:03d}.jpg"
                
                # 保存水印图像
                watermarked_path = os.path.join(watermarked_dir, base_name)
                cv2.imwrite(watermarked_path, watermarked)
                
                pair_count += 1
                
            except Exception as e:
                print(f"错误: 生成水印失败 ({image_id}, {wm_idx}): {e}")
        
        return pair_count
    
    def generate(self, image_dir, num_watermarks=200):
        """
        生成数据集
        
        Args:
            image_dir: 输入图像目录
            num_watermarks: 每张图片生成的水印数量（随机组合）
            
        Returns:
            统计信息字典
        """
        # 获取所有图像文件
        image_files = self.get_image_files(image_dir)
        
        if not image_files:
            print(f"错误: 在 {image_dir} 中找不到任何图像")
            return None
        
        print(f"找到 {len(image_files)} 张干净图像")
        print(f"每张图像将生成 {num_watermarks} 种随机水印组合")
        print(f"总计将生成约 {len(image_files) * num_watermarks} 张训练数据")
        print()
        
        total_pairs = 0
        
        # 处理每张图像
        for idx, image_path in enumerate(tqdm(image_files, desc="生成数据集")):
            pairs = self.process_image(image_path, f"img_{idx:06d}", num_watermarks)
            total_pairs += pairs
        
        # 统计信息
        train_clean_count = len(os.listdir(self.train_clean_dir))
        train_watermarked_count = len(os.listdir(self.train_watermarked_dir))
        val_clean_count = len(os.listdir(self.val_clean_dir))
        val_watermarked_count = len(os.listdir(self.val_watermarked_dir))
        
        stats = {
            'total_source_images': len(image_files),
            'train_clean': train_clean_count,
            'train_watermarked': train_watermarked_count,
            'val_clean': val_clean_count,
            'val_watermarked': val_watermarked_count,
            'total_pairs': total_pairs
        }
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description='生成大规模水印数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成数据集，每张图200种随机水印
  python generate_large_dataset.py /path/to/clean/images --num 200
  
  # 自定义输出目录和训练集比例
  python generate_large_dataset.py /path/to/images --output my_dataset --num 300 --train-ratio 0.9

水印类型（12种）:
  transparent_url, transparent_logo_text, transparent_chinese,
  semi_transparent_url, semi_transparent_chinese,
  watermark_stamp, confidential_stamp, company_logo,
  qr_code_style, gradient_text, diagonal_stripe, mixed_watermark

水印位置（10种）:
  top_left, top_right, bottom_left, bottom_right,
  top_center, bottom_center, left_center, right_center,
  center, full

数据生成公式:
  总数据量 = 源图像数 × 每张图的水印数（随机组合12×10）
  
  示例: 20张图 × 200种随机水印 = 4,000个训练样本
  示例: 100张图 × 200种随机水印 = 20,000个训练样本
        """
    )
    
    parser.add_argument('image_dir', help='干净图像目录路径')
    parser.add_argument('--output', '-o', default='data/synthetic',
                       help='输出目录（默认: data/synthetic）')
    parser.add_argument('--num', '-n', type=int, default=200,
                       help='每张图片生成的随机水印数量（默认: 200）')
    parser.add_argument('--train-ratio', type=float, default=0.85,
                       help='训练集比例（默认: 0.85）')
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.isdir(args.image_dir):
        print(f"错误: 目录不存在 {args.image_dir}")
        sys.exit(1)
    
    # 生成数据集
    generator = DatasetGenerator(args.output, args.train_ratio)
    stats = generator.generate(args.image_dir, args.num)
    
    if stats:
        print("\n" + "="*50)
        print("数据集生成完成！")
        print("="*50)
        print(f"源图像数:        {stats['total_source_images']}")
        print(f"训练集干净图:    {stats['train_clean']}")
        print(f"训练集水印图:    {stats['train_watermarked']}")
        print(f"验证集干净图:    {stats['val_clean']}")
        print(f"验证集水印图:    {stats['val_watermarked']}")
        print(f"总计图像对数:    {stats['total_pairs']}")
        print(f"输出目录:        {args.output}")
        print(f"\n💡 数据统计:")
        print(f"   水印类型数:    12种（随机组合）")
        print(f"   水印位置数:    10种（随机组合）")
        print(f"   每张图生成:    {args.num}种水印")
        print(f"   总样本数:      {stats['total_source_images']} × {args.num} = {stats['total_source_images'] * args.num}个")
        print("="*50)


if __name__ == '__main__':
    main()