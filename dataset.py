"""
数据加载器和数据增强
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class WatermarkDataset(Dataset):
    """水印去除数据集"""
    def __init__(self, clean_dir, watermarked_dir, transform=None, image_size=(256, 256), source_dir=None):
        """
        Args:
            clean_dir: 干净图像目录（用于处理已有数据集）或原始源目录（用于新生成的数据）
            watermarked_dir: 带水印图像目录
            transform: 数据增强
            image_size: 图像尺寸
            source_dir: 可选的源图像目录（如果为None，则使用clean_dir）
        """
        self.clean_dir = clean_dir
        self.watermarked_dir = watermarked_dir
        self.source_dir = source_dir or clean_dir  # 如果指定了source_dir，优先使用
        self.transform = transform
        self.image_size = image_size
        
        # 获取所有图像对
        self.image_pairs = self._get_image_pairs()
    
    def _get_image_pairs(self):
        """获取图像对列表"""
        pairs = []
        clean_images = {}
        
        # 首先收集所有源干净图像（从source_dir中）
        for clean_name in os.listdir(self.source_dir):
            if clean_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                clean_path = os.path.join(self.source_dir, clean_name)
                # 检查图像是否可以读取
                if cv2.imread(clean_path) is not None:
                    base_name = os.path.splitext(clean_name)[0]
                    clean_images[base_name] = clean_name
                else:
                    print(f"跳过无法读取的干净图像: {clean_name}")
        
        # 然后为每个水印图像找到对应的干净图像
        for watermarked_name in os.listdir(self.watermarked_dir):
            if watermarked_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                watermarked_path = os.path.join(self.watermarked_dir, watermarked_name)
                # 检查水印图像是否可以读取
                if cv2.imread(watermarked_path) is None:
                    print(f"跳过无法读取的水印图像: {watermarked_name}")
                    continue
                    
                # 水印图像命名规则: "{base_name}_wm_{index}.jpg"
                # 例如: "image_001_wm_002.jpg" -> "image_001"
                base_name = watermarked_name.rsplit('_wm_', 1)[0]
                
                if base_name in clean_images:
                    pairs.append((clean_images[base_name], watermarked_name))
        
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        clean_name, watermarked_name = self.image_pairs[idx]
        
        # 读取图像（从source_dir中读取干净图像）
        clean_path = os.path.join(self.source_dir, clean_name)
        watermarked_path = os.path.join(self.watermarked_dir, watermarked_name)
        
        clean_img = cv2.imread(clean_path)
        watermarked_img = cv2.imread(watermarked_path)
        
        # BGR to RGB
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB)
        
        # 确保图像尺寸一致（在应用transform之前）
        if self.image_size:
            clean_img = cv2.resize(clean_img, (self.image_size[1], self.image_size[0]))
            watermarked_img = cv2.resize(watermarked_img, (self.image_size[1], self.image_size[0]))
        
        # 数据增强
        if self.transform:
            # Albumentations需要同时增强两张图像
            augmented = self.transform(image=watermarked_img, mask=clean_img)
            watermarked_img = augmented['image']
            clean_img = augmented['mask']
        
        return {
            'watermarked': watermarked_img,
            'clean': clean_img,
            'filename': clean_name
        }

def get_train_transform(image_size=(256, 256)):
    """获取训练数据增强"""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        # 轻微的颜色增强
        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
            p=0.3
        ),
        # 归一化到[-1, 1]
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ], additional_targets={'mask': 'image'})

def get_val_transform(image_size=(256, 256)):
    """获取验证数据增强（仅调整大小和归一化）"""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ], additional_targets={'mask': 'image'})

def create_dataloaders(config):
    """创建训练和验证数据加载器"""
    # 如果有source_dir配置，使用它作为干净图像源；否则使用clean_dir
    train_source_dir = getattr(config, 'TRAIN_SOURCE_PATH', None)
    val_source_dir = getattr(config, 'VAL_SOURCE_PATH', None)
    
    train_dataset = WatermarkDataset(
        clean_dir=config.TRAIN_CLEAN_PATH,
        watermarked_dir=config.TRAIN_WATERMARKED_PATH,
        source_dir=train_source_dir,  # 可选的源目录
        transform=get_train_transform(config.IMAGE_SIZE),
        image_size=config.IMAGE_SIZE
    )
    
    val_dataset = WatermarkDataset(
        clean_dir=config.VAL_CLEAN_PATH,
        watermarked_dir=config.VAL_WATERMARKED_PATH,
        source_dir=val_source_dir,  # 可选的源目录
        transform=get_val_transform(config.IMAGE_SIZE),
        image_size=config.IMAGE_SIZE
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # 测试数据加载
    from config import Config
    config = Config()
    
    try:
        train_loader, val_loader = create_dataloaders(config)
        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")
        
        # 测试一个批次
        batch = next(iter(train_loader))
        print(f"带水印图像形状: {batch['watermarked'].shape}")
        print(f"干净图像形状: {batch['clean'].shape}")
    except Exception as e:
        print(f"错误: {e}")
        print("请先运行 generate_dataset.py 生成训练数据")
