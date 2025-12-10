"""
数据加载器和数据增强
"""
import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

class WatermarkDataset(Dataset):
    """水印去除数据集"""
    def __init__(self, clean_dir, watermarked_dir, transform=None, image_size=(256, 256), source_dir=None, return_patches=False, is_training=True):
        """
        Args:
            clean_dir: 干净图像目录（用于处理已有数据集）或原始源目录（用于新生成的数据）
            watermarked_dir: 带水印图像目录
            transform: 数据增强
            image_size: 图像尺寸
            source_dir: 可选的源图像目录（如果为None，则使用clean_dir）
            return_patches: 是否同时返回局部Patch（用于训练）
            is_training: 是否为训练模式（影响patch裁剪策略）
        """
        self.clean_dir = clean_dir
        self.watermarked_dir = watermarked_dir
        self.source_dir = source_dir or clean_dir  # 如果指定了source_dir，优先使用
        self.transform = transform
        self.image_size = image_size
        self.return_patches = return_patches
        self._is_training = is_training  # 标记是否为训练模式
        
        # 获取所有图像对
        self.image_pairs = self._get_image_pairs()
    
    def _read_image(self, image_path):
        """读取图像，支持多种格式"""
        try:
            # 首先尝试用cv2读取
            img = cv2.imread(image_path)
            if img is not None and img.size > 0:
                return img
            
            # 如果cv2失败，尝试用PIL读取（用于GIF等格式）
            pil_img = Image.open(image_path)
            # 转换为RGB模式（如果是RGBA或P模式）
            if pil_img.mode in ('RGBA', 'LA', 'P'):
                pil_img = pil_img.convert('RGB')
            elif pil_img.mode == 'L':
                pil_img = pil_img.convert('RGB')
            
            # 转换为numpy数组
            img = np.array(pil_img)
            # PIL读取的是RGB，转换为BGR以保持一致性
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            return img
        except Exception as e:
            return None
    
    def _get_image_pairs(self):
        """获取图像对列表"""
        pairs = []
        clean_images = {}
        
        # 首先收集所有源干净图像（从source_dir中）
        for clean_name in os.listdir(self.source_dir):
            if clean_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                clean_path = os.path.join(self.source_dir, clean_name)
                # 检查图像是否可以读取（只在初始化时检查一次）
                img = self._read_image(clean_path)
                if img is not None and img.size > 0:
                    base_name = os.path.splitext(clean_name)[0]
                    clean_images[base_name] = clean_name
                else:
                    print(f"跳过无法读取的干净图像: {clean_name}")
        
        # 然后为每个水印图像找到对应的干净图像
        for watermarked_name in os.listdir(self.watermarked_dir):
            if watermarked_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                watermarked_path = os.path.join(self.watermarked_dir, watermarked_name)
                # 检查水印图像是否可以读取（只在初始化时检查一次）
                img = self._read_image(watermarked_path)
                if img is None or img.size == 0:
                    print(f"跳过无法读取的水印图像: {watermarked_name}")
                    continue
                    
                # 水印图像命名规则: "{base_name}_wm_{index}.jpg" 或 "{base_name}_svg_wm_{index}.jpg"
                # 例如: "image_001_wm_002.jpg" -> "image_001"
                if '_svg_wm_' in watermarked_name:
                    base_name = watermarked_name.rsplit('_svg_wm_', 1)[0]
                else:
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
        
        clean_img = self._read_image(clean_path)
        watermarked_img = self._read_image(watermarked_path)
        
        # 检查图像是否成功读取
        if clean_img is None or clean_img.size == 0:
            raise RuntimeError(f"无法读取干净图像: {clean_path}")
        if watermarked_img is None or watermarked_img.size == 0:
            raise RuntimeError(f"无法读取水印图像: {watermarked_path}")
        
        # 如果图像是灰度图，转换为RGB
        if len(clean_img.shape) == 2:
            clean_img = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2RGB)
        elif clean_img.shape[2] == 1:
            clean_img = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2RGB)
        
        if len(watermarked_img.shape) == 2:
            watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_GRAY2RGB)
        elif watermarked_img.shape[2] == 1:
            watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_GRAY2RGB)
        
        # 确保图像是RGB格式
        if clean_img.shape[2] == 4:  # RGBA
            clean_img = cv2.cvtColor(clean_img, cv2.COLOR_RGBA2RGB)
        elif clean_img.shape[2] == 3 and clean_img.dtype != np.uint8:
            # 如果不是uint8，归一化到0-255
            clean_img = (clean_img * 255).astype(np.uint8)
        
        if watermarked_img.shape[2] == 4:  # RGBA
            watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_RGBA2RGB)
        elif watermarked_img.shape[2] == 3 and watermarked_img.dtype != np.uint8:
            watermarked_img = (watermarked_img * 255).astype(np.uint8)
        
        # BGR to RGB (所有图像现在都是BGR格式，需要转换为RGB)
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB)
        
        # 确保两个图像尺寸完全一致
        if clean_img.shape != watermarked_img.shape:
            watermarked_img = cv2.resize(watermarked_img, (clean_img.shape[1], clean_img.shape[0]))
        
        # 1. 全局图像 (Global): 调整大小到 image_size
        if self.image_size:
            global_clean = cv2.resize(clean_img, (self.image_size[1], self.image_size[0]))
            global_watermarked = cv2.resize(watermarked_img, (self.image_size[1], self.image_size[0]))
        else:
            global_clean = clean_img
            global_watermarked = watermarked_img
        
        # 数据增强 (Global)
        if self.transform:
            augmented = self.transform(image=global_watermarked, mask=global_clean)
            global_watermarked_t = augmented['image']
            global_clean_t = augmented['mask']
        else:
            # 如果没有transform，至少需要转换为Tensor
            global_watermarked_t = torch.from_numpy(global_watermarked.transpose(2, 0, 1)).float() / 255.0
            global_clean_t = torch.from_numpy(global_clean.transpose(2, 0, 1)).float() / 255.0
            
        result = {
            'watermarked': global_watermarked_t,
            'clean': global_clean_t,
            'filename': clean_name
        }
        
        # 2. 局部图像 (Local Patch): 随机裁剪（训练时）或中心裁剪（验证时）
        if self.return_patches and self.image_size:
            h, w = clean_img.shape[:2]
            th, tw = self.image_size
            
            # 如果图像小于目标尺寸，先调整大小
            if h < th or w < tw:
                scale = max(th/h, tw/w)
                new_h, new_w = int(h * scale) + 1, int(w * scale) + 1
                clean_img_resized = cv2.resize(clean_img, (new_w, new_h))
                watermarked_img_resized = cv2.resize(watermarked_img, (new_w, new_h))
            else:
                clean_img_resized = clean_img
                watermarked_img_resized = watermarked_img
                
            h, w = clean_img_resized.shape[:2]
            
            # 裁剪策略：训练时随机，验证时中心裁剪
            if hasattr(self, '_is_training') and self._is_training:
                # 训练时：随机裁剪
                if h > th:
                    i = random.randint(0, h - th)
                else:
                    i = 0
                
                if w > tw:
                    j = random.randint(0, w - tw)
                else:
                    j = 0
            else:
                # 验证时：中心裁剪，确保确定性
                i = max(0, (h - th) // 2)
                j = max(0, (w - tw) // 2)
                
            local_clean = clean_img_resized[i:i+th, j:j+tw]
            local_watermarked = watermarked_img_resized[i:i+th, j:j+tw]
            
            # 数据增强 (Local) - 使用相同的transform
            if self.transform:
                augmented_local = self.transform(image=local_watermarked, mask=local_clean)
                local_watermarked_t = augmented_local['image']
                local_clean_t = augmented_local['mask']
            else:
                local_watermarked_t = torch.from_numpy(local_watermarked.transpose(2, 0, 1)).float() / 255.0
                local_clean_t = torch.from_numpy(local_clean.transpose(2, 0, 1)).float() / 255.0
                
            result['local_watermarked'] = local_watermarked_t
            result['local_clean'] = local_clean_t
        
        return result

def get_train_transform(image_size=(256, 256)):
    """获取训练数据增强 (Resize由Dataset处理)"""
    return A.Compose([
        # Resize移至Dataset中处理，以支持Global/Local策略
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
    """获取验证数据增强 (Resize由Dataset处理)"""
    return A.Compose([
        # Resize移至Dataset中处理
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
        image_size=config.IMAGE_SIZE,
        return_patches=True,  # 启用Patch训练
        is_training=True  # 训练时使用随机裁剪
    )
    
    val_dataset = WatermarkDataset(
        clean_dir=config.VAL_CLEAN_PATH,
        watermarked_dir=config.VAL_WATERMARKED_PATH,
        source_dir=val_source_dir,  # 可选的源目录
        transform=get_val_transform(config.IMAGE_SIZE),
        image_size=config.IMAGE_SIZE,
        return_patches=True,  # 验证时也使用全局+局部patch，保持与训练一致
        is_training=False  # 验证时使用确定性裁剪
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
    import cv2  # 重新导入cv2确保在测试中可用
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
