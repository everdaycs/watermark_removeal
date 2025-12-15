"""
微调训练脚本 - 专门针对特定水印（如ElecFans Logo）进行微调
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from config import Config
from models.optimized_unet import AttentionResUNet
from dataset import create_dataloaders
from losses import CombinedLoss
from train import train_one_epoch, validate, save_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="微调水印去除模型")
    parser.add_argument('--pretrained_path', type=str, default='/home/kaga/Desktop/watermaker remover/watermark_removal_project/checkpoints/unet_watermark_removal_best.pth', help='预训练模型权重的路径 (.pth)')
    parser.add_argument('--data_dir', type=str, default='/home/kaga/Desktop/watermaker remover/watermark_removal_project/data/finetune', help='微调数据集根目录')
    parser.add_argument('--epochs', type=int, default=20, help='微调轮数')
    parser.add_argument('--lr', type=float, default=1e-5, help='微调学习率 (通常比预训练低)')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    return parser.parse_args()

class FinetuneConfig(Config):
    """微调配置，覆盖默认配置"""
    def __init__(self, args):
        # 路径配置
        self.PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        
        # 微调数据路径
        # 假设数据结构: data_dir/watermarked 和 data_dir/masks
        # 注意: dataset.py 需要找到对应的 clean images。
        # generate_finetune_elecfans.py 使用的是 ../20251127_no_watermark_demo 作为源
        # 我们需要确保 dataset loader 能找到原始图片。
        # dataset.py 通常通过文件名匹配。
        
        self.GENERATED_WATERMARK_DIR = os.path.join(args.data_dir, "images")
        self.GENERATED_MASK_DIR = os.path.join(args.data_dir, "masks")
        
        # 原始干净图片路径 (必须与生成脚本一致)
        self.CLEAN_IMAGES_PATH = "/home/kaga/Desktop/watermaker remover/20251201/no_watermark_20251201"
        
        self.TRAIN_WATERMARKED_PATH = self.GENERATED_WATERMARK_DIR
        self.VAL_WATERMARKED_PATH = self.GENERATED_WATERMARK_DIR
        self.TRAIN_CLEAN_PATH = self.CLEAN_IMAGES_PATH
        self.VAL_CLEAN_PATH = self.CLEAN_IMAGES_PATH
        
        # 训练参数
        self.NUM_EPOCHS = args.epochs
        self.LEARNING_RATE = args.lr
        self.BATCH_SIZE = args.batch_size
        
        # 模型保存
        self.CHECKPOINT_PATH = os.path.join(self.PROJECT_ROOT, "checkpoints_finetune")
        self.LOG_PATH = os.path.join(self.PROJECT_ROOT, "logs_finetune")
        self.MODEL_NAME = "finetune_elecfans"
        
        # 确保目录存在
        os.makedirs(self.CHECKPOINT_PATH, exist_ok=True)
        os.makedirs(self.LOG_PATH, exist_ok=True)
        
        # 其他继承自 Config 的参数...
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.SEED = 42
        self.NUM_WORKERS = 4
        self.IMAGE_SIZE = (512, 512)
        self.SAVE_FREQ = 5
        
        # 损失权重 (保持不变或微调)
        self.LOSS_L1_WEIGHT = 1.0
        self.LOSS_SSIM_WEIGHT = 0.5
        self.LOSS_PERCEPTUAL_WEIGHT = 0.1

def main():
    args = parse_args()
    config = FinetuneConfig(args)
    
    print(f"开始微调训练...")
    print(f"预训练模型: {args.pretrained_path}")
    print(f"数据目录: {args.data_dir}")
    print(f"学习率: {config.LEARNING_RATE}")
    
    # 设置随机种子
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    device = torch.device(config.DEVICE)
    
    # 创建数据加载器
    print("加载微调数据...")
    # 注意: create_dataloaders 内部会使用 config 的路径
    # 我们需要确保 config 传递正确，或者 create_dataloaders 能读取这个 config 实例
    # dataset.py 中的 create_dataloaders 接受 config 对象
    train_loader, val_loader = create_dataloaders(config)
    
    if len(train_loader) == 0:
        print("错误: 未找到训练数据，请先运行生成脚本 generate_finetune_elecfans.py")
        return

    # 创建模型
    print("创建模型...")
    model = AttentionResUNet(n_channels=3, n_classes=3)
    model = model.to(device)
    
    # 加载预训练权重
    print(f"加载权重: {args.pretrained_path}")
    checkpoint = torch.load(args.pretrained_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # 处理可能的 key 不匹配 (例如 module. 前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') # remove `module.`
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    print("权重加载成功")
    
    # 损失函数
    criterion = CombinedLoss(
        l1_weight=config.LOSS_L1_WEIGHT,
        ssim_weight=config.LOSS_SSIM_WEIGHT,
        perceptual_weight=config.LOSS_PERCEPTUAL_WEIGHT
    ).to(device)
    
    # 优化器 - 使用较小的学习率
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=config.LOG_PATH)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Finetune Epoch {epoch}/{config.NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # 训练
        train_losses = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # 验证
        val_losses = validate(model, val_loader, criterion, device, epoch)
        
        # 调度
        scheduler.step(val_losses['total'])
        
        # 记录
        writer.add_scalar('Loss/train', train_losses['total'], epoch)
        writer.add_scalar('Loss/val', val_losses['total'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\n训练损失: {train_losses['total']:.4f}")
        print(f"验证损失: {val_losses['total']:.4f}")
        
        # 保存最佳
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_checkpoint(model, optimizer, epoch, val_losses['total'], config, 
                          filename=f"{config.MODEL_NAME}_best.pth")
            print(f"✓ 最佳模型已更新")
            
    print("\n微调完成!")
    writer.close()

if __name__ == "__main__":
    main()
