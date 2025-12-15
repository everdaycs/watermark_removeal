"""
训练脚本
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from config import Config
from models.unet import UNet, ResidualUNet
from models.optimized_unet import AttentionResUNet
from dataset import create_dataloaders
from losses import CombinedLoss


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    running_losses = {'l1': 0, 'ssim': 0, 'perceptual': 0, 'total': 0}
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        watermarked_global = batch['watermarked'].to(device)
        clean_global = batch['clean'].to(device)
        
        # 如果存在局部Patch，将其与全局图像合并训练
        if 'local_watermarked' in batch:
            watermarked_local = batch['local_watermarked'].to(device)
            clean_local = batch['local_clean'].to(device)
            
            # 在batch维度拼接: [B, C, H, W] -> [2B, C, H, W]
            # 这样模型同时学习全局结构(Resize)和局部细节(Crop)
            watermarked = torch.cat([watermarked_global, watermarked_local], dim=0)
            clean = torch.cat([clean_global, clean_local], dim=0)
        else:
            watermarked = watermarked_global
            clean = clean_global
        
        # 前向传播
        optimizer.zero_grad()
        output = model(watermarked)
        
        # 计算损失
        loss, loss_dict = criterion(output, clean)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        for key in loss_dict:
            running_losses[key] += loss_dict[key]
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{running_loss/(batch_idx+1):.4f}"
        })
    
    # 计算平均损失
    num_batches = len(train_loader)
    avg_losses = {key: val / num_batches for key, val in running_losses.items()}
    
    return avg_losses

def validate(model, val_loader, criterion, device, epoch):
    """验证"""
    model.eval()
    running_loss = 0.0
    running_losses = {'l1': 0, 'ssim': 0, 'perceptual': 0, 'total': 0}
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            watermarked_global = batch['watermarked'].to(device)
            clean_global = batch['clean'].to(device)
            
            # 如果存在局部Patch，将其与全局图像合并验证
            if 'local_watermarked' in batch:
                watermarked_local = batch['local_watermarked'].to(device)
                clean_local = batch['local_clean'].to(device)
                
                # 在batch维度拼接: [B, C, H, W] -> [2B, C, H, W]
                # 保持与训练时相同的数据分布
                watermarked = torch.cat([watermarked_global, watermarked_local], dim=0)
                clean = torch.cat([clean_global, clean_local], dim=0)
            else:
                watermarked = watermarked_global
                clean = clean_global
            
            # 前向传播
            output = model(watermarked)
            
            # 计算损失
            loss, loss_dict = criterion(output, clean)
            
            # 统计
            running_loss += loss.item()
            for key in loss_dict:
                running_losses[key] += loss_dict[key]
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{running_loss/(batch_idx+1):.4f}"
            })
    
    # 计算平均损失
    num_batches = len(val_loader)
    avg_losses = {key: val / num_batches for key, val in running_losses.items()}
    
    return avg_losses

def save_checkpoint(model, optimizer, epoch, loss, config, filename=None):
    """保存检查点"""
    if filename is None:
        filename = f"{config.MODEL_NAME}_epoch_{epoch}.pth"
    
    filepath = os.path.join(config.CHECKPOINT_PATH, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    
    print(f"模型已保存: {filepath}")

def train():
    """主训练函数"""
    # 设置GPU内存优化
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 配置
    config = Config()
    
    # 设置随机种子
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # 设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("加载数据...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"训练集批次: {len(train_loader)}, 验证集批次: {len(val_loader)}")
    
    # 创建模型
    print("创建模型...")
    # model = UNet(n_channels=3, n_classes=3, bilinear=False)
    model = AttentionResUNet(n_channels=3, n_classes=3)
    model = model.to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 损失函数
    criterion = CombinedLoss(
        l1_weight=config.LOSS_L1_WEIGHT,
        ssim_weight=config.LOSS_SSIM_WEIGHT,
        perceptual_weight=config.LOSS_PERCEPTUAL_WEIGHT
    ).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=config.LOG_PATH)
    
    # 训练循环
    best_val_loss = float('inf')
    
    print("\n开始训练...")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # 训练
        train_losses = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # 验证
        val_losses = validate(model, val_loader, criterion, device, epoch)
        
        # 学习率调度
        scheduler.step(val_losses['total'])
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_losses['total'], epoch)
        writer.add_scalar('Loss/val', val_losses['total'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 打印结果
        print(f"\n训练损失: {train_losses['total']:.4f}")
        print(f"验证损失: {val_losses['total']:.4f}")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_checkpoint(model, optimizer, epoch, val_losses['total'], config, 
                          filename=f"{config.MODEL_NAME}_best.pth")
            print(f"✓ 最佳模型已更新 (验证损失: {best_val_loss:.4f})")
        
        # 定期保存
        if epoch % config.SAVE_FREQ == 0:
            save_checkpoint(model, optimizer, epoch, val_losses['total'], config)
    
    print("\n训练完成！")
    writer.close()

if __name__ == "__main__":
    train()
