"""
优化版 U-Net 模型
集成特性：
1. Residual Blocks (残差块) - 替代普通卷积，提取更深层特征
2. Attention Gates (注意力门) - 在跳跃连接中聚焦水印区域
3. PixelShuffle (亚像素卷积) - 更高质量的上采样
4. Instance Normalization - 更适合图像复原任务
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """残差块: (Conv => IN => ReLU => Conv => IN) + Identity"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_channels, affine=True)
        
        # 如果输入输出维度不一致，需要调整Identity的维度
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    """注意力门: 用于跳跃连接，抑制无关背景特征"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: 门控信号 (来自下一层，分辨率较小)
        # x: 跳跃连接特征 (来自编码器，分辨率较大)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UpsampleBlock(nn.Module):
    """上采样块: PixelShuffle + Convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # PixelShuffle 将通道数减少4倍，尺寸扩大2倍
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x

class AttentionResUNet(nn.Module):
    """集成残差、注意力和PixelShuffle的优化版U-Net"""
    def __init__(self, n_channels=3, n_classes=3):
        super().__init__()
        
        # 编码器 (Encoder)
        self.inc = ResidualBlock(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.res1 = ResidualBlock(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.res2 = ResidualBlock(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.res3 = ResidualBlock(256, 512)
        self.down4 = nn.MaxPool2d(2)
        self.res4 = ResidualBlock(512, 1024)
        
        # 注意力门 (Attention Gates)
        # F_g: 门控信号通道数 (来自上采样后的特征)
        # F_l: 跳跃连接特征通道数 (来自编码器)
        # F_int: 中间特征通道数
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        
        # 解码器 (Decoder)
        # 1024 -> 512
        self.up4 = UpsampleBlock(1024, 512)
        self.dec4 = ResidualBlock(1024, 512) # 512(up) + 512(skip)
        
        # 512 -> 256
        self.up3 = UpsampleBlock(512, 256)
        self.dec3 = ResidualBlock(512, 256) # 256(up) + 256(att_skip)
        
        # 256 -> 128
        self.up2 = UpsampleBlock(256, 128)
        self.dec2 = ResidualBlock(256, 128) # 128(up) + 128(att_skip)
        
        # 128 -> 64
        self.up1 = UpsampleBlock(128, 64)
        self.dec1 = ResidualBlock(128, 64) # 64(up) + 64(att_skip)
        
        # 输出层
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def forward(self, x):
        # 编码
        x1 = self.inc(x)          # 64, 256, 256
        
        x2 = self.down1(x1)
        x2 = self.res1(x2)        # 128, 128, 128
        
        x3 = self.down2(x2)
        x3 = self.res2(x3)        # 256, 64, 64
        
        x4 = self.down3(x3)
        x4 = self.res3(x4)        # 512, 32, 32
        
        x5 = self.down4(x4)
        x5 = self.res4(x5)        # 1024, 16, 16
        
        # 解码 + 注意力
        d5 = self.up4(x5)         # 512, 32, 32
        # 简单的拼接 (最深层通常不需要Attention，或者可以用)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.dec4(d5)        # 512, 32, 32
        
        d4 = self.up3(d5)         # 256, 64, 64
        x3_att = self.att3(g=d4, x=x3)
        d4 = torch.cat((x3_att, d4), dim=1)
        d4 = self.dec3(d4)        # 256, 64, 64
        
        d3 = self.up2(d4)         # 128, 128, 128
        x2_att = self.att2(g=d3, x=x2)
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.dec2(d3)        # 128, 128, 128
        
        d2 = self.up1(d3)         # 64, 256, 256
        x1_att = self.att1(g=d2, x=x1)
        d2 = torch.cat((x1_att, d2), dim=1)
        d2 = self.dec1(d2)        # 64, 256, 256
        
        logits = self.outc(d2)
        
        # 残差学习：预测水印，然后从原图中减去
        # 或者直接预测干净图像。这里假设直接预测干净图像。
        # 如果想用全局残差学习，可以 return x + torch.tanh(logits)
        # 但通常在去水印任务中，直接输出 tanh 也是可以的，或者在外部封装 ResidualWrapper
        
        return torch.tanh(logits)

def test_model():
    model = AttentionResUNet()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    test_model()
