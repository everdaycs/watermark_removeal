"""
改进的U-Net模型实现 - 增强版水印去除网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(卷积 => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样：MaxPool + DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样：转置卷积 + 连接 + DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出层"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """残差块 - 增强特征提取"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class AttentionGate(nn.Module):
    """注意力门控 - 增强跳跃连接"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ImprovedDoubleConv(nn.Module):
    """改进的双卷积块 - 加入残差和SE注意力"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        # 残差连接
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else None
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.residual is not None:
            residual = self.residual(residual)

        out += residual
        out = self.relu2(out)
        return out

class ImprovedDown(nn.Module):
    """改进的下采样 - 使用残差块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        # 残差块需要downsample来匹配通道数
        self.res_block = ResidualBlock(in_channels, out_channels, stride=1,
                                     downsample=nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else None)

    def forward(self, x):
        x = self.maxpool(x)
        return self.res_block(x)

class ImprovedUp(nn.Module):
    """改进的上采样 - PixelShuffle + 跳跃连接"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ImprovedDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 使用PixelShuffle替代转置卷积
            # PixelShuffle会将通道数变为原来的1/4
            up_channels = in_channels // 4
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.PixelShuffle(2)
            )
            self.conv = ImprovedDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class EnhancedUNet(nn.Module):
    """增强版U-Net - 集成多种改进"""
    def __init__(self, n_channels=3, n_classes=3, bilinear=False):
        super(EnhancedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 编码器 - 使用改进的双卷积
        self.inc = ImprovedDoubleConv(n_channels, 64)
        self.down1 = ImprovedDown(64, 128)
        self.down2 = ImprovedDown(128, 256)
        self.down3 = ImprovedDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = ImprovedDown(512, 1024 // factor)

        # 解码器 - 使用标准上采样避免复杂性
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return torch.tanh(logits)

class LightweightUNet(nn.Module):
    """轻量级U-Net - 使用深度可分离卷积"""
    def __init__(self, n_channels=3, n_classes=3):
        super(LightweightUNet, self).__init__()

        def conv_dw_pw(in_channels, out_channels, stride=1):
            """深度可分离卷积"""
            return nn.Sequential(
                # 深度卷积
                nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
                # 点卷积
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True),
            )

        # 轻量级编码器
        self.enc1 = conv_dw_pw(n_channels, 32, 2)
        self.enc2 = conv_dw_pw(32, 64, 2)
        self.enc3 = conv_dw_pw(64, 128, 2)
        self.enc4 = conv_dw_pw(128, 256, 2)

        # 轻量级解码器
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec1 = conv_dw_pw(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = conv_dw_pw(128, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec3 = conv_dw_pw(64, 32)

        self.up4 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.dec4 = conv_dw_pw(32, 16)

        self.final = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 解码器
        d1 = self.up1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        d4 = self.dec4(d4)

        return torch.tanh(self.final(d4))

def test_enhanced_model():
    """测试增强模型"""
    print("=== 增强版U-Net测试 ===")
    model = EnhancedUNet(n_channels=3, n_classes=3)
    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        output = model(x)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {total_params:,}")

    print("\n=== 轻量级U-Net测试 ===")
    light_model = LightweightUNet(n_channels=3, n_classes=3)
    with torch.no_grad():
        light_output = light_model(x)

    light_params = sum(p.numel() for p in light_model.parameters())
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {light_output.shape}")
    print(f"参数量: {light_params:,}")

if __name__ == "__main__":
    test_enhanced_model()