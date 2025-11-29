"""
损失函数定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class SSIMLoss(nn.Module):
    """结构相似性损失"""

    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2))) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class PerceptualLoss(nn.Module):
    """VGG16感知损失"""

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.vgg_layers = nn.Sequential(*list(vgg.features)[:16])  # 使用前16层
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg = self.vgg_layers(x)
        y_vgg = self.vgg_layers(y)
        return F.mse_loss(x_vgg, y_vgg)


class CombinedLoss(nn.Module):
    """组合损失函数：L1 + SSIM + 感知损失"""

    def __init__(self, l1_weight=1.0, ssim_weight=0.5, perceptual_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = PerceptualLoss()

    def forward(self, pred, target):
        # L1损失
        l1 = self.l1_loss(pred, target)

        # SSIM损失（转换为损失，SSIM值越接近1越好，所以用1-SSIM）
        ssim_val = self.ssim_loss(pred, target)
        ssim = 1 - ssim_val

        # 感知损失
        perceptual = self.perceptual_loss(pred, target)

        # 组合损失
        total_loss = (self.l1_weight * l1 +
                     self.ssim_weight * ssim +
                     self.perceptual_weight * perceptual)

        # 返回总损失和各组件损失
        loss_dict = {
            'l1': l1.item(),
            'ssim': ssim.item(),
            'perceptual': perceptual.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict
