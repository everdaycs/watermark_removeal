"""
损失函数定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CombinedLoss(nn.Module):
    """组合损失函数：L1 + SSIM + 感知损失"""
    def __init__(self, l1_weight=1.0, ssim_weight=0.5, perceptual_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        
        # 感知损失使用VGG16
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None
    
    def forward(self, pred, target):
        # L1损失
        loss_l1 = self.l1_loss(pred, target)
        
        # SSIM损失
        loss_ssim = self.ssim_loss(pred, target)
        
        # 感知损失
        loss_perceptual = 0
        if self.perceptual_loss is not None:
            loss_perceptual = self.perceptual_loss(pred, target)
        
        # 总损失
        total_loss = (
            self.l1_weight * loss_l1 +
            self.ssim_weight * loss_ssim +
            self.perceptual_weight * loss_perceptual
        )
        
        return total_loss, {
            'l1': loss_l1.item(),
            'ssim': loss_ssim.item(),
            'perceptual': loss_perceptual.item() if isinstance(loss_perceptual, torch.Tensor) else 0,
            'total': total_loss.item()
        }

class SSIMLoss(nn.Module):
    """SSIM损失函数"""
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self._create_window(window_size, self.channel)
    
    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2))) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
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
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        
        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

class PerceptualLoss(nn.Module):
    """感知损失 - 使用VGG16特征"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:16]).eval()
        
        # 冻结VGG参数
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        # 确保输入在[0, 1]范围
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        
        pred_features = self.features(pred)
        target_features = self.features(target)
        
        return F.mse_loss(pred_features, target_features)
