# 水印去除深度学习项目 - 完整复现指南

## 📋 项目概述

这是一个基于深度学习的水印去除系统，专门用于去除电路图中的水印。项目采用U-Net架构，通过在干净图像上合成多样化水印来生成训练数据，实现端到端的去水印训练和推理流程。

### 🎯 核心特性
- **多样化水印生成**：支持14种水印风格，包括ElecFans和微信SVG logo
- **可见性保证**：强制水印可见性约束，确保训练质量
- **高质量模型**：U-Net架构，结合L1、SSIM和感知损失
- **灵活推理**：支持单图和批量处理，支持高分辨率图像分块推理
- **完整工具链**：从数据生成到模型训练到推理的完整流程

### 🏗️ 技术架构
- **模型**：U-Net (编码器-解码器架构)
- **损失函数**：L1像素损失 + SSIM结构相似性损失 + VGG感知损失
- **数据增强**：Albumentations库实现多种图像变换
- **推理优化**：分块推理支持高分辨率图像处理

---

## 🔧 环境配置

### 系统要求
- **操作系统**：Linux/macOS/Windows
- **Python版本**：3.7-3.9 (推荐3.8)
- **CUDA版本**：10.2+ (如使用GPU训练)
- **内存**：至少8GB RAM，推荐16GB+
- **存储**：至少50GB可用空间

### 依赖安装

```bash
# 1. 克隆项目
git clone https://github.com/everdaycs/watermark_removeal.git
cd watermark_removeal

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

### 依赖包说明

```txt
# 核心深度学习框架
torch>=1.9.0
torchvision>=0.10.0

# 图像处理
opencv-python==4.8.1.78  # 固定版本兼容NumPy 1.x
Pillow>=8.3.0

# 数据增强
albumentations>=1.0.0

# 数值计算（重要：使用1.x版本兼容OpenCV）
numpy>=1.21.0,<2.0

# 图像质量评估
scikit-image>=0.18.0

# 训练监控
tensorboard>=2.6.0
tqdm>=4.62.0

# 可视化
matplotlib>=3.4.0

# SVG渲染（水印生成必需）
cairosvg>=2.5.0
```

### 环境验证

```bash
# 运行环境检查脚本
python test_setup.py
```

---

## 📊 数据准备

### 数据集结构

项目采用"源图像+合成水印"的数据生成策略：

```
data/
├── clean_images/          # 无水印的干净图像
├── watermarked_images/    # 带水印的图像（运行时生成）
└── masks/                 # 水印掩码（运行时生成）
```

### 水印生成配置

#### 1. 准备干净图像
将无水印的电路图放在指定目录中：
```bash
# 配置路径 (config.py)
CLEAN_IMAGES_PATH = "/path/to/your/clean/circuit/images"
```

#### 2. 水印资源准备
确保以下SVG文件存在：
```
logos/
├── elecfans-logo.svg     # ElecFans logo
├── elecfans-web.svg      # ElecFans web logo
└── WeChat.svg           # 微信logo
```

#### 3. 水印生成参数配置

```python
# generate_svg_with_masks.py 中的关键参数

# 每张图像生成的水印变体数
NUM_VARIANTS_PER_IMAGE = 16

# 透明度范围
MIN_ALPHA = 0.20
MAX_ALPHA = 0.75

# 最小可见面积比例
MIN_VISIBLE_COVERAGE = 0.0005  # 0.05%

# 启用的水印风格
ENABLED_SVG_STYLES = [
    'elecfans_logo_svg_corner',     # ElecFans LOGO角落
    'elecfans_web_svg_center',      # ElecFans Web居中
    'wechat_svg_corner_id',         # 微信角落带ID
    'elecfans_web_svg_corner',      # ElecFans Web角落
    'combined_svg_styles',          # 组合风格
]
```

#### 4. 生成训练数据

```bash
# 生成水印数据集
python generate_svg_with_masks.py
```

此脚本将为每张干净图像生成16个不同的水印变体，包括：
- 不同位置的水印（角落、居中）
- 不同大小和透明度
- 不同的水印类型组合
- 对应的二值掩码

---

## 🏋️ 模型训练

### 训练配置

#### 1. 基本配置 (config.py)

```python
class Config:
    # 数据路径
    CLEAN_IMAGES_PATH = "/path/to/clean/images"
    GENERATED_WATERMARK_DIR = "merged_watermark_images"
    GENERATED_MASK_DIR = "merged_watermark_masks"
    
    # 训练参数
    BATCH_SIZE = 16           # 根据GPU内存调整
    NUM_EPOCHS = 50           # 建议30-100轮
    LEARNING_RATE = 1e-4      # 初始学习率
    IMAGE_SIZE = (256, 256)   # 训练图像尺寸
    
    # 损失函数权重
    LOSS_L1_WEIGHT = 1.0           # L1像素损失
    LOSS_SSIM_WEIGHT = 0.5         # SSIM结构相似性损失
    LOSS_PERCEPTUAL_WEIGHT = 0.1   # VGG感知损失
```

#### 2. 数据集划分

项目自动按85%/15%划分训练/验证集，无需手动分割。

#### 3. 开始训练

```bash
# 启动训练
python train.py
```

### 训练监控

#### TensorBoard可视化
```bash
# 启动TensorBoard
tensorboard --logdir logs/

# 在浏览器中访问 http://localhost:6006
```

#### 训练输出
- **模型检查点**：`checkpoints/unet_watermark_removal_best.pth`
- **训练日志**：`logs/` 目录
- **可视化结果**：TensorBoard日志

### 训练过程详解

#### 数据增强配置 (dataset.py)
```python
# 训练时数据增强
train_transform = A.Compose([
    A.Rotate(limit=15, p=0.5),                    # 随机旋转 ±15度
    A.GaussianBlur(blur_limit=3, p=0.2),          # 高斯模糊
    A.GaussNoise(var_limit=0.01, p=0.3),          # 高斯噪声
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化到[-1,1]
    ToTensorV2(),
])

# 验证时仅归一化
val_transform = A.Compose([
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])
```

#### 损失函数 (losses.py)
```python
class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.0, ssim_weight=0.5, perceptual_weight=0.1):
        self.l1_loss = nn.L1Loss()              # 像素级差异
        self.ssim_loss = SSIMLoss()             # 结构相似性
        self.perceptual_loss = PerceptualLoss() # VGG16特征差异
    
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        
        total_loss = (self.l1_weight * l1 + 
                     self.ssim_weight * (1 - ssim) +  # SSIM转换为损失
                     self.perceptual_weight * perceptual)
        
        return total_loss, {
            'l1': l1.item(),
            'ssim': (1 - ssim).item(),
            'perceptual': perceptual.item(),
            'total': total_loss.item()
        }
```

#### U-Net模型 (models/unet.py)
```python
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        # 编码器路径 (下采样)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # 解码器路径 (上采样)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # 输出层
        self.outc = OutConv(64, n_classes)
```

---

## 🔍 模型推理

### 推理模式

项目提供两种推理脚本：

#### 1. 标准推理 (batch_inference.py)
适用于普通分辨率图像：
```bash
python batch_inference.py --input /path/to/watermarked/images --output /path/to/results
```

#### 2. 分块推理 (batch_inference_tiled.py)
适用于高分辨率图像，保持原始分辨率细节：
```bash
python batch_inference_tiled.py --input /path/to/watermarked/images --output /path/to/results
```

### 分块推理详解

#### 核心算法
```python
class TiledInference:
    def __init__(self, model, device, tile_size=256, overlap=32):
        # tile_size: 分块大小 (256x256)
        # overlap: 重叠像素 (32px，用于平滑拼接)
```

#### 处理流程
1. **图像分块**：将大图像分成256x256的块，相邻块重叠32像素
2. **逐块推理**：每个块单独通过U-Net模型
3. **权重混合**：使用高斯权重掩码平滑重叠区域
4. **结果拼接**：将处理后的块无缝拼接回完整图像

#### 优势
- **保持分辨率**：不损失任何像素细节
- **内存友好**：分块处理避免大图像内存溢出
- **无缝拼接**：重叠区域自然过渡，无拼接痕迹

### 推理参数配置

```bash
# 基本参数
--input INPUT_DIR          # 输入图像目录
--output OUTPUT_DIR        # 输出目录
--checkpoint CHECKPOINT    # 模型检查点路径

# 分块推理特有参数
--tile-size 256           # 分块大小 (默认256)
--overlap 64              # 重叠像素 (默认64，越大越平滑但越慢)
```

---

## 📈 评估与优化

### 评估指标

#### 1. 定量指标
- **PSNR**：峰值信噪比
- **SSIM**：结构相似性指数
- **MSE**：均方误差

#### 2. 视觉评估
- 水印去除完整性
- 图像细节保持度
- 伪影和失真程度

### 超参数调优

#### 训练参数
```python
# 学习率调度
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# 早停机制
early_stopping = EarlyStopping(patience=10)
```

#### 水印生成参数
```python
# 调整水印可见性
MIN_ALPHA = 0.15          # 降低最小透明度
MAX_ALPHA = 0.8           # 提高最大透明度
MIN_VISIBLE_COVERAGE = 0.001  # 增加最小可见面积
```

### 常见问题解决

#### 1. 内存不足
```python
# 减小批次大小
BATCH_SIZE = 4

# 使用梯度累积
accumulation_steps = 4
```

#### 2. 过拟合
```python
# 增加数据增强
transform = A.Compose([
    A.Rotate(limit=15),
    A.GaussianBlur(blur_limit=3),
    A.GaussNoise(var_limit=0.01),
])
```

#### 3. 水印去除不彻底
```python
# 调整损失权重
LOSS_L1_WEIGHT = 1.0      # 增加像素级损失
LOSS_SSIM_WEIGHT = 0.8    # 增加结构相似性损失
```

---

## 🚀 完整复现流程

### 快速复现（3步法）

```bash
# 1. 环境准备
git clone https://github.com/everdaycs/watermark_removeal.git
cd watermark_removeal
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 数据准备
# 将干净电路图放在指定目录，修改config.py中的CLEAN_IMAGES_PATH
python generate_svg_with_masks.py

# 3. 训练和推理
python train.py
python batch_inference_tiled.py --input /path/to/test/images
```

### 实际运行命令示例

#### 1. 完整训练流程
```bash
# 激活环境
cd /path/to/watermark_removal_project
source .venv/bin/activate

# 生成训练数据 (约5-10分钟)
python generate_svg_with_masks.py

# 开始训练 (约2-4小时，视GPU而定)
python train.py

# 监控训练 (新终端)
tensorboard --logdir logs/
# 浏览器访问: http://localhost:6006
```

#### 2. 推理测试
```bash
# 标准推理 (普通分辨率)
python batch_inference.py \
    --input "/path/to/watermarked/images" \
    --output "results/standard" \
    --checkpoint "checkpoints/unet_watermark_removal_best.pth"

# 分块推理 (高分辨率，保持细节)
python batch_inference_tiled.py \
    --input "/path/to/high_res_images" \
    --output "results/tiled" \
    --tile-size 256 \
    --overlap 64
```

#### 3. 批量处理
```bash
# 处理整个目录
python batch_inference_tiled.py \
    --input "data/test_watermarked/" \
    --output "results/batch_output/" \
    --checkpoint "checkpoints/unet_watermark_removal_best.pth"
```

---

## 📁 项目文件结构详解

```
watermark_removal_project/
├── config.py                 # 全局配置参数
├── train.py                  # 训练主脚本
├── dataset.py                # 数据加载器
├── losses.py                 # 损失函数定义
├── batch_inference.py        # 标准推理脚本
├── batch_inference_tiled.py  # 分块推理脚本
├── generate_svg_with_masks.py # 水印数据生成
├── requirements.txt          # Python依赖
├── README.md                 # 项目文档
│
├── models/                   # 模型定义
│   ├── __init__.py
│   └── unet.py              # U-Net实现
│
├── utils/                    # 工具函数
├── docs/                     # 详细文档
├── logos/                    # SVG水印资源
│   ├── elecfans-logo.svg
│   ├── elecfans-web.svg
│   └── WeChat.svg
│
├── data/                     # 数据目录
├── checkpoints/             # 模型检查点
├── logs/                    # 训练日志
├── results/                 # 推理结果
└── merged_watermark_images/ # 生成的水印图像
    merged_watermark_masks/  # 生成的掩码
```

---

## 🔧 自定义配置

### 添加新的水印类型

1. **在generate_svg_with_masks.py中添加新函数**：
```python
def custom_watermark_style(image, mask):
    """自定义水印生成函数"""
    # 实现你的水印生成逻辑
    return image, mask
```

2. **注册到ENABLED_SVG_STYLES列表**：
```python
ENABLED_SVG_STYLES = [
    'custom_watermark_style',  # 添加新风格
    # ... 其他风格
]
```

### 修改模型架构

1. **在models/unet.py中修改网络结构**
2. **调整损失函数权重**
3. **修改训练参数**

### 适配其他数据集

1. **修改config.py中的数据路径**
2. **调整数据加载器以适配新数据集格式**
3. **根据需要修改预处理流程**

---

## 📞 技术支持

### 故障排除指南

#### 常见错误及解决方案

**1. CUDA内存不足**
```bash
# 错误信息: CUDA out of memory
# 解决方案：
# 1. 减小批次大小
BATCH_SIZE = 4  # config.py中修改

# 2. 使用梯度累积
accumulation_steps = 4  # 在train.py中添加

# 3. 清空GPU缓存
torch.cuda.empty_cache()
```

**2. SVG渲染失败**
```bash
# 错误: cairosvg not available
# 解决方案：
pip install cairosvg>=2.5.0

# 或者禁用SVG水印
ENABLED_SVG_STYLES = []  # generate_svg_with_masks.py
```

**3. 图像读取失败**
```bash
# 检查图像格式
file your_image.jpg  # 确认实际格式

# 支持格式: JPG, PNG, BMP, GIF
# 如果是其他格式，转换或添加支持
```

**4. 训练不收敛**
```python
# 检查学习率
LEARNING_RATE = 1e-4  # 可能需要降低到1e-5

# 检查数据质量
# 运行可视化脚本检查生成的水印
python -c "from generate_svg_with_masks import *; test_watermark_generation()"
```

#### 性能优化

**1. GPU加速训练**
```python
# 确保CUDA可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用混合精度训练 (PyTorch 1.6+)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

**2. 数据加载优化**
```python
# 使用多线程加载
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                       num_workers=4, pin_memory=True)
```

**3. 模型推理加速**
```python
# 启用推理模式
model.eval()
with torch.no_grad():
    # 推理代码
```

### 验证复现成功

运行以下命令验证安装和配置：

```bash
# 1. 环境检查
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 2. 数据生成测试
python -c "from generate_svg_with_masks import *; print('SVG水印生成器: OK')"

# 3. 模型加载测试
python -c "from models.unet import UNet; model = UNet(); print(f'模型参数: {sum(p.numel() for p in model.parameters())}')"

# 4. 推理测试
python batch_inference_tiled.py --help
```

---

## 📜 许可证

本项目采用MIT许可证。详见LICENSE文件。

---

*最后更新：2025年12月9日*</content>
<parameter name="filePath">/home/kaga/Desktop/watermaker remover/watermark_removal_project/COMPREHENSIVE_REPRODUCTION_GUIDE.md