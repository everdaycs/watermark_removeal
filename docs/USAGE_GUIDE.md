# 🎯 模型使用指南

## 1️⃣ 训练完成后的文件

训练完成后，检查点文件保存在 `checkpoints/` 目录：

```bash
ls -lh checkpoints/
```

**文件说明**:
- `unet_watermark_removal_best.pth` - 最佳模型（验证集损失最低）
- `unet_watermark_removal_epoch_*.pth` - 每N个epoch的检查点

## 2️⃣ 单张图像去水印

### 命令行方式

```bash
python inference.py \
    --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input watermarked_image.jpg \
    --output result.jpg
```

### 参数说明

| 参数 | 说明 | 必需 |
|------|------|------|
| `--checkpoint` | 模型检查点路径 | ✅ |
| `--input` | 输入图像路径 | ✅ |
| `--output` | 输出图像保存路径 | ✅ |
| `--device` | 设备 (cuda/cpu) | ❌ (默认cuda) |

### 完整示例

```bash
# 使用GPU推理
python inference.py \
    --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input test_watermarked.jpg \
    --output test_clean.jpg \
    --device cuda

# 使用CPU推理（如果GPU不可用）
python inference.py \
    --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input test_watermarked.jpg \
    --output test_clean.jpg \
    --device cpu
```

## 3️⃣ 批量处理图像

### 命令行方式

```bash
python inference.py \
    --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input input_folder/ \
    --output output_folder/
```

### 使用场景

```bash
# 处理整个文件夹
python inference.py \
    --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input /path/to/watermarked/images/ \
    --output /path/to/cleaned/images/

# 处理后会自动创建输出目录
# 支持格式: .jpg, .jpeg, .png, .bmp
```

## 4️⃣ Python代码集成

### 基础用法

```python
import cv2
import torch
from models.unet import UNet
from inference import preprocess_image, postprocess_image, load_model

# 1. 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('checkpoints/unet_watermark_removal_best.pth', device)

# 2. 读取图像
image = cv2.imread('watermarked.jpg')

# 3. 预处理
input_tensor, original_size = preprocess_image(image)
input_tensor = input_tensor.to(device)

# 4. 推理
with torch.no_grad():
    output_tensor = model(input_tensor)

# 5. 后处理
output_image = postprocess_image(output_tensor, original_size)

# 6. 保存结果
cv2.imwrite('result.jpg', output_image)
```

### 高级用法 - 批量处理

```python
import os
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from models.unet import UNet

def batch_inference(model_path, input_dir, output_dir, batch_size=4):
    """批量处理图像"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = torch.load(model_path, map_location=device)['model_state_dict']
    net = UNet(3, 3).to(device)
    net.load_state_dict(model)
    net.eval()
    
    # 获取所有图像
    image_files = list(Path(input_dir).glob('**/*.jpg'))
    os.makedirs(output_dir, exist_ok=True)
    
    # 批量处理
    for image_path in tqdm(image_files):
        img = cv2.imread(str(image_path))
        # ... 预处理、推理、后处理 ...
        output_path = os.path.join(output_dir, image_path.name)
        cv2.imwrite(output_path, output_image)

# 调用
batch_inference(
    'checkpoints/unet_watermark_removal_best.pth',
    'watermarked_images/',
    'cleaned_images/'
)
```

## 5️⃣ 性能优化

### 加速推理

```bash
# 使用更小的图像尺寸（更快但质量略低）
# 编辑 inference.py，修改 target_size 参数
target_size = (128, 128)  # 默认(256, 256)
```

### 批量加载模型

```python
# 预加载模型一次，重复使用
model = load_model('checkpoint.pth', device)

for image_path in image_list:
    # 直接使用预加载的模型
    result = remove_watermark(model, image_path, output_path, device)
```

## 6️⃣ 常见问题

### Q: 模型文件在哪里?
```bash
ls -lh /home/kaga/Desktop/watermaker\ remover/watermark_removal_project/checkpoints/
```

### Q: 如何验证模型效果?
```bash
# 1. 生成测试样本
python demo_watermarks.py

# 2. 对测试样本进行推理
python inference.py --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input results/demo/ --output results/cleaned/

# 3. 对比原始图像和去水印结果
```

### Q: 推理速度太慢?
答: 
- 检查是否使用GPU: `nvidia-smi`
- 降低图像分辨率 (修改 preprocess_image 中的 target_size)
- 使用更快的设备或批量处理

### Q: 内存不足?
答:
- 使用CPU而不是GPU (添加 `--device cpu`)
- 降低图像分辨率
- 逐个处理而不是批量处理

## 7️⃣ 推理结果示例

| 类型 | 效果 |
|------|------|
| **输入** | 带有12种水印的图像 |
| **输出** | 干净、无水印的图像 |
| **质量** | 根据训练数据和模型大小 |
| **速度** | GPU: ~50ms/图, CPU: ~500ms/图 |

## 8️⃣ 下一步

### 评估模型性能
```bash
python evaluate.py --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --test-dir data/synthetic/val_watermarked/ \
    --clean-dir data/synthetic/val_clean/
```

### 微调或重新训练
```bash
# 修改 config.py 中的参数
# 例如: LEARNING_RATE = 1e-5 (降低学习率继续训练)
python train.py
```

### 导出为ONNX格式（跨平台部署）
```python
import torch
model = UNet(3, 3)
model.load_state_dict(torch.load('checkpoint.pth')['model_state_dict'])
torch.onnx.export(model, torch.randn(1, 3, 256, 256), 'model.onnx')
```

---

**快速开始命令**:
```bash
# 单张图像
python inference.py --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input watermarked.jpg --output result.jpg

# 整个文件夹
python inference.py --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input watermarked_folder/ --output results/
```
