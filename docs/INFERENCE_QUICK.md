# 模型使用速查表

## 单张图像去水印（最常用）

```bash
python inference.py \
    --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input watermarked.jpg \
    --output result.jpg
```

## 批量处理文件夹

```bash
python inference.py \
    --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input watermarked_images/ \
    --output cleaned_images/
```

## Python 代码调用

```python
from inference import load_model, remove_watermark
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('checkpoints/unet_watermark_removal_best.pth', device)
remove_watermark(model, 'input.jpg', 'output.jpg', device)
```

## 查看可用的检查点

```bash
ls -lh checkpoints/
```

## 推理参数

| 参数 | 用法 | 例子 |
|------|------|------|
| `--checkpoint` | 模型路径 | `checkpoints/unet_watermark_removal_best.pth` |
| `--input` | 输入(图像/文件夹) | `image.jpg` 或 `folder/` |
| `--output` | 输出路径 | `result.jpg` 或 `output_folder/` |
| `--device` | cuda\|cpu | `cuda` (默认) |

## 常用命令

```bash
# GPU推理 (快)
python inference.py --checkpoint checkpoints/unet_watermark_removal_best.pth --input test.jpg --output result.jpg --device cuda

# CPU推理 (慢但兼容)
python inference.py --checkpoint checkpoints/unet_watermark_removal_best.pth --input test.jpg --output result.jpg --device cpu

# 处理整个目录
python inference.py --checkpoint checkpoints/unet_watermark_removal_best.pth --input images/ --output results/
```

## 核心文件

- `inference.py` - 推理脚本
- `models/unet.py` - U-Net模型
- `checkpoints/` - 训练好的模型文件

---

更详细的说明见 `USAGE_GUIDE.md`
