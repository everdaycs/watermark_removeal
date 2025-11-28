# 🚀 快速参考卡 - V2.1 增强版

## 📋 12 种水印类型

```
1. transparent_url              - 透明网址
2. transparent_logo_text        - 透明LOGO+文字
3. transparent_chinese          - 透明中文
4. semi_transparent_url         - 半透明网址
5. semi_transparent_chinese     - 半透明中文
6. watermark_stamp              - 水印章 (圆形)
7. confidential_stamp           - 机密章 (倾斜)
8. company_logo                 - 公司LOGO (矩形)
9. qr_code_style                - 二维码风格 (网格)
10. gradient_text               - 渐变文字 (递变)
11. diagonal_stripe             - 对角线纹样 (几何)
12. mixed_watermark             - 混合水印 (多元)
```

## 📍 10 种水印位置

```
top_left        top_center        top_right
    
left_center     center            right_center
    
bottom_left     bottom_center     bottom_right

full (占满全图)
```

## ⚡ 常用命令

### 查看演示（120个样本）
```bash
python demo_watermarks.py
```

### 生成数据集
```bash
# 基础用法
python generate_large_dataset.py /path/to/images --num 20

# 完整用法
python generate_large_dataset.py /path/to/images \
    --num 20 \
    --output data/synthetic \
    --train-ratio 0.85
```

### 查看帮助
```bash
python generate_large_dataset.py --help
```

### 训练模型
```bash
python train.py
```

### 推理
```bash
# 单张
python inference.py \
    --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input image.jpg \
    --output result.jpg

# 批量
python inference.py \
    --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input input_folder/ \
    --output output_folder/
```

## 💻 Python API

```python
from watermark_generator import WatermarkGenerator

gen = WatermarkGenerator()

# 自动选择（随机）
result = gen.generate(image)

# 指定类型和位置
result = gen.generate(
    image,
    watermark_type='watermark_stamp',  # 12种选择
    position='top_left',                # 10种选择
    alpha=0.3,                          # 透明度
    scale=0.2                           # 相对大小
)
```

## 📊 数据量速查

| 源图数 | 变体数 | 总样本数 |
|--------|-------|---------|
| 10    | 20    | 240,000  |
| 20    | 20    | 480,000  |
| 50    | 20    | 1,200,000|
| 100   | 20    | 2,400,000|

公式: `源图数 × 12类型 × 10位置 × 变体数 = 总样本数`

## 🔧 配置调整（config.py）

### 显存不足
```python
BATCH_SIZE = 4              # 改为4
IMAGE_SIZE = (128, 128)     # 改为128×128
```

### 快速测试
```python
NUM_EPOCHS = 10             # 改为10
BATCH_SIZE = 4
```

### 高质量训练
```python
NUM_EPOCHS = 100
BATCH_SIZE = 8
IMAGE_SIZE = (256, 256)
```

## ⏱ 时间估计

| 操作 | 数据量 | GPU | 时间 |
|------|---------|-----|------|
| 演示生成 | - | - | <1分钟 |
| 数据生成 | 20张×20变体 | - | 5-10分钟 |
| 数据生成 | 100张×20变体 | - | 30-40分钟 |
| 训练 | 40K样本 | 8GB | 30分钟 |
| 训练 | 240K样本 | 8GB | 4小时 |
| 推理 | 1张 | 8GB | <1秒 |

## 📝 主要特性对比

| 特性 | V2.0 | V2.1 |
|------|------|------|
| 水印类型 | 5 | 12 ⭐ |
| 水印位置 | 4 | 10 ⭐ |
| 组合数 | 20 | 120 ⭐ |
| 单张样本 | 400 | 2,400 ⭐ |
| 自适应 | ✗ | ✓ ⭐ |
| 100张数据 | 40K | 240K ⭐ |

## 🎯 典型工作流

### 完整流程
```bash
# 1. 激活环境
source .venv/bin/activate

# 2. 查看演示（可选，1分钟）
python demo_watermarks.py

# 3. 生成数据集（10分钟-1小时）
python generate_large_dataset.py /path/to/images --num 20

# 4. 训练模型（2-4小时）
python train.py

# 5. 监控（新终端）
tensorboard --logdir logs/

# 6. 推理
python inference.py \
    --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input test.jpg \
    --output result.jpg
```

## 🐛 常见问题

### Q: 怎么只用某些水印类型？
A: 编辑 `generate_large_dataset.py`，在 `__init__` 中修改:
```python
self.watermark_types = [
    'transparent_url',
    'watermark_stamp',
    'mixed_watermark'
]
```

### Q: 怎么调整水印大小？
A: 在 `generate()` 中指定 `scale` 参数:
```python
gen.generate(image, scale=0.1)  # 10%
gen.generate(image, scale=0.5)  # 50%
```

### Q: 怎么固定某个位置？
A: 在 `generate()` 中指定 `position` 参数:
```python
gen.generate(image, position='top_left')
```

### Q: 怎么自动适应图片大小？
A: 默认已启用，无需配置。系统自动根据图片大小调整水印缩放比例。

## 📚 文档位置

- `README.md` - 完整使用说明
- `README_UPDATE.md` - V2.1升级详情
- `PROJECT_SUMMARY.txt` - 项目总结
- `USAGE.txt` - 快速参考
- `QUICK_REFERENCE.md` - 本文件

## 🌟 关键要点

✨ 12种水印类型 = 覆盖更多场景
✨ 10种位置选择 = 测试模型通用性
✨ 自适应处理 = 支持任意分辨率
✨ 6倍数据量 = 更好的模型性能
✨ 完全兼容 = 无需修改现有代码

## 🚀 立即开始

```bash
python demo_watermarks.py
```

享受 V2.1 增强版的强大功能！

