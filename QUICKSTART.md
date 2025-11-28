# 快速参考指南

本文档提供项目的快速命令参考和常见任务指南。

## 📁 项目结构

```
watermark_removal_project/
├── train.py                  # 主训练脚本
├── config.py                 # 配置文件
├── dataset.py                # 数据加载
├── losses.py                 # 损失函数
├── models/                   # 模型定义
├── scripts/                  # 工具脚本
│   ├── generate_watermarks_with_masks.py  # 生成数据
│   ├── inference.py          # 推理脚本
│   └── inference_hd.py       # 高分辨率推理
├── utils/                    # 工具函数
├── docs/                     # 详细文档
├── checkpoints/             # 模型检查点
└── data/                    # 数据目录
```

## 🚀 常用命令

### 1. 生成训练数据

```bash
# 生成水印数据集
python3 scripts/generate_watermarks_with_masks.py

# 配置参数（在脚本中修改）
INPUT_DIR = "/path/to/clean/images"
OUTPUT_DIR = "./watermark_demowen"
NUM_VARIANTS_PER_IMAGE = 32
```

### 2. 训练模型

```bash
# 标准训练
python3 train.py

# 查看训练进度（另开终端）
tensorboard --logdir logs/
```

### 3. 推理/去除水印

```bash
# 单图推理
python3 scripts/inference.py \
    --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input image.jpg \
    --output result.jpg

# 批量推理
python3 scripts/inference.py \
    --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input input_dir/ \
    --output output_dir/


python3 tools/quick_inference.py --mode batch --input "/home/kaga/Desktop/watermaker remover/transparent or background wartermark" --output results/cleaned_images --checkpoint checkpoints/unet_watermark_removal_best.pth

# 高分辨率推理
python3 scripts/inference_hd.py \
    --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input large.jpg \
    --output result.jpg \
    --tile-size 512 \
    --overlap 64
```

## ⚙️ 配置速查

### config.py 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NUM_EPOCHS` | 50 | 训练轮数 |
| `BATCH_SIZE` | 16 | 批次大小 |
| `LEARNING_RATE` | 1e-4 | 学习率 |
| `IMAGE_SIZE` | (256, 256) | 训练图像尺寸 |
| `NUM_WORKERS` | 4 | 数据加载线程 |
| `SAVE_FREQ` | 5 | 保存频率（epoch） |

### 损失权重

```python
LOSS_L1_WEIGHT = 1.0          # 像素损失
LOSS_SSIM_WEIGHT = 0.5        # 结构损失
LOSS_PERCEPTUAL_WEIGHT = 0.1  # 感知损失
```

## 📊 数据路径配置

```python
# 在 config.py 中配置
USE_NEW_GENERATED_DATA = True
CLEAN_IMAGES_PATH = "/path/to/clean/images"
GENERATED_WATERMARK_DIR = "./watermark_demowen"
```

## 🔧 故障排除

### NumPy 版本问题
```bash
pip install "numpy<2" opencv-python==4.8.1.78
```

### GPU 内存不足
- 减小 `BATCH_SIZE` (8 或 4)
- 减小 `IMAGE_SIZE` ((128, 128))

### 图像损坏
数据加载器会自动跳过并提示

## 📈 训练监控

```bash
# 启动 TensorBoard
tensorboard --logdir logs/

# 访问: http://localhost:6006
```

关键指标：
- `Loss/train`: 训练损失
- `Loss/val`: 验证损失
- `LR`: 学习率

## 🎯 最佳实践

1. **数据生成**：每张图生成 32 个变体，确保多样性
2. **训练轮数**：30-50 epochs，观察验证损失
3. **早停**：验证损失不再下降时停止
4. **批次大小**：根据 GPU 内存调整（16-32）
5. **推理**：大图使用 `inference_hd.py` 分块处理

## 📝 文件命名规则

### 生成的水印图像
- 格式：`{base_name}_wm_{index}.jpg`
- 例如：`image_001_wm_005.jpg`

### 掩码文件
- 格式：`{base_name}_wm_{index}_mask.png`
- 例如：`image_001_wm_005_mask.png`

### 检查点
- 最佳模型：`unet_watermark_removal_best.pth`
- 定期保存：`unet_watermark_removal_epoch_{N}.pth`

## 🔄 工作流程

```
1. 准备干净图像
   ↓
2. 运行 generate_watermarks_with_masks.py
   ↓
3. 配置 config.py
   ↓
4. 运行 train.py
   ↓
5. 监控 TensorBoard
   ↓
6. 使用 inference.py 测试
```

## 📚 更多文档

- [完整 README](../README.md)
- [架构说明](docs/ARCHITECTURE.md)
- [水印风格](docs/WATERMARK_STYLES_GUIDE.md)

---

**提示**：遇到问题先查看 README.md 的故障排除部分。
