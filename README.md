# 水印去除系统 (Watermark Removal System)# 水印去除深度学习项目# 水印去除深度学习项目



一个基于深度学习的水印去除系统，使用 U-Net 模型自动去除图像中的水印。



## 📋 目录基于深度学习的水印去除系统。通过在干净图像上合成多样化水印来生成训练数据，使用U-Net架构训练去水印模型。基于深度学习的电路图水印去除系统。通过在无水印图像上合成水印来生成训练数据，使用U-Net架构训练去水印模型。



- [特性](#特性)

- [项目结构](#项目结构)

- [快速开始](#快速开始)## 🚀 快速开始（3个命令）## 🚀 3步快速开始

- [使用指南](#使用指南)

- [配置说明](#配置说明)

- [文档](#文档)

```bash```bash

## ✨ 特性

# 1. 激活虚拟环境# 1. 激活虚拟环境

- **多样化水印生成**：支持 14 种水印风格，自动生成训练数据

- **可见性保证**：强制水印可见性约束，确保训练质量source .venv/bin/activatesource .venv/bin/activate

- **高质量模型**：U-Net 架构，结合 L1、SSIM 和感知损失

- **灵活推理**：支持单图和批量处理，支持高分辨率图像

- **完整工具链**：从数据生成到模型训练到推理的完整流程

# 2. 生成大规模数据集（20张干净图 → 200,000+张训练样本）# 2. 生成训练数据（首次必须，约5-10分钟）

## 📁 项目结构

python generate_large_dataset.py /home/kaga/Desktop/watermaker\ remover/20251127_no_watermark_demo --num 20python generate_dataset.py

```

watermark_removal_project/

├── README.md                 # 项目主文档

├── requirements.txt          # Python 依赖# 3. 开始训练# 3. 开始训练（约2-4小时）

├── config.py                 # 配置文件

├── train.py                  # 训练脚本python train.pypython train.py

├── dataset.py                # 数据加载器

├── losses.py                 # 损失函数``````

│

├── models/                   # 模型定义

│   ├── __init__.py

│   └── unet.py              # U-Net 模型---## 📁 项目结构

│

├── scripts/                  # 工具脚本

│   ├── generate_watermarks_with_masks.py  # 生成水印数据集

│   ├── inference.py          # 标准推理脚本## 📋 完整使用指南```

│   ├── inference_hd.py       # 高分辨率推理

│   └── demo_watermarks.py    # 水印演示watermark_removal_project/

│

├── utils/                    # 工具函数### 1️⃣ 环境准备├── .venv/                      # Python虚拟环境（已创建）

│

├── docs/                     # 详细文档├── config.py                   # 参数配置（修改这里调整训练参数）

│   ├── ARCHITECTURE.md       # 架构说明

│   ├── WATERMARK_STYLES_GUIDE.md  # 水印风格指南```bash├── generate_dataset.py         # 生成合成训练数据

│   └── ...

│# 激活虚拟环境（已预配置）├── dataset.py                  # 数据加载器

├── data/                     # 数据目录

│   └── synthetic/           # 合成训练数据cd "/home/kaga/Desktop/watermaker remover/watermark_removal_project"├── train.py                    # 训练脚本

│

├── checkpoints/             # 模型检查点source .venv/bin/activate├── inference.py                # 推理脚本

├── logs/                    # 训练日志

└── results/                 # 推理结果├── losses.py                   # 损失函数

```

# 如果需要重新安装依赖├── visualize.py                # 可视化工具

## 🚀 快速开始

pip install -r requirements.txt├── test_setup.py               # 环境检查

### 1. 环境安装

```├── requirements.txt            # 依赖列表

```bash

# 创建虚拟环境├── README.md                   # 本文件

python3 -m venv .venv

source .venv/bin/activate  # Linux/Mac### 2️⃣ 生成训练数据集├── START_HERE.md               # 快速入门指南

# .venv\Scripts\activate   # Windows

├── models/

# 安装依赖

pip install -r requirements.txt本项目采用新型数据生成方式：对同一张干净图像生成多种水印变体。│   ├── __init__.py

```

│   └── unet.py                # U-Net实现

**依赖包含**：

- PyTorch >= 1.9.0#### 基础用法└── 运行后生成:

- OpenCV >= 4.5.0

- Albumentations >= 1.0.0    ├── data/synthetic/         # 训练数据

- Pillow >= 8.3.0

- NumPy < 2.0 (兼容性考虑)```bash    ├── checkpoints/            # 模型文件



### 2. 生成训练数据python generate_large_dataset.py /path/to/clean/images    ├── results/                # 推理结果



```bash```    └── logs/                   # TensorBoard日志

# 配置输入输出路径（在 scripts/generate_watermarks_with_masks.py 中）

INPUT_DIR = "/path/to/clean/images"      # 无水印图像目录```

OUTPUT_DIR = "./watermark_demowen"       # 水印图像输出

MASK_DIR = "./watermark_demowen_masks"   # 掩码输出#### 完整命令示例



# 运行生成脚本## 📖 完整使用步骤

python3 scripts/generate_watermarks_with_masks.py

``````bash



**生成参数**：python generate_large_dataset.py /home/kaga/Desktop/watermaker\ remover/20251127_no_watermark_demo \### 1. 激活虚拟环境

- 每张图像生成 32 个水印变体

- 14 种水印风格（文本、Logo、倾斜、平铺等）    --num 20 \

- 自动可见性验证（最小透明度 30%，最小覆盖 2%）

    --output data/synthetic \```bash

### 3. 配置训练

    --train-ratio 0.85cd "/home/kaga/Desktop/watermaker remover/watermark_removal_project"

编辑 `config.py`：

```source .venv/bin/activate

```python

class Config:```

    # 数据路径

    USE_NEW_GENERATED_DATA = True#### 参数说明

    GENERATED_WATERMARK_DIR = "./watermark_demowen"

    CLEAN_IMAGES_PATH = "/path/to/clean/images"虚拟环境已配置好，包含所有必要的包：PyTorch, OpenCV, Albumentations 等。

    

    # 训练参数| 参数 | 类型 | 默认值 | 说明 |

    BATCH_SIZE = 16

    NUM_EPOCHS = 50          # 建议 30-50 epochs|------|------|--------|------|### 2. 生成训练数据（首次必须）

    LEARNING_RATE = 1e-4

    IMAGE_SIZE = (256, 256)| `image_dir` | 必需 | - | 干净图像目录路径 |

    

    # 损失权重| `--num`, `-n` | 可选 | 20 | 每种组合生成的变体数（建议20-50） |```bash

    LOSS_L1_WEIGHT = 1.0

    LOSS_SSIM_WEIGHT = 0.5| `--output`, `-o` | 可选 | data/synthetic | 输出目录 |python generate_dataset.py

    LOSS_PERCEPTUAL_WEIGHT = 0.1

```| `--train-ratio` | 可选 | 0.85 | 训练集比例 |```



### 4. 训练模型



```bash#### 生成规则这会：

python3 train.py

```- 加载500+张无水印电路图



**训练输出**：每张干净图像会生成：- 加载110张透明水印样本

- 检查点保存在 `checkpoints/`

- TensorBoard 日志在 `logs/`- **5种水印类型** × **4种位置** × **N个变体** = **20N张水印图像**- 生成约1500对训练数据对（随机位置、透明度、大小）

- 最佳模型：`checkpoints/unet_watermark_removal_best.pth`

- 自动划分为训练集(85%)和验证集(15%)

**监控训练**：

```bash**水印类型：**

tensorboard --logdir logs/

```1. `transparent_url` - 透明网址（www.example.com）预计时间：5-10分钟



### 5. 推理去除水印2. `transparent_logo_text` - 透明LOGO+文字



**单张图像**：3. `transparent_chinese` - 透明中文（机密、样品、草稿等）### 3. 开始训练

```bash

python3 scripts/inference.py \4. `semi_transparent_url` - 半透明网址

    --checkpoint checkpoints/unet_watermark_removal_best.pth \

    --input path/to/watermarked/image.jpg \5. `semi_transparent_chinese` - 半透明中文```bash

    --output result.jpg

```python train.py



**批量处理**：**水印位置：**```

```bash

python3 scripts/inference.py \1. `corner` - 四个角落之一（随机）

    --checkpoint checkpoints/unet_watermark_removal_best.pth \

    --input input_folder/ \2. `center` - 图像中心配置信息：

    --output output_folder/

```3. `diagonal` - 对角线排列（3个水印）- 默认100个epoch（可在config.py调整）



**高分辨率图像**：4. `full` - 占满全图（平铺）- 批次大小：8（GPU显存不足改为4）

```bash

python3 scripts/inference_hd.py \- 图像大小：256×256（显存不足改为128×128）

    --checkpoint checkpoints/unet_watermark_removal_best.pth \

    --input large_image.jpg \**透明度与大小：** 每个变体的透明度和大小都会随机变化，确保数据多样性。

    --output result.jpg \

    --tile-size 512 \预计时间：2-4小时（GPU）

    --overlap 64

```#### 数据生成示例



## 📖 使用指南**监控训练进度**（新开终端）：



### 数据准备若有20张干净图像，设置`--num 20`：```bash



1. **准备干净图像**：收集无水印的图像作为训练源- 总水印变体数: 20 × 5 × 4 × 20 = **40,000张**tensorboard --logdir logs/

2. **生成水印数据**：使用 `generate_watermarks_with_masks.py` 自动生成

3. **数据分布**：脚本自动划分训练集（85%）和验证集（15%）- 自动划分: 85%训练集（34,000张）+ 15%验证集（6,000张）# 浏览器打开: http://localhost:6006



### 训练技巧- 生成目录结构:```



- **Epoch 数量**：建议 30-50 epochs，观察验证损失  ```

- **批次大小**：根据 GPU 内存调整（16/32 推荐）

- **学习率**：使用默认 1e-4，自动衰减  data/synthetic/### 4. 推理去水印

- **早停**：如果验证损失不再下降，可提前停止

  ├── train_clean/          # 17,000张干净图

### 推理优化

  ├── train_watermarked/    # 34,000张水印图单张图像：

- **标准推理**：适合 256x256 - 1024x1024 图像

- **高分辨率推理**：使用 `inference_hd.py` 处理大图，分块推理避免内存溢出  ├── val_clean/            # 3,000张干净图```bash

- **批量处理**：支持目录级别的批量推理

  └── val_watermarked/      # 6,000张水印图python inference.py \

## ⚙️ 配置说明

  ```    --checkpoint checkpoints/unet_watermark_removal_best.pth \

### 主要配置项

    --input watermarked.jpg \

| 配置项 | 默认值 | 说明 |

|--------|--------|------|### 3️⃣ 训练模型    --output result.jpg

| `NUM_EPOCHS` | 50 | 训练轮数 |

| `BATCH_SIZE` | 16 | 批次大小 |```

| `LEARNING_RATE` | 1e-4 | 初始学习率 |

| `IMAGE_SIZE` | (256, 256) | 训练图像尺寸 |#### 开始训练

| `NUM_WORKERS` | 4 | 数据加载线程数 |

批量处理：

### 损失函数权重

```bash```bash

```python

LOSS_L1_WEIGHT = 1.0           # L1 像素损失python train.pypython inference.py \

LOSS_SSIM_WEIGHT = 0.5         # 结构相似性损失

LOSS_PERCEPTUAL_WEIGHT = 0.1   # VGG 感知损失```    --checkpoint checkpoints/unet_watermark_removal_best.pth \

```

    --input input_folder/ \

### 水印生成配置

训练过程会：    --output output_folder/

在 `scripts/generate_watermarks_with_masks.py` 中：

- 自动加载`data/synthetic`中的数据```

```python

NUM_VARIANTS_PER_IMAGE = 32         # 每图变体数- 使用U-Net架构

MIN_ALPHA = 0.30                    # 最小透明度

MIN_VISIBLE_COVERAGE = 0.02         # 最小可见覆盖率- 每5个epoch保存一次模型## ⚙️ 配置调整

ENABLE_VISIBILITY_VALIDATION = True # 启用可见性验证

```- 将日志保存到`logs/`目录



## 📚 文档编辑 `config.py` 修改参数：



- **[架构说明](docs/ARCHITECTURE.md)**：系统架构和模型设计#### 自定义训练参数

- **[水印风格指南](docs/WATERMARK_STYLES_GUIDE.md)**：14 种水印风格详解

- **[快速参考](docs/QUICK_REFERENCE.md)**：常用命令速查**快速测试**（5分钟）：

- **[可见性数据集](docs/VISIBILITY_DATASET_README.md)**：数据集生成说明

编辑 `config.py` 修改：```python

## 🔧 故障排除

BATCH_SIZE = 4

### 常见问题

```pythonNUM_EPOCHS = 10

**1. NumPy 版本冲突**

```bash# 训练参数IMAGE_SIZE = (128, 128)

# 降级到兼容版本

pip install "numpy<2" opencv-python==4.8.1.78BATCH_SIZE = 8              # GPU显存不足改为4```

```

NUM_EPOCHS = 100            # 训练轮数

**2. 图像尺寸不匹配**

- 问题：Albumentations 报错 "Height and Width should be equal"LEARNING_RATE = 1e-4        # 学习率**标准训练**（2小时）：

- 解决：已在 `dataset.py` 中添加预处理，自动调整尺寸

IMAGE_SIZE = (256, 256)     # 图像大小```python

**3. 损坏的图像文件**

- 问题：某些图像无法读取BATCH_SIZE = 8

- 解决：数据加载器会自动跳过损坏文件并输出提示

# 数据参数NUM_EPOCHS = 50

**4. GPU 内存不足**

- 减小 `BATCH_SIZE`（例如改为 8 或 4）TRAIN_RATIO = 0.85          # 训练集比例```

- 减小 `IMAGE_SIZE`（例如改为 (128, 128)）

NUM_WATERMARKS_PER_IMAGE = 3  # 每张图水印数

### 性能优化

**高质量**（4小时，推荐）：

- **多核加载**：增加 `NUM_WORKERS` 加速数据加载

- **混合精度**：在训练脚本中启用 AMP (Automatic Mixed Precision)# 损失函数权重```python

- **梯度累积**：小批次 + 梯度累积模拟大批次训练

LOSS_L1_WEIGHT = 1.0BATCH_SIZE = 8

## 📊 训练监控

LOSS_PERCEPTUAL_WEIGHT = 0.1NUM_EPOCHS = 100

查看训练进度：

LOSS_SSIM_WEIGHT = 0.5```

```bash

# 启动 TensorBoard```

tensorboard --logdir logs/

**GPU不足**：

# 浏览器访问

http://localhost:6006#### 监控训练进度```python

```

BATCH_SIZE = 4

**关键指标**：

- `Loss/train`：训练损失曲线```bashIMAGE_SIZE = (128, 128)

- `Loss/val`：验证损失曲线（用于早停判断）

- `LR`：学习率变化# 新开一个终端，激活环境后运行```



## 🎯 项目特点tensorboard --logdir logs/



### 可见性约束系统## ⚡ 常用命令



所有生成的水印都经过严格验证：# 浏览器打开: http://localhost:6006

- ✅ 最小透明度 30%（确保可见）

- ✅ 自适应颜色对比（根据背景亮度选择文本颜色）``````bash

- ✅ 最小覆盖面积 2%（避免过小水印）

- ✅ 失败重试机制（最多 3 次尝试）source .venv/bin/activate                    # 激活虚拟环境



### 多样化水印风格### 4️⃣ 推理（去水印）python test_setup.py                         # 环境检查



支持 14 种风格（详见 [docs/WATERMARK_STYLES_GUIDE.md](docs/WATERMARK_STYLES_GUIDE.md)）：python generate_dataset.py                   # 生成数据

1. 角落单文本

2. 居中单文本#### 单张图像python train.py                              # 训练

3. 平铺文本

4. 对角线带状tensorboard --logdir logs/                   # 监控训练

5. 多行居中文本

6. Logo 角落```bashpython inference.py --checkpoint checkpoints/unet_watermark_removal_best.pth --input in.jpg --output out.jpg  # 单张推理

7. 二维码风格

8. 描边文本python inference.py \python inference.py --checkpoint checkpoints/unet_watermark_removal_best.pth --input input_dir/ --output output_dir/  # 批量推理

9. 阴影文本

10. 渐变透明度    --checkpoint checkpoints/unet_watermark_removal_best.pth \deactivate                                   # 退出虚拟环境

11. 噪声侵蚀

12. 横幅框    --input watermarked.jpg \```

13. 曲线文本

14. 组合风格    --output result.jpg



## 📝 许可证```## 🎯 模型架构



本项目仅供学习和研究使用。



## 🤝 贡献#### 批量处理- **U-Net**: 编码-解码 + 跳跃连接



欢迎提交 Issue 和 Pull Request！- **参数量**: 约31M



## 📧 联系```bash- **损失函数**: L1 + SSIM + 感知损失



如有问题或建议，请提交 Issue。python inference.py \



---    --checkpoint checkpoints/unet_watermark_removal_best.pth \## 📊 预期结果



**最后更新**: 2025-11-28    --input input_folder/ \


    --output output_folder/ \训练50-100个epoch后：

    --batch-size 8- PSNR: 28-32 dB

```- SSIM: 0.92-0.96



#### 参数说明## 🔧 故障排除



| 参数 | 说明 |**虚拟环境问题**：

|------|------|```bash

| `--checkpoint` | 训练好的模型路径 |python3 -m venv .venv

| `--input` | 输入图像或文件夹 |source .venv/bin/activate

| `--output` | 输出路径（文件或文件夹） |pip install -r requirements.txt

| `--batch-size` | 批处理大小（默认8） |```



---**GPU显存不足**：

编辑 `config.py`，设置 `BATCH_SIZE = 4`

## 📁 项目结构

**找不到数据**：

```检查 `config.py` 中的路径是否正确

watermark_removal_project/

├── README.md                       # 本文件 - 完整使用指南## ✨ 项目特点

│

├── 🔧 核心脚本✅ 完全自动化  

├── watermark_generator.py          # 水印生成器（支持5种类型，4种位置）✅ 灵活配置  

├── generate_large_dataset.py       # 批量数据集生成（对同一图生成多个变体）✅ GPU加速  

├── train.py                        # 训练脚本✅ 完善文档  

├── inference.py                    # 推理脚本✅ 易于使用  

│

├── ⚙️ 配置文件---

├── config.py                       # 参数配置（修改这里调整训练参数）

└── requirements.txt                # Python依赖列表**立即开始：**

│```bash

├── 📊 模型与数据source .venv/bin/activate

├── models/python generate_dataset.py

│   ├── __init__.pypython train.py

│   └── unet.py                     # U-Net网络定义```

├── data/
│   └── synthetic/                  # 生成的训练数据（自动创建）
├── checkpoints/                    # 保存的模型（自动创建）
├── results/                        # 推理结果（自动创建）
└── logs/                          # TensorBoard日志（自动创建）
│
├── 🛠️ 工具脚本
├── dataset.py                      # 数据加载器
├── losses.py                       # 自定义损失函数
├── visualize_watermarks.py         # 水印可视化工具
└── test_setup.py                   # 环境检查工具
```

---

## 🎯 典型工作流

### 场景1: 从零开始训练（推荐新手）

```bash
# 1. 进入项目目录
cd "/home/kaga/Desktop/watermaker remover/watermark_removal_project"
source .venv/bin/activate

# 2. 生成数据集（仅需一次）
python generate_large_dataset.py /home/kaga/Desktop/watermaker\ remover/20251127_no_watermark_demo --num 20

# 3. 开始训练（2-4小时，取决于GPU）
python train.py

# 4. 监控训练（新开终端）
tensorboard --logdir logs/
```

### 场景2: 继续之前的训练

```bash
# 直接运行train.py，会自动加载最新的模型继续训练
python train.py
```

### 场景3: 生成更多数据（模型欠拟合）

```bash
# 重新生成数据集，增加变体数
python generate_large_dataset.py /path/to/images --num 50

# 清除已有模型并重新训练
rm -rf checkpoints/logs/
python train.py
```

### 场景4: 推理测试

```bash
# 对新图像进行去水印
python inference.py \
    --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input test_watermarked.jpg \
    --output test_result.jpg
```

---

## 🔧 常见问题

### Q: 显存不足怎么办？

**A:** 编辑 `config.py` 降低批次大小和图像尺寸：
```python
BATCH_SIZE = 4              # 从8改为4
IMAGE_SIZE = (128, 128)     # 从256改为128
```

### Q: 如何增加数据多样性？

**A:** 增加 `--num` 参数：
```bash
# 从20变体改为50变体（图像数变5倍）
python generate_large_dataset.py /path/to/images --num 50
```

### Q: 怎样修改水印类型？

**A:** 编辑 `watermark_generator.py` 中的水印生成函数。项目预设了5种类型，可根据需要扩展。

### Q: 如何只对特定水印类型训练？

**A:** 编辑 `generate_large_dataset.py` 中的 `self.watermark_types` 列表，注释掉不需要的类型。

### Q: 训练中断了如何恢复？

**A:** 运行 `python train.py` 会自动加载 `checkpoints/` 中最新的模型继续训练。

---

## 📊 预期效果

| 配置 | 数据量 | 训练时间 | 显存需求 |
|------|--------|--------|----------|
| 20张图 × 20变体 | 40,000对 | 30分钟 | 8GB+ |
| 20张图 × 50变体 | 100,000对 | 2小时 | 8GB+ |
| 100张图 × 20变体 | 200,000对 | 4小时 | 12GB+ |
| 100张图 × 50变体 | 500,000对 | 8小时 | 16GB+ |

---

## 🐛 调试与检查

### 检查环境

```bash
python test_setup.py
```

### 检查数据生成

```bash
# 生成单张图的水印预览
python watermark_generator.py test_image.jpg --output watermark_preview --num 1
```

### 查看水印效果

```bash
python visualize_watermarks.py
```

---

## 📝 许可证与致谢

此项目为教学用途。

---

## 联系方式

遇到问题或有建议？检查文件中的注释或提出Issue。
