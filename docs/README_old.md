# 水印去除深度学习项目# 水印去除深度学习项目



基于深度学习的水印去除系统。通过在干净图像上合成多样化水印来生成训练数据，使用U-Net架构训练去水印模型。基于深度学习的电路图水印去除系统。通过在无水印图像上合成水印来生成训练数据，使用U-Net架构训练去水印模型。



## 🚀 快速开始（3个命令）## 🚀 3步快速开始



```bash```bash

# 1. 激活虚拟环境# 1. 激活虚拟环境

source .venv/bin/activatesource .venv/bin/activate



# 2. 生成大规模数据集（20张干净图 → 200,000+张训练样本）# 2. 生成训练数据（首次必须，约5-10分钟）

python generate_large_dataset.py /home/kaga/Desktop/watermaker\ remover/20251127_no_watermark_demo --num 20python generate_dataset.py



# 3. 开始训练# 3. 开始训练（约2-4小时）

python train.pypython train.py

``````



---## 📁 项目结构



## 📋 完整使用指南```

watermark_removal_project/

### 1️⃣ 环境准备├── .venv/                      # Python虚拟环境（已创建）

├── config.py                   # 参数配置（修改这里调整训练参数）

```bash├── generate_dataset.py         # 生成合成训练数据

# 激活虚拟环境（已预配置）├── dataset.py                  # 数据加载器

cd "/home/kaga/Desktop/watermaker remover/watermark_removal_project"├── train.py                    # 训练脚本

source .venv/bin/activate├── inference.py                # 推理脚本

├── losses.py                   # 损失函数

# 如果需要重新安装依赖├── visualize.py                # 可视化工具

pip install -r requirements.txt├── test_setup.py               # 环境检查

```├── requirements.txt            # 依赖列表

├── README.md                   # 本文件

### 2️⃣ 生成训练数据集├── START_HERE.md               # 快速入门指南

├── models/

本项目采用新型数据生成方式：对同一张干净图像生成多种水印变体。│   ├── __init__.py

│   └── unet.py                # U-Net实现

#### 基础用法└── 运行后生成:

    ├── data/synthetic/         # 训练数据

```bash    ├── checkpoints/            # 模型文件

python generate_large_dataset.py /path/to/clean/images    ├── results/                # 推理结果

```    └── logs/                   # TensorBoard日志

```

#### 完整命令示例

## 📖 完整使用步骤

```bash

python generate_large_dataset.py /home/kaga/Desktop/watermaker\ remover/20251127_no_watermark_demo \### 1. 激活虚拟环境

    --num 20 \

    --output data/synthetic \```bash

    --train-ratio 0.85cd "/home/kaga/Desktop/watermaker remover/watermark_removal_project"

```source .venv/bin/activate

```

#### 参数说明

虚拟环境已配置好，包含所有必要的包：PyTorch, OpenCV, Albumentations 等。

| 参数 | 类型 | 默认值 | 说明 |

|------|------|--------|------|### 2. 生成训练数据（首次必须）

| `image_dir` | 必需 | - | 干净图像目录路径 |

| `--num`, `-n` | 可选 | 20 | 每种组合生成的变体数（建议20-50） |```bash

| `--output`, `-o` | 可选 | data/synthetic | 输出目录 |python generate_dataset.py

| `--train-ratio` | 可选 | 0.85 | 训练集比例 |```



#### 生成规则这会：

- 加载500+张无水印电路图

每张干净图像会生成：- 加载110张透明水印样本

- **5种水印类型** × **4种位置** × **N个变体** = **20N张水印图像**- 生成约1500对训练数据对（随机位置、透明度、大小）

- 自动划分为训练集(85%)和验证集(15%)

**水印类型：**

1. `transparent_url` - 透明网址（www.example.com）预计时间：5-10分钟

2. `transparent_logo_text` - 透明LOGO+文字

3. `transparent_chinese` - 透明中文（机密、样品、草稿等）### 3. 开始训练

4. `semi_transparent_url` - 半透明网址

5. `semi_transparent_chinese` - 半透明中文```bash

python train.py

**水印位置：**```

1. `corner` - 四个角落之一（随机）

2. `center` - 图像中心配置信息：

3. `diagonal` - 对角线排列（3个水印）- 默认100个epoch（可在config.py调整）

4. `full` - 占满全图（平铺）- 批次大小：8（GPU显存不足改为4）

- 图像大小：256×256（显存不足改为128×128）

**透明度与大小：** 每个变体的透明度和大小都会随机变化，确保数据多样性。

预计时间：2-4小时（GPU）

#### 数据生成示例

**监控训练进度**（新开终端）：

若有20张干净图像，设置`--num 20`：```bash

- 总水印变体数: 20 × 5 × 4 × 20 = **40,000张**tensorboard --logdir logs/

- 自动划分: 85%训练集（34,000张）+ 15%验证集（6,000张）# 浏览器打开: http://localhost:6006

- 生成目录结构:```

  ```

  data/synthetic/### 4. 推理去水印

  ├── train_clean/          # 17,000张干净图

  ├── train_watermarked/    # 34,000张水印图单张图像：

  ├── val_clean/            # 3,000张干净图```bash

  └── val_watermarked/      # 6,000张水印图python inference.py \

  ```    --checkpoint checkpoints/unet_watermark_removal_best.pth \

    --input watermarked.jpg \

### 3️⃣ 训练模型    --output result.jpg

```

#### 开始训练

批量处理：

```bash```bash

python train.pypython inference.py \

```    --checkpoint checkpoints/unet_watermark_removal_best.pth \

    --input input_folder/ \

训练过程会：    --output output_folder/

- 自动加载`data/synthetic`中的数据```

- 使用U-Net架构

- 每5个epoch保存一次模型## ⚙️ 配置调整

- 将日志保存到`logs/`目录

编辑 `config.py` 修改参数：

#### 自定义训练参数

**快速测试**（5分钟）：

编辑 `config.py` 修改：```python

BATCH_SIZE = 4

```pythonNUM_EPOCHS = 10

# 训练参数IMAGE_SIZE = (128, 128)

BATCH_SIZE = 8              # GPU显存不足改为4```

NUM_EPOCHS = 100            # 训练轮数

LEARNING_RATE = 1e-4        # 学习率**标准训练**（2小时）：

IMAGE_SIZE = (256, 256)     # 图像大小```python

BATCH_SIZE = 8

# 数据参数NUM_EPOCHS = 50

TRAIN_RATIO = 0.85          # 训练集比例```

NUM_WATERMARKS_PER_IMAGE = 3  # 每张图水印数

**高质量**（4小时，推荐）：

# 损失函数权重```python

LOSS_L1_WEIGHT = 1.0BATCH_SIZE = 8

LOSS_PERCEPTUAL_WEIGHT = 0.1NUM_EPOCHS = 100

LOSS_SSIM_WEIGHT = 0.5```

```

**GPU不足**：

#### 监控训练进度```python

BATCH_SIZE = 4

```bashIMAGE_SIZE = (128, 128)

# 新开一个终端，激活环境后运行```

tensorboard --logdir logs/

## ⚡ 常用命令

# 浏览器打开: http://localhost:6006

``````bash

source .venv/bin/activate                    # 激活虚拟环境

### 4️⃣ 推理（去水印）python test_setup.py                         # 环境检查

python generate_dataset.py                   # 生成数据

#### 单张图像python train.py                              # 训练

tensorboard --logdir logs/                   # 监控训练

```bashpython inference.py --checkpoint checkpoints/unet_watermark_removal_best.pth --input in.jpg --output out.jpg  # 单张推理

python inference.py \python inference.py --checkpoint checkpoints/unet_watermark_removal_best.pth --input input_dir/ --output output_dir/  # 批量推理

    --checkpoint checkpoints/unet_watermark_removal_best.pth \deactivate                                   # 退出虚拟环境

    --input watermarked.jpg \```

    --output result.jpg

```## 🎯 模型架构



#### 批量处理- **U-Net**: 编码-解码 + 跳跃连接

- **参数量**: 约31M

```bash- **损失函数**: L1 + SSIM + 感知损失

python inference.py \

    --checkpoint checkpoints/unet_watermark_removal_best.pth \## 📊 预期结果

    --input input_folder/ \

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
