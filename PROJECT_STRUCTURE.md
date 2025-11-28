# 项目结构说明

本文档详细说明重构后的项目结构和各个文件的作用。

## 📂 完整目录结构

```
watermark_removal_project/
│
├── README.md                    # 项目主文档（完整指南）
├── QUICKSTART.md                # 快速开始指南
├── PATH_CONFIG.md               # 路径配置说明
├── PROJECT_STRUCTURE.md         # 本文件
├── requirements.txt             # Python 依赖
├── .gitignore                   # Git 忽略配置
│
├── config.py                    # 核心配置文件
├── train.py                     # 训练主脚本
├── dataset.py                   # 数据加载器
├── losses.py                    # 损失函数定义
│
├── models/                      # 模型定义目录
│   ├── __init__.py
│   └── unet.py                 # U-Net 模型实现
│
├── scripts/                     # 工具脚本目录
│   ├── generate_watermarks_with_masks.py  # 生成水印数据集（主要）
│   ├── generate_large_dataset.py          # 大规模数据生成
│   ├── demo_watermarks.py                 # 水印演示
│   ├── watermark_generator.py             # 水印生成器
│   ├── inference.py                       # 标准推理脚本
│   └── inference_hd.py                    # 高分辨率推理
│
├── utils/                       # 工具函数库
│   ├── __init__.py
│   ├── image_utils.py          # 图像处理工具
│   └── validation.py           # 验证工具
│
├── docs/                        # 详细文档目录
│   ├── README_old.md           # 旧版 README（备份）
│   ├── ARCHITECTURE.md         # 系统架构说明
│   ├── WATERMARK_STYLES_GUIDE.md  # 水印风格指南
│   ├── QUICK_REFERENCE.md      # 命令快速参考
│   ├── VISIBILITY_DATASET_README.md  # 可见性数据集说明
│   ├── USAGE_GUIDE.md          # 使用指南
│   ├── USAGE_NEW_DATASET.md    # 新数据集使用
│   ├── CONFIGURATION_EXAMPLES.md  # 配置示例
│   └── INFERENCE_QUICK.md      # 推理快速指南
│
├── data/                        # 数据目录
│   ├── .gitkeep
│   └── synthetic/              # 合成训练数据（可选）
│
├── checkpoints/                 # 模型检查点
│   ├── .gitkeep
│   └── *.pth                   # 训练好的模型文件
│
├── logs/                        # 训练日志
│   ├── .gitkeep
│   └── events.out.tfevents.*   # TensorBoard 日志
│
├── results/                     # 推理结果
│   └── .gitkeep
│
├── watermark_demowen/           # 生成的水印图像（由脚本生成）
└── watermark_demowen_masks/     # 生成的掩码图像（由脚本生成）
```

## 📄 核心文件说明

### 配置和主脚本

| 文件 | 作用 | 主要内容 |
|------|------|---------|
| `config.py` | 全局配置 | 路径、训练参数、模型参数 |
| `train.py` | 训练脚本 | 模型训练主循环 |
| `dataset.py` | 数据加载 | 数据集类和数据增强 |
| `losses.py` | 损失函数 | L1、SSIM、感知损失 |

### models/ 目录

| 文件 | 作用 |
|------|------|
| `unet.py` | U-Net 模型实现，包括编码器、解码器和跳跃连接 |
| `__init__.py` | 模块初始化文件 |

### scripts/ 目录

| 脚本 | 用途 | 何时使用 |
|------|------|---------|
| `generate_watermarks_with_masks.py` | 生成训练数据集 | **最常用**，训练前必须运行 |
| `inference.py` | 标准推理 | 去除图像水印（单图或批量） |
| `inference_hd.py` | 高分辨率推理 | 处理大尺寸图像（分块处理） |
| `demo_watermarks.py` | 水印演示 | 查看水印效果 |
| `generate_large_dataset.py` | 大规模生成 | 生成大量训练数据 |
| `watermark_generator.py` | 水印生成器 | 基础水印生成类 |

### utils/ 目录

| 模块 | 功能 |
|------|------|
| `image_utils.py` | 图像处理工具：目录创建、文件搜索、尺寸获取 |
| `validation.py` | 验证工具：水印可见性验证、图像有效性检查 |

### docs/ 目录

| 文档 | 内容 |
|------|------|
| `ARCHITECTURE.md` | 系统架构、模型设计、数据流 |
| `WATERMARK_STYLES_GUIDE.md` | 14 种水印风格详细说明 |
| `QUICK_REFERENCE.md` | 常用命令快速查询 |
| `VISIBILITY_DATASET_README.md` | 可见性约束数据集说明 |
| `USAGE_GUIDE.md` | 详细使用指南 |

## 🔄 数据流图

```
干净图像 (CLEAN_IMAGES_PATH)
    ↓
[generate_watermarks_with_masks.py]
    ↓
水印图像 (watermark_demowen/) + 掩码 (watermark_demowen_masks/)
    ↓
[dataset.py] → 数据加载和增强
    ↓
[train.py] → 模型训练
    ↓
模型检查点 (checkpoints/*.pth)
    ↓
[inference.py / inference_hd.py]
    ↓
去水印结果 (results/)
```

## 🎯 常用工作流程

### 1. 首次使用

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 修改配置
# 编辑 config.py 中的 CLEAN_IMAGES_PATH

# 3. 生成数据
python3 scripts/generate_watermarks_with_masks.py

# 4. 训练模型
python3 train.py

# 5. 推理测试
python3 scripts/inference.py --checkpoint checkpoints/unet_watermark_removal_best.pth --input test.jpg --output result.jpg
```

### 2. 继续训练

```bash
# 修改 config.py 中的 NUM_EPOCHS
# 加载已有检查点继续训练（需修改 train.py）
python3 train.py
```

### 3. 批量推理

```bash
python3 scripts/inference.py \
    --checkpoint checkpoints/unet_watermark_removal_best.pth \
    --input input_dir/ \
    --output output_dir/
```

## 📝 配置文件层次

```
config.py (主配置)
    ↓
scripts/*.py (脚本级配置)
    ↓
命令行参数 (运行时覆盖)
```

**优先级**：命令行参数 > 脚本配置 > config.py

## 🔧 自定义和扩展

### 添加新的水印风格

编辑 `scripts/generate_watermarks_with_masks.py`：
```python
# 在 ENABLED_STYLES 中添加新风格
ENABLED_STYLES = [
    # ... 现有风格
    'your_new_style',  # 新风格
]

# 实现新风格函数
def your_new_style(draw, img_width, img_height, ...):
    # 你的实现
    pass
```

### 添加新的损失函数

编辑 `losses.py`：
```python
class YourNewLoss(nn.Module):
    def forward(self, pred, target):
        # 你的实现
        pass

# 在 CombinedLoss 中集成
```

### 添加新的工具函数

在 `utils/` 中创建新模块：
```python
# utils/your_module.py
def your_function():
    pass

# utils/__init__.py 中导出
from .your_module import your_function
```

## 📊 文件大小估算

| 类型 | 大小范围 |
|------|---------|
| 干净图像 (500张) | ~100-500 MB |
| 生成的水印数据 (16000张) | ~3-8 GB |
| 模型检查点 | ~120 MB/个 |
| 训练日志 | ~10-50 MB |

**建议**：
- 将大数据集存储在外部驱动器
- 定期清理旧的检查点
- 使用 `.gitignore` 排除数据目录

## 🔍 查找特定功能

| 需求 | 位置 |
|------|------|
| 修改训练参数 | `config.py` |
| 调整水印风格 | `scripts/generate_watermarks_with_masks.py` |
| 修改模型结构 | `models/unet.py` |
| 调整损失权重 | `config.py` (LOSS_*_WEIGHT) |
| 添加数据增强 | `dataset.py` (get_train_transform) |
| 修改推理流程 | `scripts/inference.py` |

## 📚 延伸阅读

- [README.md](README.md) - 项目概览和快速开始
- [QUICKSTART.md](QUICKSTART.md) - 快速参考命令
- [PATH_CONFIG.md](PATH_CONFIG.md) - 路径配置详解
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - 系统架构

## 🤝 贡献指南

如果你想扩展项目功能：

1. **保持结构清晰**：新功能放在对应目录
2. **更新文档**：修改后更新相关文档
3. **添加注释**：为复杂逻辑添加注释
4. **测试功能**：确保新代码不破坏现有功能

---

**最后更新**: 2025-11-28
