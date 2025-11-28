# 项目路径配置说明

本文档说明如何配置项目中的各种路径，适应不同的数据存储位置。

## 📋 路径配置概览

项目主要涉及以下路径：
1. **干净图像路径**：无水印的原始图像
2. **生成数据路径**：生成的水印图像和掩码
3. **模型保存路径**：检查点和日志
4. **推理输入输出**：待处理图像和结果

## 🔧 配置文件：config.py

### 基础路径配置

```python
class Config:
    # 项目根目录（自动获取）
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # 干净图像源路径（需要修改为你的路径）
    CLEAN_IMAGES_PATH = "/home/kaga/Desktop/watermaker remover/20251127_no_watermark_demo"
    
    # 生成的水印数据路径
    GENERATED_WATERMARK_DIR = os.path.join(PROJECT_ROOT, "watermark_demowen")
    GENERATED_MASK_DIR = os.path.join(PROJECT_ROOT, "watermark_demowen_masks")
    
    # 模型和结果路径
    CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints")
    RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")
    LOG_PATH = os.path.join(PROJECT_ROOT, "logs")
```

### 使用新生成数据

```python
# 启用新生成的数据集
USE_NEW_GENERATED_DATA = True

if USE_NEW_GENERATED_DATA:
    # 训练时使用的路径
    TRAIN_WATERMARKED_PATH = GENERATED_WATERMARK_DIR
    VAL_WATERMARKED_PATH = GENERATED_WATERMARK_DIR
    
    # 干净图像源（用于配对）
    TRAIN_SOURCE_PATH = CLEAN_IMAGES_PATH
    VAL_SOURCE_PATH = CLEAN_IMAGES_PATH
    
    # clean_dir 作为后备
    TRAIN_CLEAN_PATH = CLEAN_IMAGES_PATH
    VAL_CLEAN_PATH = CLEAN_IMAGES_PATH
```

## 🎯 生成脚本路径配置

在 `scripts/generate_watermarks_with_masks.py` 中：

```python
# 输入：干净图像目录
INPUT_DIR = "/home/kaga/Desktop/watermaker remover/20251127_no_watermark_demo"

# 输出：生成的水印图像
OUTPUT_DIR = "./watermark_demowen"

# 输出：生成的掩码
MASK_DIR = "./watermark_demowen_masks"
```

### 配置示例

**场景 1：数据在同一磁盘**
```python
INPUT_DIR = "/home/user/data/clean_images"
OUTPUT_DIR = "./watermark_demowen"
MASK_DIR = "./watermark_demowen_masks"
```

**场景 2：数据在外部磁盘**
```python
INPUT_DIR = "/media/external_drive/clean_images"
OUTPUT_DIR = "/media/external_drive/watermark_data"
MASK_DIR = "/media/external_drive/watermark_masks"
```

**场景 3：使用相对路径**
```python
INPUT_DIR = "../data/clean_images"
OUTPUT_DIR = "../data/generated/watermarks"
MASK_DIR = "../data/generated/masks"
```

## 📂 推荐目录结构

### 选项 1：项目内存储（小数据集）

```
watermark_removal_project/
├── data/
│   ├── clean_images/          # 原始图像
│   ├── watermark_demowen/     # 生成的水印
│   └── watermark_demowen_masks/  # 生成的掩码
├── checkpoints/
├── logs/
└── results/
```

配置：
```python
CLEAN_IMAGES_PATH = os.path.join(PROJECT_ROOT, "data/clean_images")
GENERATED_WATERMARK_DIR = os.path.join(PROJECT_ROOT, "data/watermark_demowen")
```

### 选项 2：外部存储（大数据集）

```
/media/external_drive/
├── clean_images/
├── watermark_data/
└── watermark_masks/

watermark_removal_project/
├── checkpoints/
├── logs/
└── results/
```

配置：
```python
CLEAN_IMAGES_PATH = "/media/external_drive/clean_images"
GENERATED_WATERMARK_DIR = "/media/external_drive/watermark_data"
```

### 选项 3：网络存储（NAS）

```
/mnt/nas/
└── watermark_project_data/
    ├── clean_images/
    ├── generated_watermarks/
    └── generated_masks/
```

配置：
```python
DATA_ROOT = "/mnt/nas/watermark_project_data"
CLEAN_IMAGES_PATH = os.path.join(DATA_ROOT, "clean_images")
GENERATED_WATERMARK_DIR = os.path.join(DATA_ROOT, "generated_watermarks")
```

## 🔄 路径同步

确保 `config.py` 和生成脚本中的路径一致：

### config.py
```python
CLEAN_IMAGES_PATH = "/path/to/clean/images"
GENERATED_WATERMARK_DIR = "./watermark_demowen"
```

### scripts/generate_watermarks_with_masks.py
```python
INPUT_DIR = "/path/to/clean/images"  # 与 CLEAN_IMAGES_PATH 一致
OUTPUT_DIR = "./watermark_demowen"   # 与 GENERATED_WATERMARK_DIR 一致
```

## ⚠️ 常见问题

### 1. 路径不存在错误

**问题**：`FileNotFoundError: [Errno 2] No such file or directory`

**解决**：
- 检查路径是否正确
- 确保目录存在或脚本会自动创建
- Linux 路径区分大小写

### 2. 权限错误

**问题**：`PermissionError: [Errno 13] Permission denied`

**解决**：
```bash
# 修改目录权限
chmod -R 755 /path/to/directory

# 或使用 sudo（不推荐）
sudo python3 script.py
```

### 3. 跨平台路径

**Windows**：
```python
CLEAN_IMAGES_PATH = "D:\\data\\clean_images"
# 或使用原始字符串
CLEAN_IMAGES_PATH = r"D:\data\clean_images"
```

**Linux/Mac**：
```python
CLEAN_IMAGES_PATH = "/home/user/data/clean_images"
```

**跨平台兼容**：
```python
import os
CLEAN_IMAGES_PATH = os.path.join(os.path.expanduser("~"), "data", "clean_images")
```

### 4. 相对路径 vs 绝对路径

**相对路径**（相对于脚本位置）：
```python
INPUT_DIR = "./data/clean_images"
OUTPUT_DIR = "../output/watermarks"
```

**绝对路径**（推荐，避免混淆）：
```python
INPUT_DIR = "/absolute/path/to/clean_images"
OUTPUT_DIR = "/absolute/path/to/watermarks"
```

## 📝 配置检查清单

配置完成后，检查以下内容：

- [ ] `CLEAN_IMAGES_PATH` 指向干净图像目录
- [ ] `GENERATED_WATERMARK_DIR` 存在或可创建
- [ ] `GENERATED_MASK_DIR` 存在或可创建
- [ ] `INPUT_DIR` (生成脚本) 与 `CLEAN_IMAGES_PATH` 一致
- [ ] 所有路径有读写权限
- [ ] 磁盘空间充足（生成数据需要大量空间）

## 🧪 验证配置

运行以下命令验证路径配置：

```bash
# 检查配置
python3 -c "
from config import Config
c = Config()
import os

print('CLEAN_IMAGES_PATH:', c.CLEAN_IMAGES_PATH)
print('Exists:', os.path.exists(c.CLEAN_IMAGES_PATH))

print('\nGENERATED_WATERMARK_DIR:', c.GENERATED_WATERMARK_DIR)
print('Exists:', os.path.exists(c.GENERATED_WATERMARK_DIR))
"
```

## 💡 最佳实践

1. **使用绝对路径**：避免路径混淆
2. **统一配置**：所有路径在 `config.py` 中集中管理
3. **分离数据和代码**：大数据集存储在外部驱动器
4. **版本控制**：`.gitignore` 中排除数据目录
5. **文档化**：记录你的路径配置决策

## 📚 相关文档

- [README.md](../README.md) - 项目概览
- [QUICKSTART.md](../QUICKSTART.md) - 快速开始

---

**提示**：修改路径配置后，重新运行数据生成和训练脚本。
