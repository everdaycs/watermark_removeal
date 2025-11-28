# 使用新生成的水印数据集进行训练

## 数据集位置

生成的数据集保存在以下位置：

```
watermark_removal_project/
├── watermark_demowen/          # 生成的水印图像（约10,000-12,000张）
├── watermark_demowen_masks/    # 对应的水印掩码（二值PNG）
└── generate_watermarks_with_masks.py  # 生成脚本
```

源干净图像位置：
```
/home/kaga/Desktop/watermaker remover/20251127_no_watermark_demo/
```

## 训练脚本修改说明

已对以下文件进行了修改以支持新生成的数据：

### 1. **dataset.py** 修改内容
- 添加了 `source_dir` 参数到 `WatermarkDataset` 类
- 改进了 `_get_image_pairs()` 方法，支持新的文件命名规则：
  - 水印图像：`{base_name}_wm_{index}.jpg`
  - 源干净图像：原始文件名
- 修改了 `__getitem__` 方法以从 `source_dir` 加载干净图像
- 更新了 `create_dataloaders()` 函数以传递 `source_dir` 参数

### 2. **config.py** 修改内容
添加了新配置：
```python
# 新生成的水印数据路径
GENERATED_WATERMARK_DIR = "./watermark_demowen"
GENERATED_MASK_DIR = "./watermark_demowen_masks"

# 使用新生成的数据
USE_NEW_GENERATED_DATA = True

# 自动配置路径指向新数据
TRAIN_WATERMARKED_PATH = GENERATED_WATERMARK_DIR
VAL_WATERMARKED_PATH = GENERATED_WATERMARK_DIR
TRAIN_SOURCE_PATH = CLEAN_IMAGES_PATH  # 源干净图像目录
VAL_SOURCE_PATH = CLEAN_IMAGES_PATH
```

## 如何训练

1. **确保数据已生成**：
   ```bash
   python3 generate_watermarks_with_masks.py
   ```
   这将生成约 10,000-12,000 张水印图像及其掩码。

2. **开始训练**：
   ```bash
   python3 train.py
   ```

3. **验证数据加载**（可选）：
   ```bash
   python3 dataset.py
   ```
   这将打印训练集和验证集的批次数和样本形状。

## 工作原理

```
源干净图像 (20251127_no_watermark_demo)
         ↓
    生成水印 (generate_watermarks_with_masks.py)
         ↓
生成的水印图像 (watermark_demowen/*.jpg)
         ↓
    训练时自动配对：
    - 水印图像：watermark_demowen/{base}_wm_{idx}.jpg
    - 干净图像：20251127_no_watermark_demo/{base}.jpg (自动查找)
         ↓
      训练模型 (train.py)
```

## 关键特性

✓ **自动文件配对**：无需手动复制或组织文件
✓ **支持14种水印风格**：多样化的训练数据
✓ **可见性约束**：所有水印都清晰可见（α >= 0.30）
✓ **自适应颜色**：根据背景自动选择对比色
✓ **训练/验证分割**：自动从所有生成的水印中抽样（比例在dataset中随机）

## 切换回旧数据格式

如果需要使用旧的数据格式（synthetic_data），修改 `config.py`：
```python
USE_NEW_GENERATED_DATA = False
```

然后确保 `data/synthetic/` 目录中有正确的子目录和数据文件。
