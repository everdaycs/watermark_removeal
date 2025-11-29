# 批量水印去除推理脚本使用说明

## 脚本文件
`batch_inference.py` - 一键批量处理水印去除的脚本

## 功能特点
- 🚀 **一键运行**：无需配置参数，直接运行即可
- 📁 **自动路径**：自动识别输入输出目录
- 🧠 **智能加载**：自动加载最佳训练模型
- ⚡ **GPU加速**：支持CUDA GPU加速处理
- 📊 **进度显示**：实时显示处理进度
- 🛡️ **错误处理**：自动跳过损坏的图像文件

## 使用方法

### 1. 激活虚拟环境
```bash
cd "/home/kaga/Desktop/watermaker remover/watermark_removal_project"
source .venv/bin/activate
```

### 2. 运行脚本
```bash
python3 batch_inference.py
```

## 输入输出说明

### 输入目录
```
/media/kaga/本地磁盘/20251121_watermark_demo/data_trans/transparent or background wartermark/
```
- 包含待处理的带水印图像
- 支持格式：`.jpg`, `.jpeg`, `.png`, `.bmp`

### 输出目录
```
watermark_removal_project/results/
```
- 保存去水印后的结果图像
- 文件名与输入保持一致
- 图像质量：JPEG 95%质量，PNG无损压缩

### 模型文件
```
watermark_removal_project/checkpoints/unet_watermark_removal_best.pth
```
- 自动加载最佳训练模型
- 使用U-Net架构，31M参数

## 运行示例

```
============================================================
🚀 批量水印去除推理脚本
   输入: transparent or background wartermark 文件夹
   输出: results 文件夹
============================================================
📁 输入目录: /media/kaga/本地磁盘/20251121_watermark_demo/...
📁 输出目录: /home/kaga/Desktop/watermaker remover/...
🧠 模型路径: /home/kaga/Desktop/watermaker remover/...
⚡ 使用设备: cuda
✓ 模型已加载: checkpoints/unet_watermark_removal_best.pth
✓ 训练轮数: 7
✓ 找到 110 张图像待处理
处理进度: 100%|██████████████████████████████████████████| 110/110
✓ 完成! 成功处理 107/110 张图像
✓ 结果保存在: results/

============================================================
🎉 所有图像处理完成！
   共处理: 107 张图像
   结果位置: results/
============================================================
```

## 注意事项

1. **依赖环境**：确保已激活Python虚拟环境
2. **模型文件**：确保 `checkpoints/unet_watermark_removal_best.pth` 存在
3. **输入路径**：确保输入目录路径正确且可访问
4. **磁盘空间**：确保有足够的磁盘空间保存结果
5. **文件权限**：确保对输入输出目录有读写权限

## 故障排除

### 常见问题

**Q: 提示"输入目录不存在"**
A: 检查输入路径是否正确：`/media/kaga/本地磁盘/20251121_watermark_demo/data_trans/transparent or background wartermark/`

**Q: 提示"模型文件不存在"**
A: 确保已完成模型训练，或检查 `checkpoints/` 目录

**Q: 处理速度慢**
A: 这是正常现象，GPU处理速度约为80-100张/秒

**Q: 某些图像处理失败**
A: 可能是图像文件损坏，脚本会自动跳过并继续处理其他图像

## 技术规格

- **模型架构**：U-Net (31M参数)
- **输入尺寸**：256x256 (自动调整)
- **处理精度**：FP32
- **内存占用**：约2GB GPU内存
- **支持格式**：JPG, PNG, BMP