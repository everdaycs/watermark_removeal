"""
配置文件 - 水印去除模型训练配置

使用方法：
1. 修改 CLEAN_IMAGES_PATH 为你的干净图像目录
2. 运行 scripts/generate_watermarks_with_masks.py 生成训练数据
3. 调整训练参数（BATCH_SIZE, NUM_EPOCHS 等）
4. 运行 train.py 开始训练
"""
import os

class Config:
    # ========================================================================
    # 路径配置
    # ========================================================================
    
    # 项目根目录（自动获取）
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # 干净图像路径（无水印的原始图像）
    # 修改为你的实际路径
    CLEAN_IMAGES_PATH = "/home/kaga/Desktop/watermaker remover/20251127_no_watermark_demo"
    
    # 旧的水印图像路径（已废弃，保留用于兼容）
    WATERMARK_IMAGES_PATH = "/media/kaga/本地磁盘/20251121_watermark_demo/data_trans/transparent or background wartermark"
    
    # 生成的水印数据路径
    # 由 scripts/generate_watermarks_with_masks.py 生成
    GENERATED_WATERMARK_DIR = os.path.join(PROJECT_ROOT, "merged_watermark_images")
    GENERATED_MASK_DIR = os.path.join(PROJECT_ROOT, "merged_watermark_masks")
    
    # 训练数据配置
    USE_NEW_GENERATED_DATA = True  # 使用新生成的数据集
    
    if USE_NEW_GENERATED_DATA:
        # 水印图像路径
        TRAIN_WATERMARKED_PATH = GENERATED_WATERMARK_DIR
        VAL_WATERMARKED_PATH = GENERATED_WATERMARK_DIR
        
        # 干净图像路径（用于配对）
        TRAIN_SOURCE_PATH = CLEAN_IMAGES_PATH
        VAL_SOURCE_PATH = CLEAN_IMAGES_PATH
        TRAIN_CLEAN_PATH = CLEAN_IMAGES_PATH
        VAL_CLEAN_PATH = CLEAN_IMAGES_PATH
    else:
        # 旧方式：使用预分割的数据（已废弃）
        SYNTHETIC_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic")
        TRAIN_CLEAN_PATH = os.path.join(SYNTHETIC_DATA_PATH, "train_clean")
        TRAIN_WATERMARKED_PATH = os.path.join(SYNTHETIC_DATA_PATH, "train_watermarked")
        VAL_CLEAN_PATH = os.path.join(SYNTHETIC_DATA_PATH, "val_clean")
        VAL_WATERMARKED_PATH = os.path.join(SYNTHETIC_DATA_PATH, "val_watermarked")
    
    # 模型输出路径
    CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints")
    RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")
    LOG_PATH = os.path.join(PROJECT_ROOT, "logs")
    
    # ========================================================================
    # 数据集参数
    # ========================================================================
    
    # 数据集划分比例（自动划分）
    TRAIN_RATIO = 0.85
    VAL_RATIO = 0.15
    
    # 水印合成参数（已废弃，保留用于兼容旧脚本）
    WATERMARK_ALPHA_RANGE = (0.15, 0.6)
    WATERMARK_SCALE_RANGE = (0.1, 0.3)
    NUM_WATERMARKS_PER_IMAGE = 3
    
    # ========================================================================
    # 训练参数
    # ========================================================================
    
    BATCH_SIZE = 16           # 批次大小（GPU内存不足时可减小到4或8）
    NUM_EPOCHS = 10           # 训练轮数（建议 30-50）
    LEARNING_RATE = 1e-4      # 初始学习率
    NUM_WORKERS = 4           # 数据加载线程数
    
    # ========================================================================
    # 图像参数
    # ========================================================================
    
    IMAGE_SIZE = (256, 256)   # 训练图像尺寸
    IMAGE_CHANNELS = 3        # RGB 图像
    
    # ========================================================================
    # 模型参数
    # ========================================================================
    
    MODEL_NAME = "unet_watermark_removal"
    SAVE_FREQ = 5             # 每 N 个 epoch 保存一次模型
    
    # ========================================================================
    # 损失函数权重
    # ========================================================================
    
    LOSS_L1_WEIGHT = 1.0           # L1 像素损失
    LOSS_SSIM_WEIGHT = 0.5         # 结构相似性损失
    LOSS_PERCEPTUAL_WEIGHT = 0.1   # VGG 感知损失
    
    # ========================================================================
    # 运行环境
    # ========================================================================
    
    DEVICE = "cuda"           # 使用 GPU（如无 GPU 改为 "cpu"）
    SEED = 42                 # 随机种子（确保可重现性）
