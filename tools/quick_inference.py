"""
快速推理工具 - 使用训练好的模型处理数据
支持内存处理，不生成额外文件
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import Config
from models.unet import UNet


class WatermarkRemover:
    """水印去除工具类"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        初始化模型
        
        Args:
            checkpoint_path: 模型检查点路径
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(checkpoint_path)
        print(f"✓ 模型已加载到 {self.device}")
    
    def _load_model(self, checkpoint_path):
        """加载模型权重"""
        model = UNet(n_channels=3, n_classes=3, bilinear=False)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _preprocess(self, image, target_size=(256, 256)):
        """
        预处理图像
        
        Args:
            image: OpenCV 格式图像 (BGR)
            target_size: 模型输入尺寸
            
        Returns:
            tensor, 原始尺寸
        """
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # 调整尺寸
        image_resized = cv2.resize(image, target_size)
        
        # BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # 归一化到[-1, 1]
        image_normalized = (image_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
        
        # 转为tensor: (H, W, C) -> (1, C, H, W)
        tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device), original_size
    
    def _postprocess(self, output_tensor, original_size):
        """
        后处理输出
        
        Args:
            output_tensor: 模型输出 (1, C, H, W)
            original_size: 原始图像尺寸 (width, height)
            
        Returns:
            OpenCV 格式图像 (BGR)
        """
        # 移除batch维度: (1, C, H, W) -> (C, H, W)
        output = output_tensor.squeeze(0).cpu().numpy()
        
        # 从[-1, 1]转回[0, 255]
        output = ((output * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        
        # (C, H, W) -> (H, W, C)
        output = output.transpose(1, 2, 0)
        
        # RGB to BGR
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        # 恢复原始尺寸
        output = cv2.resize(output, original_size)
        
        return output
    
    def process(self, image):
        """
        处理单张图像（内存处理，返回numpy数组）
        
        Args:
            image: OpenCV 格式图像 (BGR)
            
        Returns:
            去水印后的图像 (BGR)
        """
        # 预处理
        input_tensor, original_size = self._preprocess(image)
        
        # 推理
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        # 后处理
        output_image = self._postprocess(output_tensor, original_size)
        
        return output_image
    
    def process_file(self, input_path):
        """
        从文件读取并处理
        
        Args:
            input_path: 输入图像路径
            
        Returns:
            去水印后的图像 (numpy array)
        """
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"无法读取图像: {input_path}")
        
        return self.process(image)
    
    def batch_process(self, input_dir):
        """
        批量处理目录中的图像（内存处理）
        
        Args:
            input_dir: 输入目录
            
        Yields:
            (文件名, 处理后的图像)
        """
        input_path = Path(input_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        image_files = [f for f in input_path.rglob('*') 
                      if f.suffix.lower() in image_extensions]
        
        print(f"找到 {len(image_files)} 张图像")
        
        for image_file in image_files:
            try:
                result = self.process_file(image_file)
                yield image_file.name, result
            except Exception as e:
                print(f"处理失败: {image_file} - {e}")


# ============================================================================
# 使用示例
# ============================================================================

def example_single_image():
    """示例：处理单张图像"""
    config = Config()
    
    # 初始化去水印工具
    remover = WatermarkRemover(
        checkpoint_path='checkpoints/unet_watermark_removal_best.pth',
        device=config.DEVICE
    )
    
    # 处理图像（返回numpy数组，不保存文件）
    result = remover.process_file('test_image.jpg')
    
    # 在内存中使用结果
    print(f"结果图像形状: {result.shape}")
    print(f"结果数据类型: {result.dtype}")
    
    # 如果需要保存，手动保存
    # cv2.imwrite('output.jpg', result)
    
    return result


def example_batch_process():
    """示例：批量处理图像"""
    config = Config()
    
    remover = WatermarkRemover(
        checkpoint_path='checkpoints/unet_watermark_removal_best.pth',
        device=config.DEVICE
    )
    
    # 批量处理（内存处理，不自动保存）
    results = {}
    for filename, processed_image in remover.batch_process('input_folder'):
        results[filename] = processed_image
        print(f"✓ 已处理: {filename}")
    
    # 在内存中处理结果，按需保存
    return results


def example_in_memory_processing():
    """示例：纯内存处理（推荐）"""
    config = Config()
    
    remover = WatermarkRemover(
        checkpoint_path='checkpoints/unet_watermark_removal_best.pth',
        device=config.DEVICE
    )
    
    # 读取图像
    image = cv2.imread('input.jpg')
    
    # 处理
    result = remover.process(image)
    
    # 在内存中的进一步处理（不写文件）
    # 例如：
    # - 转换格式
    # - 与其他图像合并
    # - 通过API返回
    # - 实时显示
    
    print(f"处理完成，结果形状: {result.shape}")
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='快速推理工具 - 处理数据不生成文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  # 处理单张图像（返回numpy数组）
  python3 quick_inference.py --mode single --input test.jpg
  
  # 批量处理（返回字典）
  python3 quick_inference.py --mode batch --input input_folder
  
  # 在Python代码中使用
  from quick_inference import WatermarkRemover
  remover = WatermarkRemover('checkpoints/best.pth')
  result = remover.process(cv_image)
        '''
    )
    
    parser.add_argument('--mode', choices=['single', 'batch'], default='single',
                       help='处理模式')
    parser.add_argument('--input', required=True,
                       help='输入图像或文件夹路径')
    parser.add_argument('--output', 
                       help='输出目录（可选，指定后自动保存结果）')
    parser.add_argument('--checkpoint', default='checkpoints/unet_watermark_removal_best.pth',
                       help='模型检查点路径')
    parser.add_argument('--device', default='cuda',
                       help='运行设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 初始化
    remover = WatermarkRemover(args.checkpoint, args.device)
    
    if args.mode == 'single':
        # 单张处理
        result = remover.process_file(args.input)
        print(f"✓ 处理完成")
        print(f"  结果形状: {result.shape}")
        print(f"  数据类型: {result.dtype}")
        print(f"  值范围: [{result.min()}, {result.max()}]")
        
        # 如果指定了输出目录，保存结果
        if args.output:
            import cv2
            import os
            os.makedirs(args.output, exist_ok=True)
            
            # 构造输出文件名
            input_name = os.path.basename(args.input)
            name_without_ext = os.path.splitext(input_name)[0]
            output_path = os.path.join(args.output, f"{name_without_ext}_clean.jpg")
            
            cv2.imwrite(output_path, result)
            print(f"✓ 结果已保存: {output_path}")
    
    else:  # batch
        # 批量处理
        count = 0
        saved_count = 0
        
        for filename, result in remover.batch_process(args.input):
            count += 1
            print(f"✓ {count}. {filename} - 形状:{result.shape}")
            
            # 如果指定了输出目录，保存结果
            if args.output:
                import cv2
                import os
                os.makedirs(args.output, exist_ok=True)
                
                # 构造输出文件名
                name_without_ext = os.path.splitext(filename)[0]
                output_path = os.path.join(args.output, f"{name_without_ext}_clean.jpg")
                
                cv2.imwrite(output_path, result)
                saved_count += 1
        
        if args.output:
            print(f"\n✓ 批量处理完成！共处理 {count} 张图像，保存 {saved_count} 张到 {args.output}")
        else:
            print(f"\n✓ 批量处理完成！共处理 {count} 张图像（内存处理，未保存）")
