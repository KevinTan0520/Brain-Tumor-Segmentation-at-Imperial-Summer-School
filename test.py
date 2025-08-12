import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
# import cv2  # 移除cv2依赖
from PIL import Image
import argparse
import os

# 获取当前文件所在目录并设置为工作目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR)
print(f"Current working directory set to: {CURRENT_DIR}")

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 下采样路径
        for feature in features:
            self.downs.append(self.double_conv(in_channels, feature))
            in_channels = feature
        
        # 上采样路径
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(self.double_conv(feature*2, feature))
        
        # 瓶颈层
        self.bottleneck = self.double_conv(features[-1], features[-1]*2)
        
        # 最终分类层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        skip_connections = []
        
        # 下采样
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # 瓶颈层
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # 上采样
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            # 处理尺寸不匹配
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return torch.sigmoid(self.final_conv(x))

def load_model(model_path, device):
    """加载训练好的模型"""
    # 如果model_path不是绝对路径，则基于当前目录构建路径
    if not os.path.isabs(model_path):
        model_path = os.path.join(CURRENT_DIR, model_path)
    
    print(f"Loading model from: {model_path}")
    
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_slice):
    """预处理图像切片"""
    # 归一化
    image_slice = (image_slice - image_slice.mean()) / (image_slice.std() + 1e-8)
    # 转换为tensor并添加batch和channel维度
    image_tensor = torch.FloatTensor(image_slice).unsqueeze(0).unsqueeze(0)
    return image_tensor

def postprocess_mask(mask, threshold=0.5):
    """后处理分割掩码"""
    mask = mask.squeeze().cpu().numpy()
    binary_mask = (mask > threshold).astype(np.uint8)
    return binary_mask

def overlay_mask_on_image(image, mask, color=(1.0, 0.0, 0.0), alpha=0.5):
    """在图像上叠加红色肿瘤掩码 - 使用numpy替代cv2"""
    # 确保图像是RGB格式
    if len(image.shape) == 2:
        # 灰度图转RGB
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image.copy()
    
    # 创建红色掩码
    colored_mask = np.zeros_like(image_rgb)
    colored_mask[mask == 1] = color
    
    # 叠加掩码
    result = image_rgb * (1 - alpha) + colored_mask * alpha
    result = np.clip(result, 0, 1)  # 确保值在[0,1]范围内
    return result

def test_on_nifti_file(model, fla_path, device, output_dir='test_results'):
    """对NIfTI文件进行测试并保存结果"""
    # 如果输入路径不是绝对路径，则基于当前目录构建路径
    if not os.path.isabs(fla_path):
        fla_path = os.path.join(CURRENT_DIR, fla_path)
    
    # 如果输出目录不是绝对路径，则基于当前目录构建路径
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(CURRENT_DIR, output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Loading NIfTI file from: {fla_path}")
    print(f"Output directory: {output_dir}")
    
    # 加载NIfTI文件
    fla_img = nib.load(fla_path)
    fla_data = fla_img.get_fdata()
    
    patient_id = os.path.basename(fla_path).replace('_fla.nii.gz', '')
    
    print(f"Processing patient: {patient_id}")
    print(f"Image shape: {fla_data.shape}")
    
    # 对所有切片进行预测
    num_slices = fla_data.shape[2]
    predictions = []
    
    with torch.no_grad():
        for slice_idx in range(num_slices):
            # 获取切片
            image_slice = fla_data[:, :, slice_idx]
            
            # 跳过空切片
            if image_slice.sum() == 0:
                predictions.append(np.zeros_like(image_slice))
                continue
            
            # 预处理
            image_tensor = preprocess_image(image_slice).to(device)
            
            # 预测
            pred_mask = model(image_tensor)
            binary_mask = postprocess_mask(pred_mask)
            
            predictions.append(binary_mask)
            
            # 保存一些示例切片
            if slice_idx % 10 == 0 and binary_mask.sum() > 0:  # 只保存有肿瘤的切片
                # 归一化图像用于显示
                display_image = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
                
                # 叠加掩码
                overlay_image = overlay_mask_on_image(display_image, binary_mask)
                
                # 保存结果
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(display_image, cmap='gray')
                plt.title(f'Original Slice {slice_idx}')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(binary_mask, cmap='Reds')
                plt.title('Predicted Tumor Mask')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(overlay_image)
                plt.title('Overlay Result')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/{patient_id}_slice_{slice_idx:03d}.png', dpi=150, bbox_inches='tight')
                plt.close()
    
    # 统计信息
    total_tumor_voxels = sum(pred.sum() for pred in predictions)
    print(f"Total predicted tumor voxels: {total_tumor_voxels}")
    
    return predictions

def test_on_single_image(model, image_path, device, output_dir='test_results'):
    """对单张图像进行测试"""
    # 如果输入路径不是绝对路径，则基于当前目录构建路径
    if not os.path.isabs(image_path):
        image_path = os.path.join(CURRENT_DIR, image_path)
    
    # 如果输出目录不是绝对路径，则基于当前目录构建路径
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(CURRENT_DIR, output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Loading image from: {image_path}")
    print(f"Output directory: {output_dir}")
    
    # 加载图像
    if image_path.endswith('.nii.gz'):
        # NIfTI文件
        img = nib.load(image_path)
        image_data = img.get_fdata()
        if len(image_data.shape) == 3:
            image_data = image_data[:, :, image_data.shape[2]//2]  # 取中间切片
    else:
        # 普通图像文件 - 使用PIL替代cv2
        try:
            img = Image.open(image_path).convert('L')  # 转为灰度图
            image_data = np.array(img, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Cannot load image: {image_path}. Error: {e}")
    
    # 预处理
    image_tensor = preprocess_image(image_data).to(device)
    
    # 预测
    with torch.no_grad():
        pred_mask = model(image_tensor)
        binary_mask = postprocess_mask(pred_mask)
    
    # 归一化图像用于显示
    display_image = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    
    # 叠加掩码
    overlay_image = overlay_mask_on_image(display_image, binary_mask)
    
    # 显示和保存结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(display_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(binary_mask, cmap='Reds')
    plt.title('Predicted Tumor Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay_image)
    plt.title('Overlay Result')
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    output_filename = os.path.splitext(os.path.basename(image_path))[0] + '_result.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Result saved to: {output_path}")
    
    return binary_mask, overlay_image

def main():
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation Testing')
    parser.add_argument('--model_path', default='best_brain_tumor_model.pth', 
                       help='Path to trained model')
    parser.add_argument('--input_path', required=True,
                       help='Path to input image or NIfTI file')
    parser.add_argument('--output_dir', default='test_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    try:
        model = load_model(args.model_path, device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 测试
    try:
        if args.input_path.endswith('_fla.nii.gz'):
            # NIfTI文件测试
            predictions = test_on_nifti_file(model, args.input_path, device, args.output_dir)
            print(f"Results saved to {args.output_dir}")
        else:
            # 单张图像测试
            mask, overlay = test_on_single_image(model, args.input_path, device, args.output_dir)
            print(f"Results saved to {args.output_dir}")
            
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    # 如果没有命令行参数，提供默认测试示例
    import sys
    if len(sys.argv) == 1:
        # 示例用法
        print("Usage examples:")
        print("python test.py --input_path path/to/patient_fla.nii.gz")
        print("python test.py --input_path path/to/brain_image.png --model_path best_brain_tumor_model.pth")
        print(f"\nCurrent working directory: {CURRENT_DIR}")
        print("All relative paths will be resolved relative to this directory.")
        
        # 可以在这里添加默认测试
        # test_on_single_image(model, 'example_image.png', device)
    else:
        main()