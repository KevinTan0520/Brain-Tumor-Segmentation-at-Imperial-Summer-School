import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from skimage import measure, morphology
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# 获取当前文件所在目录并设置为工作目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR)
print(f"Current working directory set to: {CURRENT_DIR}")

# 原版UNet模型定义（与train.py保持一致）
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 下采样路径 (编码器)
        for feature in features:
            self.downs.append(self.double_conv(in_channels, feature))
            in_channels = feature
        
        # 瓶颈层
        self.bottleneck = self.double_conv(features[-1], features[-1]*2)
        
        # 上采样路径 (解码器)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(self.double_conv(feature*2, feature))
        
        # 最终分类层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
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
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], 
                                mode='bilinear', align_corners=False)
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        # 最终输出
        x = self.final_conv(x)
        return self.sigmoid(x)

def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 创建模型实例
    model = UNet(in_channels=1, out_channels=1)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model loaded from checkpoint format")
        if 'best_dice' in checkpoint:
            print(f"   Model Best Dice: {checkpoint['best_dice']:.6f}")
        if 'epoch' in checkpoint:
            print(f"   Trained for {checkpoint['epoch']} epochs")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Model loaded from state dict format")
    
    model.to(device)
    model.eval()
    
    return model

def preprocess_slice(slice_data):
    """改进的预处理函数 - 避免信息丢失"""
    # 更宽松的空切片判断
    if slice_data.max() <= slice_data.min() + 1e-6:
        return np.zeros_like(slice_data, dtype=np.float32)
    
    # 保存原始范围
    original_min, original_max = slice_data.min(), slice_data.max()
    
    # 更温和的归一化方法
    if original_max > original_min:
        # 首先进行0-1归一化
        slice_data = (slice_data - original_min) / (original_max - original_min)
        
        # 然后进行更温和的Z-score归一化
        mean = slice_data.mean()
        std = slice_data.std() + 1e-8
        
        if std > 1e-6:  # 只有在有足够变化时才进行标准化
            slice_data = (slice_data - mean) / std
            # 使用更宽松的截断范围
            slice_data = np.clip(slice_data, -5, 5)
    
    return slice_data.astype(np.float32)

def resize_with_padding(image, target_size=(240, 240)):
    """带填充的尺寸调整，避免信息丢失"""
    from skimage.transform import resize
    
    if image.shape == target_size:
        return image
    
    # 计算缩放比例
    scale_y = target_size[0] / image.shape[0]
    scale_x = target_size[1] / image.shape[1]
    scale = min(scale_y, scale_x)  # 使用较小的缩放比例保持比例
    
    # 计算新尺寸
    new_height = int(image.shape[0] * scale)
    new_width = int(image.shape[1] * scale)
    
    # 调整尺寸
    resized = resize(image, (new_height, new_width), preserve_range=True, anti_aliasing=True)
    
    # 创建目标尺寸的数组并居中放置
    result = np.zeros(target_size, dtype=image.dtype)
    
    # 计算居中位置
    y_offset = (target_size[0] - new_height) // 2
    x_offset = (target_size[1] - new_width) // 2
    
    result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return result

def resize_prediction_back(prediction, original_shape, target_size=(240, 240)):
    """将预测结果调整回原始尺寸"""
    from skimage.transform import resize
    
    if prediction.shape == original_shape:
        return prediction
    
    # 如果是从带填充的图像预测的，需要先提取有效区域
    if prediction.shape == target_size:
        # 计算原始图像在目标尺寸中的位置
        scale_y = target_size[0] / original_shape[0]
        scale_x = target_size[1] / original_shape[1]
        scale = min(scale_y, scale_x)
        
        new_height = int(original_shape[0] * scale)
        new_width = int(original_shape[1] * scale)
        
        y_offset = (target_size[0] - new_height) // 2
        x_offset = (target_size[1] - new_width) // 2
        
        # 提取有效预测区域
        valid_prediction = prediction[y_offset:y_offset+new_height, x_offset:x_offset+new_width]
        
        # 调整回原始尺寸
        return resize(valid_prediction, original_shape, preserve_range=True, anti_aliasing=True)
    else:
        # 直接调整
        return resize(prediction, original_shape, preserve_range=True, anti_aliasing=True)

def process_single_patient(input_path, model, device, output_base_dir):
    """修复版本的患者处理函数"""
    print(f"\n{'='*80}")
    print(f"PROCESSING: {input_path}")
    print(f"{'='*80}")
    
    # 获取患者ID
    patient_id = Path(input_path).parent.name
    print(f"Patient ID: {patient_id}")
    
    # 创建患者输出目录
    patient_output_dir = os.path.join(output_base_dir, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)
    
    # 加载NIfTI文件
    print("Loading NIfTI file...")
    nifti_img = nib.load(input_path)
    nifti_data = nifti_img.get_fdata()
    
    print(f"   Image shape: {nifti_data.shape}")
    print(f"   Value range: [{nifti_data.min():.2f}, {nifti_data.max():.2f}]")
    
    # 3D分割预测
    print("Running 3D segmentation...")
    depth = nifti_data.shape[2]
    predictions = np.zeros_like(nifti_data)
    
    # 记录处理的切片数量
    processed_slices = 0
    skipped_slices = 0
    
    # 逐切片处理
    for slice_idx in tqdm(range(depth), desc="Processing slices"):
        slice_data = nifti_data[:, :, slice_idx].copy()
        
        # 更宽松的空切片判断
        if slice_data.max() <= slice_data.min() + 1e-6:
            skipped_slices += 1
            continue
        
        # 改进的预处理
        processed_slice = preprocess_slice(slice_data)
        
        # 改进的尺寸调整
        if processed_slice.shape != (240, 240):
            processed_slice = resize_with_padding(processed_slice, (240, 240))
        
        # 转换为tensor
        input_tensor = torch.FloatTensor(processed_slice).unsqueeze(0).unsqueeze(0).to(device)
        
        # 模型预测
        with torch.no_grad():
            pred = model(input_tensor)
            pred = pred.squeeze().cpu().numpy()
        
        # 改进的预测结果调整
        if pred.shape != nifti_data.shape[:2]:
            pred = resize_prediction_back(pred, nifti_data.shape[:2], (240, 240))
        
        predictions[:, :, slice_idx] = pred
        processed_slices += 1
    
    print(f"   Processed slices: {processed_slices}/{depth}")
    print(f"   Skipped empty slices: {skipped_slices}")
    
    # 检查每个切片的预测情况
    slice_with_predictions = 0
    for slice_idx in range(depth):
        if np.sum(predictions[:, :, slice_idx] > 0.1) > 0:  # 使用更低的阈值检查
            slice_with_predictions += 1
    
    print(f"   Slices with predictions: {slice_with_predictions}/{depth}")
    
    # 特别检查后部切片（靠近155的切片）
    posterior_slices = max(1, depth // 4)  # 检查后1/4的切片
    posterior_predictions = 0
    for slice_idx in range(depth - posterior_slices, depth):
        if np.sum(predictions[:, :, slice_idx] > 0.1) > 0:
            posterior_predictions += 1
    
    print(f"   Posterior slices with predictions: {posterior_predictions}/{posterior_slices}")
    
    # 改进的后处理
    print("Applying improved post-processing...")
    
    # 使用更低的阈值进行二值化，特别是对于后部切片
    binary_mask = np.zeros_like(predictions, dtype=np.uint8)
    
    for slice_idx in range(depth):
        slice_pred = predictions[:, :, slice_idx]
        
        # 对后部切片使用更低的阈值
        if slice_idx > depth * 0.7:  # 后30%的切片
            threshold = 0.3
        else:
            threshold = 0.5
        
        binary_mask[:, :, slice_idx] = (slice_pred > threshold).astype(np.uint8)
    
    print(f"   Using adaptive thresholds: 0.5 (anterior), 0.3 (posterior)")
    
    # 3D形态学闭运算
    selem = morphology.ball(1)
    binary_mask = morphology.binary_closing(binary_mask, selem)
    
    # 连通组件分析和小组件移除
    labeled_mask, num_components = measure.label(binary_mask, return_num=True)
    
    if num_components > 0:
        component_sizes = []
        component_locations = []
        
        for i in range(1, num_components + 1):
            component_mask = (labeled_mask == i)
            size = np.sum(component_mask)
            
            # 计算组件的位置（检查是否在后部）
            coords = np.where(component_mask)
            avg_z = np.mean(coords[2]) if len(coords[2]) > 0 else 0
            
            component_sizes.append((size, i))
            component_locations.append((avg_z, i))
        
        # 更智能的组件保留策略
        final_mask = np.zeros_like(binary_mask)
        
        for size, label in sorted(component_sizes, reverse=True):
            # 对于后部切片的小组件，使用更宽松的阈值
            avg_z = next(z for z, l in component_locations if l == label)
            
            if avg_z > depth * 0.7:  # 后部切片
                min_size = 20  # 更小的阈值
            else:
                min_size = 50  # 标准阈值
            
            if size >= min_size:
                final_mask[labeled_mask == label] = 1
                print(f"   Kept component {label}: size={size}, z_avg={avg_z:.1f}")
            else:
                print(f"   Removed component {label}: size={size}, z_avg={avg_z:.1f} (too small)")
    else:
        final_mask = binary_mask
    
    # 3D孔洞填充
    final_mask = ndimage.binary_fill_holes(final_mask)
    
    # 最终统计
    tumor_voxels = int(np.sum(final_mask))
    print(f"   Final tumor voxels: {tumor_voxels:,}")
    
    # 检查各个区域的分割情况
    anterior_third = np.sum(final_mask[:, :, :depth//3])
    middle_third = np.sum(final_mask[:, :, depth//3:2*depth//3])
    posterior_third = np.sum(final_mask[:, :, 2*depth//3:])
    
    print(f"   Distribution: Anterior={anterior_third}, Middle={middle_third}, Posterior={posterior_third}")
    
    # 保存NIfTI分割结果
    segmentation_output_path = os.path.join(patient_output_dir, f"{patient_id}_seg.nii.gz")
    
    # 创建NIfTI图像
    mask_img = nib.Nifti1Image(
        final_mask.astype(np.float32),
        nifti_img.affine,
        nifti_img.header
    )
    
    # 保存
    nib.save(mask_img, segmentation_output_path)
    print(f"   Segmentation saved: {segmentation_output_path}")
    
    # 创建增强可视化，特别关注后部切片
    print("Creating enhanced visualization...")
    
    # 找到有肿瘤的切片，包括后部
    tumor_slices = []
    for i in range(depth):
        if np.sum(final_mask[:, :, i]) > 0:
            tumor_slices.append(i)
    
    # 确保包含后部切片
    all_slices_with_signal = []
    for i in range(depth):
        if np.sum(predictions[:, :, i] > 0.1) > 10:  # 有明显信号的切片
            all_slices_with_signal.append(i)
    
    # 组合切片索引
    combined_slices = sorted(list(set(tumor_slices + all_slices_with_signal)))
    
    if len(combined_slices) > 0:
        # 选择代表性切片，确保覆盖整个范围
        if len(combined_slices) > 8:
            selected_slices = np.linspace(0, len(combined_slices)-1, 8, dtype=int)
            slice_indices = [combined_slices[i] for i in selected_slices]
        else:
            slice_indices = combined_slices
        
        # 创建拼接图
        cols = 4
        rows = 2
        fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
        
        for i in range(8):  # 固定8个子图
            row, col = divmod(i, cols)
            
            if i < len(slice_indices):
                slice_idx = slice_indices[i]
                
                # 显示原始图像
                img_slice = nifti_data[:, :, slice_idx]
                axes[row, col].imshow(img_slice, cmap='gray', alpha=0.8)
                
                # 叠加原始预测（半透明）
                pred_slice = predictions[:, :, slice_idx]
                if np.sum(pred_slice > 0.1) > 0:
                    axes[row, col].imshow(pred_slice, cmap='hot', alpha=0.3, vmin=0, vmax=1)
                
                # 叠加最终分割掩码
                mask_slice = final_mask[:, :, slice_idx]
                if np.sum(mask_slice) > 0:
                    axes[row, col].contour(mask_slice, levels=[0.5], colors=['red'], linewidths=2)
                    title = f'Slice {slice_idx} - Final: {np.sum(mask_slice)} voxels'
                else:
                    title = f'Slice {slice_idx} - Max pred: {pred_slice.max():.3f}'
                
                axes[row, col].set_title(title, fontsize=10)
            else:
                # 空白子图
                axes[row, col].set_title('No data', fontsize=10)
            
            axes[row, col].axis('off')
        
        plt.suptitle(f'Enhanced Final Test Results - {patient_id}\n'
                    f'Total slices: {depth}, Tumor slices: {len(tumor_slices)}, '
                    f'Signal slices: {len(all_slices_with_signal)}', fontsize=14)
        plt.tight_layout()
        
        # 保存图像
        montage_path = os.path.join(patient_output_dir, f"{patient_id}_enhanced_slice_montage.png")
        plt.savefig(montage_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Enhanced visualization saved: {montage_path}")
    
    print(f"✅ Patient {patient_id} completed")
    
    return patient_output_dir

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Final Test Brain Tumor Segmentation')
    parser.add_argument('--test_dir', default='final_test',
                       help='Directory containing test data')
    parser.add_argument('--model_path', default='best_brain_tumor_model.pth',
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Processing device: {device}")
    
    # 加载模型
    try:
        model = load_model(args.model_path, device)
        print("Model loaded successfully\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 创建输出目录
    current_time = datetime.now()
    date_str = current_time.strftime('%Y%m%d')
    time_str = current_time.strftime('%H%M%S')
    output_base_dir = f"final_test_result_{date_str}_{time_str}"
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"Output directory: {output_base_dir}")
    
    # 查找测试文件
    if not os.path.exists(args.test_dir):
        print(f"Test directory not found: {args.test_dir}")
        return
    
    # 查找所有 {id}/{id}_fla.nii.gz 文件
    pattern = os.path.join(args.test_dir, "*", "*_fla.nii.gz")
    test_files = glob.glob(pattern)
    
    if not test_files:
        print(f"No test files found in {args.test_dir}")
        print("Expected structure: final_test/{id}/{id}_fla.nii.gz")
        return
    
    print(f"Found {len(test_files)} test files:")
    for i, file_path in enumerate(test_files, 1):
        patient_id = Path(file_path).parent.name
        print(f"  {i}. {patient_id}")
    
    # 处理所有测试文件
    print(f"\nStarting processing...")
    
    successful = 0
    failed = 0
    
    for i, test_file in enumerate(test_files, 1):
        try:
            patient_id = Path(test_file).parent.name
            print(f"\n[{i}/{len(test_files)}] Processing {patient_id}...")
            
            process_single_patient(test_file, model, device, output_base_dir)
            successful += 1
            
        except Exception as e:
            patient_id = Path(test_file).parent.name
            print(f"Failed to process {patient_id}: {e}")
            failed += 1
    
    # 总结
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETED")
    print(f"{'='*80}")
    print(f"Total files: {len(test_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_base_dir}")
    print(f"Results format: {{patient_id}}_seg.nii.gz")

if __name__ == "__main__":
    main()