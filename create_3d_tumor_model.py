import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse
import os
from pathlib import Path
from skimage import measure
from skimage.morphology import binary_closing, ball
from scipy import ndimage
import glob

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
    if not os.path.isabs(model_path):
        model_path = os.path.join(CURRENT_DIR, model_path)
    
    print(f"Loading model from: {model_path}")
    
    model = UNet().to(device)
    
    # 直接使用 weights_only=False（如果你信任模型文件）
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model from checkpoint")
        
        # 如果需要，可以打印额外信息
        if 'best_dice' in checkpoint:
            print(f"Model's best validation Dice: {checkpoint['best_dice']:.4f}")
        if 'epoch' in checkpoint:
            print(f"Model was trained for {checkpoint['epoch']} epochs")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights directly")
    
    model.eval()
    return model

def preprocess_image(image_slice):
    """预处理图像切片"""
    # 归一化
    if image_slice.std() > 1e-8:
        image_slice = (image_slice - image_slice.mean()) / image_slice.std()
    else:
        image_slice = image_slice - image_slice.mean()
    
    # 转换为tensor并添加batch和channel维度
    image_tensor = torch.FloatTensor(image_slice).unsqueeze(0).unsqueeze(0)
    return image_tensor

def postprocess_mask(mask, threshold=0.5):
    """后处理分割掩码"""
    mask = mask.squeeze().cpu().numpy()
    binary_mask = (mask > threshold).astype(np.uint8)
    return binary_mask

def predict_3d_segmentation(model, nifti_data, device, batch_size=8):
    """对3D NIfTI数据进行分割预测"""
    height, width, depth = nifti_data.shape
    predictions = np.zeros((height, width, depth), dtype=np.uint8)
    
    print(f"Processing {depth} slices...")
    
    with torch.no_grad():
        for i in range(0, depth, batch_size):
            end_idx = min(i + batch_size, depth)
            batch_slices = []
            valid_slices = []
            
            # 准备批次数据
            for slice_idx in range(i, end_idx):
                image_slice = nifti_data[:, :, slice_idx]
                
                # 跳过空切片
                if image_slice.sum() == 0:
                    predictions[:, :, slice_idx] = 0
                    continue
                
                # 预处理
                image_tensor = preprocess_image(image_slice)
                batch_slices.append(image_tensor)
                valid_slices.append(slice_idx)
            
            if not batch_slices:
                continue
            
            # 批量预测
            batch_tensor = torch.cat(batch_slices, dim=0).to(device)
            batch_predictions = model(batch_tensor)
            
            # 处理预测结果
            for j, slice_idx in enumerate(valid_slices):
                binary_mask = postprocess_mask(batch_predictions[j])
                predictions[:, :, slice_idx] = binary_mask
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {min(i + batch_size, depth)}/{depth} slices")
    
    return predictions

def post_process_3d_mask(mask, min_size=100, closing_radius=2):
    """3D后处理：去除小连通域并进行形态学闭运算"""
    print("Applying 3D post-processing...")
    
    # 形态学闭运算，填补小空洞
    if closing_radius > 0:
        structuring_element = ball(closing_radius)
        mask = binary_closing(mask, structuring_element)
    
    # 连通域分析，去除小的噪声区域
    labeled_mask = measure.label(mask, connectivity=3)
    regions = measure.regionprops(labeled_mask)
    
    # 保留足够大的连通域
    filtered_mask = np.zeros_like(mask)
    kept_regions = 0
    
    for region in regions:
        if region.area >= min_size:
            filtered_mask[labeled_mask == region.label] = 1
            kept_regions += 1
    
    print(f"Kept {kept_regions} regions with size >= {min_size} voxels")
    
    return filtered_mask.astype(np.uint8)

def create_3d_mesh(mask, step_size=2):
    """使用Marching Cubes算法创建3D网格"""
    print("Creating 3D mesh using Marching Cubes...")
    
    try:
        # 使用Marching Cubes算法生成网格
        vertices, faces, normals, values = measure.marching_cubes(
            mask, level=0.5, step_size=step_size, allow_degenerate=False
        )
        
        print(f"Generated mesh with {len(vertices)} vertices and {len(faces)} faces")
        return vertices, faces, normals
    
    except Exception as e:
        print(f"Error creating 3D mesh: {e}")
        return None, None, None

def create_interactive_3d_plot(vertices, faces, mask, patient_id, output_dir):
    """创建交互式3D可视化"""
    print("Creating interactive 3D visualization...")
    
    if vertices is None or faces is None:
        print("No valid mesh data for visualization")
        return
    
    # 创建3D mesh图
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1], 
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='red',
            opacity=0.8,
            name='Tumor',
            lighting=dict(ambient=0.3, diffuse=0.8, specular=0.8),
            lightposition=dict(x=100, y=100, z=100)
        )
    ])
    
    # 设置布局
    fig.update_layout(
        title=f'3D Brain Tumor Segmentation - Patient {patient_id}',
        scene=dict(
            xaxis_title='X (voxels)',
            yaxis_title='Y (voxels)',
            zaxis_title='Z (voxels)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='data'
        ),
        font=dict(size=12),
        width=800,
        height=600
    )
    
    # 保存交互式HTML文件
    html_path = os.path.join(output_dir, f'{patient_id}_3d_tumor_interactive.html')
    fig.write_html(html_path)
    print(f"Interactive 3D model saved to: {html_path}")
    
    # 保存静态图像
    static_path = os.path.join(output_dir, f'{patient_id}_3d_tumor_static.png')
    fig.write_image(static_path, width=800, height=600, scale=2)
    print(f"Static 3D image saved to: {static_path}")

def create_matplotlib_3d_plot(mask, patient_id, output_dir):
    """使用matplotlib创建基本3D可视化"""
    print("Creating matplotlib 3D visualization...")
    
    # 找到所有肿瘤体素的坐标
    tumor_coords = np.where(mask == 1)
    
    if len(tumor_coords[0]) == 0:
        print("No tumor voxels found for visualization")
        return
    
    # 为了性能，对大量点进行下采样
    max_points = 5000
    if len(tumor_coords[0]) > max_points:
        indices = np.random.choice(len(tumor_coords[0]), max_points, replace=False)
        x = tumor_coords[0][indices]
        y = tumor_coords[1][indices]
        z = tumor_coords[2][indices]
    else:
        x, y, z = tumor_coords
    
    # 创建3D散点图
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制肿瘤点
    scatter = ax.scatter(x, y, z, c=z, cmap='Reds', alpha=0.6, s=1)
    
    # 设置标签和标题
    ax.set_xlabel('X (voxels)')
    ax.set_ylabel('Y (voxels)')
    ax.set_zlabel('Z (voxels)')
    ax.set_title(f'3D Brain Tumor Visualization - Patient {patient_id}')
    
    # 添加颜色条
    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, label='Z coordinate')
    
    # 保存图像
    plt_path = os.path.join(output_dir, f'{patient_id}_3d_tumor_matplotlib.png')
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Matplotlib 3D visualization saved to: {plt_path}")

def create_slice_montage(original_data, mask_data, patient_id, output_dir, num_slices=16):
    """创建切片拼接图"""
    print(f"Creating slice montage with {num_slices} slices...")
    
    depth = original_data.shape[2]
    slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for i, slice_idx in enumerate(slice_indices):
        if i >= len(axes):
            break
            
        # 获取原始切片和掩码
        original_slice = original_data[:, :, slice_idx]
        mask_slice = mask_data[:, :, slice_idx]
        
        # 归一化原始图像
        if original_slice.max() > original_slice.min():
            original_slice = (original_slice - original_slice.min()) / (original_slice.max() - original_slice.min())
        
        # 创建叠加图像
        overlay = np.stack([original_slice, original_slice, original_slice], axis=-1)
        overlay[:, :, 0] = np.where(mask_slice == 1, 1.0, overlay[:, :, 0])  # 红色肿瘤
        
        axes[i].imshow(overlay)
        axes[i].set_title(f'Slice {slice_idx}')
        axes[i].axis('off')
    
    # 隐藏多余的subplot
    for i in range(len(slice_indices), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Tumor Segmentation Montage - Patient {patient_id}', fontsize=16)
    plt.tight_layout()
    
    # 保存拼接图
    montage_path = os.path.join(output_dir, f'{patient_id}_slice_montage.png')
    plt.savefig(montage_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Slice montage saved to: {montage_path}")

def save_nifti_segmentation(mask, original_nifti_img, output_path):
    """保存分割结果为NIfTI文件"""
    print(f"Saving segmentation to: {output_path}")
    
    # 创建新的NIfTI图像，使用原始图像的header和affine
    segmentation_img = nib.Nifti1Image(
        mask.astype(np.uint8),
        original_nifti_img.affine,
        original_nifti_img.header
    )
    
    # 保存文件
    nib.save(segmentation_img, output_path)
    print(f"NIfTI segmentation saved successfully")

def calculate_3d_statistics(mask, voxel_spacing=(1.0, 1.0, 1.0)):
    """计算3D分割统计信息"""
    tumor_voxels = np.sum(mask == 1)
    
    # 计算体积（假设体素间距）
    voxel_volume = np.prod(voxel_spacing)  # mm³
    tumor_volume_mm3 = tumor_voxels * voxel_volume
    tumor_volume_ml = tumor_volume_mm3 / 1000  # 转换为毫升
    
    # 连通域分析
    labeled_mask = measure.label(mask, connectivity=3)
    regions = measure.regionprops(labeled_mask)
    
    num_components = len(regions)
    largest_component_size = max([r.area for r in regions]) if regions else 0
    
    # 计算边界框
    if tumor_voxels > 0:
        coords = np.where(mask == 1)
        bbox_min = [np.min(coords[i]) for i in range(3)]
        bbox_max = [np.max(coords[i]) for i in range(3)]
        bbox_size = [bbox_max[i] - bbox_min[i] + 1 for i in range(3)]
    else:
        bbox_min = bbox_max = bbox_size = [0, 0, 0]
    
    stats = {
        'tumor_voxels': int(tumor_voxels),
        'tumor_volume_mm3': float(tumor_volume_mm3),
        'tumor_volume_ml': float(tumor_volume_ml),
        'num_components': int(num_components),
        'largest_component_size': int(largest_component_size),
        'bbox_min': bbox_min,
        'bbox_max': bbox_max,
        'bbox_size': bbox_size
    }
    
    return stats

def process_single_patient(input_path, model, device, output_dir):
    """处理单个患者的数据"""
    print(f"\n{'='*60}")
    print(f"Processing: {input_path}")
    print(f"{'='*60}")
    
    # 获取患者ID
    patient_id = Path(input_path).stem.replace('_fla', '')
    patient_output_dir = os.path.join(output_dir, f"patient_{patient_id}")
    os.makedirs(patient_output_dir, exist_ok=True)
    
    # 加载NIfTI文件
    print("Loading NIfTI file...")
    nifti_img = nib.load(input_path)
    nifti_data = nifti_img.get_fdata()
    
    print(f"Image shape: {nifti_data.shape}")
    print(f"Image data type: {nifti_data.dtype}")
    print(f"Voxel spacing: {nifti_img.header.get_zooms()[:3]}")
    
    # 进行3D分割预测
    print("Running 3D segmentation...")
    raw_predictions = predict_3d_segmentation(model, nifti_data, device)
    
    # 3D后处理
    processed_mask = post_process_3d_mask(raw_predictions, min_size=50, closing_radius=1)
    
    # 计算统计信息
    voxel_spacing = nifti_img.header.get_zooms()[:3]
    stats = calculate_3d_statistics(processed_mask, voxel_spacing)
    
    print(f"\n3D Segmentation Statistics:")
    print(f"  Tumor voxels: {stats['tumor_voxels']:,}")
    print(f"  Tumor volume: {stats['tumor_volume_ml']:.2f} ml")
    print(f"  Number of components: {stats['num_components']}")
    print(f"  Largest component: {stats['largest_component_size']:,} voxels")
    print(f"  Bounding box size: {stats['bbox_size']}")
    
    # 保存NIfTI分割结果
    segmentation_output_path = os.path.join(patient_output_dir, f"{patient_id}_predicted_segmentation.nii.gz")
    save_nifti_segmentation(processed_mask, nifti_img, segmentation_output_path)
    
    # 创建3D网格
    vertices, faces, normals = create_3d_mesh(processed_mask, step_size=2)
    
    # 创建可视化
    if vertices is not None:
        # 交互式3D可视化
        try:
            create_interactive_3d_plot(vertices, faces, processed_mask, patient_id, patient_output_dir)
        except Exception as e:
            print(f"Warning: Could not create interactive plot: {e}")
    
    # matplotlib 3D可视化
    create_matplotlib_3d_plot(processed_mask, patient_id, patient_output_dir)
    
    # 创建切片拼接图
    create_slice_montage(nifti_data, processed_mask, patient_id, patient_output_dir)
    
    # 保存统计信息
    stats_file = os.path.join(patient_output_dir, f"{patient_id}_3d_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write(f"3D Tumor Segmentation Statistics - Patient {patient_id}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Original image shape: {nifti_data.shape}\n")
        f.write(f"Voxel spacing: {voxel_spacing}\n\n")
        
        f.write("Tumor Statistics:\n")
        for key, value in stats.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"\nResults saved to: {patient_output_dir}")
    return patient_output_dir, stats

def find_nifti_files(input_dir, pattern="*_fla.nii.gz"):
    """查找目录中的NIfTI文件"""
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    search_pattern = os.path.join(input_dir, "**", pattern)
    nifti_files = glob.glob(search_pattern, recursive=True)
    
    if not nifti_files:
        # 尝试其他常见模式
        alternative_patterns = ["*.nii.gz", "*flair*.nii.gz", "*t1*.nii.gz", "*T1*.nii.gz"]
        for alt_pattern in alternative_patterns:
            search_pattern = os.path.join(input_dir, "**", alt_pattern)
            nifti_files = glob.glob(search_pattern, recursive=True)
            if nifti_files:
                print(f"Found {len(nifti_files)} files with pattern: {alt_pattern}")
                break
    
    return sorted(nifti_files)

def main():
    parser = argparse.ArgumentParser(description='Create 3D Brain Tumor Segmentation Model')
    parser.add_argument('--input_dir', required=True,
                       help='Input directory containing .nii.gz files')
    parser.add_argument('--model_path', default='best_brain_tumor_model.pth',
                       help='Path to trained model (default: best_brain_tumor_model.pth)')
    parser.add_argument('--output_dir', default='3d_tumor_results',
                       help='Output directory for results (default: 3d_tumor_results)')
    parser.add_argument('--pattern', default='*_fla.nii.gz',
                       help='File pattern to search for (default: *_fla.nii.gz)')
    parser.add_argument('--single_file', default=None,
                       help='Process single NIfTI file instead of directory')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 解析路径
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(CURRENT_DIR, args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # 加载模型
    try:
        model = load_model(args.model_path, device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 处理文件
    try:
        if args.single_file:
            # 处理单个文件
            nifti_files = [args.single_file]
        else:
            # 查找所有NIfTI文件
            nifti_files = find_nifti_files(args.input_dir, args.pattern)
        
        if not nifti_files:
            print(f"No NIfTI files found in {args.input_dir} with pattern {args.pattern}")
            return
        
        print(f"Found {len(nifti_files)} NIfTI files to process")
        
        # 处理所有文件
        all_stats = []
        for i, nifti_file in enumerate(nifti_files, 1):
            print(f"\n{'#'*60}")
            print(f"Processing file {i}/{len(nifti_files)}")
            print(f"{'#'*60}")
            
            try:
                patient_dir, stats = process_single_patient(
                    nifti_file, model, device, args.output_dir
                )
                stats['patient_id'] = Path(nifti_file).stem.replace('_fla', '')
                stats['input_file'] = nifti_file
                all_stats.append(stats)
                
            except Exception as e:
                print(f"Error processing {nifti_file}: {e}")
                continue
        
        # 生成总结报告
        if all_stats:
            summary_file = os.path.join(args.output_dir, "processing_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("3D Brain Tumor Segmentation - Processing Summary\n")
                f.write("="*60 + "\n\n")
                f.write(f"Total files processed: {len(all_stats)}\n")
                f.write(f"Model used: {args.model_path}\n")
                f.write(f"Device: {device}\n\n")
                
                f.write("Individual Results:\n")
                f.write("-"*40 + "\n")
                for stats in all_stats:
                    f.write(f"Patient {stats['patient_id']}:\n")
                    f.write(f"  Volume: {stats['tumor_volume_ml']:.2f} ml\n")
                    f.write(f"  Components: {stats['num_components']}\n")
                    f.write(f"  Voxels: {stats['tumor_voxels']:,}\n\n")
                
                # 统计汇总
                total_volume = sum(s['tumor_volume_ml'] for s in all_stats)
                avg_volume = total_volume / len(all_stats)
                total_voxels = sum(s['tumor_voxels'] for s in all_stats)
                
                f.write("Summary Statistics:\n")
                f.write("-"*20 + "\n")
                f.write(f"Total tumor volume: {total_volume:.2f} ml\n")
                f.write(f"Average tumor volume: {avg_volume:.2f} ml\n")
                f.write(f"Total tumor voxels: {total_voxels:,}\n")
            
            print(f"\n{'='*60}")
            print("PROCESSING COMPLETED")
            print(f"{'='*60}")
            print(f"Processed {len(all_stats)} patients successfully")
            print(f"Total tumor volume: {total_volume:.2f} ml")
            print(f"Average tumor volume: {avg_volume:.2f} ml")
            print(f"Results saved to: {args.output_dir}")
            print(f"Summary report: {summary_file}")
        
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("3D Brain Tumor Segmentation and Visualization Tool")
        print("="*55)
        print("\nUsage examples:")
        print("python create_3d_tumor_model.py --input_dir dataset_folder --model_path model.pth")
        print("python create_3d_tumor_model.py --single_file patient_001_fla.nii.gz")
        print("python create_3d_tumor_model.py --input_dir data --output_dir results --pattern '*T1*.nii.gz'")
        
        print("\nFeatures:")
        print("- Process single file or entire directories")
        print("- Generate 3D tumor segmentation (.nii.gz)")
        print("- Create interactive 3D visualizations (HTML)")
        print("- Generate static 3D plots and slice montages")
        print("- Calculate detailed 3D statistics")
        print("- Batch processing with summary reports")
        
        print("\nRequired packages:")
        print("pip install torch nibabel numpy matplotlib plotly scikit-image scipy")
        
        print("\nOutput files for each patient:")
        print("- {patient_id}_predicted_segmentation.nii.gz  # 3D segmentation")
        print("- {patient_id}_3d_tumor_interactive.html      # Interactive 3D model")
        print("- {patient_id}_3d_tumor_static.png            # Static 3D plot")
        print("- {patient_id}_3d_tumor_matplotlib.png        # Matplotlib 3D plot")
        print("- {patient_id}_slice_montage.png              # Slice overview")
        print("- {patient_id}_3d_statistics.txt              # Detailed statistics")
    else:
        main()