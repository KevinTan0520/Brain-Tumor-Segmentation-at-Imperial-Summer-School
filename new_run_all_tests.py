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

# 从 new_train.py 导入模型架构类
class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv1(attention)
        return x * self.sigmoid(attention)

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SelfAttention(nn.Module):
    """简化的自注意力机制"""
    def __init__(self, in_dim, num_heads=8):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.proj = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    """简化的Transformer块"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TransXAI_UNet(nn.Module):
    """
    基于TransXAI设计的混合CNN-Transformer架构 - 推理版本
    """
    def __init__(self, in_channels=1, out_channels=1, features=[24, 48, 96, 192]):
        super(TransXAI_UNet, self).__init__()
        
        # CNN特征提取器
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        # 下采样路径
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for i, feature in enumerate(features):
            if i == 0:
                continue
            self.downs.append(self.double_conv(features[i-1], feature))
        
        # 瓶颈层
        self.bottleneck = self.double_conv(features[-1], features[-1]*2)
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(features[-1]*2, num_heads=8) for _ in range(4)
        ])
        
        # 位置编码
        self.pos_embed_conv = nn.Conv2d(features[-1]*2, features[-1]*2, 1)
        
        # 上采样路径
        self.ups = nn.ModuleList()
        up_features = list(reversed(features))
        
        for i in range(len(features)):
            if i == 0:
                self.ups.append(nn.ConvTranspose2d(features[-1]*2, up_features[i], kernel_size=2, stride=2))
                self.ups.append(
                    nn.Sequential(
                        self.double_conv(up_features[i]*2, up_features[i]),
                        SpatialAttention(),
                        ChannelAttention(up_features[i])
                    )
                )
            else:
                self.ups.append(nn.ConvTranspose2d(up_features[i-1], up_features[i], kernel_size=2, stride=2))
                if i < len(features) - 1:
                    self.ups.append(
                        nn.Sequential(
                            self.double_conv(up_features[i]*2, up_features[i]),
                            SpatialAttention(),
                            ChannelAttention(up_features[i])
                        )
                    )
                else:
                    self.ups.append(self.double_conv(up_features[i]*2, up_features[i]))
        
        # 最终分类层
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0]//2, 3, padding=1),
            nn.BatchNorm2d(features[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0]//2, out_channels, 1)
        )
        
        # 深度监督输出
        self.deep_supervision = nn.ModuleList([
            nn.Conv2d(up_features[i], out_channels, 1) for i in range(len(features)-1)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
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
        # 初始卷积
        x = self.initial_conv(x)
        skip_connections = [x]
        
        # 下采样路径
        for down in self.downs:
            x = self.pool(x)
            x = down(x)
            skip_connections.append(x)
        
        # 瓶颈层 + Transformer处理
        x = self.pool(x)
        x = self.bottleneck(x)
        
        # 添加位置编码
        x = x + self.pos_embed_conv(x)
        
        # Transformer编码
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        
        for transformer in self.transformer_layers:
            x_flat = transformer(x_flat, H, W)
        
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        skip_connections = skip_connections[::-1]
        
        # 上采样路径
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        # 最终输出 - 推理时直接应用sigmoid
        final_output = self.final_conv(x)
        return torch.sigmoid(final_output)

def load_hybrid_model(model_path, device):
    """加载训练好的混合模型"""
    print(f"Loading hybrid model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 检查模型架构信息
    if 'model_architecture' in checkpoint:
        print(f"   Model architecture: {checkpoint['model_architecture']}")
    
    if 'features' in checkpoint:
        features = checkpoint['features']
        print(f"   Model features: {features}")
    else:
        # 默认特征配置
        features = [24, 48, 96, 192]
        print(f"   Using default features: {features}")
    
    # 创建模型实例
    model = TransXAI_UNet(in_channels=1, out_channels=1, features=features)
    
    # 加载模型权重
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
    """预处理单个切片数据（与训练保持一致）"""
    if slice_data.max() <= slice_data.min() + 1e-6:
        return np.zeros_like(slice_data, dtype=np.float32)
    
    # 基于体积级别的归一化（简化版）
    volume_mean = slice_data[slice_data > 0].mean() if slice_data[slice_data > 0].size > 0 else 0
    volume_std = slice_data[slice_data > 0].std() if slice_data[slice_data > 0].size > 0 else 1
    
    if volume_std > 1e-8:
        slice_data = (slice_data - volume_mean) / volume_std
    else:
        slice_data = slice_data - volume_mean
    
    # 稳健的值域限制
    slice_data = np.clip(slice_data, -4, 4)
    
    return slice_data.astype(np.float32)

def process_single_case(fla_path, seg_path, model, device, output_base_dir):
    """处理单个病例（训练集中的数据）"""
    print(f"\n{'='*80}")
    print(f"PROCESSING CASE: {fla_path}")
    print(f"{'='*80}")
    
    # 获取患者ID
    patient_id = Path(fla_path).parent.name
    print(f"Patient ID: {patient_id}")
    
    # 创建患者输出目录
    patient_output_dir = os.path.join(output_base_dir, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)
    
    # 加载NIfTI文件
    print("Loading NIfTI files...")
    fla_img = nib.load(fla_path)
    seg_img = nib.load(seg_path)
    
    fla_data = fla_img.get_fdata()
    seg_data = seg_img.get_fdata()
    
    print(f"   FLA image shape: {fla_data.shape}")
    print(f"   SEG image shape: {seg_data.shape}")
    print(f"   FLA value range: [{fla_data.min():.2f}, {fla_data.max():.2f}]")
    print(f"   Ground truth tumor voxels: {int(np.sum(seg_data > 0)):,}")
    
    # 3D分割预测
    print("Running 3D segmentation...")
    depth = fla_data.shape[2]
    predictions = np.zeros_like(fla_data)
    
    # 逐切片处理
    for slice_idx in tqdm(range(depth), desc="Processing slices"):
        slice_data = fla_data[:, :, slice_idx].copy()
        
        # 跳过空切片
        if slice_data.max() <= slice_data.min() + 1e-6:
            continue
        
        # 预处理
        processed_slice = preprocess_slice(slice_data)
        
        # 调整到240x240（如果需要）
        if processed_slice.shape != (240, 240):
            from skimage.transform import resize
            processed_slice = resize(processed_slice, (240, 240), preserve_range=True, anti_aliasing=True)
        
        # 转换为tensor
        input_tensor = torch.FloatTensor(processed_slice).unsqueeze(0).unsqueeze(0).to(device)
        
        # 模型预测（sigmoid已在模型内部应用）
        with torch.no_grad():
            pred = model(input_tensor)
            pred = pred.squeeze().cpu().numpy()
        
        # 调整回原始尺寸（如果需要）
        if pred.shape != fla_data.shape[:2]:
            from skimage.transform import resize
            pred = resize(pred, fla_data.shape[:2], preserve_range=True, anti_aliasing=True)
        
        predictions[:, :, slice_idx] = pred
    
    # 后处理
    print("Applying post-processing...")
    
    # 二值化
    binary_mask = (predictions > 0.5).astype(np.uint8)
    
    # 3D形态学闭运算
    selem = morphology.ball(1)
    binary_mask = morphology.binary_closing(binary_mask, selem)
    
    # 连通组件分析
    labeled_mask, num_components = measure.label(binary_mask, return_num=True)
    
    if num_components > 0:
        # 保留大组件
        component_sizes = [(np.sum(labeled_mask == i), i) for i in range(1, num_components + 1)]
        final_mask = np.zeros_like(binary_mask)
        
        for size, label in sorted(component_sizes, reverse=True):
            if size >= 50:  # 最小尺寸阈值
                final_mask[labeled_mask == label] = 1
    else:
        final_mask = binary_mask
    
    # 3D孔洞填充
    final_mask = ndimage.binary_fill_holes(final_mask)
    
    # 统计信息
    predicted_tumor_voxels = int(np.sum(final_mask))
    ground_truth_tumor_voxels = int(np.sum(seg_data > 0))
    
    print(f"   Predicted tumor voxels: {predicted_tumor_voxels:,}")
    print(f"   Ground truth tumor voxels: {ground_truth_tumor_voxels:,}")
    
    # 保存预测的分割结果（与calc.py兼容的格式）
    pred_output_path = os.path.join(patient_output_dir, f"{patient_id}_seg.nii.gz")
    
    # 创建预测NIfTI图像
    pred_img = nib.Nifti1Image(
        final_mask.astype(np.float32),
        fla_img.affine,
        fla_img.header
    )
    nib.save(pred_img, pred_output_path)
    print(f"   Predicted segmentation saved: {pred_output_path}")
    
    # 复制真实分割到输出目录（用于calc.py比较）
    gt_output_path = os.path.join(patient_output_dir, f"{patient_id}_seg_gt.nii.gz")
    
    gt_img = nib.Nifti1Image(
        (seg_data > 0).astype(np.float32),
        seg_img.affine,
        seg_img.header
    )
    nib.save(gt_img, gt_output_path)
    print(f"   Ground truth segmentation saved: {gt_output_path}")
    
    # 创建对比可视化
    print("Creating comparison visualization...")
    
    # 找到有肿瘤的切片
    gt_tumor_slices = []
    pred_tumor_slices = []
    
    for i in range(depth):
        if np.sum(seg_data[:, :, i] > 0) > 0:
            gt_tumor_slices.append(i)
        if np.sum(final_mask[:, :, i]) > 0:
            pred_tumor_slices.append(i)
    
    # 选择有代表性的切片进行可视化
    all_tumor_slices = sorted(list(set(gt_tumor_slices + pred_tumor_slices)))
    
    if len(all_tumor_slices) > 0:
        # 选择最多8个切片
        if len(all_tumor_slices) > 8:
            selected_indices = np.linspace(0, len(all_tumor_slices)-1, 8, dtype=int)
            selected_slices = [all_tumor_slices[i] for i in selected_indices]
        else:
            selected_slices = all_tumor_slices
        
        # 创建对比图
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, slice_idx in enumerate(selected_slices):
            if i >= 8:
                break
            
            # 显示原始图像
            img_slice = fla_data[:, :, slice_idx]
            axes[i].imshow(img_slice, cmap='gray', alpha=0.8)
            
            # 叠加真实分割（绿色）
            gt_slice = seg_data[:, :, slice_idx]
            if np.sum(gt_slice > 0) > 0:
                axes[i].contour(gt_slice, levels=[0.5], colors=['green'], linewidths=2, alpha=0.8)
            
            # 叠加预测分割（红色）
            pred_slice = final_mask[:, :, slice_idx]
            if np.sum(pred_slice) > 0:
                axes[i].contour(pred_slice, levels=[0.5], colors=['red'], linewidths=2, alpha=0.8)
            
            gt_count = int(np.sum(gt_slice > 0))
            pred_count = int(np.sum(pred_slice))
            
            axes[i].set_title(f'Slice {slice_idx}\nGT: {gt_count}, Pred: {pred_count}', fontsize=10)
            axes[i].axis('off')
        
        # 隐藏未使用的子图
        for i in range(len(selected_slices), 8):
            axes[i].axis('off')
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Ground Truth'),
            Line2D([0], [0], color='red', lw=2, label='Prediction')
        ]
        fig.legend(handles=legend_elements, loc='upper right')
        
        plt.suptitle(f'Segmentation Comparison - {patient_id}\n'
                    f'GT: {ground_truth_tumor_voxels:,} voxels, Pred: {predicted_tumor_voxels:,} voxels', 
                    fontsize=16)
        plt.tight_layout()
        
        # 保存图像
        comparison_path = os.path.join(patient_output_dir, f"{patient_id}_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   Comparison visualization saved: {comparison_path}")
    
    print(f"✅ Case {patient_id} completed")
    
    return {
        'patient_id': patient_id,
        'predicted_voxels': predicted_tumor_voxels,
        'ground_truth_voxels': ground_truth_tumor_voxels,
        'pred_path': pred_output_path,
        'gt_path': gt_output_path,
        'comparison_path': comparison_path if 'comparison_path' in locals() else None
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Batch Test Brain Tumor Segmentation on Training Set')
    parser.add_argument('--data_dir', default='dataset_segmentation/train',
                       help='Directory containing training data')
    parser.add_argument('--model_path', default='best_brain_tumor_model.pth',
                       help='Path to trained hybrid model')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Processing device: {device}")
    
    # 加载混合模型
    try:
        model = load_hybrid_model(args.model_path, device)
        print("Hybrid model loaded successfully\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 创建输出目录
    current_time = datetime.now()
    date_str = current_time.strftime('%Y%m%d')
    time_str = current_time.strftime('%H%M%S')
    output_base_dir = f"batch_tests_result_{date_str}_{time_str}"
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"Output directory: {output_base_dir}")
    
    # 查找训练数据文件
    if not os.path.exists(args.data_dir):
        print(f"Data directory not found: {args.data_dir}")
        return
    
    # 查找所有 {id}/{id}_fla.nii.gz 和对应的 {id}_seg.nii.gz 文件
    fla_pattern = os.path.join(args.data_dir, "*", "*_fla.nii.gz")
    fla_files = glob.glob(fla_pattern)
    
    if not fla_files:
        print(f"No FLA files found in {args.data_dir}")
        print("Expected structure: dataset_segmentation/train/{id}/{id}_fla.nii.gz")
        return
    
    # 验证对应的seg文件存在
    valid_cases = []
    for fla_path in fla_files:
        seg_path = fla_path.replace('_fla.nii.gz', '_seg.nii.gz')
        if os.path.exists(seg_path):
            valid_cases.append((fla_path, seg_path))
        else:
            patient_id = Path(fla_path).parent.name
            print(f"Warning: Missing seg file for {patient_id}")
    
    if not valid_cases:
        print("No valid case pairs found!")
        return
    
    print(f"Found {len(valid_cases)} valid cases:")
    for i, (fla_path, seg_path) in enumerate(valid_cases, 1):
        patient_id = Path(fla_path).parent.name
        print(f"  {i}. {patient_id}")
    
    # 处理所有病例
    print(f"\nStarting batch testing...")
    
    results = []
    successful = 0
    failed = 0
    
    for i, (fla_path, seg_path) in enumerate(valid_cases, 1):
        try:
            patient_id = Path(fla_path).parent.name
            print(f"\n[{i}/{len(valid_cases)}] Processing {patient_id}...")
            
            result = process_single_case(fla_path, seg_path, model, device, output_base_dir)
            results.append(result)
            successful += 1
            
        except Exception as e:
            patient_id = Path(fla_path).parent.name
            print(f"Failed to process {patient_id}: {e}")
            failed += 1
    
    # 创建结果汇总
    print("\nCreating summary report...")
    
    summary_path = os.path.join(output_base_dir, "batch_test_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("BATCH TEST SUMMARY REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Data Directory: {args.data_dir}\n")
        f.write(f"Output Directory: {output_base_dir}\n\n")
        
        f.write("PROCESSING RESULTS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total cases: {len(valid_cases)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")
        
        if results:
            f.write("INDIVIDUAL CASE RESULTS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Patient ID':<15} {'GT Voxels':<12} {'Pred Voxels':<12} {'Ratio':<8}\n")
            f.write("-" * 50 + "\n")
            
            total_gt_voxels = 0
            total_pred_voxels = 0
            
            for result in results:
                gt_voxels = result['ground_truth_voxels']
                pred_voxels = result['predicted_voxels']
                ratio = pred_voxels / gt_voxels if gt_voxels > 0 else float('inf')
                
                f.write(f"{result['patient_id']:<15} {gt_voxels:<12,} {pred_voxels:<12,} {ratio:<8.3f}\n")
                
                total_gt_voxels += gt_voxels
                total_pred_voxels += pred_voxels
            
            f.write("-" * 50 + "\n")
            f.write(f"{'TOTAL':<15} {total_gt_voxels:<12,} {total_pred_voxels:<12,} "
                   f"{total_pred_voxels/total_gt_voxels if total_gt_voxels > 0 else float('inf'):<8.3f}\n")
        
        f.write("\nFILE STRUCTURE FOR calc.py:\n")
        f.write("-" * 25 + "\n")
        f.write("Each patient directory contains:\n")
        f.write("  - {patient_id}_seg.nii.gz      (predicted segmentation)\n")
        f.write("  - {patient_id}_seg_gt.nii.gz   (ground truth segmentation)\n")
        f.write("  - {patient_id}_comparison.png  (visual comparison)\n\n")
        f.write("To calculate metrics with calc.py:\n")
        f.write(f"python calc.py --pred_dir {output_base_dir} --gt_dir {output_base_dir}\n")
    
    # 最终总结
    print(f"\n{'='*80}")
    print("BATCH TESTING COMPLETED")
    print(f"{'='*80}")
    print(f"Total cases: {len(valid_cases)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_base_dir}")
    print(f"Summary report: {summary_path}")
    print("\n📊 Results format compatible with calc.py:")
    print(f"   Predicted: {{patient_id}}_seg.nii.gz")
    print(f"   Ground truth: {{patient_id}}_seg_gt.nii.gz")
    print(f"\n🧮 To calculate all metrics:")
    print(f"python calc.py --pred_dir {output_base_dir} --gt_dir {output_base_dir}")

if __name__ == "__main__":
    main()