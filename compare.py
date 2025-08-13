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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

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
    """加载训练好的模型 - 兼容新旧格式和PyTorch版本"""
    # 确保torch模块可用
    import torch as torch_module
    
    if not os.path.isabs(model_path):
        model_path = os.path.join(CURRENT_DIR, model_path)
    
    print(f"Loading model from: {model_path}")
    print(f"PyTorch version: {torch_module.__version__}")
    
    model = UNet().to(device)
    
    try:
        # 方法1: 首先尝试安全加载（仅权重）- 适用于PyTorch 2.6+
        print("Attempting safe loading (weights_only=True)...")
        checkpoint = torch_module.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print("✓ Model loaded successfully with weights_only=True")
        
    except Exception as e1:
        print(f"Safe loading failed: {e1}")
        
        try:
            # 方法2: 尝试加载完整检查点（兼容旧格式）
            print("Attempting full checkpoint loading...")
            checkpoint = torch_module.load(model_path, map_location=device, weights_only=False)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # 新格式：包含完整训练信息的检查点
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("✓ Model loaded from checkpoint dict")
                    
                    # 打印训练信息
                    if 'best_dice' in checkpoint:
                        print(f"  → Model's best validation Dice: {checkpoint['best_dice']:.4f}")
                    if 'epoch' in checkpoint:
                        print(f"  → Model was trained for {checkpoint['epoch']} epochs")
                    if 'early_stopping_info' in checkpoint:
                        es_info = checkpoint['early_stopping_info']
                        if es_info.get('triggered', False):
                            print(f"  → Training stopped early at epoch {es_info.get('best_epoch', 'N/A')}")
                        else:
                            print("  → Training completed without early stopping")
                
                else:
                    # 直接是state_dict格式
                    model.load_state_dict(checkpoint)
                    print("✓ Model loaded as direct state_dict")
            else:
                # 旧格式：直接加载
                model.load_state_dict(checkpoint)
                print("✓ Model loaded directly")
                
        except Exception as e2:
            print(f"Standard loading failed: {e2}")
            
            try:
                # 方法3: 使用安全全局对象上下文
                print("Attempting loading with safe globals...")
                # 明确导入torch.serialization
                import torch.serialization as torch_serialization
                
                with torch_serialization.safe_globals([
                    'numpy.core.multiarray.scalar',
                    'numpy.dtype', 
                    'numpy.ndarray',
                    'collections.OrderedDict',
                    'builtins.dict',
                    'builtins.int',
                    'builtins.float',
                    'builtins.bool'
                ]):
                    checkpoint = torch_module.load(model_path, map_location=device, weights_only=True)
                    
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        print("✓ Model loaded with safe globals from checkpoint")
                    else:
                        model.load_state_dict(checkpoint)
                        print("✓ Model loaded with safe globals directly")
                        
            except Exception as e3:
                # 方法4: 最后的尝试 - 完全不安全加载（仅限可信源）
                print(f"Safe globals loading failed: {e3}")
                print("⚠️  Attempting unsafe loading (use only for trusted models)...")
                
                try:
                    # 完全禁用安全检查（危险，仅限可信模型）
                    checkpoint = torch_module.load(model_path, map_location=device, weights_only=False)
                    
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        print("✓ Model loaded via unsafe method from checkpoint")
                    else:
                        model.load_state_dict(checkpoint)
                        print("✓ Model loaded via unsafe method directly")
                        
                except Exception as e4:
                    print(f"All loading methods failed!")
                    print(f"Final error: {e4}")
                    print(f"Original errors: {e1}, {e2}, {e3}")
                    
                    # 提供调试信息
                    print("\n🔍 Debugging information:")
                    try:
                        checkpoint = torch_module.load(model_path, map_location='cpu', weights_only=False)
                        print(f"  Checkpoint type: {type(checkpoint)}")
                        if isinstance(checkpoint, dict):
                            print(f"  Checkpoint keys: {list(checkpoint.keys())}")
                            
                            # 更详细的调试信息
                            for key, value in checkpoint.items():
                                if hasattr(value, 'shape'):
                                    print(f"    {key}: tensor with shape {value.shape}")
                                elif isinstance(value, dict):
                                    print(f"    {key}: dict with keys {list(value.keys())}")
                                else:
                                    print(f"    {key}: {type(value)} - {str(value)[:100]}...")
                                    
                    except Exception as debug_e:
                        print(f"  Cannot inspect checkpoint: {debug_e}")
                    
                    raise RuntimeError(f"Unable to load model from {model_path}. Please check the file format and PyTorch version compatibility.")
    
    model.eval()
    print(f"✓ Model set to evaluation mode")
    return model

def preprocess_image(image_slice):
    """预处理图像切片"""
    # 改进的归一化：与训练时保持一致
    if image_slice.std() > 1e-8:
        image_slice = (image_slice - image_slice.mean()) / image_slice.std()
    else:
        image_slice = image_slice - image_slice.mean()
    
    # 确保数值范围合理（与训练时保持一致）
    image_slice = np.clip(image_slice, -5, 5)
    
    # 转换为tensor并添加batch和channel维度
    image_tensor = torch.FloatTensor(image_slice).unsqueeze(0).unsqueeze(0)
    return image_tensor

def postprocess_mask(mask, threshold=0.5):
    """后处理分割掩码"""
    mask = mask.squeeze().cpu().numpy()
    binary_mask = (mask > threshold).astype(np.uint8)
    return binary_mask

def calculate_metrics(pred_mask, true_mask):
    """计算分割指标"""
    # 展平数组
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()
    
    # 确保是二值
    pred_flat = (pred_flat > 0).astype(int)
    true_flat = (true_flat > 0).astype(int)
    
    # 计算各种指标
    metrics = {}
    
    # 基本分类指标
    metrics['accuracy'] = accuracy_score(true_flat, pred_flat)
    metrics['precision'] = precision_score(true_flat, pred_flat, zero_division=0)
    metrics['recall'] = recall_score(true_flat, pred_flat, zero_division=0)
    metrics['f1_score'] = f1_score(true_flat, pred_flat, zero_division=0)
    
    # IoU (Intersection over Union) / Jaccard Index
    metrics['iou'] = jaccard_score(true_flat, pred_flat, zero_division=0)
    
    # Dice系数
    intersection = np.sum(pred_flat * true_flat)
    dice = (2. * intersection) / (np.sum(pred_flat) + np.sum(true_flat) + 1e-8)
    metrics['dice'] = dice
    
    # Specificity (True Negative Rate)
    tn = np.sum((1 - pred_flat) * (1 - true_flat))
    fp = np.sum(pred_flat * (1 - true_flat))
    specificity = tn / (tn + fp + 1e-8)
    metrics['specificity'] = specificity
    
    # 体积相关指标
    pred_volume = np.sum(pred_flat)
    true_volume = np.sum(true_flat)
    
    # 体积误差
    volume_error = abs(pred_volume - true_volume) / (true_volume + 1e-8)
    metrics['volume_error'] = volume_error
    
    # 相对体积差异
    relative_volume_diff = (pred_volume - true_volume) / (true_volume + 1e-8)
    metrics['relative_volume_diff'] = relative_volume_diff
    
    return metrics

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

def overlay_comparison_masks(image, pred_mask, true_mask, alpha=0.5):
    """在图像上叠加预测和真实掩码的比较"""
    # 确保图像是RGB格式
    if len(image.shape) == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image.copy()
    
    # 创建比较掩码
    colored_mask = np.zeros_like(image_rgb)
    
    # True Positive: 绿色 (预测对的肿瘤区域)
    tp_mask = (pred_mask == 1) & (true_mask == 1)
    colored_mask[tp_mask] = (0, 1.0, 0)  # 绿色
    
    # False Positive: 红色 (误检的肿瘤区域)
    fp_mask = (pred_mask == 1) & (true_mask == 0)
    colored_mask[fp_mask] = (1.0, 0, 0)  # 红色
    
    # False Negative: 蓝色 (漏检的肿瘤区域)
    fn_mask = (pred_mask == 0) & (true_mask == 1)
    colored_mask[fn_mask] = (0, 0, 1.0)  # 蓝色
    
    # 叠加掩码
    result = image_rgb * (1 - alpha) + colored_mask * alpha
    result = np.clip(result, 0, 1)
    return result

def test_on_nifti_file_with_ground_truth(model, fla_path, seg_path, device, output_dir='compare_results'):
    """对NIfTI文件进行测试并计算准确度指标（需要ground truth）"""
    # 如果输入路径不是绝对路径，则基于当前目录构建路径
    if not os.path.isabs(fla_path):
        fla_path = os.path.join(CURRENT_DIR, fla_path)
    if not os.path.isabs(seg_path):
        seg_path = os.path.join(CURRENT_DIR, seg_path)
    
    # 如果输出目录不是绝对路径，则基于当前目录构建路径
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(CURRENT_DIR, output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Loading NIfTI files from: {fla_path} and {seg_path}")
    print(f"Output directory: {output_dir}")
    
    # 加载NIfTI文件
    fla_img = nib.load(fla_path)
    seg_img = nib.load(seg_path)
    fla_data = fla_img.get_fdata()
    seg_data = seg_img.get_fdata()
    
    patient_id = os.path.basename(fla_path).replace('_fla.nii.gz', '')
    
    print(f"Processing patient: {patient_id}")
    print(f"Image shape: {fla_data.shape}")
    print(f"Segmentation shape: {seg_data.shape}")
    
    # 对所有切片进行预测
    num_slices = fla_data.shape[2]
    predictions = []
    ground_truths = []
    all_metrics = []
    
    print("Processing slices...")
    with torch.no_grad():
        for slice_idx in range(num_slices):
            if (slice_idx + 1) % 20 == 0:
                print(f"  Processed {slice_idx + 1}/{num_slices} slices")
            
            # 获取切片
            image_slice = fla_data[:, :, slice_idx]
            true_mask = seg_data[:, :, slice_idx]
            true_mask = (true_mask > 0).astype(np.uint8)
            
            # 跳过空切片
            if image_slice.sum() == 0:
                predictions.append(np.zeros_like(image_slice))
                ground_truths.append(true_mask)
                continue
            
            # 预处理（使用改进的预处理方法）
            image_tensor = preprocess_image(image_slice).to(device)
            
            # 预测
            pred_mask = model(image_tensor)
            binary_mask = postprocess_mask(pred_mask)
            
            predictions.append(binary_mask)
            ground_truths.append(true_mask)
            
            # 计算当前切片的指标
            if true_mask.sum() > 0 or binary_mask.sum() > 0:  # 只对有肿瘤或有预测的切片计算指标
                slice_metrics = calculate_metrics(binary_mask, true_mask)
                slice_metrics['slice_idx'] = slice_idx
                all_metrics.append(slice_metrics)
            
            # 保存一些示例切片（包含比较）
            if slice_idx % 10 == 0 and (binary_mask.sum() > 0 or true_mask.sum() > 0):
                # 归一化图像用于显示
                display_image = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
                
                # 创建比较图像
                comparison_image = overlay_comparison_masks(display_image, binary_mask, true_mask)
                overlay_image = overlay_mask_on_image(display_image, binary_mask)
                true_overlay = overlay_mask_on_image(display_image, true_mask, color=(0, 1.0, 0))
                
                # 保存结果
                plt.figure(figsize=(20, 8))
                
                plt.subplot(2, 3, 1)
                plt.imshow(display_image, cmap='gray')
                plt.title(f'Original Slice {slice_idx}')
                plt.axis('off')
                
                plt.subplot(2, 3, 2)
                plt.imshow(true_mask, cmap='Greens')
                plt.title('Ground Truth')
                plt.axis('off')
                
                plt.subplot(2, 3, 3)
                plt.imshow(binary_mask, cmap='Reds')
                plt.title('Prediction')
                plt.axis('off')
                
                plt.subplot(2, 3, 4)
                plt.imshow(true_overlay)
                plt.title('GT Overlay (Green)')
                plt.axis('off')
                
                plt.subplot(2, 3, 5)
                plt.imshow(overlay_image)
                plt.title('Pred Overlay (Red)')
                plt.axis('off')
                
                plt.subplot(2, 3, 6)
                plt.imshow(comparison_image)
                plt.title('Comparison (TP:Green, FP:Red, FN:Blue)')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/{patient_id}_comparison_slice_{slice_idx:03d}.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
    
    # 计算整体指标
    all_predictions = np.concatenate([p.flatten() for p in predictions])
    all_ground_truths = np.concatenate([g.flatten() for g in ground_truths])
    
    overall_metrics = calculate_metrics(all_predictions, all_ground_truths)
    
    # 计算切片级别的平均指标
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            if key != 'slice_idx':
                avg_metrics[f'avg_{key}'] = np.mean([m[key] for m in all_metrics])
                avg_metrics[f'std_{key}'] = np.std([m[key] for m in all_metrics])
    
    # 输出结果
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS FOR PATIENT: {patient_id}")
    print("="*60)
    
    print("\nOVERALL METRICS (全体素级别):")
    print(f"Accuracy:           {overall_metrics['accuracy']:.4f}")
    print(f"Dice Coefficient:   {overall_metrics['dice']:.4f}")
    print(f"IoU (Jaccard):      {overall_metrics['iou']:.4f}")
    print(f"Precision:          {overall_metrics['precision']:.4f}")
    print(f"Recall (Sensitivity): {overall_metrics['recall']:.4f}")
    print(f"Specificity:        {overall_metrics['specificity']:.4f}")
    print(f"F1 Score:           {overall_metrics['f1_score']:.4f}")
    print(f"Volume Error:       {overall_metrics['volume_error']:.4f}")
    print(f"Relative Vol Diff:  {overall_metrics['relative_volume_diff']:.4f}")
    
    if all_metrics:
        print(f"\nSLICE-LEVEL AVERAGE METRICS (切片级别平均):")
        print(f"Avg Dice:           {avg_metrics['avg_dice']:.4f} ± {avg_metrics['std_dice']:.4f}")
        print(f"Avg IoU:            {avg_metrics['avg_iou']:.4f} ± {avg_metrics['std_iou']:.4f}")
        print(f"Avg Precision:      {avg_metrics['avg_precision']:.4f} ± {avg_metrics['std_precision']:.4f}")
        print(f"Avg Recall:         {avg_metrics['avg_recall']:.4f} ± {avg_metrics['std_recall']:.4f}")
    
    print(f"\nSTATISTICS:")
    print(f"Total slices:       {num_slices}")
    print(f"Slices with tumor:  {sum(1 for g in ground_truths if g.sum() > 0)}")
    print(f"Predicted tumors:   {sum(1 for p in predictions if p.sum() > 0)}")
    print(f"True tumor volume:  {sum(g.sum() for g in ground_truths)} voxels")
    print(f"Pred tumor volume:  {sum(p.sum() for p in predictions)} voxels")
    
    # 保存详细结果到文件
    results_file = os.path.join(output_dir, f'{patient_id}_metrics.txt')
    with open(results_file, 'w') as f:
        f.write(f"EVALUATION RESULTS FOR PATIENT: {patient_id}\n")
        f.write("="*60 + "\n\n")
        
        f.write("OVERALL METRICS:\n")
        for key, value in overall_metrics.items():
            f.write(f"{key}: {value:.6f}\n")
        
        if all_metrics:
            f.write("\nSLICE-LEVEL METRICS:\n")
            for key, value in avg_metrics.items():
                f.write(f"{key}: {value:.6f}\n")
        
        f.write(f"\nDETAILED SLICE METRICS:\n")
        for metrics in all_metrics:
            f.write(f"Slice {metrics['slice_idx']}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return predictions, overall_metrics, all_metrics

def test_on_nifti_file(model, fla_path, device, output_dir='compare_results'):
    """对NIfTI文件进行测试并保存结果（无ground truth）"""
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
    
    print("Processing slices...")
    with torch.no_grad():
        for slice_idx in range(num_slices):
            if (slice_idx + 1) % 20 == 0:
                print(f"  Processed {slice_idx + 1}/{num_slices} slices")
            
            # 获取切片
            image_slice = fla_data[:, :, slice_idx]
            
            # 跳过空切片
            if image_slice.sum() == 0:
                predictions.append(np.zeros_like(image_slice))
                continue
            
            # 预处理（使用改进的预处理方法）
            image_tensor = preprocess_image(image_slice).to(device)
            
            # 预测
            pred_mask = model(image_tensor)
            binary_mask = postprocess_mask(pred_mask)
            
            predictions.append(binary_mask)
            
            # 保存一些示例切片
            if slice_idx % 10 == 0 and binary_mask.sum() > 0:  # 只保存有肿瘤的切片
                # 归一化图像用于显示
                display_image = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
                
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

def test_on_single_image(model, image_path, device, output_dir='compare_results'):
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
    
    # 预处理（使用改进的预处理方法）
    image_tensor = preprocess_image(image_data).to(device)
    
    # 预测
    with torch.no_grad():
        pred_mask = model(image_tensor)
        binary_mask = postprocess_mask(pred_mask)
    
    # 归一化图像用于显示
    display_image = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    
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
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation Comparison and Testing')
    parser.add_argument('--model_path', default='best_brain_tumor_model.pth', 
                       help='Path to trained model')
    parser.add_argument('--input_path', required=True,
                       help='Path to input image or NIfTI file')
    parser.add_argument('--ground_truth_path', default=None,
                       help='Path to ground truth segmentation file (optional)')
    parser.add_argument('--output_dir', default='compare_results',
                       help='Output directory for results (default: compare_results)')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 确保输出目录存在
    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(CURRENT_DIR, args.output_dir)
    else:
        output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # 加载模型
    try:
        model = load_model(args.model_path, device)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试
    try:
        if args.input_path.endswith('_fla.nii.gz'):
            # NIfTI文件测试
            if args.ground_truth_path:
                # 有ground truth，计算准确度指标
                predictions, metrics, slice_metrics = test_on_nifti_file_with_ground_truth(
                    model, args.input_path, args.ground_truth_path, device, args.output_dir)
                print(f"✅ Results and metrics saved to {args.output_dir}")
            else:
                # 检查是否可以自动找到对应的seg文件
                auto_seg_path = args.input_path.replace('_fla.nii.gz', '_seg.nii.gz')
                if os.path.exists(auto_seg_path):
                    print(f"🔍 Found ground truth file: {auto_seg_path}")
                    predictions, metrics, slice_metrics = test_on_nifti_file_with_ground_truth(
                        model, args.input_path, auto_seg_path, device, args.output_dir)
                    print(f"✅ Results and metrics saved to {args.output_dir}")
                else:
                    predictions = test_on_nifti_file(model, args.input_path, device, args.output_dir)
                    print(f"✅ Results saved to {args.output_dir} (no ground truth available)")
        else:
            # 单张图像测试
            mask, overlay = test_on_single_image(model, args.input_path, device, args.output_dir)
            print(f"✅ Results saved to {args.output_dir}")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 如果没有命令行参数，提供默认测试示例
    import sys
    if len(sys.argv) == 1:
        # 示例用法
        print("Brain Tumor Segmentation Model Comparison Tool")
        print("="*50)
        print("Usage examples:")
        print("  python compare.py --input_path path/to/patient_fla.nii.gz")
        print("  python compare.py --input_path path/to/patient_fla.nii.gz --ground_truth_path path/to/patient_seg.nii.gz")
        print("  python compare.py --input_path path/to/brain_image.png --model_path best_brain_tumor_model.pth")
        print("  python compare.py --input_path path/to/patient_fla.nii.gz --output_dir custom_compare_results")
        
        print(f"\nCurrent working directory: {CURRENT_DIR}")
        print("All relative paths will be resolved relative to this directory.")
        
        print(f"\nPyTorch version: {torch.__version__}")
        print("Required packages:")
        print("  pip install torch torchvision nibabel numpy matplotlib pillow scikit-learn")
        
        print("\nFeatures:")
        print("  ✓ Compatible with PyTorch 2.6+ security changes")
        print("  ✓ Automatic accuracy evaluation when ground truth is available")
        print("  ✓ Comprehensive metrics: Dice, IoU, Precision, Recall, F1, Specificity")
        print("  ✓ Visual comparison with color-coded overlays")
        print("  ✓ Detailed results saved to text files")
        print("  ✓ Improved preprocessing matching training pipeline")
        
        print("\nOutput:")
        print("  → All results saved to 'compare_results' folder by default")
        print("  → Comparison images with color-coded overlays")
        print("  → Detailed metrics text files")
        print("  → Individual slice visualizations")
        
        print("\nColor coding in comparison images:")
        print("  🟢 Green: True Positive (correctly predicted tumor)")
        print("  🔴 Red: False Positive (incorrectly predicted tumor)")
        print("  🔵 Blue: False Negative (missed tumor)")
    else:
        main()