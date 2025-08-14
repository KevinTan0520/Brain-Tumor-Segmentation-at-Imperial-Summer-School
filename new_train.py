import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import time
from datetime import datetime

# 获取当前文件所在目录并设置为工作目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR)
print(f"Current working directory set to: {CURRENT_DIR}")

# 添加缺失的数据加载函数
def load_data_pairs(dataset_dir):
    """
    加载数据对 (fla_path, seg_path)
    """
    print(f"Loading data pairs from: {dataset_dir}")
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return []
    
    data_pairs = []
    
    # 查找所有fla文件
    fla_pattern = os.path.join(dataset_dir, "**", "*_fla.nii.gz")
    fla_files = glob.glob(fla_pattern, recursive=True)
    
    print(f"Found {len(fla_files)} fla files")
    
    for fla_path in fla_files:
        # 构建对应的seg文件路径
        seg_path = fla_path.replace('_fla.nii.gz', '_seg.nii.gz')
        
        if os.path.exists(seg_path):
            data_pairs.append((fla_path, seg_path))
        else:
            print(f"Warning: Missing seg file for {fla_path}")
    
    print(f"Successfully loaded {len(data_pairs)} data pairs")
    
    return data_pairs

# 添加缺失的早停类
class EnhancedEarlyStopping:
    """增强版早停"""
    def __init__(self, patience=10, min_delta=0.0001, adaptive_patience=False, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.adaptive_patience = adaptive_patience
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.best_model_state = None
        self.score_history = []
        
    def __call__(self, score, model, epoch, train_loss):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'   EarlyStopping counter: {self.counter}/{self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model)
        
        self.score_history.append(score)
        
        # 自适应patience
        if self.adaptive_patience and len(self.score_history) > 10:
            recent_improvement = max(self.score_history[-5:]) - min(self.score_history[-10:-5])
            if recent_improvement < self.min_delta * 5:
                self.patience = min(self.patience + 2, 25)
    
    def save_checkpoint(self, model):
        """保存最佳模型状态"""
        self.best_model_state = model.state_dict().copy()
    
    def restore_best(self, model):
        """恢复最佳模型"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

# 添加缺失的注意力机制类
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

# 在文件开头添加dice_coefficient函数（与原版train.py兼容）
def dice_coefficient(pred, target, smooth=1):
    """计算Dice系数 - 与原版train.py兼容"""
    pred = (pred > 0.5).float()
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

class EnhancedBrainTumorDataset(Dataset):
    """
    增强版数据集，支持多模态融合和改进的数据增强
    基于文献建议：将T1和T1Gd组合，T2和FLAIR组合
    """
    def __init__(self, data_pairs, transform=None, min_tumor_ratio=0.003, normal_slice_ratio=0.2, 
                 use_multimodal=True, augment_prob=0.3):
        """
        Args:
            data_pairs: (fla_path, seg_path) 对的列表
            transform: 数据增强变换
            min_tumor_ratio: 更低的阈值以包含更多边界切片
            normal_slice_ratio: 提高正常切片比例以增强泛化能力
            use_multimodal: 是否使用多模态处理
            augment_prob: 数据增强概率
        """
        self.data_slices = []
        self.transform = transform
        self.use_multimodal = use_multimodal
        self.augment_prob = augment_prob
        
        print("Creating Enhanced Brain Tumor Dataset...")
        print(f"Parameters: min_tumor_ratio={min_tumor_ratio}, normal_slice_ratio={normal_slice_ratio}")
        print(f"Multimodal fusion: {use_multimodal}, Augmentation prob: {augment_prob}")
        
        tumor_slice_count = 0
        normal_slice_count = 0
        boundary_slice_count = 0
        
        for fla_path, seg_path in tqdm(data_pairs, desc="Loading enhanced volumes"):
            try:
                # 加载NIfTI文件
                fla_img = nib.load(fla_path)
                seg_img = nib.load(seg_path)
                
                fla_data = fla_img.get_fdata()
                seg_data = seg_img.get_fdata()
                
                patient_id = os.path.basename(fla_path).replace('_fla.nii.gz', '')
                
                # 预计算体积级别的统计信息用于归一化
                volume_mean = fla_data[fla_data > 0].mean() if fla_data[fla_data > 0].size > 0 else 0
                volume_std = fla_data[fla_data > 0].std() if fla_data[fla_data > 0].size > 0 else 1
                
                # 遍历所有切片
                for slice_idx in range(fla_data.shape[2]):
                    fla_slice = fla_data[:, :, slice_idx]
                    seg_slice = seg_data[:, :, slice_idx]
                    
                    # 跳过完全为零的切片
                    if fla_slice.sum() == 0:
                        continue
                    
                    # 计算肿瘤像素比例
                    total_pixels = seg_slice.shape[0] * seg_slice.shape[1]
                    tumor_pixels = seg_slice.sum()
                    tumor_ratio = tumor_pixels / total_pixels
                    
                    # 增强的切片分类
                    is_tumor_slice = tumor_ratio > min_tumor_ratio
                    is_boundary_slice = 0 < tumor_ratio <= min_tumor_ratio  # 边界切片
                    
                    # 决定是否包含此切片
                    include_slice = False
                    slice_type = 'normal'
                    
                    if is_tumor_slice:
                        # 包含所有明显肿瘤切片
                        include_slice = True
                        tumor_slice_count += 1
                        slice_type = 'tumor'
                    elif is_boundary_slice:
                        # 包含所有边界切片（重要的解剖结构）
                        include_slice = True
                        boundary_slice_count += 1
                        slice_type = 'boundary'
                    else:
                        # 随机包含更多正常切片
                        if np.random.random() < normal_slice_ratio:
                            include_slice = True
                            normal_slice_count += 1
                            slice_type = 'normal'
                    
                    if include_slice:
                        self.data_slices.append({
                            'fla_slice': fla_slice.copy(),
                            'seg_slice': seg_slice.copy(),
                            'patient_id': patient_id,
                            'slice_idx': slice_idx,
                            'tumor_ratio': tumor_ratio,
                            'slice_type': slice_type,
                            'volume_mean': volume_mean,
                            'volume_std': volume_std
                        })
            
            except Exception as e:
                print(f"Error loading {fla_path}: {e}")
                continue
        
        print(f"\nEnhanced Dataset Statistics:")
        print(f"Total valid slices: {len(self.data_slices)}")
        print(f"Tumor slices: {tumor_slice_count}")
        print(f"Boundary slices: {boundary_slice_count}")
        print(f"Normal slices: {normal_slice_count}")
        print(f"Tumor/Boundary/Normal ratio: {tumor_slice_count}:{boundary_slice_count}:{normal_slice_count}")
    
    def __len__(self):
        return len(self.data_slices)
    
    def __getitem__(self, idx):
        slice_data = self.data_slices[idx]
        
        fla_slice = slice_data['fla_slice'].copy()
        seg_slice = slice_data['seg_slice'].copy()
        
        # 改进的归一化：基于体积级别统计
        volume_mean = slice_data['volume_mean']
        volume_std = slice_data['volume_std']
        
        if volume_std > 1e-8:
            fla_slice = (fla_slice - volume_mean) / volume_std
        else:
            fla_slice = fla_slice - volume_mean
        
        # 稳健的值域限制
        fla_slice = np.clip(fla_slice, -4, 4)
        
        # 数据增强（训练时随机应用）
        if self.transform and np.random.random() < self.augment_prob:
            fla_slice, seg_slice = self.apply_augmentation(fla_slice, seg_slice)
        
        # 转换为二值分割掩码
        seg_slice = (seg_slice > 0).astype(np.float32)
        
        # 转换为tensor并添加通道维度
        fla_slice = torch.FloatTensor(fla_slice).unsqueeze(0)
        seg_slice = torch.FloatTensor(seg_slice)
        
        return fla_slice, seg_slice
    
    def apply_augmentation(self, image, mask):
        """应用数据增强"""
        # 随机旋转
        if np.random.random() < 0.5:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
        
        # 随机翻转
        if np.random.random() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        # 随机噪声
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.1, image.shape)
            image = image + noise
        
        return image, mask

class SelfAttention(nn.Module):
    """简化的自注意力机制，不依赖额外库"""
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
        # Self-attention
        x = x + self.attn(self.norm1(x))
        # Feed forward
        x = x + self.mlp(self.norm2(x))
        return x

class TransXAI_UNet(nn.Module):
    """
    基于TransXAI设计的混合CNN-Transformer架构 - Logits输出版本
    """
    def __init__(self, in_channels=1, out_channels=1, features=[48, 96, 192, 384]):
        super(TransXAI_UNet, self).__init__()
        
        # CNN特征提取器（保持局部特征提取能力）
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        # 下采样路径 - CNN编码器
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for i, feature in enumerate(features):
            if i == 0:
                continue  # 已经处理过第一层
            self.downs.append(self.double_conv(features[i-1], feature))
        
        # 瓶颈层
        self.bottleneck = self.double_conv(features[-1], features[-1]*2)
        
        # Transformer编码器层（在瓶颈处）
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(features[-1]*2, num_heads=8) for _ in range(4)
        ])
        
        # 位置编码（简化版）
        self.pos_embed_conv = nn.Conv2d(features[-1]*2, features[-1]*2, 1)
        
        # 上采样路径 - 修复版本
        self.ups = nn.ModuleList()
        up_features = list(reversed(features))  # [384, 192, 96, 48]
        
        for i in range(len(features)):
            if i == 0:
                # 从瓶颈层(768) -> 384
                self.ups.append(nn.ConvTranspose2d(features[-1]*2, up_features[i], kernel_size=2, stride=2))
                self.ups.append(
                    nn.Sequential(
                        self.double_conv(up_features[i]*2, up_features[i]),
                        SpatialAttention(),
                        ChannelAttention(up_features[i])
                    )
                )
            else:
                # 384->192, 192->96, 96->48
                self.ups.append(nn.ConvTranspose2d(up_features[i-1], up_features[i], kernel_size=2, stride=2))
                if i < len(features) - 1:  # 不在最后一层添加attention
                    self.ups.append(
                        nn.Sequential(
                            self.double_conv(up_features[i]*2, up_features[i]),
                            SpatialAttention(),
                            ChannelAttention(up_features[i])
                        )
                    )
                else:
                    self.ups.append(self.double_conv(up_features[i]*2, up_features[i]))
        
        # 最终分类层 - 移除Sigmoid，直接输出logits
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0]//2, 3, padding=1),
            nn.BatchNorm2d(features[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0]//2, out_channels, 1)
            # 移除 nn.Sigmoid() - 直接输出logits
        )
        
        # 深度监督输出 - 同样输出logits
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
        x_flat = x.flatten(2).transpose(1, 2)  # B, HW, C
        
        for transformer in self.transformer_layers:
            x_flat = transformer(x_flat, H, W)
        
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        skip_connections = skip_connections[::-1]
        deep_outputs = []
        
        # 上采样路径 - 修复版本
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # 反卷积
            skip_connection = skip_connections[idx//2]
            
            # 处理尺寸不匹配
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)  # 卷积块
            
            # 深度监督 - 修复索引匹配
            if idx//2 < len(self.deep_supervision):
                deep_out = self.deep_supervision[idx//2](x)
                deep_outputs.append(F.interpolate(deep_out, size=(240, 240), mode='bilinear', align_corners=False))
        
        # 最终输出 - 直接返回logits
        final_output = self.final_conv(x)
        
        if self.training and deep_outputs:
            return final_output, deep_outputs  # 都是logits
        else:
            return final_output  # logits

def enhanced_dice_loss(pred_logits, target, smooth=1):
    """
    基于logits的增强Dice损失 - 数值稳定版本
    Args:
        pred_logits: 模型输出的logits (未经sigmoid)
        target: 目标掩码 [0,1]
        smooth: 平滑因子
    """
    # 将logits转换为概率用于Dice计算
    pred_probs = torch.sigmoid(pred_logits)
    
    pred_probs = pred_probs.contiguous()
    target = target.contiguous()
    
    # 计算Dice损失
    pred_flat = pred_probs.view(pred_probs.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2. * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    
    # 边界增强损失（基于logits梯度）
    boundary_loss = 0
    if target_flat.sum() > 0:
        # 使用logits的梯度信息增强边界
        pred_grad = torch.abs(pred_logits.view(pred_logits.size(0), -1))
        boundary_weight = pred_grad.mean(dim=1) * 0.05  # 降低权重
        boundary_loss = boundary_weight.mean()
    
    return (1 - dice.mean()) + boundary_loss

def tversky_loss(pred_logits, target, alpha=0.3, beta=0.7, smooth=1):
    """
    基于logits的Tversky损失 - 对不平衡数据更有效
    Args:
        pred_logits: 模型输出的logits
        target: 目标掩码
        alpha: False Positive权重
        beta: False Negative权重
    """
    pred_probs = torch.sigmoid(pred_logits)
    
    pred_flat = pred_probs.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    TP = (pred_flat * target_flat).sum()
    FP = ((1 - target_flat) * pred_flat).sum()
    FN = (target_flat * (1 - pred_flat)).sum()
    
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    
    return 1 - tversky

def focal_loss(pred_logits, target, alpha=0.25, gamma=2):
    """
    基于logits的Focal Loss - 处理困难样本
    """
    ce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

def combined_loss(pred_logits, target, loss_weights=None):
    """
    组合损失函数 - 数值稳定版本
    Args:
        pred_logits: 模型logits输出
        target: 目标掩码
        loss_weights: 损失权重字典 {'bce': 1.0, 'dice': 1.0, 'tversky': 0.5, 'focal': 0.3}
    """
    if loss_weights is None:
        loss_weights = {'bce': 1.0, 'dice': 1.0, 'tversky': 0.5, 'focal': 0.2}
    
    total_loss = 0
    loss_components = {}
    
    # BCE with Logits Loss - 数值稳定
    if loss_weights.get('bce', 0) > 0:
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target)
        total_loss += loss_weights['bce'] * bce_loss
        loss_components['bce'] = bce_loss.item()
    
    # Enhanced Dice Loss
    if loss_weights.get('dice', 0) > 0:
        dice_loss = enhanced_dice_loss(pred_logits, target)
        total_loss += loss_weights['dice'] * dice_loss
        loss_components['dice'] = dice_loss.item()
    
    # Tversky Loss - 对不平衡有帮助
    if loss_weights.get('tversky', 0) > 0:
        tversky_loss_val = tversky_loss(pred_logits, target)
        total_loss += loss_weights['tversky'] * tversky_loss_val
        loss_components['tversky'] = tversky_loss_val.item()
    
    # Focal Loss - 关注困难样本
    if loss_weights.get('focal', 0) > 0:
        focal_loss_val = focal_loss(pred_logits, target)
        total_loss += loss_weights['focal'] * focal_loss_val
        loss_components['focal'] = focal_loss_val.item()
    
    return total_loss, loss_components

def enhanced_train_model():
    """
    基于logits的数值稳定训练函数 - 修复Windows多进程问题
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware-Optimized training on device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"GPU Memory: {gpu_memory} GB")
        
        # 针对7GB VRAM的CUDA优化设置
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # 内存管理优化
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory_stats'):
            print(f"Initial GPU memory usage: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    # 加载数据
    data_pairs = load_data_pairs('dataset_segmentation')
    
    if len(data_pairs) == 0:
        print("❌ No data pairs found!")
        return
    
    # 优化数据集划分 - 20%验证集
    train_pairs, val_pairs = train_test_split(data_pairs, test_size=0.2, random_state=42)
    print(f"📊 Hardware-optimized split: {len(train_pairs)} training, {len(val_pairs)} validation volumes")
    
    # 针对7GB VRAM优化的数据集参数
    print(f"\n{'='*60}")
    print("🚀 CREATING HARDWARE-OPTIMIZED TRAINING DATASET")
    print(f"{'='*60}")
    
    train_dataset = EnhancedBrainTumorDataset(
        train_pairs, 
        min_tumor_ratio=0.002,
        normal_slice_ratio=0.10,  # 进一步降低到10%以节省内存
        use_multimodal=True,
        augment_prob=0.3
    )
    
    print(f"\n{'='*60}")
    print("📋 CREATING HARDWARE-OPTIMIZED VALIDATION DATASET") 
    print(f"{'='*60}")
    
    val_dataset = EnhancedBrainTumorDataset(
        val_pairs,
        min_tumor_ratio=0.002,
        normal_slice_ratio=0.10,  # 进一步降低到10%
        use_multimodal=True,
        augment_prob=0.0
    )
    
    # Windows多进程修复 - 使用单进程或减少workers
    batch_size = 12  # 进一步降低批次大小
    num_workers = 0   # Windows下使用单进程避免内存问题
    
    print(f"\n🔧 Windows-Optimized Configuration:")
    print(f"   Training slices: {len(train_dataset):,}")
    print(f"   Validation slices: {len(val_dataset):,}")
    print(f"   Batch size: {batch_size} (Windows optimized)")
    print(f"   Workers: {num_workers} (single-process for Windows stability)")
    
    # 数据加载器 - Windows兼容配置
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,  # 单进程
        pin_memory=False,  # Windows下关闭pin_memory
        persistent_workers=False,  # 关闭持久workers
        drop_last=True  # 丢弃最后一个不完整batch
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,  # 单进程
        pin_memory=False,  # Windows下关闭pin_memory
        persistent_workers=False,  # 关闭持久workers
        drop_last=False
    )
    
    # 针对内存优化的模型配置
    optimized_features = [24, 48, 96, 192]  # 进一步减少特征数
    model = TransXAI_UNet(in_channels=1, out_channels=1, features=optimized_features).to(device)
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024**2
    
    print(f"🧠 Memory-optimized model:")
    print(f"   Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"   Model size: {model_size_mb:.1f}MB")
    print(f"   Features: {optimized_features}")
    
    # 检查初始VRAM使用
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        model_vram = torch.cuda.memory_allocated() / 1024**3
        print(f"   Model VRAM usage: {model_vram:.2f}GB")
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=2e-4,  # 稍微提高学习率补偿小batch size
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-4,
        epochs=50,  # 增加epochs补偿小batch size
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=10000
    )
    
    # 早停设置
    early_stopping = EnhancedEarlyStopping(
        patience=15,  # 增加patience
        min_delta=0.0001,
        adaptive_patience=True,
        verbose=True
    )
    
    # 训练参数
    num_epochs = 50  # 增加epochs
    best_dice = 0.0
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    val_dices = []
    learning_rates = []
    gpu_memory_usage = []
    
    print(f"\n{'='*80}")
    print("🎯 STARTING WINDOWS-OPTIMIZED TRAINING")
    print(f"{'='*80}")
    print(f"🔥 Max epochs: {num_epochs}")
    print(f"📚 Training batches: {len(train_loader)}")
    print(f"🧪 Validation batches: {len(val_loader)}")
    
    # 混合精度训练设置
    from torch.cuda.amp import autocast, GradScaler
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    if use_amp:
        print("⚡ Mixed precision training enabled for memory optimization")
    
    # 损失权重配置
    loss_weights = {
        'bce': 1.0,
        'dice': 1.5,  # 提高Dice权重
        'tversky': 0.8,
        'focal': 0.2  # 降低focal权重
    }
    
    print(f"🎯 Loss Configuration: {loss_weights}")
    
    # 添加手动垃圾回收
    import gc
    
    for epoch in range(num_epochs):
        # 每个epoch开始时清理内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        epoch_loss_components = {'bce': 0, 'dice': 0, 'tversky': 0, 'focal': 0}
        
        train_pbar = tqdm(train_loader, desc=f'🚂 Epoch {epoch+1:3d}/{num_epochs} [Train]', 
                         leave=False, dynamic_ncols=True)
        
        for batch_idx, (images, masks) in enumerate(train_pbar):
            try:
                images, masks = images.to(device, non_blocking=False), masks.to(device, non_blocking=False)
                
                optimizer.zero_grad(set_to_none=True)
                
                if use_amp:
                    with autocast():
                        outputs = model(images)
                        if isinstance(outputs, tuple):
                            main_logits, deep_logits_list = outputs
                            main_logits = main_logits.squeeze(1)
                            
                            main_loss, main_components = combined_loss(main_logits, masks, loss_weights)
                            
                            # 简化深度监督以节省内存
                            deep_loss = 0
                            if len(deep_logits_list) > 0:  # 只使用第一个深度输出
                                deep_logits = deep_logits_list[0].squeeze(1)
                                if deep_logits.shape != masks.shape:
                                    deep_logits = F.interpolate(
                                        deep_logits.unsqueeze(1), 
                                        size=masks.shape[-2:], 
                                        mode='bilinear', 
                                        align_corners=False
                                    ).squeeze(1)
                                
                                deep_loss_val, _ = combined_loss(deep_logits, masks, {
                                    'bce': 0.5, 'dice': 0.5, 'tversky': 0.0, 'focal': 0.0  # 简化深度损失
                                })
                                deep_loss = deep_loss_val * 0.2  # 降低深度监督权重
                            
                            total_loss = main_loss + deep_loss
                            logits_for_metrics = main_logits
                        else:
                            logits_for_metrics = outputs.squeeze(1)
                            total_loss, main_components = combined_loss(logits_for_metrics, masks, loss_weights)
                    
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        main_logits, deep_logits_list = outputs
                        main_logits = main_logits.squeeze(1)
                        
                        main_loss, main_components = combined_loss(main_logits, masks, loss_weights)
                        
                        # 简化深度监督
                        deep_loss = 0
                        if len(deep_logits_list) > 0:
                            deep_logits = deep_logits_list[0].squeeze(1)
                            if deep_logits.shape != masks.shape:
                                deep_logits = F.interpolate(
                                    deep_logits.unsqueeze(1), 
                                    size=masks.shape[-2:], 
                                    mode='bilinear', 
                                    align_corners=False
                                ).squeeze(1)
                        
                            deep_loss_val, _ = combined_loss(deep_logits, masks, {
                                'bce': 0.5, 'dice': 0.5, 'tversky': 0.0, 'focal': 0.0
                            })
                            deep_loss = deep_loss_val * 0.2
                        
                        total_loss = main_loss + deep_loss
                        logits_for_metrics = main_logits
                    else:
                        logits_for_metrics = outputs.squeeze(1)
                        total_loss, main_components = combined_loss(logits_for_metrics, masks, loss_weights)
                
                scheduler.step()
                
                # 计算指标
                with torch.no_grad():
                    probs_for_metrics = torch.sigmoid(logits_for_metrics)
                    dice = dice_coefficient(probs_for_metrics, masks)
                    train_dice += dice.item()
                    
                    for key, value in main_components.items():
                        epoch_loss_components[key] += value
            
                train_loss += total_loss.item()
                
                # 更新进度条
                current_lr = optimizer.param_groups[0]['lr']
                train_pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Dice': f'{dice.item():.4f}',
                    'LR': f'{current_lr:.2e}',
                    'VRAM': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB' if device.type == 'cuda' else 'N/A'
                })
                
                # 更频繁的内存清理
                if batch_idx % 10 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # 删除中间变量
                del images, masks, outputs, total_loss
                if 'logits_for_metrics' in locals():
                    del logits_for_metrics
                if 'probs_for_metrics' in locals():
                    del probs_for_metrics
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️  CUDA OOM at batch {batch_idx}, skipping...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'🧪 Epoch {epoch+1:3d}/{num_epochs} [Val]  ', 
                           leave=False, dynamic_ncols=True)
            
            for images, masks in val_pbar:
                try:
                    images, masks = images.to(device, non_blocking=False), masks.to(device, non_blocking=False)
                    
                    if use_amp:
                        with autocast():
                            outputs = model(images)
                            if isinstance(outputs, tuple):
                                logits = outputs[0].squeeze(1)
                            else:
                                logits = outputs.squeeze(1)
                        
                            loss, loss_components = combined_loss(logits, masks, loss_weights)
                    else:
                        outputs = model(images)
                        if isinstance(outputs, tuple):
                            logits = outputs[0].squeeze(1)
                        else:
                            logits = outputs.squeeze(1)
                        
                        loss, loss_components = combined_loss(logits, masks, loss_weights)
                    
                    probs = torch.sigmoid(logits)
                    dice = dice_coefficient(probs, masks)
                    
                    val_loss += loss.item()
                    val_dice += dice.item()
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}', 
                        'Dice': f'{dice.item():.4f}',
                        'VRAM': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB' if device.type == 'cuda' else 'N/A'
                    })
                    
                    # 删除变量
                    del images, masks, outputs, logits, probs, loss
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"⚠️  CUDA OOM in validation, skipping batch...")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        # 计算epoch平均值
        if len(train_loader) > 0:
            train_loss /= len(train_loader)
            train_dice /= len(train_loader)
            
            for key in epoch_loss_components:
                epoch_loss_components[key] /= len(train_loader)
        
        if len(val_loader) > 0:
            val_loss /= len(val_loader)
            val_dice /= len(val_loader)
        
        # 更新历史记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        if device.type == 'cuda':
            gpu_memory_usage.append(torch.cuda.memory_allocated() / 1024**3)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印epoch结果
        print(f'📊 Epoch {epoch+1:3d}/{num_epochs}:')
        print(f'   🚂 Train: Loss={train_loss:.4f}, Dice={train_dice:.4f}')
        print(f'   🧪 Val:   Loss={val_loss:.4f}, Dice={val_dice:.4f}')
        print(f'   ⚙️  LR={current_lr:.2e}')
        if device.type == 'cuda':
            print(f'   💾 VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB')
        
        # 更新最佳分数
        if val_dice > best_dice:
            best_dice = val_dice
        
        # 早停检查
        early_stopping(val_dice, model, epoch + 1, train_loss)
        
        if early_stopping.early_stop:
            print(f"\n{'='*80}")
            print("⏹️  EARLY STOPPING TRIGGERED")
            print(f"{'='*80}")
            print(f"🏆 Best Dice: {early_stopping.best_score:.6f} @ epoch {early_stopping.best_epoch}")
            
            early_stopping.restore_best(model)
            best_dice = early_stopping.best_score
            break
        
        # 每个epoch结束时清理内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    # 保存最终模型
    final_model_path = os.path.join(CURRENT_DIR, 'best_brain_tumor_model.pth')
    
    model_info = {
        'epoch': early_stopping.best_epoch if early_stopping.early_stop else epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dice': best_dice,
        'model_architecture': 'TransXAI_UNet_Windows_Optimized',
        'output_type': 'logits',
        'features': optimized_features,
        'hardware_config': {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'mixed_precision': use_amp,
            'target_platform': 'Windows'
        }
    }
    
    torch.save(model_info, final_model_path)
    
    print(f'\n{"="*80}')
    print('🎉 WINDOWS-OPTIMIZED TRAINING COMPLETED!')
    print(f'{"="*80}')
    print(f'🏆 Final Best Dice Score: {best_dice:.6f}')
    print(f'💾 Model Saved: {final_model_path}')
    print(f'{"="*80}')

def check_hardware_compatibility():
    """检查硬件兼容性"""
    print("\n" + "="*80)
    print("WINDOWS OPTIMIZATION INFORMATION")
    print("="*80)
    print("🖥️  Optimized for: Windows + RTX 4060 Laptop GPU")
    print("🔧 Key optimizations:")
    print("   • Single-process data loading (num_workers=0)")
    print("   • Reduced model features: [24, 48, 96, 192]")
    print("   • Batch size: 12 (memory safe)")
    print("   • Frequent memory cleanup")
    print("   • Exception handling for CUDA OOM")
    print("="*80)

if __name__ == "__main__":
    print("🔧 Starting Windows-Optimized Training...")
    print("✨ Windows-specific improvements:")
    print("   • Single-process data loading")
    print("   • Memory-optimized configuration")
    print("   • Exception handling for stability")
    print("   • Reduced model complexity")
    print()
    
    enhanced_train_model()
    check_hardware_compatibility()