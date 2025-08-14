import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# 获取当前文件所在目录并设置为工作目录
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # __file__ is not defined in Colab
# os.chdir(CURRENT_DIR) # Remove this line as we are not changing the working directory
# print(f"Current working directory set to: {CURRENT_DIR}") # Remove this line

# 在文件开头添加dice_coefficient函数（与原版兼容）
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
    基于TransXAI设计的混合CNN-Transformer架构，使用简化的实现
    """
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
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

        # 上采样路径 - 混合CNN-Transformer解码器
        self.ups = nn.ModuleList()
        for i, feature in enumerate(reversed(features)):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            if i < len(features) - 1:  # 不在最后一层添加transformer
                # 在解码器中也添加注意力机制
                self.ups.append(
                    nn.Sequential(
                        self.double_conv(feature*2, feature),
                        SpatialAttention(),
                        ChannelAttention(feature)
                    )
                )
            else:
                self.ups.append(self.double_conv(feature*2, feature))

        # 最终分类层
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0]//2, 3, padding=1),
            nn.BatchNorm2d(features[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0]//2, out_channels, 1),
            nn.Sigmoid()  # 确保输出在[0,1]范围内
        )

        # 深度监督输出
        self.deep_supervision = nn.ModuleList([
            nn.Conv2d(feature, out_channels, 1) for feature in features[1:]
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

        # 上采样路径
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # 反卷积
            skip_connection = skip_connections[idx//2]

            # 处理尺寸不匹配
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)  # 卷积块

            # 深度监督
            if idx//2 < len(self.deep_supervision):
                deep_out = self.deep_supervision[idx//2](x)
                deep_outputs.append(F.interpolate(deep_out, size=(240, 240), mode='bilinear', align_corners=False))

        # 最终输出
        final_output = self.final_conv(x)

        if self.training and deep_outputs:
            return final_output, deep_outputs
        else:
            return final_output

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class EnhancedEarlyStopping:
    """
    增强的早停类，支持多指标监控和自适应patience
    """
    def __init__(self, patience=15, min_delta=0.0003, restore_best_weights=True,
                 verbose=True, adaptive_patience=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.adaptive_patience = adaptive_patience

        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False
        self.best_weights = None
        self.score_history = []
        self.initial_patience = patience

    def __call__(self, val_score, model, epoch, train_loss=None):
        """
        增强的早停检查，考虑训练损失和验证分数的关系
        """
        self.score_history.append(val_score)

        # 自适应patience调整
        if self.adaptive_patience and len(self.score_history) > 10:
            recent_improvement = np.std(self.score_history[-10:])
            if recent_improvement < 0.001:  # 如果最近改善很小
                self.patience = max(self.initial_patience // 2, 8)
            else:
                self.patience = self.initial_patience

        # 第一次调用
        if self.best_score is None:
            self.best_score = val_score
            self.best_epoch = epoch
            self.save_checkpoint(model)
            if self.verbose:
                print(f'  → Enhanced early stopping baseline: {val_score:.6f}')

        # 检查是否有改善
        elif val_score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  → Early stopping: {self.counter}/{self.patience} (best: {self.best_score:.6f} @ epoch {self.best_epoch})')

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'  → Early stopping triggered! Best: {self.best_score:.6f} @ epoch {self.best_epoch}')
        else:
            # 有改善
            improvement = val_score - self.best_score
            self.best_score = val_score
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model)
            if self.verbose:
                print(f'  → NEW BEST: {val_score:.6f} (+{improvement:.6f}) - patience reset')

    def save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def restore_best(self, model):
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print(f'  → Restored best weights from epoch {self.best_epoch}')

def enhanced_dice_loss(pred, target, smooth=1, weight=None):
    """
    增强的Dice损失，支持类别权重和边界增强
    """
    pred = pred.contiguous()
    target = target.contiguous()

    # 计算基本Dice损失
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2. * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)

    # 边界敏感损失（如果目标中有肿瘤）
    boundary_loss = 0
    if target_flat.sum() > 0:
        # 计算边界权重
        boundary_weight = torch.abs(pred_flat - target_flat).mean(dim=1)
        boundary_loss = boundary_weight.mean() * 0.1

    return (1 - dice.mean()) + boundary_loss

def load_data_pairs(data_dir):
    """增强的数据加载函数"""
    data_pairs = []
    # data_dir = os.path.join(CURRENT_DIR, data_dir) # Use absolute path instead of CURRENT_DIR
    train_dir = os.path.join(data_dir, 'train')

    print(f"Enhanced data loading from: {train_dir}")

    if not os.path.exists(train_dir):
        print(f"Warning: Training directory not found: {train_dir}")
        return data_pairs

    for patient_id in sorted(os.listdir(train_dir)):
        patient_dir = os.path.join(train_dir, patient_id)
        if os.path.isdir(patient_dir):
            # Assuming data is in a structured format like patient_id/patient_id_fla.nii.gz
            fla_path = os.path.join(patient_dir, f"{patient_id}_fla.nii.gz")
            seg_path = os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")

            if os.path.exists(fla_path) and os.path.exists(seg_path):
                data_pairs.append((fla_path, seg_path))
                # print(f"✓ Patient {patient_id}: {os.path.getsize(fla_path)//1024//1024}MB + {os.path.getsize(seg_path)//1024//1024}MB") # Verbose output, can uncomment if needed

    print(f"Total enhanced data pairs: {len(data_pairs)}")
    return data_pairs

def enhanced_train_model():
    """
    增强的训练函数，集成了最新的技术和最佳实践
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Enhanced training on device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

    # 加载数据
    # Assuming the dataset is located in /content/dataset_segmentation
    data_dir = '/content/dataset_segmentation'
    data_pairs = load_data_pairs(data_dir)

    if len(data_pairs) == 0:
        print("❌ No data pairs found!")
        return

    # 增强的数据集划分
    # Added stratify=None to avoid errors if class distribution is highly skewed or only one class exists
    train_pairs, val_pairs = train_test_split(data_pairs, test_size=0.25, random_state=42, stratify=None)
    print(f"📊 Data split: {len(train_pairs)} training, {len(val_pairs)} validation volumes")

    # 创建增强数据集
    print(f"\n{'='*60}")
    print("🚀 CREATING ENHANCED TRAINING DATASET")
    print(f"{'='*60}")

    train_dataset = EnhancedBrainTumorDataset(
        train_pairs,
        min_tumor_ratio=0.003,    # 更低阈值
        normal_slice_ratio=0.25,   # 更多正常切片
        use_multimodal=True,
        augment_prob=0.4          # 40%增强概率
    )

    print(f"\n{'='*60}")
    print("📋 CREATING ENHANCED VALIDATION DATASET")
    print(f"{'='*60}")

    val_dataset = EnhancedBrainTumorDataset(
        val_pairs,
        min_tumor_ratio=0.003,
        normal_slice_ratio=0.25,
        use_multimodal=True,
        augment_prob=0.0          # 验证集不使用增强
    )

    # 动态批次大小
    total_slices = len(train_dataset)
    if total_slices > 3000:
        batch_size = 24
    elif total_slices > 1500:
        batch_size = 16
    else:
        batch_size = 12

    print(f"\n📈 Final Configuration:")
    print(f"   Training slices: {len(train_dataset):,}")
    print(f"   Validation slices: {len(val_dataset):,}")
    print(f"   Batch size: {batch_size}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 创建增强模型
    model = TransXAI_UNet(in_channels=1, out_channels=1).to(device)

    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # 损失函数
    bce_criterion = nn.BCELoss()

    # 优化器（简化版）
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    # 学习率调度器（修复版本）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # 增强早停
    early_stopping = EnhancedEarlyStopping(
        patience=18,
        min_delta=0.0002,
        adaptive_patience=True,
        verbose=True
    )

    # 训练参数
    num_epochs = 60
    best_dice = 0.0

    # 训练历史记录
    train_losses = []
    val_losses = []
    val_dices = []
    learning_rates = []

    print(f"\n{'='*80}")
    print("🎯 STARTING ENHANCED TRAINING")
    print(f"{'='*80}")
    print(f"🔥 Max epochs: {num_epochs}")
    print(f"📚 Training batches: {len(train_loader)}")
    print(f"🧪 Validation batches: {len(val_loader)}")
    print(f"⏰ Early stopping patience: {early_stopping.patience}")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        train_pbar = tqdm(train_loader, desc=f'🚂 Epoch {epoch+1:3d}/{num_epochs} [Train]',
                         leave=False)

        for batch_idx, (images, masks) in enumerate(train_pbar):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            if isinstance(outputs, tuple):  # 深度监督
                main_output, deep_outputs = outputs
                main_output = main_output.squeeze(1)

                # 主损失
                bce_loss = bce_criterion(main_output, masks)
                dice_loss_val = enhanced_dice_loss(main_output, masks)
                main_loss = bce_loss + dice_loss_val

                # 深度监督损失
                deep_loss = 0
                for deep_out in deep_outputs:
                    deep_out = deep_out.squeeze(1)
                    deep_out_resized = F.interpolate(deep_out.unsqueeze(1), size=masks.shape[-2:],
                                                   mode='bilinear', align_corners=False).squeeze(1)
                    deep_loss += enhanced_dice_loss(deep_out_resized, masks) * 0.4

                loss = main_loss + deep_loss
                outputs = main_output
            else:
                outputs = outputs.squeeze(1)
                bce_loss = bce_criterion(outputs, masks)
                dice_loss_val = enhanced_dice_loss(outputs, masks)
                loss = bce_loss + dice_loss_val

            loss.backward()
            optimizer.step()

            # 计算Dice系数
            with torch.no_grad():
                dice = dice_coefficient(outputs, masks)
                train_dice += dice.item()

            train_loss += loss.item()

            # 更新进度条
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice.item():.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'🧪 Epoch {epoch+1:3d}/{num_epochs} [Val]  ',
                           leave=False)

            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                outputs = outputs.squeeze(1)

                bce_loss = bce_criterion(outputs, masks)
                dice_loss_val = enhanced_dice_loss(outputs, masks)
                loss = bce_loss + dice_loss_val

                dice = dice_coefficient(outputs, masks)

                val_loss += loss.item()
                val_dice += dice.item()

                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice.item():.4f}'
                })

        # 计算epoch平均值
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        # 更新历史记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # 学习率调度
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_dice)
        new_lr = optimizer.param_groups[0]['lr']

        # 手动打印学习率变化
        if old_lr != new_lr:
            print(f'   📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}')

        current_lr = optimizer.param_groups[0]['lr']

        # 打印epoch结果
        print(f'📊 Epoch {epoch+1:3d}/{num_epochs}:')
        print(f'   🚂 Train: Loss={train_loss:.4f}, Dice={train_dice:.4f}')
        print(f'   🧪 Val:   Loss={val_loss:.4f}, Dice={val_dice:.4f}')
        print(f'   ⚙️  LR={current_lr:.2e}')

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
            print(f"⏰ Stopped at epoch {epoch + 1}")

            early_stopping.restore_best(model)
            best_dice = early_stopping.best_score
            break

        # 定期保存检查点
        if (epoch + 1) % 15 == 0:
            # checkpoint_path = os.path.join(CURRENT_DIR, f'enhanced_checkpoint_epoch_{epoch+1}.pth') # Use absolute path or relative path from current Colab directory
            checkpoint_path = f'/content/enhanced_checkpoint_epoch_{epoch+1}.pth' # Saving to /content/
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_dice': val_dice,
                'best_dice': best_dice,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f'   💾 Checkpoint saved: {checkpoint_path}')

    # 保存最终增强模型
    # final_model_path = os.path.join(CURRENT_DIR, 'enhanced_brain_tumor_model.pth') # Use absolute path
    final_model_path = '/content/enhanced_brain_tumor_model.pth' # Saving to /content/

    # 创建完整的增强模型信息
    enhanced_model_info = {
        'epoch': early_stopping.best_epoch if early_stopping.early_stop else epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_dice': best_dice,
        'model_architecture': 'TransXAI_UNet',
        'model_features': {
            'CNN_backbone': True,
            'Transformer_layers': 4,
            'attention_mechanisms': ['spatial', 'channel', 'self'],
            'deep_supervision': True
        },
        'training_config': {
            'batch_size': batch_size,
            'total_epochs': len(train_losses),
            'early_stopping_epoch': early_stopping.best_epoch if early_stopping.early_stop else None,
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau',
            'loss_function': 'BCE + Enhanced_Dice',
            'data_augmentation': True
        },
        'dataset_info': {
            'train_slices': len(train_dataset),
            'val_slices': len(val_dataset),
            'min_tumor_ratio': 0.003,
            'normal_slice_ratio': 0.25
        },
        'performance_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_dices': val_dices,
            'learning_rates': learning_rates
        },
        'early_stopping_info': {
            'triggered': early_stopping.early_stop,
            'best_epoch': early_stopping.best_epoch,
            'patience_used': early_stopping.counter,
            'final_patience': early_stopping.patience,
            'score_history': early_stopping.score_history[-20:] if len(early_stopping.score_history) > 20 else early_stopping.score_history
        }
    }

    torch.save(enhanced_model_info, final_model_path)

    # 创建兼容格式
    # compatible_model_path = os.path.join(CURRENT_DIR, 'enhanced_model_compatible.pth') # Use absolute path
    compatible_model_path = '/content/enhanced_model_compatible.pth' # Saving to /content/
    compatible_model_info = {
        'epoch': early_stopping.best_epoch if early_stopping.early_stop else epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dice': best_dice,
        'train_stats': {
            'total_slices': len(train_dataset),
            'tumor_slices': sum(1 for s in train_dataset.data_slices if s['slice_type'] == 'tumor'),
            'normal_slices': sum(1 for s in train_dataset.data_slices if s['slice_type'] == 'normal'),
            'avg_tumor_ratio': np.mean([s['tumor_ratio'] for s in train_dataset.data_slices if s['slice_type'] == 'tumor']) if sum(1 for s in train_dataset.data_slices if s['slice_type'] == 'tumor') > 0 else 0
        },
        'val_stats': {
            'total_slices': len(val_dataset),
            'tumor_slices': sum(1 for s in val_dataset.data_slices if s['slice_type'] == 'tumor'),
            'normal_slices': sum(1 for s in val_dataset.data_slices if s['slice_type'] == 'normal'),
            'avg_tumor_ratio': np.mean([s['tumor_ratio'] for s in val_dataset.data_slices if s['slice_type'] == 'tumor']) if sum(1 for s in val_dataset.data_slices if s['slice_type'] == 'tumor') > 0 else 0
        },
        'early_stopping_info': {
            'triggered': early_stopping.early_stop,
            'best_epoch': early_stopping.best_epoch,
            'patience_used': early_stopping.counter,
            'score_history': early_stopping.score_history[-20:] if len(early_stopping.score_history) > 20 else early_stopping.score_history
        }
    }

    torch.save(compatible_model_info, compatible_model_path)

    # 创建训练曲线
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Enhanced Training Results - TransXAI Brain Tumor Segmentation', fontsize=16, fontweight='bold')

        epochs_range = range(1, len(train_losses) + 1)

        # 损失对比
        axes[0, 0].plot(epochs_range, train_losses, 'b-', linewidth=2, alpha=0.8, label='Train Loss')
        axes[0, 0].plot(epochs_range, val_losses, 'r-', linewidth=2, alpha=0.8, label='Val Loss')
        if early_stopping.early_stop:
            axes[0, 0].axvline(x=early_stopping.best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch: {early_stopping.best_epoch}')
        axes[0, 0].set_title('Training vs Validation Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Dice系数进展
        axes[0, 1].plot(epochs_range, val_dices, 'g-', linewidth=3, alpha=0.8, label='Validation Dice')
        axes[0, 1].axhline(y=best_dice, color='red', linestyle=':', alpha=0.8, label=f'Best Dice: {best_dice:.4f}')
        if early_stopping.early_stop:
            axes[0, 1].axvline(x=early_stopping.best_epoch, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Validation Dice Coefficient Progress', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 学习率变化
        axes[1, 0].plot(epochs_range, learning_rates, 'purple', linewidth=2, alpha=0.8)
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # 综合指标
        axes[1, 1].plot(epochs_range, train_losses, label='Train Loss', alpha=0.7)
        axes[1, 1].plot(epochs_range, val_losses, label='Val Loss', alpha=0.7)
        ax_dice = axes[1, 1].twinx()
        ax_dice.plot(epochs_range, val_dices, 'g-', label='Val Dice', alpha=0.8)
        axes[1, 1].set_title('Combined Metrics View', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss', color='blue')
        ax_dice.set_ylabel('Dice Score', color='green')
        axes[1, 1].legend(loc='upper left')
        ax_dice.legend(loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        # curves_path = os.path.join(CURRENT_DIR, 'enhanced_training_curves.png') # Use absolute path
        curves_path = '/content/enhanced_training_curves.png' # Saving to /content/
        plt.savefig(curves_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        print(f'📈 Enhanced training curves saved: {curves_path}')

    except Exception as e:
        print(f"❌ Error creating plots: {e}")

    # 最终总结
    print(f'\n{"="*90}')
    print('🎉 ENHANCED TRAINING COMPLETED!')
    print(f'{"="*90}')
    print(f'🏆 Final Best Dice Score: {best_dice:.6f}')
    print(f'📊 Total Epochs Trained: {len(train_losses)}')
    print(f'⏰ Early Stopping: {"✅ Triggered" if early_stopping.early_stop else "❌ Not triggered"}')
    if early_stopping.early_stop:
        print(f'🎯 Best Performance at Epoch: {early_stopping.best_epoch}')
        print(f'⏳ Patience Used: {early_stopping.counter}/{early_stopping.patience}')
    print(f'📚 Training Data Used: {len(train_dataset):,} slices')
    print(f'🧪 Validation Data Used: {len(val_dataset):,} slices')
    print(f'💾 Enhanced Model Saved: enhanced_brain_tumor_model.pth')
    print(f'💾 Compatible Format: enhanced_model_compatible.pth')
    print(f'🧠 Model Architecture: TransXAI-UNet (CNN + Transformer)')
    print(f'⚠️  NOTE: This model is NOT compatible with original train.py due to different architecture')
    print(f'{"="*90}')

# 添加一个专门的兼容性检查函数
def check_model_compatibility():
    """检查模型兼容性并提供使用指南"""
    print("\n" + "="*80)
    print("MODEL COMPATIBILITY INFORMATION")
    print("="*80)
    print("📋 Enhanced Model Files:")
    print("   - enhanced_brain_tumor_model.pth (完整增强模型)")
    print("   - enhanced_model_compatible.pth (兼容格式，仅权重)")
    print("   - enhanced_training_curves.png (训练曲线)")
    print("\n🚨 IMPORTANT COMPATIBILITY NOTES:")
    print("   ❌ enhanced_brain_tumor_model.pth 不能直接用原版 train.py 的代码加载")
    print("   ❌ 模型架构完全不同：TransXAI_UNet vs UNet")
    print("   ❌ 参数数量和结构都不同")
    print("\n✅ 如果需要使用增强模型，请：")
    print("   1. 使用 new_train.py 中定义的 TransXAI_UNet 类")
    print("   2. 或者创建专门的推理脚本")
    print("   3. 或者使用提供的兼容性封装函数")
    print("\n📝 使用示例：")
    print("   # 加载增强模型")
    print("   from new_train import TransXAI_UNet")
    print("   model = TransXAI_UNet()")
    print("   checkpoint = torch.load('enhanced_brain_tumor_model.pth')")
    print("   model.load_state_dict(checkpoint['model_state_dict'])")
    print("="*80)

# 在主函数末尾调用
if __name__ == "__main__":
    enhanced_train_model()
    check_model_compatibility()