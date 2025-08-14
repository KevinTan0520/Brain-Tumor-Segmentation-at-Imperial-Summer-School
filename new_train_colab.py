import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import os
import gc
import psutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import sys

# Colab环境检测和设置
def setup_colab_environment():
    """设置Colab环境"""
    if 'google.colab' in sys.modules:
        print("🔧 Running in Google Colab")
        from google.colab import drive
        try:
            drive.mount('/content/drive')
            print("✅ Google Drive mounted successfully")
        except:
            print("⚠️ Google Drive mount failed or already mounted")
        
        # 创建必要目录
        os.makedirs('/content/checkpoints', exist_ok=True)
        os.makedirs('/content/models', exist_ok=True)
        
        return True
    return False

# 内存监控函数
def get_memory_info():
    """获取内存使用信息"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_cached = torch.cuda.memory_reserved() / 1024**3
    else:
        gpu_memory = gpu_allocated = gpu_cached = 0
    
    ram_info = psutil.virtual_memory()
    ram_total = ram_info.total / 1024**3
    ram_used = ram_info.used / 1024**3
    ram_percent = ram_info.percent
    
    return {
        'ram_total': ram_total,
        'ram_used': ram_used, 
        'ram_percent': ram_percent,
        'gpu_total': gpu_memory,
        'gpu_allocated': gpu_allocated,
        'gpu_cached': gpu_cached
    }

def print_memory_usage(stage=""):
    """打印内存使用情况"""
    info = get_memory_info()
    print(f"💾 {stage} Memory Usage:")
    print(f"   RAM: {info['ram_used']:.1f}/{info['ram_total']:.1f}GB ({info['ram_percent']:.1f}%)")
    if info['gpu_total'] > 0:
        print(f"   GPU: {info['gpu_allocated']:.1f}GB allocated, {info['gpu_cached']:.1f}GB cached")

def cleanup_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Dice系数计算函数
def dice_coefficient(pred, target, smooth=1):
    """计算Dice系数"""
    pred = (pred > 0.5).float()
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

class MemoryEfficientBrainTumorDataset(Dataset):
    """
    内存优化版本的脑肿瘤数据集 - 适用于Colab 12GB RAM
    采用懒加载策略，只在需要时加载数据
    """
    def __init__(self, data_pairs, min_tumor_ratio=0.005, normal_slice_ratio=0.15, 
                 max_slices_per_patient=20, augment_prob=0.3):
        """
        Args:
            data_pairs: (fla_path, seg_path) 对的列表
            min_tumor_ratio: 肿瘤像素最小比例阈值
            normal_slice_ratio: 正常切片包含比例
            max_slices_per_patient: 每个患者最大切片数（内存控制）
            augment_prob: 数据增强概率
        """
        self.data_pairs = data_pairs
        self.min_tumor_ratio = min_tumor_ratio
        self.normal_slice_ratio = normal_slice_ratio
        self.max_slices_per_patient = max_slices_per_patient
        self.augment_prob = augment_prob
        
        print(f"🔄 Creating Memory-Efficient Dataset...")
        print(f"   Max slices per patient: {max_slices_per_patient}")
        print(f"   Min tumor ratio: {min_tumor_ratio}")
        print(f"   Normal slice ratio: {normal_slice_ratio}")
        
        # 预扫描数据，只存储切片索引而不加载实际数据
        self.slice_indices = []
        self._prescan_data()
        
        print(f"✅ Dataset created with {len(self.slice_indices)} slices")
        print_memory_usage("Dataset Creation")
    
    def _prescan_data(self):
        """预扫描数据，构建切片索引"""
        tumor_count = 0
        normal_count = 0
        boundary_count = 0
        
        for patient_idx, (fla_path, seg_path) in enumerate(tqdm(self.data_pairs, desc="Pre-scanning data")):
            try:
                # 只加载分割掩码来分析切片
                seg_img = nib.load(seg_path)
                seg_data = seg_img.get_fdata()
                
                patient_slices = []
                
                # 分析每个切片
                for slice_idx in range(seg_data.shape[2]):
                    seg_slice = seg_data[:, :, slice_idx]
                    
                    if seg_slice.sum() == 0:
                        continue
                    
                    total_pixels = seg_slice.shape[0] * seg_slice.shape[1]
                    tumor_pixels = seg_slice.sum()
                    tumor_ratio = tumor_pixels / total_pixels
                    
                    slice_type = 'normal'
                    include = False
                    
                    if tumor_ratio > self.min_tumor_ratio:
                        slice_type = 'tumor'
                        include = True
                        tumor_count += 1
                    elif tumor_ratio > 0:
                        slice_type = 'boundary'
                        include = True
                        boundary_count += 1
                    elif np.random.random() < self.normal_slice_ratio:
                        slice_type = 'normal'
                        include = True
                        normal_count += 1
                    
                    if include:
                        patient_slices.append({
                            'patient_idx': patient_idx,
                            'fla_path': fla_path,
                            'seg_path': seg_path,
                            'slice_idx': slice_idx,
                            'tumor_ratio': tumor_ratio,
                            'slice_type': slice_type
                        })
                
                # 限制每个患者的切片数量
                if len(patient_slices) > self.max_slices_per_patient:
                    # 优先保留肿瘤切片
                    tumor_slices = [s for s in patient_slices if s['slice_type'] == 'tumor']
                    boundary_slices = [s for s in patient_slices if s['slice_type'] == 'boundary']
                    normal_slices = [s for s in patient_slices if s['slice_type'] == 'normal']
                    
                    selected_slices = tumor_slices[:self.max_slices_per_patient//2]
                    remaining = self.max_slices_per_patient - len(selected_slices)
                    
                    if remaining > 0:
                        selected_slices.extend(boundary_slices[:remaining//2])
                        remaining = self.max_slices_per_patient - len(selected_slices)
                        
                    if remaining > 0:
                        selected_slices.extend(normal_slices[:remaining])
                    
                    patient_slices = selected_slices
                
                self.slice_indices.extend(patient_slices)
                
                # 清理内存
                del seg_img, seg_data
                cleanup_memory()
                
            except Exception as e:
                print(f"❌ Error pre-scanning {seg_path}: {e}")
                continue
        
        print(f"📊 Pre-scan Statistics:")
        print(f"   Tumor slices: {tumor_count}")
        print(f"   Boundary slices: {boundary_count}")  
        print(f"   Normal slices: {normal_count}")
    
    def __len__(self):
        return len(self.slice_indices)
    
    def __getitem__(self, idx):
        """懒加载数据 - 修复版本"""
        slice_info = self.slice_indices[idx]
        
        try:
            # 只在需要时加载数据
            fla_img = nib.load(slice_info['fla_path'])
            seg_img = nib.load(slice_info['seg_path'])
            
            fla_data = fla_img.get_fdata()
            seg_data = seg_img.get_fdata()
            
            slice_idx = slice_info['slice_idx']
            fla_slice = fla_data[:, :, slice_idx].copy()  # 立即创建副本
            seg_slice = seg_data[:, :, slice_idx].copy()  # 立即创建副本
            
            # 立即释放大数据
            del fla_img, seg_img, fla_data, seg_data
            
            # 数据预处理
            if fla_slice.max() > 0:
                mean = fla_slice.mean()
                std = fla_slice.std() + 1e-8
                fla_slice = (fla_slice - mean) / std
        
            # 确保连续的内存布局
            fla_slice = np.ascontiguousarray(fla_slice)
            seg_slice = np.ascontiguousarray(seg_slice)
            
            fla_slice = np.clip(fla_slice, -3, 3)
            
            # 数据增强
            if np.random.random() < self.augment_prob:
                fla_slice, seg_slice = self._apply_augmentation(fla_slice, seg_slice)
            
            # 转换为二值掩码
            seg_slice = (seg_slice > 0).astype(np.float32)
            
            # 确保数组是连续的，然后转换为tensor
            fla_slice = np.ascontiguousarray(fla_slice)
            seg_slice = np.ascontiguousarray(seg_slice)
            
            fla_slice = torch.FloatTensor(fla_slice).unsqueeze(0)
            seg_slice = torch.FloatTensor(seg_slice)
            
            return fla_slice, seg_slice
        
        except Exception as e:
            print(f"❌ Error loading slice {idx}: {e}")
            # 返回连续的空数据避免训练中断
            empty_image = torch.zeros(1, 240, 240)
            empty_mask = torch.zeros(240, 240)
            return empty_image, empty_mask

    def _apply_augmentation(self, image, mask):
        """轻量级数据增强 - 完全修复版本"""
        # 确保输入数组是连续的
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        
        # 随机翻转 - 直接操作避免负步长
        if np.random.random() < 0.5:
            image = image[:, ::-1].copy()  # 水平翻转
            mask = mask[:, ::-1].copy()
        
        if np.random.random() < 0.5:
            image = image[::-1, :].copy()  # 垂直翻转
            mask = mask[::-1, :].copy()
        
        # 随机旋转90度 - 使用更安全的方法
        if np.random.random() < 0.5:
            k = np.random.randint(1, 4)
            for _ in range(k):
                image = np.transpose(image)[:, ::-1].copy()
                mask = np.transpose(mask)[:, ::-1].copy()
        
        # 轻微随机噪声 (可选)
        if np.random.random() < 0.2:
            noise = np.random.normal(0, 0.05, image.shape)
            image = image + noise
        
        return image, mask

class LightweightUNet(nn.Module):
    """
    轻量级U-Net - 适用于Colab内存限制
    大幅减少参数数量以适应12GB RAM + 16GB GPU
    """
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super(LightweightUNet, self).__init__()
        
        self.features = features
        
        # 编码器
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # 第一层
        self.encoder.append(self._make_conv_block(in_channels, features[0]))
        
        # 其他编码器层
        for i in range(1, len(features)):
            self.encoder.append(self._make_conv_block(features[i-1], features[i]))
        
        # 瓶颈层
        self.bottleneck = self._make_conv_block(features[-1], features[-1] * 2)
        
        # 解码器
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for i in range(len(features)):
            if i == 0:
                self.upconvs.append(nn.ConvTranspose2d(features[-1] * 2, features[-1], 2, 2))
                self.decoder.append(self._make_conv_block(features[-1] * 2, features[-1]))
            else:
                self.upconvs.append(nn.ConvTranspose2d(features[-i], features[-i-1], 2, 2))
                self.decoder.append(self._make_conv_block(features[-i], features[-i-1]))
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0]//2, 3, padding=1),
            nn.BatchNorm2d(features[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0]//2, out_channels, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_conv_block(self, in_channels, out_channels):
        """创建卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 编码路径
        skip_connections = []
        
        for i, encoder_layer in enumerate(self.encoder):
            x = encoder_layer(x)
            skip_connections.append(x)
            if i < len(self.encoder) - 1:
                x = self.pool(x)
        
        # 瓶颈层
        x = self.pool(x)
        x = self.bottleneck(x)
        
        # 解码路径
        skip_connections = skip_connections[::-1]
        
        for i, (upconv, decoder_layer) in enumerate(zip(self.upconvs, self.decoder)):
            x = upconv(x)
            
            # 处理尺寸不匹配
            skip_connection = skip_connections[i]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([skip_connection, x], dim=1)
            x = decoder_layer(x)
        
        return self.final_conv(x)

class MemoryEfficientEarlyStopping:
    """内存优化的早停机制"""
    def __init__(self, patience=12, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False
        
    def __call__(self, val_score, model, epoch):
        if self.best_score is None:
            self.best_score = val_score
            self.best_epoch = epoch
            if self.verbose:
                print(f'  → Early stopping baseline: {val_score:.6f}')
                
        elif val_score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  → Early stopping: {self.counter}/{self.patience}')
                
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'  → Early stopping triggered!')
        else:
            improvement = val_score - self.best_score
            self.best_score = val_score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f'  → NEW BEST: {val_score:.6f} (+{improvement:.6f})')

def enhanced_dice_loss(pred, target, smooth=1):
    """增强Dice损失"""
    pred = pred.contiguous()
    target = target.contiguous()
    
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2. * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    
    return 1 - dice.mean()

def load_data_pairs(data_dir):
    """加载数据对"""
    data_pairs = []
    train_dir = os.path.join(data_dir, 'train')
    
    print(f"🔍 Loading data from: {train_dir}")
    
    if not os.path.exists(train_dir):
        print(f"❌ Training directory not found: {train_dir}")
        return data_pairs
    
    for patient_id in sorted(os.listdir(train_dir)):
        patient_dir = os.path.join(train_dir, patient_id)
        if os.path.isdir(patient_dir):
            fla_path = os.path.join(patient_dir, f"{patient_id}_fla.nii.gz")
            seg_path = os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")
            
            if os.path.exists(fla_path) and os.path.exists(seg_path):
                data_pairs.append((fla_path, seg_path))
    
    print(f"✅ Found {len(data_pairs)} data pairs")
    return data_pairs

def colab_optimized_train():
    """
    Colab优化的训练函数
    针对12GB RAM + 16GB Tesla T4优化
    """
    # 设置Colab环境
    is_colab = setup_colab_environment()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Training device: {device}")
    
    if device.type == 'cuda':
        print(f"📱 GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"💾 GPU Memory: {gpu_memory} GB")
    
    print_memory_usage("Initial")
    
    # 加载数据
    data_dir = '/content/dataset_segmentation'
    data_pairs = load_data_pairs(data_dir)
    
    if len(data_pairs) == 0:
        print("❌ No data pairs found!")
        return
    
    print(f"📊 Total patients: {len(data_pairs)}")
    
    # 数据划分
    train_pairs, val_pairs = train_test_split(data_pairs, test_size=0.2, random_state=42)
    print(f"🔄 Data split: {len(train_pairs)} train, {len(val_pairs)} validation")
    
    # 创建内存优化数据集
    print("\n" + "="*50)
    print("🚀 CREATING MEMORY-EFFICIENT DATASETS")
    print("="*50)
    
    train_dataset = MemoryEfficientBrainTumorDataset(
        train_pairs,
        max_slices_per_patient=15,  # 减少切片数
        min_tumor_ratio=0.005,
        normal_slice_ratio=0.15,
        augment_prob=0.4
    )
    
    val_dataset = MemoryEfficientBrainTumorDataset(
        val_pairs,
        max_slices_per_patient=10,  # 验证集更少切片
        min_tumor_ratio=0.005,
        normal_slice_ratio=0.15,
        augment_prob=0.0
    )
    
    print_memory_usage("Dataset Created")
    
    # 动态批次大小 - 根据数据集大小和GPU内存调整
    total_slices = len(train_dataset)
    if total_slices > 2000:
        batch_size = 8
    elif total_slices > 1000:
        batch_size = 12
    else:
        batch_size = 16
    
    print(f"\n📈 Final Configuration:")
    print(f"   Training slices: {len(train_dataset):,}")
    print(f"   Validation slices: {len(val_dataset):,}")
    print(f"   Batch size: {batch_size}")
    
    # 创建数据加载器 - Colab优化
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,  # Colab推荐使用1个worker
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=False  # 节省内存
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=False
    )
    
    # 创建轻量级模型
    model = LightweightUNet(in_channels=1, out_channels=1).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    print_memory_usage("Model Created")
    
    # 损失函数和优化器
    bce_criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=8
    )
    
    # 早停
    early_stopping = MemoryEfficientEarlyStopping(patience=15, min_delta=0.001)
    
    # 训练参数
    num_epochs = 50
    best_dice = 0.0
    
    # 训练历史
    train_losses = []
    val_losses = []
    val_dices = []
    
    print(f"\n{'='*60}")
    print("🎯 STARTING COLAB-OPTIMIZED TRAINING")
    print(f"{'='*60}")
    print(f"🔥 Max epochs: {num_epochs}")
    print(f"📚 Training batches: {len(train_loader)}")
    print(f"🧪 Validation batches: {len(val_loader)}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:2d}/{num_epochs} [Train]', leave=False)
        
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images).squeeze(1)
            
            # 损失计算
            bce_loss = bce_criterion(outputs, masks)
            dice_loss = enhanced_dice_loss(outputs, masks)
            loss = bce_loss + dice_loss
            
            loss.backward()
            optimizer.step()
            
            # 指标计算
            with torch.no_grad():
                dice = dice_coefficient(outputs, masks)
                train_dice += dice.item()
            
            train_loss += loss.item()
            
            # 更新进度条
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.3f}',
                'Dice': f'{dice.item():.3f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.1e}'
            })
            
            # 定期清理内存
            if batch_idx % 10 == 0:
                cleanup_memory()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1:2d}/{num_epochs} [Val]  ', leave=False)
            
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images).squeeze(1)
                
                bce_loss = bce_criterion(outputs, masks)
                dice_loss = enhanced_dice_loss(outputs, masks)
                loss = bce_loss + dice_loss
                
                dice = dice_coefficient(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice.item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.3f}',
                    'Dice': f'{dice.item():.3f}'
                })
        
        # 计算平均值
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        
        # 学习率调度
        scheduler.step(val_dice)
        
        # 打印结果
        print(f'📊 Epoch {epoch+1:2d}/{num_epochs}: '
              f'Train Loss={train_loss:.4f}, Dice={train_dice:.4f} | '
              f'Val Loss={val_loss:.4f}, Dice={val_dice:.4f}')
        
        # 更新最佳分数
        if val_dice > best_dice:
            best_dice = val_dice
        
        # 早停检查
        early_stopping(val_dice, model, epoch + 1)
        
        if early_stopping.early_stop:
            print(f"⏹️  Early stopping at epoch {epoch + 1}")
            print(f"🏆 Best Dice: {early_stopping.best_score:.6f} @ epoch {early_stopping.best_epoch}")
            break
        
        # 定期保存和内存清理
        if (epoch + 1) % 10 == 0:
            # 保存检查点
            checkpoint_path = f'/content/checkpoints/checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'best_dice': best_dice
            }, checkpoint_path)
            print(f'💾 Checkpoint saved: {checkpoint_path}')
            
            # 清理内存
            cleanup_memory()
            print_memory_usage(f"Epoch {epoch+1}")
    
    # 保存最终模型
    final_model_path = '/content/models/brain_tumor_model_colab.pth'
    torch.save({
        'epoch': early_stopping.best_epoch if early_stopping.early_stop else epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dice': best_dice,
        'model_architecture': 'LightweightUNet',
        'training_config': {
            'batch_size': batch_size,
            'total_epochs': len(train_losses),
            'colab_optimized': True
        }
    }, final_model_path)
    
    # 创建训练曲线
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        epochs_range = range(1, len(train_losses) + 1)
        
        # 损失曲线
        axes[0].plot(epochs_range, train_losses, 'b-', label='Train Loss')
        axes[0].plot(epochs_range, val_losses, 'r-', label='Val Loss')
        axes[0].set_title('Training vs Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Dice曲线
        axes[1].plot(epochs_range, val_dices, 'g-', linewidth=2, label='Validation Dice')
        axes[1].axhline(y=best_dice, color='red', linestyle='--', alpha=0.8, label=f'Best: {best_dice:.4f}')
        axes[1].set_title('Validation Dice Progress')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 内存使用（示意）
        memory_info = get_memory_info()
        axes[2].bar(['RAM Used', 'RAM Free', 'GPU Used'], 
                   [memory_info['ram_used'], memory_info['ram_total']-memory_info['ram_used'], memory_info['gpu_allocated']])
        axes[2].set_title('Memory Usage (GB)')
        axes[2].set_ylabel('Memory (GB)')
        
        plt.tight_layout()
        curves_path = '/content/training_curves_colab.png'
        plt.savefig(curves_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f'📈 Training curves saved: {curves_path}')
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
    
    # 最终总结
    print(f'\n{"="*60}')
    print('🎉 COLAB TRAINING COMPLETED!')
    print(f'{"="*60}')
    print(f'🏆 Best Dice Score: {best_dice:.6f}')
    print(f'📊 Total Epochs: {len(train_losses)}')
    print(f'📚 Training Slices: {len(train_dataset):,}')
    print(f'🧪 Validation Slices: {len(val_dataset):,}')
    print(f'💾 Model Saved: {final_model_path}')
    print(f'🧠 Model: Lightweight UNet ({total_params:,} params)')
    print(f'⚡ Colab Optimized: ✅')
    print_memory_usage("Final")
    print(f'{"="*60}')

def check_colab_compatibility():
    """检查Colab兼容性"""
    print("\n" + "="*60)
    print("COLAB COMPATIBILITY CHECK")
    print("="*60)
    
    memory_info = get_memory_info()
    print(f"💾 Available RAM: {memory_info['ram_total']:.1f}GB")
    print(f"💾 Used RAM: {memory_info['ram_used']:.1f}GB ({memory_info['ram_percent']:.1f}%)")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {memory_info['gpu_total']:.1f}GB")
        print(f"💾 GPU Allocated: {memory_info['gpu_allocated']:.1f}GB")
    
    print(f"\n✅ Optimizations Applied:")
    print(f"   - Lazy data loading")
    print(f"   - Reduced model size")
    print(f"   - Memory-efficient dataset")
    print(f"   - Automatic garbage collection")
    print(f"   - Smaller batch sizes")
    print(f"   - Limited slices per patient")
    print("="*60)

# 主函数
if __name__ == "__main__":
    check_colab_compatibility()
    colab_optimized_train()