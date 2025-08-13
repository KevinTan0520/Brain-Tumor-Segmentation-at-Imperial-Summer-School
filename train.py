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

# 获取当前文件所在目录并设置为工作目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR)
print(f"Current working directory set to: {CURRENT_DIR}")

class BrainTumorDataset(Dataset):
    def __init__(self, data_pairs, transform=None, min_tumor_ratio=0.005, normal_slice_ratio=0.15):
        """
        改进的数据集类：预提取所有有效切片
        
        Args:
            data_pairs: (fla_path, seg_path) 对的列表
            transform: 数据增强变换
            min_tumor_ratio: 最小肿瘤像素比例，低于此值的切片被认为是正常切片
            normal_slice_ratio: 保留正常切片的比例
        """
        self.data_slices = []
        self.transform = transform
        
        print("Extracting all valid slices from dataset...")
        print(f"Parameters: min_tumor_ratio={min_tumor_ratio}, normal_slice_ratio={normal_slice_ratio}")
        
        tumor_slice_count = 0
        normal_slice_count = 0
        
        for fla_path, seg_path in tqdm(data_pairs, desc="Loading volumes"):
            try:
                # 加载NIfTI文件
                fla_img = nib.load(fla_path)
                seg_img = nib.load(seg_path)
                
                fla_data = fla_img.get_fdata()
                seg_data = seg_img.get_fdata()
                
                patient_id = os.path.basename(fla_path).replace('_fla.nii.gz', '')
                
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
                    
                    # 分类切片
                    is_tumor_slice = tumor_ratio > min_tumor_ratio
                    
                    # 决定是否包含此切片
                    include_slice = False
                    
                    if is_tumor_slice:
                        # 包含所有肿瘤切片
                        include_slice = True
                        tumor_slice_count += 1
                    else:
                        # 随机包含一定比例的正常切片
                        if np.random.random() < normal_slice_ratio:
                            include_slice = True
                            normal_slice_count += 1
                    
                    if include_slice:
                        self.data_slices.append({
                            'fla_slice': fla_slice.copy(),
                            'seg_slice': seg_slice.copy(),
                            'patient_id': patient_id,
                            'slice_idx': slice_idx,
                            'tumor_ratio': tumor_ratio,
                            'is_tumor_slice': is_tumor_slice
                        })
            
            except Exception as e:
                print(f"Error loading {fla_path}: {e}")
                continue
        
        print(f"\nDataset Statistics:")
        print(f"Total valid slices: {len(self.data_slices)}")
        print(f"Tumor slices: {tumor_slice_count}")
        print(f"Normal slices: {normal_slice_count}")
        print(f"Tumor/Normal ratio: {tumor_slice_count/(normal_slice_count+1e-8):.2f}")
    
    def __len__(self):
        return len(self.data_slices)
    
    def __getitem__(self, idx):
        slice_data = self.data_slices[idx]
        
        fla_slice = slice_data['fla_slice'].copy()
        seg_slice = slice_data['seg_slice'].copy()
        
        # 改进的归一化：更稳定的方法
        if fla_slice.std() > 1e-8:
            fla_slice = (fla_slice - fla_slice.mean()) / fla_slice.std()
        else:
            fla_slice = fla_slice - fla_slice.mean()
        
        # 确保数值范围合理
        fla_slice = np.clip(fla_slice, -5, 5)
        
        # 转换为二值分割掩码
        seg_slice = (seg_slice > 0).astype(np.float32)
        
        # 数据增强（如果需要）
        if self.transform:
            fla_slice, seg_slice = self.transform(fla_slice, seg_slice)
        
        # 转换为tensor并添加通道维度
        fla_slice = torch.FloatTensor(fla_slice).unsqueeze(0)
        seg_slice = torch.FloatTensor(seg_slice)
        
        return fla_slice, seg_slice
    
    def get_statistics(self):
        """返回数据集统计信息"""
        tumor_slices = sum(1 for s in self.data_slices if s['is_tumor_slice'])
        normal_slices = len(self.data_slices) - tumor_slices
        
        tumor_ratios = [s['tumor_ratio'] for s in self.data_slices if s['is_tumor_slice']]
        avg_tumor_ratio = np.mean(tumor_ratios) if tumor_ratios else 0
        
        return {
            'total_slices': len(self.data_slices),
            'tumor_slices': tumor_slices,
            'normal_slices': normal_slices,
            'avg_tumor_ratio': avg_tumor_ratio
        }

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

class EarlyStopping:
    """改进的早停类，支持多种停止策略"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True, verbose=True):
        """
        Args:
            patience: 多少个epoch没有改善后停止训练
            min_delta: 最小改善阈值
            restore_best_weights: 是否恢复到最佳权重
            verbose: 是否打印详细信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False
        self.best_weights = None
        self.score_history = []
    
    def __call__(self, val_score, model, epoch):
        """
        Args:
            val_score: 当前的验证分数（越高越好）
            model: 模型对象
            epoch: 当前epoch
        """
        self.score_history.append(val_score)
        
        # 第一次调用
        if self.best_score is None:
            self.best_score = val_score
            self.best_epoch = epoch
            self.save_checkpoint(model)
            if self.verbose:
                print(f'  → Early stopping baseline set: {val_score:.6f}')
        
        # 检查是否有改善
        elif val_score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  → Early stopping counter: {self.counter}/{self.patience} (best: {self.best_score:.6f} at epoch {self.best_epoch})')
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'  → Early stopping triggered! No improvement for {self.patience} epochs.')
                    print(f'  → Best score: {self.best_score:.6f} at epoch {self.best_epoch}')
        
        else:
            # 有改善
            improvement = val_score - self.best_score
            self.best_score = val_score
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model)
            if self.verbose:
                print(f'  → New best score: {val_score:.6f} (improvement: +{improvement:.6f})')
    
    def save_checkpoint(self, model):
        """保存当前最佳模型权重"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
    
    def restore_best(self, model):
        """恢复最佳权重"""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print(f'  → Restored best weights from epoch {self.best_epoch}')

def dice_loss(pred, target, smooth=1):
    """计算Dice损失函数"""
    pred = pred.contiguous()
    target = target.contiguous()
    
    # 将tensor展平为 (batch_size, -1)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # 计算交集
    intersection = (pred_flat * target_flat).sum(dim=1)
    
    # 计算Dice系数，然后转换为损失
    dice = (2. * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    
    # 返回1减去Dice系数作为损失
    return 1 - dice.mean()

def dice_coefficient(pred, target, smooth=1):
    """计算Dice系数"""
    pred = (pred > 0.5).float()
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def load_data_pairs(data_dir):
    data_pairs = []
    # 使用绝对路径，基于当前脚本所在目录
    data_dir = os.path.join(CURRENT_DIR, data_dir)
    train_dir = os.path.join(data_dir, 'train')
    
    print(f"Looking for data in: {train_dir}")
    
    if not os.path.exists(train_dir):
        print(f"Warning: Training directory not found: {train_dir}")
        return data_pairs
    
    for patient_id in os.listdir(train_dir):
        patient_dir = os.path.join(train_dir, patient_id)
        if os.path.isdir(patient_dir):
            fla_path = os.path.join(patient_dir, f"{patient_id}_fla.nii.gz")
            seg_path = os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")
            
            if os.path.exists(fla_path) and os.path.exists(seg_path):
                data_pairs.append((fla_path, seg_path))
                print(f"Found data pair for patient: {patient_id}")
    
    print(f"Total data pairs found: {len(data_pairs)}")
    return data_pairs

def train_model():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # 加载数据
    data_pairs = load_data_pairs('dataset_segmentation')
    
    if len(data_pairs) == 0:
        print("Error: No data pairs found. Please check your dataset structure.")
        print("Expected structure: dataset_segmentation/train/{patient_id}/{patient_id}_fla.nii.gz and {patient_id}_seg.nii.gz")
        return
    
    # 划分训练集和验证集
    train_pairs, val_pairs = train_test_split(data_pairs, test_size=0.2, random_state=42)
    print(f"Split data: {len(train_pairs)} training volumes, {len(val_pairs)} validation volumes")
    
    # 创建改进的数据集（预提取所有切片）
    print("\n" + "="*50)
    print("CREATING TRAINING DATASET")
    print("="*50)
    train_dataset = BrainTumorDataset(
        train_pairs, 
        min_tumor_ratio=0.005,  # 更低的阈值以包含更多切片
        normal_slice_ratio=0.15  # 15%的正常切片
    )
    
    print("\n" + "="*50)
    print("CREATING VALIDATION DATASET") 
    print("="*50)
    val_dataset = BrainTumorDataset(
        val_pairs,
        min_tumor_ratio=0.005,
        normal_slice_ratio=0.15
    )
    
    # 打印数据集统计
    train_stats = train_dataset.get_statistics()
    val_stats = val_dataset.get_statistics()
    
    print(f"\nFinal Dataset Statistics:")
    print(f"Training: {train_stats['total_slices']} slices ({train_stats['tumor_slices']} tumor, {train_stats['normal_slices']} normal)")
    print(f"Validation: {val_stats['total_slices']} slices ({val_stats['tumor_slices']} tumor, {val_stats['normal_slices']} normal)")
    
    # 调整批次大小（因为数据量增加了）
    batch_size = 16 if len(train_dataset) > 1000 else 8
    print(f"Using batch size: {batch_size}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 创建模型
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 修复学习率调度器的兼容性问题
    try:
        # 新版本PyTorch
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
    except TypeError:
        try:
            # 旧版本PyTorch可能不支持verbose参数
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5
            )
            print("Note: Using ReduceLROnPlateau without verbose parameter")
        except:
            # 如果还是有问题，使用StepLR作为替代
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            print("Note: Using StepLR scheduler as fallback")
    
    # 创建改进的早停对象
    early_stopping = EarlyStopping(
        patience=12,        # 12个epoch没有改善就停止
        min_delta=0.0005,   # 最小改善阈值：0.0005 (0.05%)
        restore_best_weights=True,
        verbose=True
    )
    
    # 训练参数
    num_epochs = 20    # 最大训练轮数
    best_dice = 0.0
    
    train_losses = []
    val_losses = []
    val_dices = []
    
    print(f"\n" + "="*60)
    print(f"STARTING TRAINING")
    print(f"="*60)
    print(f"Training samples: {len(train_dataset)} slices from {len(train_pairs)} volumes")
    print(f"Validation samples: {len(val_dataset)} slices from {len(val_pairs)} volumes")
    print(f"Max epochs: {num_epochs}, Batch size: {batch_size}")
    print(f"Early stopping: patience={early_stopping.patience}, min_delta={early_stopping.min_delta}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:3d}/{num_epochs} [Train]')
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            # 调试信息：仅在第一个epoch的第一个batch打印
            if batch_idx == 0 and epoch == 0:
                print(f"\nFirst batch info:")
                print(f"  Image shape: {images.shape}")
                print(f"  Mask shape: {masks.shape}")
                print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
                print(f"  Mask range: [{masks.min():.3f}, {masks.max():.3f}]")
            
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            
            # 计算组合损失
            bce_loss = criterion(outputs, masks)
            dice_loss_val = dice_loss(outputs, masks)
            loss = bce_loss + dice_loss_val
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'BCE': f'{bce_loss.item():.4f}',
                'Dice': f'{dice_loss_val.item():.4f}'
            })
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1:3d}/{num_epochs} [Val]  ')
            for images, masks in val_pbar:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                
                outputs = model(images).squeeze(1)
                
                bce_loss = criterion(outputs, masks)
                dice_loss_val = dice_loss(outputs, masks)
                loss = bce_loss + dice_loss_val
                
                dice = dice_coefficient(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice.item()
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}', 
                    'Dice': f'{dice.item():.4f}'
                })
        
        # 计算平均值
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        
        # 学习率调度
        try:
            if 'ReduceLROnPlateau' in str(type(scheduler)):
                scheduler.step(val_dice)
            else:
                scheduler.step()
        except Exception as e:
            print(f"Scheduler step failed: {e}")
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch+1:3d}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.6f} | LR: {current_lr:.2e}')
        
        # 更新最佳Dice（用于保存模型）
        if val_dice > best_dice:
            best_dice = val_dice
        
        # 早停检查
        early_stopping(val_dice, model, epoch + 1)
        
        if early_stopping.early_stop:
            print(f"\n" + "="*60)
            print("EARLY STOPPING TRIGGERED")
            print("="*60)
            print(f"Training stopped at epoch {epoch + 1}")
            print(f"Best Dice score: {early_stopping.best_score:.6f} at epoch {early_stopping.best_epoch}")
            
            # 恢复最佳权重
            early_stopping.restore_best(model)
            best_dice = early_stopping.best_score
            break
        
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(CURRENT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'train_stats': train_stats,
                'val_stats': val_stats
            }, checkpoint_path)
            print(f'  → Checkpoint saved: {checkpoint_path}')
    
    # 保存最终模型
    final_model_path = os.path.join(CURRENT_DIR, 'best_brain_tumor_model.pth')
    torch.save({
        'epoch': early_stopping.best_epoch if early_stopping.early_stop else epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dice': best_dice,
        'train_stats': train_stats,
        'val_stats': val_stats,
        'early_stopping_info': {
            'triggered': early_stopping.early_stop,
            'best_epoch': early_stopping.best_epoch,
            'patience_used': early_stopping.counter,
            'score_history': early_stopping.score_history[-20:]  # 保存最后20个epoch的分数
        }
    }, final_model_path)
    
    # 绘制训练曲线
    try:
        plt.figure(figsize=(20, 6))
        
        plt.subplot(1, 4, 1)
        plt.plot(train_losses, label='Train Loss', linewidth=2, alpha=0.8)
        plt.plot(val_losses, label='Val Loss', linewidth=2, alpha=0.8)
        if early_stopping.early_stop:
            plt.axvline(x=early_stopping.best_epoch-1, color='red', linestyle='--', alpha=0.7, label='Best Epoch')
        plt.title('Training and Validation Loss', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 4, 2)
        plt.plot(val_dices, label='Val Dice', color='green', linewidth=2, alpha=0.8)
        if early_stopping.early_stop:
            plt.axvline(x=early_stopping.best_epoch-1, color='red', linestyle='--', alpha=0.7, label='Best Epoch')
            plt.axhline(y=early_stopping.best_score, color='red', linestyle=':', alpha=0.7, label=f'Best Score: {early_stopping.best_score:.4f}')
        plt.title('Validation Dice Coefficient', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 4, 3)
        plt.plot(train_losses, label='Train Loss', alpha=0.7)
        plt.plot(val_losses, label='Val Loss', alpha=0.7)
        plt.plot([d * 2 for d in val_dices], label='Val Dice * 2', alpha=0.7)
        if early_stopping.early_stop:
            plt.axvline(x=early_stopping.best_epoch-1, color='red', linestyle='--', alpha=0.7, label='Best Epoch')
        plt.title('All Metrics Combined', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 新增：早停历史
        plt.subplot(1, 4, 4)
        patience_history = []
        for i, score in enumerate(early_stopping.score_history):
            if i == 0:
                patience_history.append(0)
            else:
                if score <= early_stopping.score_history[max(0, i-1)] + early_stopping.min_delta:
                    patience_history.append(min(patience_history[-1] + 1, early_stopping.patience))
                else:
                    patience_history.append(0)
        
        plt.plot(patience_history, label='Patience Counter', color='orange', linewidth=2)
        plt.axhline(y=early_stopping.patience, color='red', linestyle='--', alpha=0.7, label=f'Patience Limit: {early_stopping.patience}')
        plt.title('Early Stopping Progress', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Patience Counter')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        curves_save_path = os.path.join(CURRENT_DIR, 'training_curves.png')
        plt.savefig(curves_save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f'Training curves saved to: {curves_save_path}')
        
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    print(f'\n' + "="*60)
    print('TRAINING COMPLETED!')
    print("="*60)
    print(f'Final Validation Dice: {best_dice:.6f}')
    print(f'Total epochs trained: {len(train_losses)}')
    print(f'Early stopping triggered: {early_stopping.early_stop}')
    if early_stopping.early_stop:
        print(f'Best epoch: {early_stopping.best_epoch}')
        print(f'Patience used: {early_stopping.counter}/{early_stopping.patience}')
    print(f'Total training slices used: {len(train_dataset):,}')
    print(f'Total validation slices used: {len(val_dataset):,}')
    print(f'Final model saved to: {final_model_path}')

if __name__ == "__main__":
    train_model()