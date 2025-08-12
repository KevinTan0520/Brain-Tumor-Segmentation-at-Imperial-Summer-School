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
    def __init__(self, data_pairs, transform=None):
        self.data_pairs = data_pairs
        self.transform = transform
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        fla_path, seg_path = self.data_pairs[idx]
        
        # 加载NIfTI文件
        fla_img = nib.load(fla_path)
        seg_img = nib.load(seg_path)
        
        fla_data = fla_img.get_fdata()
        seg_data = seg_img.get_fdata()
        
        # 获取俯视横截面（axial slices）
        # 随机选择一个切片
        slice_idx = np.random.randint(0, fla_data.shape[2])
        
        fla_slice = fla_data[:, :, slice_idx]
        seg_slice = seg_data[:, :, slice_idx]
        
        # 归一化
        fla_slice = (fla_slice - fla_slice.mean()) / (fla_slice.std() + 1e-8)
        
        # 转换为二值分割掩码
        seg_slice = (seg_slice > 0).astype(np.float32)
        
        # 转换为tensor并添加通道维度
        fla_slice = torch.FloatTensor(fla_slice).unsqueeze(0)
        seg_slice = torch.FloatTensor(seg_slice)
        
        return fla_slice, seg_slice

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
    
    # 加载数据
    data_pairs = load_data_pairs('dataset_segmentation')
    
    if len(data_pairs) == 0:
        print("Error: No data pairs found. Please check your dataset structure.")
        print("Expected structure: dataset_segmentation/train/{patient_id}/{patient_id}_fla.nii.gz and {patient_id}_seg.nii.gz")
        return
    
    train_pairs, val_pairs = train_test_split(data_pairs, test_size=0.2, random_state=42)
    
    # 创建数据加载器
    train_dataset = BrainTumorDataset(train_pairs)
    val_dataset = BrainTumorDataset(val_pairs)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)  # 设置为0避免多进程问题
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # 创建模型
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练参数
    num_epochs = 50
    best_dice = 0.0
    
    train_losses = []
    val_losses = []
    val_dices = []
    
    print(f"Starting training with {len(train_pairs)} training samples and {len(val_pairs)} validation samples")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images, masks = images.to(device), masks.to(device)
            
            # 调试信息：打印张量形状
            if batch_idx == 0 and epoch == 0:
                print(f"Image shape: {images.shape}")
                print(f"Mask shape: {masks.shape}")
            
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            
            # 调试信息：打印输出形状
            if batch_idx == 0 and epoch == 0:
                print(f"Output shape after squeeze: {outputs.shape}")
            
            # 计算损失
            bce_loss = criterion(outputs, masks)
            dice_loss_val = dice_loss(outputs, masks)
            loss = bce_loss + dice_loss_val
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images).squeeze(1)
                
                bce_loss = criterion(outputs, masks)
                dice_loss_val = dice_loss(outputs, masks)
                loss = bce_loss + dice_loss_val
                
                dice = dice_coefficient(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice.item()
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Dice': f'{dice.item():.4f}'})
        
        # 计算平均值
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
        
        # 保存最佳模型（保存在当前目录）
        if val_dice > best_dice:
            best_dice = val_dice
            model_save_path = os.path.join(CURRENT_DIR, 'best_brain_tumor_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f'New best model saved with Dice: {best_dice:.4f} at {model_save_path}')
    
    # 绘制训练曲线（保存在当前目录）
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(val_dices, label='Val Dice')
    plt.title('Validation Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    
    plt.tight_layout()
    curves_save_path = os.path.join(CURRENT_DIR, 'training_curves.png')
    plt.savefig(curves_save_path)
    plt.show()
    
    print(f'Training completed! Best Dice: {best_dice:.4f}')
    print(f'Training curves saved to: {curves_save_path}')

if __name__ == "__main__":
    train_model()