import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from balanced_dataset import BalancedTrafficDataset
from model_improved_resnet50 import ImprovedResNet50
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import numpy as np
import os
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time
from torch.cuda.amp import GradScaler, autocast
import pandas as pd

# 设置随机种子，确保实验可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 数据增强函数 - 适度增强，避免过度变形
class TrafficDataAugmentation:
    def __init__(self):
        # 训练时的数据增强变换 - 增强力度以减少过拟合
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((240, 240)),  # 稍大一点以进行随机裁剪
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=0.05),  # 小概率转为灰度
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),  # 随机擦除
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 验证/测试时的变换
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # 直接调整到目标大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# 早期停止
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='best_model_balanced.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        
    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        print(f'模型已保存到 {self.path}')

# 训练一个周期
def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, targets) in enumerate(loader):
        # 将数据移动到设备上
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 混合精度训练
        if scaler is not None:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 常规训练
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # 更新学习率调度器
        if isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 每10个批次打印一次统计信息
        if (i+1) % 10 == 0:
            print(f'批次: {i+1}/{len(loader)}, 损失: {running_loss/(i+1):.4f}, 准确率: {100.*correct/total:.2f}%')
    
    return running_loss / len(loader), 100.*correct/total

# 验证
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(loader)
    
    print(f'验证 - 损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%')
    return avg_loss, accuracy, all_preds, all_targets

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_names, filename='confusion_matrix_balanced.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    # 归一化混淆矩阵
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 创建带百分比和原始计数的标签
    labels = np.zeros_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                labels[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
    
    sns.heatmap(
        cm_norm, 
        annot=labels,
        fmt='',
        cmap='Blues', 
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵 (平衡数据集)')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    return cm, cm_norm

# 绘制训练历史
def plot_history(train_losses, val_losses, train_accs, val_accs, filename='training_history_balanced.png'):
    plt.figure(figsize=(15, 6))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('周期')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.xlabel('周期')
    plt.ylabel('准确率(%)')
    plt.title('训练和验证准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_classification_report(y_true, y_pred, class_names, filename='classification_report_balanced.csv'):
    """保存分类报告为CSV文件"""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(filename)
    return df

def main():
    # 设置随机种子
    set_seed(42)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据增强
    data_aug = TrafficDataAugmentation()
    
    # 创建平衡的数据集
    print("正在创建平衡数据集...")
    train_dataset = BalancedTrafficDataset(
        root_dir='E:/code/qianru/amap_traffic_final_train_data',
        json_file='E:/code/qianru/amap_traffic_final_train.json',
        train=True,
        transform=data_aug.train_transform,
        balance_classes=True  # 启用类别平衡
    )
    
    # 验证集不需要平衡
    val_dataset = BalancedTrafficDataset(
        root_dir='E:/code/qianru/amap_traffic_final_train_data',
        json_file='E:/code/qianru/amap_traffic_final_train.json',
        train=False,
        transform=data_aug.val_transform,
        balance_classes=False  # 不平衡验证集
    )
    
    # 数据加载器
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           num_workers=4, pin_memory=True)
    
    # 初始化改进版模型
    print("初始化ResNet50改进模型...")
    model = ImprovedResNet50(dropout_rate=0.65).to(device)  # 增加dropout率
    
    # 训练参数，增加权重衰减
    criterion = CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=2e-4)  # 增加权重衰减
    
    # 使用OneCycleLR学习率调度器
    total_epochs = 50  # 增加总周期数
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=0.002, steps_per_epoch=steps_per_epoch, 
                          epochs=total_epochs, pct_start=0.1, div_factor=25, final_div_factor=1000)
    
    # 早期停止 - 增加耐心值
    early_stopping = EarlyStopping(patience=2, path='best_model_resnet50_balanced_1.pth')
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 记录训练历史
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    '''
    # 训练循环
    print(f"开始训练 {total_epochs} 个周期...")
    start_time = time.time()
    
    for epoch in range(total_epochs):
        print(f"周期 {epoch+1}/{total_epochs}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        val_loss, val_acc, all_preds, all_targets = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 早期停止检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("触发早期停止")
            break
        
        # 如果不是OneCycleLR，则在每个周期结束时调整学习率
        if not isinstance(scheduler, OneCycleLR):
            scheduler.step(val_loss)
    
    elapsed_time = time.time() - start_time
    print(f"训练完成，耗时: {elapsed_time/60:.2f} 分钟")
    '''
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model_resnet50_balanced_1.pth', map_location=device))
    
    # 最终验证
    print("进行最终验证...")
    _, final_acc, final_preds, final_targets = validate(model, val_loader, criterion, device)
    print(f"最终验证准确率: {final_acc:.2f}%")
    
    # 绘制混淆矩阵
    class_names = ['畅通', '缓行', '拥堵', '封闭']
    cm, cm_norm = plot_confusion_matrix(final_targets, final_preds, class_names)
    
    # 保存分类报告
    report_df = save_classification_report(final_targets, final_preds, class_names)
    print("分类报告:")
    print(report_df)
    
    # 绘制训练历史
    plot_history(train_losses, val_losses, train_accs, val_accs)
    
    # 保存模型用于集成
    torch.save(model.state_dict(), 'model_resnet50_balanced_final.pth', _use_new_zipfile_serialization=False)
    
    print("训练和评估已完成。最终模型已保存。")

if __name__ == "__main__":
    main() 