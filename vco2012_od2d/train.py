"""训练模块

实现模型的训练流程，包括数据加载、模型训练、验证和提前停止。
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from dataset import VOCDataset
from net import AnchorFreeDetector, HeatmapLoss
from config import (
    BATCH_SIZE, EPOCHS as NUM_EPOCHS, INITIAL_LR as LEARNING_RATE, NUM_WORKERS, 
    MODEL_PATH, OUTPUT_DIR, WARMUP_EPOCHS, WARMUP_LR, MIN_LR, LR_SCHEDULER,
    QUICK_TEST, QUICK_TEST_BATCHES, QUICK_TEST_EPOCHS
)

# Early Stopping配置
EARLY_STOP_PATIENCE = 5  # 连续5个epoch验证loss没改善就停止
MIN_DELTA = 1e-4  # 最小改善阈值

class WarmupScheduler:
    """Warmup学习率调度器
    
    在前warmup_epochs个epoch中，学习率从warmup_lr线性增加到base_lr，
    然后使用指定的调度器（cosine/step/plateau）。
    """
    
    def __init__(self, optimizer, warmup_epochs, warmup_lr, base_lr, scheduler_type='cosine', 
                 num_epochs=50, min_lr=1e-6):
        """初始化
        
        Args:
            optimizer: 优化器
            warmup_epochs: 预热epoch数量
            warmup_lr: 预热起始学习率
            base_lr: 基础学习率
            scheduler_type: 调度器类型 ('cosine', 'step', 'plateau')
            num_epochs: 总epoch数量
            min_lr: 最小学习率
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.current_epoch = 0
        self.scheduler_type = scheduler_type
        
        # 创建主调度器
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=num_epochs - warmup_epochs, 
                eta_min=min_lr
            )
        elif scheduler_type == 'step':
            self.scheduler = StepLR(
                optimizer, 
                step_size=10, 
                gamma=0.1
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.1, 
                patience=3, 
                min_lr=min_lr
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step(self, metric=None):
        """更新学习率
        
        Args:
            metric: 用于ReduceLROnPlateau的指标（如val loss）
        """
        if self.current_epoch < self.warmup_epochs:
            # Warmup阶段：线性增加学习率
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * \
                 (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # 主调度器阶段
            if self.scheduler_type == 'plateau':
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()
        
        self.current_epoch += 1
    
    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']

class EarlyStopping:
    """提前停止类
    
    监控验证loss，如果连续patience个epoch没有改善，则停止训练。
    """
    
    def __init__(self, patience=5, min_delta=1e-4, mode='min'):
        """初始化
        
        Args:
            patience: 容忍的epoch数量
            min_delta: 最小改善阈值
            mode: 'min'表示loss越小越好，'max'表示指标越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """更新状态
        
        Args:
            score: 当前分数（loss或指标）
            
        Returns:
            is_best: 是否为最佳分数
        """
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def validate(model, dataloader, criterion, device):
    """验证模型
    
    Args:
        model: 模型
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        
    Returns:
        平均验证loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (images, heatmaps, offsets, wh) in enumerate(dataloader):
            # 移动数据到设备
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            offsets = offsets.to(device)
            wh = wh.to(device)
            
            # 前向传播
            pred_heatmaps, pred_offsets, pred_wh = model(images)
            
            # 计算损失
            loss = criterion(pred_heatmaps, pred_offsets, pred_wh, heatmaps, offsets, wh)
            
            # 跳过nan loss
            if torch.isnan(loss):
                continue
            
            total_loss += loss.item()
            
            # 快速测试
            if QUICK_TEST and batch_idx + 1 >= QUICK_TEST_BATCHES:
                break
    
    return total_loss / (batch_idx + 1)

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    """训练一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        num_epochs: 总epoch数
        
    Returns:
        平均损失
    """
    # 设置模型为训练模式
    model.train()
    total_loss = 0.0
    
    # 使用tqdm显示进度条
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
    # 遍历数据加载器
    for batch_idx, (images, heatmaps, offsets, wh) in enumerate(pbar):
        # 移动数据到设备
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        offsets = offsets.to(device)
        wh = wh.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        pred_heatmaps, pred_offsets, pred_wh = model(images)
        
        # 计算损失
        loss = criterion(pred_heatmaps, pred_offsets, pred_wh, heatmaps, offsets, wh)
        
        # 检查loss是否为nan
        if torch.isnan(loss):
            print(f"\n❌ Loss is NaN at batch {batch_idx+1}!")
            print(f"   pred_heatmaps range: [{pred_heatmaps.min().item():.4f}, {pred_heatmaps.max().item():.4f}]")
            print(f"   pred_offsets range: [{pred_offsets.min().item():.4f}, {pred_offsets.max().item():.4f}]")
            print(f"   pred_wh range: [{pred_wh.min().item():.4f}, {pred_wh.max().item():.4f}]")
            print(f"   heatmaps range: [{heatmaps.min().item():.4f}, {heatmaps.max().item():.4f}]")
            print(f"   offsets range: [{offsets.min().item():.4f}, {offsets.max().item():.4f}]")
            print(f"   wh range: [{wh.min().item():.4f}, {wh.max().item():.4f}]")
            # 跳过这个batch
            continue
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 累加损失
        total_loss += loss.item()
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 快速测试：只运行指定数量的batch
        if QUICK_TEST and batch_idx + 1 >= QUICK_TEST_BATCHES:
            print(f"\n⏸️  Quick test mode: Stopping after {QUICK_TEST_BATCHES} batches")
            break
    
    # 计算平均损失
    return total_loss / (batch_idx + 1)

def plot_training_history(history, output_dir):
    """绘制训练历史曲线
    
    Args:
        history: 训练历史字典
        output_dir: 输出目录
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 绘制学习率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['lr'], label='Learning Rate', linewidth=2, color='orange')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    print("=" * 80)
    print("🚀 Starting Training Pipeline")
    print("=" * 80)
    
    # 阶段1：设置设备
    print("\n📍 Stage 1/7: Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   ✅ Using device: {device}")
    
    # 阶段2：创建数据集
    print("\n📍 Stage 2/7: Creating dataset...")
    train_dataset = VOCDataset(split='train')
    val_dataset = VOCDataset(split='val')
    print(f"   ✅ Train dataset size: {len(train_dataset)}")
    print(f"   ✅ Val dataset size: {len(val_dataset)}")
    
    # 阶段3：创建数据加载器
    print("\n📍 Stage 3/7: Creating dataloader...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    print(f"   ✅ Batch size: {BATCH_SIZE}")
    print(f"   ✅ Number of workers: {NUM_WORKERS}")
    print(f"   ✅ Train batches: {len(train_loader)}")
    print(f"   ✅ Val batches: {len(val_loader)}")
    
    if QUICK_TEST:
        print(f"   ⚡ Quick test mode: Will run only {QUICK_TEST_BATCHES} batches")
    
    # 阶段4：初始化模型
    print("\n📍 Stage 4/7: Initializing model...")
    model = AnchorFreeDetector()
    model.to(device)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✅ Total parameters: {total_params:,}")
    print(f"   ✅ Trainable parameters: {trainable_params:,}")
    
    # 阶段5：初始化优化器和损失函数
    print("\n📍 Stage 5/7: Initializing optimizer and loss...")
    criterion = HeatmapLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"   ✅ Optimizer: Adam")
    print(f"   ✅ Initial learning rate: {LEARNING_RATE}")
    print(f"   ✅ Loss function: HeatmapLoss")
    
    # 阶段6：初始化学习率调度器
    print("\n📍 Stage 6/7: Initializing learning rate scheduler...")
    lr_scheduler = WarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_lr=WARMUP_LR,
        base_lr=LEARNING_RATE,
        scheduler_type=LR_SCHEDULER,
        num_epochs=NUM_EPOCHS,
        min_lr=MIN_LR
    )
    print(f"   ✅ Scheduler type: {LR_SCHEDULER}")
    print(f"   ✅ Warmup epochs: {WARMUP_EPOCHS}")
    print(f"   ✅ Warmup LR: {WARMUP_LR}")
    print(f"   ✅ Min LR: {MIN_LR}")
    
    # 阶段7：初始化Early Stopping
    print("\n📍 Stage 7/7: Initializing early stopping...")
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=MIN_DELTA, mode='min')
    print(f"   ✅ Early stopping patience: {EARLY_STOP_PATIENCE}")
    print(f"   ✅ Min delta: {MIN_DELTA}")
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    # 最佳模型路径
    best_model_path = MODEL_PATH.replace('.pth', '_best.pth')
    
    print("\n" + "=" * 80)
    print("🔥 Training Started!")
    print("=" * 80)
    
    # 开始训练
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*80}")
        print(f"📊 Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"{'='*80}")
        
        # 训练一个epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(lr_scheduler.get_lr())
        
        print(f"\n✅ Epoch [{epoch+1}/{NUM_EPOCHS}] completed!")
        print(f"   📉 Train Loss: {train_loss:.4f}")
        print(f"   📉 Val Loss: {val_loss:.4f}")
        print(f"   📊 Learning Rate: {lr_scheduler.get_lr():.6f}")
        
        # 检查是否为最佳模型
        is_best = early_stopping(val_loss)
        
        if is_best:
            # 保存最佳模型
            torch.save(model.state_dict(), best_model_path)
            print(f"   ⭐ New best model! Saved to: {best_model_path}")
        else:
            print(f"   ⏳ No improvement. Patience: {early_stopping.counter}/{EARLY_STOP_PATIENCE}")
        
        # 更新学习率
        if LR_SCHEDULER == 'plateau':
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()
        
        # 检查是否需要提前停止
        if early_stopping.early_stop:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs!")
            print(f"   Best val loss: {early_stopping.best_score:.4f}")
            break
        
        # 检查loss是否为nan
        if train_loss != train_loss or val_loss != val_loss:  # nan != nan
            print("\n❌ Training stopped due to NaN loss!")
            print("   Possible reasons:")
            print("   1. Learning rate too high")
            print("   2. Gradient explosion")
            print("   3. Numerical instability in loss function")
            break
        
        # 快速测试：只运行1个epoch
        if QUICK_TEST:
            print("\n⏸️  Quick test mode: Stopping after 1 epoch")
            break
    
    # 保存最后的模型
    print("\n" + "=" * 80)
    print("💾 Saving model...")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"   ✅ Last model saved to: {MODEL_PATH}")
    
    if os.path.exists(best_model_path):
        print(f"   ✅ Best model saved to: {best_model_path}")
    
    # 绘制训练历史
    plot_training_history(history, OUTPUT_DIR)
    print(f"   ✅ Training history plot saved to: {OUTPUT_DIR}/training_history.png")
    
    # 保存训练历史到txt
    history_path = os.path.join(OUTPUT_DIR, 'training_history.txt')
    with open(history_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Training History\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Epoch':<10} {'Train Loss':<15} {'Val Loss':<15} {'LR':<15}\n")
        f.write("-" * 80 + "\n")
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1:<10} {history['train_loss'][i]:<15.4f} {history['val_loss'][i]:<15.4f} {history['lr'][i]:<15.6f}\n")
        f.write("=" * 80 + "\n")
    print(f"   ✅ Training history saved to: {history_path}")
    
    print("\n" + "=" * 80)
    print("🎉 Training completed!")
    print(f"   Total epochs: {len(history['train_loss'])}")
    print(f"   Best val loss: {early_stopping.best_score:.4f}")
    print("=" * 80)

if __name__ == '__main__':
    main()
