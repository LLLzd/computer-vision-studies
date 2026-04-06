import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 导入配置和模型
from config import DEVICE, UPSCALE_FACTOR, NUM_CHANNELS, VAL_HR, VAL_LR, MODEL_PATH, OUTPUT_DIR, MODEL_NAME
from models import ESPCN, EDSR

# ===================== 1. 数据集加载器 =====================
class SRDataset(Dataset):
    """超分辨率数据集加载器
    用于加载低分辨率和高分辨率图像对
    """
    def __init__(self, hr_dir, lr_dir):
        """初始化数据集
        Args:
            hr_dir: 高分辨率图像目录
            lr_dir: 低分辨率图像目录
        """
        # 加载高分辨率图像路径
        self.hr_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith(('png', 'jpg'))])
        # 加载低分辨率图像路径
        self.lr_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.endswith(('png', 'jpg'))])

    def __len__(self):
        """返回数据集大小"""
        return len(self.hr_files)

    def __getitem__(self, idx):
        """获取数据项
        Args:
            idx: 数据索引
        Returns:
            lr: 低分辨率图像张量
            hr: 高分辨率图像张量
        """
        # 读取高分辨率图像并转换为RGB
        hr = Image.open(self.hr_files[idx]).convert("RGB")
        # 读取低分辨率图像并转换为RGB
        lr = Image.open(self.lr_files[idx]).convert("RGB")
        
        # 转换为numpy数组，然后转为tensor
        # 形状从 (H, W, C) 转为 (C, H, W)，并归一化到 [0, 1]
        hr = torch.from_numpy(np.array(hr)).permute(2, 0, 1).float() / 255.0
        lr = torch.from_numpy(np.array(lr)).permute(2, 0, 1).float() / 255.0
        
        return lr, hr

# ===================== 2. 评估函数 =====================
def evaluate():
    """评估模型性能"""
    # 加载模型
    print(f"📦 加载模型: {MODEL_PATH}")
    if MODEL_NAME == 'espcn':
        model = ESPCN(UPSCALE_FACTOR, NUM_CHANNELS).to(DEVICE)
    elif MODEL_NAME == 'edsr':
        model = EDSR(UPSCALE_FACTOR, NUM_CHANNELS).to(DEVICE)
    else:
        raise ValueError(f"不支持的模型名称: {MODEL_NAME}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 加载验证数据集
    print(f"📊 加载验证数据集: {VAL_HR}")
    val_dataset = SRDataset(VAL_HR, VAL_LR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # 评估指标
    total_psnr = 0
    total_ssim = 0
    
    # 创建可视化
    plt.figure(figsize=(15, 10))
    
    with torch.no_grad():
        for idx, (lr, hr) in enumerate(tqdm(val_loader, desc="评估中")):
            # 将数据移至指定设备
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            
            # 前向传播：生成超分辨率图像
            sr = model(lr)
            
            # 转换为numpy数组用于计算指标
            # 形状从 (C, H, W) 转为 (H, W, C)
            sr_np = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()
            hr_np = hr.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # 计算PSNR和SSIM
            # 确保图像在 [0, 1] 范围内
            sr_np = np.clip(sr_np, 0, 1)
            hr_np = np.clip(hr_np, 0, 1)
            
            # 计算PSNR
            current_psnr = psnr(hr_np, sr_np, data_range=1.0)
            total_psnr += current_psnr
            
            # 计算SSIM
            current_ssim = ssim(hr_np, sr_np, multichannel=True, data_range=1.0, channel_axis=2)
            total_ssim += current_ssim
            
            # 可视化前5个样本
            if idx < 5:
                plt.subplot(5, 3, idx*3 + 1)
                plt.imshow(lr.squeeze(0).permute(1, 2, 0).cpu().numpy())
                plt.title(f"输入 (LR)")
                plt.axis('off')
                
                plt.subplot(5, 3, idx*3 + 2)
                plt.imshow(sr_np)
                plt.title(f"输出 (SR) PSNR: {current_psnr:.2f}")
                plt.axis('off')
                
                plt.subplot(5, 3, idx*3 + 3)
                plt.imshow(hr_np)
                plt.title(f"真值 (HR) SSIM: {current_ssim:.4f}")
                plt.axis('off')
    
    # 计算平均指标
    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)
    
    print(f"\n📊 评估结果:")
    print(f"平均 PSNR: {avg_psnr:.2f} dB")
    print(f"平均 SSIM: {avg_ssim:.4f}")
    
    # 保存可视化结果
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "evaluation_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化结果已保存至: {output_path}")
    
    # 保存评估结果到文本文件
    results_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.txt")
    with open(results_path, 'w') as f:
        f.write(f"评估结果\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"模型: {MODEL_PATH}\n")
        f.write(f"验证集: {VAL_HR}\n")
        f.write(f"超分倍数: {UPSCALE_FACTOR}x\n")
        f.write(f"验证样本数: {len(val_loader)}\n")
        f.write(f"\n平均 PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"平均 SSIM: {avg_ssim:.4f}\n")
    print(f"📝 评估指标已保存至: {results_path}")

if __name__ == "__main__":
    print("🚀 开始评估模型性能...")
    evaluate()
