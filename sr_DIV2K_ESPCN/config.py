import torch
import os

# ===================== 1. 设备配置 =====================
# 设备选择：使用M1芯片的GPU加速
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ===================== 2. 模型配置 =====================
# 模型名称：espcn 或 edsr
MODEL_NAME = "edsr"
# 超分倍数：3倍放大
UPSCALE_FACTOR = 2
# 输入图像通道数（RGB为3）
NUM_CHANNELS = 3

# ===================== 3. 训练配置 =====================
# 批次大小：M1 16G内存完美支持的批次大小
BATCH_SIZE = 2 if MODEL_NAME == "edsr" else 8
# 训练轮次：平衡训练效果和时间
EPOCHS = 1
# 学习率：根据模型类型设置不同的学习率
LR = 1e-4 if MODEL_NAME == "edsr" else 1e-3

# ===================== 4. 数据集路径 =====================
# 训练集路径
TRAIN_HR = "data/DIV2K_train_HR"
TRAIN_LR = f"data/DIV2K_train_LR_bicubic/X{UPSCALE_FACTOR}"

# 验证集路径
VAL_HR = "data/DIV2K_valid_HR"
VAL_LR = f"data/DIV2K_valid_LR_bicubic/X{UPSCALE_FACTOR}"

# 测试集路径
TEST_HR = "data/DIV2K_test_HR"
TEST_LR = f"data/DIV2K_test_LR_bicubic/X{UPSCALE_FACTOR}"

# ===================== 5. 输出路径 =====================
# 模型保存路径
MODEL_PATH = f"weights/{MODEL_NAME}_x{UPSCALE_FACTOR}_best.pth"
# 输出目录
OUTPUT_DIR = "outputs"

# 创建必要的目录
os.makedirs("weights", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
