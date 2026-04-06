"""
推理脚本
用于对单张 MNIST 图片进行分割预测
"""

# 导入必要的库
import os  # 用于文件路径操作
import torch  # PyTorch 核心库
from PIL import Image  # 用于图像加载
import matplotlib.pyplot as plt  # 用于可视化
import numpy as np  # 用于数组操作

# 从配置文件导入必要的配置
from config import DATA_DIR, MODEL_DIR, OUTPUT_DIR, device, transform

# 从模型文件导入 U-Net 模型
from net import UNet


def infer_single_image(model, image, device):
    """
    对单张图片进行分割预测
    Args:
        model: 模型
        image: PIL 图像
        device: 设备
    Returns:
        原始图像、处理得到的 label 图、预测掩码、预测的前景图
    """
    # 转换为灰度格式
    image = image.convert('L')
    
    # 预处理图片
    input_tensor = transform(image).unsqueeze(0).to(device)  # 添加批次维度并移动到设备
    
    # 模型预测
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        output = model(input_tensor)  # 前向传播
        # 应用 sigmoid
        sigmoid_output = torch.sigmoid(output).squeeze().cpu().numpy()
        # 二值化
        mask = sigmoid_output > 0.5
    
    # 转换原始图像格式
    original_image = np.array(image)  # 转换为 numpy 数组
    
    # 生成处理得到的 label 图（二值化）
    label = (original_image > 128).astype(bool)
    
    # 生成分割的前景图（背景为白色）
    foreground = np.ones_like(original_image) * 255  # 背景为白色
    foreground[mask] = original_image[mask]  # 前景为原始图像
    
    return original_image, label, mask, foreground


def visualize_result(images_list, save_path):
    """
    可视化分割结果
    Args:
        images_list: 包含多张图片结果的列表，每个元素是 (original, label, mask, foreground)
        save_path: 保存路径
    """
    # 创建画布
    fig, axes = plt.subplots(len(images_list), 4, figsize=(20, 5 * len(images_list)))  # 3行4列
    
    # 标题列表
    titles = ['Original Image', 'Processed Label', 'Predicted Mask', 'Foreground (White Background)']
    
    # 遍历每张图片
    for i, (original, label, mask, foreground) in enumerate(images_list):
        # 遍历每个子图
        for j, (img, title) in enumerate(zip([original, label, mask, foreground], titles)):
            ax = axes[i, j]
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存结果
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"分割结果已保存到: {save_path}")


def main():
    """
    主推理函数
    """
    print("开始推理 MNIST 分割模型...")
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = UNet(in_channels=1, out_channels=1).to(device)  # 移动模型到设备
    
    # 加载预训练模型
    model_path = os.path.join(MODEL_DIR, 'unet_mnist.pth')
    if not os.path.exists(model_path):
        print("❌ 模型文件不存在，请先训练模型")
        return
    
    model.load_state_dict(torch.load(model_path))  # 加载模型权重
    print(f"✅ 加载模型: {model_path}")
    
    # 从测试集中随机选择 3 张图片
    from torchvision.datasets import MNIST
    dataset = MNIST(root=DATA_DIR, train=False, download=False)
    
    # 随机选择 3 个索引
    import random
    indices = random.sample(range(len(dataset)), 3)
    print(f"随机选择的图片索引: {indices}")
    
    # 推理并收集结果
    images_list = []
    for i, idx in enumerate(indices):
        image, _ = dataset[idx]
        # 推理
        original, label, mask, foreground = infer_single_image(model, image, device)
        images_list.append((original, label, mask, foreground))
        print(f"处理第 {i+1} 张图片完成")
    
    # 保存结果
    save_path = os.path.join(OUTPUT_DIR, 'infer_result.png')
    visualize_result(images_list, save_path)
    
    print("\n🎉 推理完成！")


if __name__ == '__main__':
    # 调用主函数
    main()
