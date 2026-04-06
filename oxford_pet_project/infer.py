"""
推理脚本
用于对单张图片进行分割预测
"""

# 导入必要的库
import os  # 用于文件路径操作
import torch  # PyTorch 核心库
from PIL import Image  # 用于图像加载
import matplotlib.pyplot as plt  # 用于可视化
import numpy as np  # 用于数组操作
import random  # 用于随机选择样本
import glob  # 用于查找文件

# 从配置文件导入必要的配置
from config import DATA_DIR, MODEL_DIR, OUTPUT_DIR, device, transform

# 从模型文件导入 U-Net 模型
from net import UNet


def visualize_result(original_images, labels, masks, softmax_outputs, save_path):
    """
    可视化分割结果
    Args:
        original_images: 原始图像列表
        labels: 原始标签列表
        masks: 预测掩码列表
        softmax_outputs: 原始softmax输出列表
        save_path: 保存路径
    """
    # 创建画布
    num_samples = len(original_images)
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))  # 每行4列
    
    for i in range(num_samples):
        original_image = original_images[i]
        label = labels[i]
        mask = masks[i]
        
        # 调整掩码尺寸以匹配原始图像
        from PIL import Image
        # 处理预测掩码：转换为可视化格式
        mask_processed = np.where(mask == 1, 255, np.where(mask == 2, 0, 127))
        mask_resized = np.array(Image.fromarray(mask_processed.astype(np.uint8)).resize(
            (original_image.shape[1], original_image.shape[0]), 
            Image.NEAREST
        ), dtype=np.uint8)
        
        # 调整标签尺寸以匹配原始图像
        # 原始标签值：1=前景，2=背景，3=未分类
        label_np = label.squeeze().numpy()
        # 转换标签值：前景为255（白色），背景为0（黑色），未分类为127（灰色）
        label_processed = np.where(label_np == 1, 255, np.where(label_np == 2, 0, 127))
        label_resized = np.array(Image.fromarray(label_processed.astype(np.uint8)).resize(
            (original_image.shape[1], original_image.shape[0]), 
            Image.NEAREST
        ), dtype=np.uint8)
        
        # 显示原始图像
        axes[i, 0].imshow(original_image)
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')
        
        # 显示原始标签
        axes[i, 1].imshow(label_resized, cmap='gray')
        axes[i, 1].set_title('Original Label')
        axes[i, 1].axis('off')
        
        # 显示预测掩码（白色前景，黑色背景）
        axes[i, 2].imshow(mask_resized, cmap='gray')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')
        
        # 显示预测的前景和背景图
        black_background = np.zeros_like(original_image)
        black_background[mask_resized == 255] = original_image[mask_resized == 255]
        axes[i, 3].imshow(black_background)
        axes[i, 3].set_title('Foreground on Black')
        axes[i, 3].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存结果
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"分割结果已保存到: {save_path}")


def inference(model, image, device):
    """
    对单张图片进行分割预测
    Args:
        model: 模型
        image: 预处理后的图像张量
        device: 设备
    Returns:
        预测掩码和sigmoid输出
    """
    # 模型预测
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        output = model(image)  # 前向传播
        # 打印原始输出的统计信息
        output_np = output.squeeze().cpu().numpy()
        print(f"原始输出形状: {output_np.shape}")
        print(f"原始输出 min: {output_np.min()}")
        print(f"原始输出 max: {output_np.max()}")
        print(f"原始输出 mean: {output_np.mean()}")
        
        # 对三分类输出应用 softmax
        # softmax 函数将输出转换为概率分布，每个通道的值表示对应类别的概率
        # dim=1 表示在通道维度上应用 softmax，确保每个像素的三个通道值总和为1
        # 输入形状: [1, 3, 256, 256]（batch_size=1, channels=3, height=256, width=256）
        # 输出形状: [1, 3, 256, 256]
        softmax_output = torch.softmax(output, dim=1)
        # squeeze() 去除 batch 维度，因为 batch_size=1
        # 输出形状: [3, 256, 256]
        softmax_output = softmax_output.squeeze().cpu().numpy()
        print(f"Softmax output shape: {softmax_output.shape}")
        
        # 获取每个像素的预测类别
        # argmax(axis=0) 在通道维度上取最大值的索引
        # 例如：如果某个像素的三个通道值为 [0.8, 0.1, 0.1]，则 argmax 结果为 0
        # 加1是因为类别从1开始（1=前景，2=背景，3=未分类）
        # 输出形状: [256, 256]
        mask = np.argmax(softmax_output, axis=0) + 1  # 加1是因为类别从1开始
        print(f"预测掩码形状: {mask.shape}")
        print(f"预测掩码值范围: {mask.min()} - {mask.max()}")
    
    return mask, softmax_output


def load_model(model, model_path, device):
    """
    加载模型权重
    Args:
        model: 模型
        model_path: 模型路径
        device: 设备
    Returns:
        epoch: 训练轮数
        loss: 损失值
    """
    # 加载检查点文件
    checkpoint = torch.load(model_path, map_location=device)
    
    # 检查是否是完整的检查点（包含 model_state_dict）
    if 'model_state_dict' in checkpoint:
        # 从完整的检查点中加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        loss = checkpoint.get('loss', 'N/A')
        print(f"✅ 加载检查点: {model_path}")
        print(f"   - Epoch: {epoch}")
        print(f"   - Loss: {loss:.4f}")
        return epoch, loss
    else:
        # 直接加载模型权重（兼容旧格式）
        model.load_state_dict(checkpoint)
        print(f"✅ 加载模型: {model_path}")
        return 'N/A', 'N/A'


def test_single_epoch(model, dataset, indices, device):
    """
    使用指定模型测试一组固定的图片
    Args:
        model: 模型
        dataset: 数据集
        indices: 图片索引列表
        device: 设备
    Returns:
        original_images: 原始图像列表
        labels: 标签列表
        masks: 预测掩码列表
        softmax_outputs: softmax输出列表
    """
    # 存储结果
    original_images = []
    labels = []
    masks = []
    softmax_outputs = []
    
    for i, idx in enumerate(indices):
        print(f"\n处理第 {i+1} 张图片...")
        # 获取图片和标签
        image, label = dataset[idx]
        # 获取图片路径
        img_path = os.path.join(DATA_DIR, 'oxford-iiit-pet', 'images', dataset._images[idx])
        original_image = Image.open(img_path).convert('RGB')
        original_image = np.array(original_image)
        
        print(f"使用测试图片: {dataset._images[idx]}")
        print(f"预处理后图像形状: {image.shape}")
        print(f"原始图像形状: {original_image.shape}")
        
        # 推理
        input_tensor = image.unsqueeze(0).to(device)  # 使用预处理后的图像
        mask, softmax_output = inference(model, input_tensor, device)
        
        # 存储结果
        original_images.append(original_image)
        labels.append(label)
        masks.append(mask)
        softmax_outputs.append(softmax_output)
    
    return original_images, labels, masks, softmax_outputs


def main():
    """
    主推理函数
    测试所有 epoch 开头的模型，使用同一组固定图片
    """
    print("开始推理 Oxford-IIIT Pet 图像分割...")
    print(f"使用设备: {device}")
    
    # 加载数据集以获取原始标签
    from torchvision.datasets import OxfordIIITPet
    from config import label_transform
    
    dataset = OxfordIIITPet(
        root=DATA_DIR,
        split='test',
        target_types='segmentation',
        transform=transform,  # 使用与训练相同的预处理
        target_transform=label_transform,
        download=False
    )
    
    # 随机挑选4张图片，固定种子以确保每次选择相同的图片
    random.seed(42)
    num_samples = 4
    indices = random.sample(range(len(dataset)), num_samples)
    print(f"\n选择的图片索引: {indices}")
    
    # 查找所有 epoch 开头的模型文件
    epoch_pattern = os.path.join(MODEL_DIR, 'epoch*_unet_pet.pth')
    epoch_models = sorted(glob.glob(epoch_pattern))
    
    if not epoch_models:
        print("❌ 未找到 epoch 开头的模型文件，请先训练模型")
        return
    
    print(f"\n找到 {len(epoch_models)} 个 epoch 模型文件")
    
    # 初始化模型
    model = UNet(in_channels=3, out_channels=3).to(device)  # 移动模型到设备
    
    # 遍历所有 epoch 模型进行测试
    for model_path in epoch_models:
        # 从文件名中提取 epoch 编号
        filename = os.path.basename(model_path)
        # 提取 epoch 编号（例如：epoch0_unet_pet.pth -> epoch0）
        epoch_name = filename.split('_')[0]
        
        print(f"\n{'='*60}")
        print(f"测试 {epoch_name} 模型")
        print(f"{'='*60}")
        
        # 加载模型
        epoch, loss = load_model(model, model_path, device)
        
        # 测试模型
        original_images, labels, masks, softmax_outputs = test_single_epoch(
            model, dataset, indices, device
        )
        
        # 保存结果
        save_path = os.path.join(OUTPUT_DIR, f'infer_{epoch_name}_result.png')
        visualize_result(original_images, labels, masks, softmax_outputs, save_path)
    
    print(f"\n{'='*60}")
    print("🎉 所有模型测试完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    # 调用主函数
    main()
