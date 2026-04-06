"""
推理脚本
用于对 Pascal VOC 2012 数据集进行分割预测
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
from config import DATA_DIR, MODEL_DIR, OUTPUT_DIR, device, transform, CLASS_NAMES, NUM_CLASSES

# 从模型文件导入 U-Net 模型
from net import UNet
from visual_dataset import PascalVOC2012Dataset


def get_color_map(num_classes):
    """
    生成颜色映射，用于可视化分割结果
    
    Args:
        num_classes: 类别数
    
    Returns:
        颜色映射字典，键为类别索引，值为 RGB 颜色
    """
    # 定义 21 个类别的颜色（包括背景）
    color_map = {
        0: [0, 0, 0],       # 背景 - 黑色
        1: [128, 0, 0],     # 飞机 - 红色
        2: [0, 128, 0],     # 自行车 - 绿色
        3: [128, 128, 0],   # 鸟 - 黄色
        4: [0, 0, 128],     # 船 - 蓝色
        5: [128, 0, 128],   # 瓶子 - 紫色
        6: [0, 128, 128],   # 公交车 - 青色
        7: [128, 128, 128], # 汽车 - 灰色
        8: [64, 0, 0],      # 猫 - 深红色
        9: [192, 0, 0],     # 椅子 - 亮红色
        10: [64, 128, 0],   # 牛 - 深绿色
        11: [192, 128, 0],  # 餐桌 - 亮黄色
        12: [64, 0, 128],   # 狗 - 深蓝色
        13: [192, 0, 128],  # 马 - 亮紫色
        14: [64, 128, 128], # 摩托车 - 深青色
        15: [192, 128, 128],# 人 - 亮灰色
        16: [0, 64, 0],     # 盆栽植物 - 暗绿色
        17: [128, 64, 0],   # 羊 - 橙黄色
        18: [0, 192, 0],    # 沙发 - 鲜绿色
        19: [128, 192, 0],  # 火车 - 黄绿色
        20: [0, 64, 128]    # 电视监视器 - 蓝绿色
    }
    return color_map


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
    
    # 获取颜色映射
    color_map = get_color_map(NUM_CLASSES)
    
    for i in range(num_samples):
        original_image = original_images[i]
        label = labels[i]
        mask = masks[i]
        
        # 调整掩码尺寸以匹配原始图像
        # 处理预测掩码：转换为彩色可视化格式
        mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cls_idx, color in color_map.items():
            mask_color[mask == cls_idx] = color
        mask_resized = np.array(Image.fromarray(mask_color).resize(
            (original_image.shape[1], original_image.shape[0]), 
            Image.NEAREST
        ), dtype=np.uint8)
        
        # 调整标签尺寸以匹配原始图像
        label_np = label.squeeze().numpy()
        # 转换标签值为彩色
        label_color = np.zeros((label_np.shape[0], label_np.shape[1], 3), dtype=np.uint8)
        for cls_idx, color in color_map.items():
            label_color[label_np == cls_idx] = color
        label_resized = np.array(Image.fromarray(label_color).resize(
            (original_image.shape[1], original_image.shape[0]), 
            Image.NEAREST
        ), dtype=np.uint8)
        
        # 显示原始图像
        axes[i, 0].imshow(original_image)
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')
        
        # 显示原始标签
        axes[i, 1].imshow(label_resized)
        axes[i, 1].set_title('Original Label')
        axes[i, 1].axis('off')
        
        # 显示预测掩码
        axes[i, 2].imshow(mask_resized)
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')
        
        # 显示预测的前景和背景图
        # 创建一个彩色覆盖图
        overlay = original_image.copy()
        # 降低原始图像的亮度，以便突出显示分割结果
        overlay = overlay * 0.5
        # 将分割结果叠加到原始图像上
        overlay[mask_resized != 0] = overlay[mask_resized != 0] * 0.5 + mask_resized[mask_resized != 0] * 0.5
        axes[i, 3].imshow(overlay.astype(np.uint8))
        axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存结果
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # print(f"分割结果已保存到: {save_path}")


def inference(model, image, device):
    """
    对单张图片进行分割预测
    Args:
        model: 模型
        image: 预处理后的图像张量
        device: 设备
    Returns:
        预测掩码和softmax输出
    """
    # 模型预测
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        output = model(image)  # 前向传播
        # 打印原始输出的统计信息
        output_np = output.squeeze().cpu().numpy()
        # print(f"原始输出形状: {output_np.shape}")
        # print(f"原始输出 min: {output_np.min()}")
        # print(f"原始输出 max: {output_np.max()}")
        # print(f"原始输出 mean: {output_np.mean()}")
        
        # 对21分类输出应用 softmax
        # softmax 函数将输出转换为概率分布，每个通道的值表示对应类别的概率
        # dim=1 表示在通道维度上应用 softmax，确保每个像素的21个通道值总和为1
        # 输入形状: [1, 21, 160, 160]（batch_size=1, channels=21, height=160, width=160）
        # 输出形状: [1, 21, 160, 160]
        softmax_output = torch.softmax(output, dim=1)
        # squeeze() 去除 batch 维度，因为 batch_size=1
        # 输出形状: [21, 160, 160]
        softmax_output = softmax_output.squeeze().cpu().numpy()
        # print(f"Softmax output shape: {softmax_output.shape}")
        
        # 获取每个像素的预测类别
        # argmax(axis=0) 在通道维度上取最大值的索引
        # 输出形状: [160, 160]
        mask = np.argmax(softmax_output, axis=0)  # 类别从0开始
        # print(f"预测掩码形状: {mask.shape}")
        # print(f"预测掩码值范围: {mask.min()} - {mask.max()}")
    
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
        img_path = dataset.images[idx]
        original_image = Image.open(img_path).convert('RGB')
        original_image = np.array(original_image)
        
        # print(f"使用测试图片: {os.path.basename(img_path)}")
        # print(f"预处理后图像形状: {image.shape}")
        # print(f"原始图像形状: {original_image.shape}")
        
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
    测试所有 epoch 开头的模型，使用随机选择的图片
    """
    print("开始推理 Pascal VOC 2012 图像分割...")
    print(f"使用设备: {device}")
    print(f"类别数: {NUM_CLASSES}")
    
    # 加载数据集以获取原始标签
    from config import label_transform
    
    dataset = PascalVOC2012Dataset(
        root=DATA_DIR,
        split='val',  # 使用验证集
        transform=transform,  # 使用与训练相同的预处理
        target_transform=label_transform,
        only_with_annotation=True
    )
    
    # 随机挑选4张图片，每次运行都选择不同的图片
    num_samples = 4
    indices = random.sample(range(len(dataset)), num_samples)
    print(f"\n选择的图片索引: {indices}")
    
    # 查找所有 epoch 开头的模型文件
    epoch_pattern = os.path.join(MODEL_DIR, 'epoch*_unet_voc2012.pth')
    epoch_models = sorted(glob.glob(epoch_pattern))
    
    if not epoch_models:
        print("❌ 未找到 epoch 开头的模型文件，请先训练模型")
        return
    
    print(f"\n找到 {len(epoch_models)} 个 epoch 模型文件")
    
    # 初始化模型
    model = UNet(in_channels=3, out_channels=NUM_CLASSES).to(device)  # 移动模型到设备
    
    # 遍历所有 epoch 模型进行测试
    for model_path in epoch_models:
        # 从文件名中提取 epoch 编号
        filename = os.path.basename(model_path)
        # 提取 epoch 编号（例如：epoch0_unet_voc2012.pth -> epoch0）
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
