"""
推理脚本
用于对单张手写数字图片进行推理并可视化结果
"""
# 导入 PyTorch 库
import torch
# 导入 torchvision 库，用于加载数据集
import torchvision
# 导入 matplotlib 库，用于可视化
import matplotlib.pyplot as plt
# 导入 numpy 库，用于数组操作
import numpy as np

# 从配置文件导入必要的配置
from config import DATA_DIR, MODEL_PATH, INFER_PATH, TRANSFORM, CLASSES
# 从模型文件导入网络结构
from net import Net


def load_model(device):
    """加载训练好的模型"""
    # 创建网络实例并移动到指定设备
    net = Net().to(device)
    # 加载训练好的模型权重
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # 设置模型为评估模式（关闭 dropout 等训练特有的层）
    net.eval()
    # 返回加载好的模型
    return net


def get_sample_image(index=0):
    """
    获取测试集中的指定样本
    Args:
        index: 样本索引，默认为0
    Returns:
        image: 图像张量
        label: 真实标签
    """
    # 加载 MNIST 测试集
    testset = torchvision.datasets.MNIST(
        root=DATA_DIR,  # 数据集保存路径
        train=False,  # 加载测试集（train=True 表示加载训练集）
        download=True,  # 如果数据集不存在则下载
        transform=TRANSFORM  # 应用数据预处理转换
    )
    # 获取指定索引的样本（图像和标签）
    image, label = testset[index]
    # 返回图像和标签
    return image, label


def predict(net, image, device):
    """
    对单张图片进行预测
    Args:
        net: 模型
        image: 图像张量
        device: 计算设备
    Returns:
        predicted_class: 预测的类别索引
        confidence: 预测置信度（概率）
    """
    # 禁用梯度计算，提高推理速度
    with torch.no_grad():
        # 添加批次维度（从 [1, 28, 28] 变为 [1, 1, 28, 28]）
        image = image.unsqueeze(0).to(device)
        # 前向传播，获取模型输出
        output = net(image)
        # 计算预测概率（使用 softmax）
        probabilities = torch.softmax(output, dim=1)
        # 获取最大概率值和对应的类别索引
        confidence, predicted_class = torch.max(probabilities, 1)
        # 将张量转换为标量并返回
        return predicted_class.item(), confidence.item()


def visualize_result(image, true_label, pred_label, confidence, save_path):
    """
    可视化推理结果
    Args:
        image: 图像张量
        true_label: 真实标签
        pred_label: 预测标签
        confidence: 预测置信度
        save_path: 保存路径
    """
    # 反归一化并转换格式用于显示
    # 将像素值从 [-1, 1] 转换回 [0, 1]
    image = image.numpy() / 2 + 0.5
    # 从 [1, 28, 28] 转换为 [28, 28] 格式（用于 matplotlib 显示）
    image = image.squeeze()
    
    # 创建图形
    plt.figure(figsize=(6, 6))
    # 显示图像，使用灰度颜色映射
    plt.imshow(image, cmap='gray')
    # 设置标题，显示真实标签、预测标签和置信度
    plt.title(f"True: {CLASSES[true_label]} | Pred: {CLASSES[pred_label]}\nConfidence: {confidence:.2%}")
    # 关闭坐标轴
    plt.axis("off")
    # 调整布局
    plt.tight_layout()
    # 保存结果到指定路径
    plt.savefig(save_path)
    # 打印保存路径
    print(f"推理结果已保存到: {save_path}")


def main():
    """主推理函数"""
    # 设置设备（优先使用 GPU，如果没有则使用 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 打印使用的设备
    print(f"使用设备：{device}")
    
    # 加载模型
    print("加载模型...")
    net = load_model(device)
    
    # 获取测试样本（默认第一张）
    sample_index = 0
    print(f"获取测试样本 {sample_index}...")
    image, true_label = get_sample_image(sample_index)
    
    # 进行推理
    print("进行推理...")
    pred_label, confidence = predict(net, image, device)
    
    # 打印结果
    print(f"\n推理结果：")
    print(f"真实标签: {CLASSES[true_label]}")
    print(f"预测标签: {CLASSES[pred_label]}")
    print(f"置信度: {confidence:.2%}")
    # 打印预测是否正确
    print(f"预测{'正确' if pred_label == true_label else '错误'}")
    
    # 可视化并保存结果
    visualize_result(image, true_label, pred_label, confidence, INFER_PATH)


if __name__ == '__main__':
    # 调用主函数
    main()
