"""
测试/评估脚本
用于在测试集上评估已训练模型的性能
"""
# 导入 PyTorch 库
import torch
# 导入 torchvision 库，用于加载数据集
import torchvision

# 从配置文件导入必要的配置
from config import DATA_DIR, MODEL_PATH, TRANSFORM, BATCH_SIZE, CLASSES
# 从模型文件导入网络结构
from net import Net


def get_test_loader():
    """获取测试数据加载器"""
    # 加载 CIFAR-10 测试集
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,  # 数据集保存路径
        train=False,  # 加载测试集（train=True 表示加载训练集）
        download=True,  # 如果数据集不存在则下载
        transform=TRANSFORM  # 应用数据预处理转换
    )
    # 创建测试数据加载器
    testloader = torch.utils.data.DataLoader(
        testset,  # 数据集
        batch_size=BATCH_SIZE,  # 批次大小
        shuffle=False  # 测试时不需要打乱数据
    )
    # 返回数据加载器
    return testloader


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


def evaluate(net, testloader, device):
    """
    在测试集上评估模型
    Returns:
        accuracy: 测试集准确率
        predictions: 预测结果列表
        ground_truths: 真实标签列表
    """
    # 初始化正确预测数
    correct = 0
    # 初始化总样本数
    total = 0
    # 初始化预测结果列表
    predictions = []
    # 初始化真实标签列表
    ground_truths = []
    
    # 禁用梯度计算，提高评估速度
    with torch.no_grad():
        # 遍历测试数据加载器
        for images, labels in testloader:
            # 将图像和标签移动到指定设备
            images, labels = images.to(device), labels.to(device)
            # 前向传播，获取模型输出
            outputs = net(images)
            # 获取预测结果（最大值和索引）
            _, predicted = torch.max(outputs.data, 1)
            
            # 累加总样本数
            total += labels.size(0)
            # 累加正确预测数
            correct += (predicted == labels).sum().item()
            
            # 将预测结果和真实标签添加到列表中（转换为 numpy 数组）
            predictions.extend(predicted.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())
    
    # 计算准确率
    accuracy = 100 * correct / total
    # 返回准确率、预测结果和真实标签
    return accuracy, predictions, ground_truths


def main():
    """主测试函数"""
    # 设置设备（优先使用 GPU，如果没有则使用 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 打印使用的设备
    print(f"使用设备：{device}")
    
    # 加载模型
    print("加载模型...")
    net = load_model(device)
    
    # 获取测试数据加载器
    testloader = get_test_loader()
    
    # 评估模型
    print("开始评估...")
    accuracy, predictions, ground_truths = evaluate(net, testloader, device)
    
    # 打印测试集准确率
    print(f"\n测试集准确率: {accuracy:.2f}%")
    
    # 显示前10个测试样本的预测结果
    print("\n前10个测试样本的预测结果：")
    # 遍历前10个样本
    for i in range(10):
        # 打印预测结果和真实标签
        print(f"样本 {i+1}: 预测: {CLASSES[predictions[i]]}, 真实: {CLASSES[ground_truths[i]]}")


if __name__ == '__main__':
    # 调用主函数
    main()
