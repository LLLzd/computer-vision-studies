"""
训练脚本
用于训练 MNIST 手写数字分类模型
"""
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 PyTorch 的优化器模块
import torch.optim as optim
# 导入 torchvision 库，用于加载数据集
import torchvision
# 导入 matplotlib 库，用于绘制训练曲线
import matplotlib.pyplot as plt

# 从配置文件导入必要的配置
from config import DATA_DIR, MODEL_PATH, PLOT_PATH, TRANSFORM, BATCH_SIZE, LEARNING_RATE, EPOCHS, CLASSES
# 从模型文件导入网络结构
from net import Net


def get_data_loaders():
    """获取训练和测试数据加载器"""
    # 加载 MNIST 训练集
    trainset = torchvision.datasets.MNIST(
        root=DATA_DIR,  # 数据集保存路径
        train=True,  # 加载训练集
        download=True,  # 如果数据集不存在则下载
        transform=TRANSFORM  # 应用数据预处理转换
    )
    # 加载 MNIST 测试集
    testset = torchvision.datasets.MNIST(
        root=DATA_DIR,  # 数据集保存路径
        train=False,  # 加载测试集
        download=True,  # 如果数据集不存在则下载
        transform=TRANSFORM  # 应用数据预处理转换
    )
    
    # 创建训练数据加载器
    trainloader = torch.utils.data.DataLoader(
        trainset,  # 数据集
        batch_size=BATCH_SIZE,  # 批次大小
        shuffle=True  # 训练时打乱数据
    )
    # 创建测试数据加载器
    testloader = torch.utils.data.DataLoader(
        testset,  # 数据集
        batch_size=BATCH_SIZE,  # 批次大小
        shuffle=False  # 测试时不需要打乱数据
    )
    
    # 返回训练和测试数据加载器
    return trainloader, testloader


def train_epoch(net, trainloader, criterion, optimizer, device):
    """
    训练一个epoch
    Returns:
        avg_loss: 平均损失
        avg_acc: 平均准确率
    """
    # 设置模型为训练模式（启用 dropout、batch norm 等）
    net.train()
    # 初始化当前轮的累计损失
    running_loss = 0.0
    # 初始化正确预测数
    correct = 0
    # 初始化总样本数
    total = 0
    
    # 遍历训练数据加载器
    for inputs, labels in trainloader:
        # 将输入和标签移动到指定设备
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 清零优化器的梯度
        optimizer.zero_grad()
        # 前向传播，获取模型输出
        outputs = net(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播，计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()
        
        # 累加损失值
        running_loss += loss.item()
        # 获取预测结果（最大值和索引）
        _, predicted = torch.max(outputs, 1)
        # 累加总样本数
        total += labels.size(0)
        # 累加正确预测数
        correct += (predicted == labels).sum().item()
    
    # 计算平均损失
    avg_loss = running_loss / len(trainloader)
    # 计算准确率
    avg_acc = 100 * correct / total
    # 返回平均损失和准确率
    return avg_loss, avg_acc


def evaluate(net, testloader, device):
    """
    在测试集上评估模型
    Returns:
        accuracy: 测试集准确率
    """
    # 设置模型为评估模式（关闭 dropout 等训练特有的层）
    net.eval()
    # 初始化正确预测数
    correct = 0
    # 初始化总样本数
    total = 0
    
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
    
    # 计算准确率
    return 100 * correct / total


def plot_curves(train_loss_history, train_acc_history):
    """绘制训练曲线"""
    # 创建图形，大小为 12x4
    plt.figure(figsize=(12, 4))
    
    # 绘制 Loss 曲线（子图 1）
    plt.subplot(1, 2, 1)
    # 绘制训练损失曲线，标签为 "Train Loss"
    plt.plot(train_loss_history, label="Train Loss")
    # 设置标题
    plt.title("Loss Curve")
    # 设置 x 轴标签
    plt.xlabel("Epoch")
    # 设置 y 轴标签
    plt.ylabel("Loss")
    # 显示图例
    plt.legend()
    
    # 绘制 Accuracy 曲线（子图 2）
    plt.subplot(1, 2, 2)
    # 绘制训练准确率曲线，标签为 "Train Acc"，颜色为橙色
    plt.plot(train_acc_history, label="Train Acc", color="orange")
    # 设置标题
    plt.title("Accuracy Curve")
    # 设置 x 轴标签
    plt.xlabel("Epoch")
    # 设置 y 轴标签
    plt.ylabel("Accuracy (%)")
    # 显示图例
    plt.legend()
    
    # 调整布局
    plt.tight_layout()
    # 保存图形到指定路径
    plt.savefig(PLOT_PATH)
    # 打印保存路径
    print(f"训练曲线已保存到：{PLOT_PATH}")


def main():
    """主训练函数"""
    # 设置设备（优先使用 GPU，如果没有则使用 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 打印使用的设备
    print(f"使用设备：{device}")
    
    # 获取训练和测试数据加载器
    trainloader, testloader = get_data_loaders()
    
    # 创建网络实例并移动到指定设备
    net = Net().to(device)
    
    # 定义损失函数（交叉熵损失）
    criterion = nn.CrossEntropyLoss()
    # 定义优化器（Adam 优化器）
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    # 初始化训练损失历史列表
    train_loss_history = []
    # 初始化训练准确率历史列表
    train_acc_history = []
    
    # 打印开始训练的提示
    print("开始训练...")
    # 遍历每个训练轮次
    for epoch in range(EPOCHS):
        # 训练一个轮次，获取损失和准确率
        loss, acc = train_epoch(net, trainloader, criterion, optimizer, device)
        # 保存损失到历史记录
        train_loss_history.append(loss)
        # 保存准确率到历史记录
        train_acc_history.append(acc)
        # 打印当前轮的损失和准确率
        print(f"[{epoch+1}/{EPOCHS}] loss: {loss:.3f} | acc: {acc:.2f}%")
    
    # 保存模型权重
    torch.save(net.state_dict(), MODEL_PATH)
    # 打印保存路径
    print(f"训练完成！模型已保存到：{MODEL_PATH}")
    
    # 绘制训练曲线
    plot_curves(train_loss_history, train_acc_history)
    
    # 评估模型
    test_acc = evaluate(net, testloader, device)
    # 打印测试集准确率
    print(f"\n测试集准确率: {test_acc:.2f}%")
    
    # 显示前10个测试样本的预测结果
    print("\n前10个测试样本的预测结果：")
    # 创建测试数据迭代器
    dataiter = iter(testloader)
    # 获取第一个批次的图像和标签
    images, labels = next(dataiter)
    # 将图像移动到指定设备
    images = images.to(device)
    # 前向传播，获取模型输出
    outputs = net(images)
    # 获取预测结果（最大值和索引）
    _, predicted = torch.max(outputs, 1)
    
    # 遍历前10个样本
    for i in range(10):
        # 打印预测结果和真实标签
        print(f"样本 {i+1}: 预测: {CLASSES[predicted[i]]}, 真实: {CLASSES[labels[i]]}")


if __name__ == '__main__':
    # 调用主函数
    main()
