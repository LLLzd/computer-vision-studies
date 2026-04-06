"""
测试 MNIST 数据集下载
"""
from torchvision import datasets, transforms

print("开始下载 MNIST 数据集...")
print("如果下载速度慢，请耐心等待...")

transform = transforms.ToTensor()

# 先下载训练集
print("\n下载训练集...")
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
print(f"✓ 训练集下载完成，大小: {len(train_dataset)}")

# 再下载测试集
print("\n下载测试集...")
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
print(f"✓ 测试集下载完成，大小: {len(test_dataset)}")

print("\n✓ 所有数据下载完成！")
print("现在可以运行 train.py 开始训练")
