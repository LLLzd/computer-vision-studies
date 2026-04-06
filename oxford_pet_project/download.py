"""
数据集下载脚本
用于下载 Oxford-IIIT Pet 数据集并测试加载

数据集格式说明：
- images.tar.gz: 包含所有宠物图像
- annotations.tar.gz: 包含分割标注和其他元数据
- 解压后目录结构：
  oxford-iiit-pet/
  ├── images/          # 宠物图像
  ├── annotations/     # 标注文件
  │   ├── trimaps/     # 分割标注（1=前景，2=背景，3=未分类）
  │   ├── xmls/        # 边界框标注
  │   ├── list.txt     # 数据集列表
  │   ├── trainval.txt # 训练验证集划分
  │   └── test.txt     # 测试集划分
"""

# 导入必要的库
import os  # 用于文件路径操作
import urllib.request  # 用于下载文件
import tarfile  # 用于解压 tar.gz 文件

# 从配置文件导入必要的配置
from config import DATA_DIR, dataset_url, annotations_url, transform, label_transform


# 从 torchvision 导入数据集
from torchvision.datasets import OxfordIIITPet


def download_file(url, save_path):
    """
    下载文件
    Args:
        url: 文件下载链接
        save_path: 保存路径
    Returns:
        bool: 下载是否成功
    """
    print(f"正在下载: {url}")  # 打印下载信息
    try:
        # 使用 urllib 下载文件
        urllib.request.urlretrieve(url, save_path)
        print(f"✓ 下载完成: {save_path}")  # 下载成功
        return True
    except Exception as e:
        print(f"✗ 下载失败: {e}")  # 下载失败
        return False


def extract_tar(tar_path, extract_path):
    """
    解压 tar.gz 文件
    Args:
        tar_path: tar.gz 文件路径
        extract_path: 解压路径
    Returns:
        bool: 解压是否成功
    """
    print(f"正在解压: {tar_path}")  # 打印解压信息
    try:
        # 打开并解压 tar.gz 文件
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_path)
        print(f"✓ 解压完成: {extract_path}")  # 解压成功
        return True
    except Exception as e:
        print(f"✗ 解压失败: {e}")  # 解压失败
        return False


def test_dataset():
    """
    测试数据集加载
    Returns:
        bool: 测试是否成功
    """
    print("\n=== 测试数据集加载 ===")
    print(f"Data directory: {DATA_DIR}")  # 打印数据目录路径
    print(f"Data directory exists: {os.path.exists(DATA_DIR)}")  # 检查数据目录是否存在

    # 检查数据目录结构
    if os.path.exists(DATA_DIR):  # 如果数据目录存在
        print("\nData directory contents:")  # 打印目录内容
        for root, dirs, files in os.walk(DATA_DIR):  # 遍历目录
            level = root.replace(DATA_DIR, '').count(os.sep)  # 计算目录层级
            indent = ' ' * 2 * level  # 缩进
            print(f"{indent}{os.path.basename(root)}/")  # 打印目录名
            subindent = ' ' * 2 * (level + 1)  # 子目录缩进
            for file in files[:5]:  # 只显示前5个文件
                print(f"{subindent}{file}")  # 打印文件名
            if len(files) > 5:  # 如果文件数量超过5个
                print(f"{subindent}... and {len(files) - 5} more files")  # 显示剩余文件数量

    # 尝试加载数据集
    try:
        print("\nAttempting to load dataset...")  # 打印尝试加载数据集信息
        dataset = OxfordIIITPet(
            root=DATA_DIR,  # 数据集保存路径
            split='trainval',  # 使用训练+验证集
            target_types='segmentation',  # 使用分割标注
            transform=transform,  # 图像预处理
            target_transform=label_transform,  # 标签预处理
            download=False  # 不下载数据
        )
        print(f"✅ Dataset loaded successfully! Size: {len(dataset)}")  # 打印数据集加载成功信息
        return True
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")  # 打印加载失败信息
        return False


def main():
    """
    主函数
    下载并解压 Oxford-IIIT Pet 数据集，然后测试加载
    """
    print("开始处理 Oxford-IIIT Pet 数据集...")  # 打印开始信息
    
    # 检查是否已经有数据
    images_dir = os.path.join(DATA_DIR, 'oxford-iiit-pet', 'images')
    annotations_dir = os.path.join(DATA_DIR, 'oxford-iiit-pet', 'annotations')
    
    data_exists = os.path.exists(images_dir) and os.path.exists(annotations_dir)
    
    if data_exists:
        print("⚠️  数据集已存在，跳过下载")
        # 测试数据集
        if test_dataset():
            print("\n🎉 数据集测试成功！")
        else:
            print("\n❌ 数据集测试失败，尝试重新下载...")
            data_exists = False  # 强制重新下载
    
    if not data_exists:
        # 下载图像文件
        images_tar = os.path.join(DATA_DIR, 'images.tar.gz')  # 图像文件保存路径
        if not os.path.exists(images_tar):  # 检查文件是否已存在
            if not download_file(dataset_url, images_tar):  # 下载文件
                print("❌ 图像文件下载失败")  # 下载失败
                return
        else:
            print("⚠️  图像文件已存在，跳过下载")  # 文件已存在，跳过
        
        # 下载标注文件
        annotations_tar = os.path.join(DATA_DIR, 'annotations.tar.gz')  # 标注文件保存路径
        if not os.path.exists(annotations_tar):  # 检查文件是否已存在
            if not download_file(annotations_url, annotations_tar):  # 下载文件
                print("❌ 标注文件下载失败")  # 下载失败
                return
        else:
            print("⚠️  标注文件已存在，跳过下载")  # 文件已存在，跳过
        
        # 解压图像文件
        if not os.path.exists(images_dir):  # 检查目录是否已存在
            if not extract_tar(images_tar, DATA_DIR):  # 解压文件
                print("❌ 图像文件解压失败")  # 解压失败
                return
        else:
            print("⚠️  图像目录已存在，跳过解压")  # 目录已存在，跳过
        
        # 解压标注文件
        if not os.path.exists(annotations_dir):  # 检查目录是否已存在
            if not extract_tar(annotations_tar, DATA_DIR):  # 解压文件
                print("❌ 标注文件解压失败")  # 解压失败
                return
        else:
            print("⚠️  标注目录已存在，跳过解压")  # 目录已存在，跳过
        
        print("\n🎉 Oxford-IIIT Pet 数据集下载完成！")  # 下载完成
        print(f"图像目录: {images_dir}")  # 打印图像目录
        print(f"标注目录: {annotations_dir}")  # 打印标注目录
        
        # 测试数据集
        if test_dataset():
            print("\n🎉 数据集测试成功！")
        else:
            print("\n❌ 数据集测试失败")


if __name__ == '__main__':
    # 调用主函数
    main()
