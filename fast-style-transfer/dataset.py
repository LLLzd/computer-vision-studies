import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class StyleTransferDataset(Dataset):
    """风格迁移数据集"""
    def __init__(self, content_dir, image_size=256):
        """初始化数据集
        
        Args:
            content_dir: 内容图像目录
            image_size: 图像尺寸
        """
        self.content_dir = content_dir
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 获取所有图像路径
        self.image_paths = []
        for root, _, files in os.walk(content_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(self.image_paths)} content images in {content_dir}")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            image: 预处理后的图像
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image

def get_dataloader(content_dir, batch_size=4, image_size=256, num_workers=2):
    """获取数据加载器
    
    Args:
        content_dir: 内容图像目录
        batch_size: 批量大小
        image_size: 图像尺寸
        num_workers: 工作线程数
        
    Returns:
        dataloader: 数据加载器
    """
    dataset = StyleTransferDataset(content_dir, image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader
