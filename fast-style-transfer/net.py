import torch
import torch.nn as nn

class TransformerNetwork(nn.Module):
    """风格迁移网络"""
    def __init__(self):
        super(TransformerNetwork, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            # 第一层
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 第二层
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 第三层
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU()
        )
        
        # 残差块
        self.residuals = nn.Sequential(
            self._residual_block(128),
            self._residual_block(128),
            self._residual_block(128),
            self._residual_block(128),
            self._residual_block(128)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            # 上采样层1
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 上采样层2
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 输出层
            nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )
    
    def _residual_block(self, channels):
        """创建残差块"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        """前向传播"""
        x = self.encoder(x)
        x = x + self.residuals(x)  # 残差连接
        x = self.decoder(x)
        return x

class VGG19Features(nn.Module):
    """VGG19特征提取网络"""
    def __init__(self):
        super(VGG19Features, self).__init__()
        from torchvision.models import vgg19
        vgg = vgg19(pretrained=True)
        
        # 选择特定的层用于特征提取
        self.features = nn.Sequential(
            *list(vgg.features)[:30]  # 截取到conv4_4层
        )
        
        # 冻结参数
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """前向传播"""
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            # 保存特定层的特征
            if i in [3, 8, 13, 22, 31]:  # relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
                features.append(x)
        return features

class ContentLoss(nn.Module):
    """内容损失"""
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, content_features, target_features):
        """计算内容损失"""
        return self.mse(content_features, target_features)

class StyleLoss(nn.Module):
    """风格损失"""
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, style_features, target_features):
        """计算风格损失"""
        loss = 0
        for style, target in zip(style_features, target_features):
            style_gram = self._gram_matrix(style)
            target_gram = self._gram_matrix(target)
            loss += self.mse(style_gram, target_gram)
        return loss
    
    def _gram_matrix(self, x):
        """计算Gram矩阵"""
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size * channels, height * width)
        return torch.mm(x, x.t()) / (batch_size * channels * height * width)

class TotalVariationLoss(nn.Module):
    """总变分损失"""
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
    
    def forward(self, x):
        """计算总变分损失"""
        batch_size, channels, height, width = x.size()
        horizontal = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        vertical = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        return (horizontal + vertical) / (batch_size * channels * height * width)
