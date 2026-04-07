"""模型网络模块

实现anchor-free的目标检测模型，包括特征提取和预测头。
精简版：减少参数量，提高训练速度。
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from config import NUM_CLASSES, INPUT_CHANNELS, HIDDEN_CHANNELS, AVG_BOX_WIDTH, AVG_BOX_HEIGHT

class AnchorFreeDetector(nn.Module):
    """Anchor-free目标检测模型（精简版）"""
    
    def __init__(self):
        """初始化模型
        """
        super(AnchorFreeDetector, self).__init__()
        
        # 特征提取网络（精简版，带BatchNorm）
        self.backbone = nn.Sequential(
            # 第一层卷积：输入通道3，输出通道64
            # 输入：[batch_size, 3, 512, 512]
            # 输出：[batch_size, 64, 256, 256]
            nn.Conv2d(INPUT_CHANNELS, HIDDEN_CHANNELS, 3, padding=1),
            nn.BatchNorm2d(HIDDEN_CHANNELS),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 第二层卷积：输入通道64，输出通道128
            # 输入：[batch_size, 64, 256, 256]
            # 输出：[batch_size, 128, 128, 128]
            nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS*2, 3, padding=1),
            nn.BatchNorm2d(HIDDEN_CHANNELS*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 第三层卷积：输入通道128，输出通道256
            # 输入：[batch_size, 128, 128, 128]
            # 输出：[batch_size, 256, 128, 128]
            nn.Conv2d(HIDDEN_CHANNELS*2, HIDDEN_CHANNELS*4, 3, padding=1),
            nn.BatchNorm2d(HIDDEN_CHANNELS*4),
            nn.ReLU(inplace=True),
        )
        
        # Heatmap 预测头：输出通道数为类别数
        # 输入：[batch_size, 256, 128, 128]
        # 输出：[batch_size, NUM_CLASSES, 128, 128]
        self.heatmap_head = nn.Conv2d(HIDDEN_CHANNELS*4, NUM_CLASSES, 1)
        
        # 偏移量预测头：输出通道数为2（x, y偏移）
        # 输入：[batch_size, 256, 128, 128]
        # 输出：[batch_size, 2, 128, 128]
        self.offset_head = nn.Conv2d(HIDDEN_CHANNELS*4, 2, 1)
        
        # 宽高预测头：输出通道数为2（width, height）
        # 输入：[batch_size, 256, 128, 128]
        # 输出：[batch_size, 2, 128, 128]
        self.wh_head = nn.Conv2d(HIDDEN_CHANNELS*4, 2, 1)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重
        
        对卷积层使用He初始化，对BatchNorm层使用默认初始化。
        这样可以加快训练收敛，提高训练稳定性。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 对卷积层使用He初始化（适合ReLU激活函数）
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # 偏置初始化为0
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm层默认初始化：weight=1，bias=0
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        
        # 对预测头进行特殊初始化
        # Heatmap头的偏置初始化为一个较小的负值（类似CenterNet的做法）
        # 这样可以避免初始阶段输出过多的假阳性
        init.constant_(self.heatmap_head.bias, -2.19)
        
        # Offset头的偏置初始化为0
        init.constant_(self.offset_head.bias, 0)
        
        # Wh头的偏置初始化为box的平均大小
        # 这样可以让模型从一个合理的初始值开始预测宽高
        # wh_head的输出通道顺序是[width, height]
        # bias[0]对应width，bias[1]对应height
        with torch.no_grad():
            self.wh_head.bias[0] = AVG_BOX_WIDTH
            self.wh_head.bias[1] = AVG_BOX_HEIGHT
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入图像张量，形状为 [batch_size, 3, 512, 512]
            
        Returns:
            heatmap: 类别heatmap，形状为 [batch_size, NUM_CLASSES, 128, 128]
            offsets: 偏移量，形状为 [batch_size, 2, 128, 128]
            wh: 宽高，形状为 [batch_size, 2, 128, 128]
        """
        # 提取特征
        # 输入：[batch_size, 3, 512, 512]
        # 输出：[batch_size, 256, 128, 128]
        features = self.backbone(x)
        
        # 预测heatmap（使用sigmoid激活函数）
        heatmap = torch.sigmoid(self.heatmap_head(features))
        
        # 预测偏移量（使用tanh激活函数，输出范围[-1, 1]，然后缩放到[-0.5, 0.5]）
        offsets = torch.tanh(self.offset_head(features)) * 0.5
        
        # 预测宽高（使用ReLU激活函数，确保宽高为正）
        wh = torch.relu(self.wh_head(features))
        
        return heatmap, offsets, wh

class HeatmapLoss(nn.Module):
    """Heatmap损失函数
    
    实现了Focal Loss用于heatmap分类，L1 Loss用于offset回归，Smooth L1 Loss用于wh回归。
    添加数值稳定性保护，防止NaN。
    """
    
    def __init__(self, alpha=2, beta=4):
        """初始化损失函数
        
        Args:
            alpha: Focal Loss的alpha参数，控制难易样本的权重
            beta: Focal Loss的beta参数，控制正负样本的权重
        """
        super(HeatmapLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-7  # 防止log(0)的epsilon
    
    def _l1_loss(self, pred, target):
        """自己实现的L1 Loss
        
        公式：loss = |pred - target|
        """
        return torch.abs(pred - target)
    
    def _smooth_l1_loss(self, pred, target, beta=1.0):
        """自己实现的Smooth L1 Loss
        
        公式：
        loss = {
            0.5 * (pred - target)^2 / beta,  if |pred - target| < beta
            |pred - target| - 0.5 * beta,    otherwise
        }
        """
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < beta,
            0.5 * (diff ** 2) / beta,
            diff - 0.5 * beta
        )
        return loss
    
    def forward(self, pred_heatmap, pred_offsets, pred_wh, target_heatmap, target_offsets, target_wh):
        """计算损失
        
        Args:
            pred_heatmap: 预测的heatmap
            pred_offsets: 预测的偏移量
            pred_wh: 预测的宽高
            target_heatmap: 目标heatmap
            target_offsets: 目标偏移量
            target_wh: 目标宽高
            
        Returns:
            总损失
        """
        # Heatmap 损失 (focal loss)
        # 正样本掩码：标记目标中心位置（值为1.0的区域）
        pos_mask = target_heatmap == 1.0
        # 负样本掩码：标记非目标中心位置（值小于1.0的区域）
        neg_mask = target_heatmap < 1.0
        
        # Focal loss for heatmap
        # 添加epsilon防止log(0)导致NaN
        pred_heatmap = torch.clamp(pred_heatmap, min=self.eps, max=1.0 - self.eps)
        
        # 正样本损失：对难分类的正样本（预测概率低的样本）给予更高的权重
        # 公式：- (1 - p)^alpha * log(p)
        # 其中p是预测概率，alpha是调节因子
        pos_loss = -torch.pow(1 - pred_heatmap[pos_mask], self.alpha) * torch.log(pred_heatmap[pos_mask])
        
        # 负样本损失：对难分类的负样本（预测概率高的样本）给予更高的权重
        # 公式：- (1 - t)^beta * p^alpha * log(1 - p)
        # 其中t是目标值，p是预测概率，alpha和beta是调节因子
        # 这里使用target_heatmap[neg_mask]作为(1 - t)，因为target_heatmap在负样本区域接近0
        neg_loss = -torch.pow(1 - target_heatmap[neg_mask], self.beta) * torch.pow(pred_heatmap[neg_mask], self.alpha) * torch.log(1 - pred_heatmap[neg_mask])
        
        # 处理空的情况
        if pos_loss.numel() == 0:
            pos_loss = torch.tensor(0.0, device=pred_heatmap.device)
        else:
            pos_loss = torch.mean(pos_loss)
        
        if neg_loss.numel() == 0:
            neg_loss = torch.tensor(0.0, device=pred_heatmap.device)
        else:
            neg_loss = torch.mean(neg_loss)
        
        heatmap_loss = pos_loss + neg_loss
        
        # Offset loss (only for positive locations)
        max_cls_mask = target_heatmap.max(dim=1, keepdim=True)[0] == 1.0
        offset_pos_mask = max_cls_mask.repeat(1, 2, 1, 1)
        
        if offset_pos_mask.sum() > 0:
            offset_loss = torch.mean(self._l1_loss(pred_offsets[offset_pos_mask], target_offsets[offset_pos_mask]))
        else:
            offset_loss = torch.tensor(0.0, device=pred_heatmap.device)
        
        # Width/Height loss (only for positive locations)
        if offset_pos_mask.sum() > 0:
            wh_loss = torch.mean(self._smooth_l1_loss(pred_wh[offset_pos_mask], target_wh[offset_pos_mask], beta=1.0))
        else:
            wh_loss = torch.tensor(0.0, device=pred_heatmap.device)
        
        return heatmap_loss + 0.1 * offset_loss + 0.1 * wh_loss
