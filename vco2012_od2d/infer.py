"""推理模块

实现模型的推理流程，包括模型加载、图像预处理、推理和结果可视化。
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from net import AnchorFreeDetector
from dataset import get_test_image_paths
from config import (
    JPEGIMAGES_DIR, ANNOTATIONS_DIR, OUTPUT_DIR, IMAGE_SIZE, HEATMAP_SIZE, 
    VOC_CLASSES, MODEL_PATH, THRESHOLD, IOU_THRESHOLD, MAX_DETECTIONS
)
from preprocess import Preprocessor

class InferenceRunner:
    """推理器类"""
    
    def __init__(self, model_path, device='cpu'):
        """初始化推理器
        
        Args:
            model_path: 模型路径
            device: 设备
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.model.to(device)
        self.model.eval()
        self.preprocessor = Preprocessor(target_size=IMAGE_SIZE)
    
    def _load_model(self, model_path):
        """加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            加载好的模型
        """
        model = AnchorFreeDetector()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def predict(self, image_path, threshold=THRESHOLD):
        """预测单个图像
        
        Args:
            image_path: 图像路径
            threshold: 检测阈值
            
        Returns:
            image: 预处理后的图像
            detections: 检测结果
        """
        # 预处理图像（只处理图像，不处理标注）
        image, _ = self.preprocessor.preprocess_image(image_path)
        
        # 转换为张量
        image_tensor = torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32) / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # 前向传播
        with torch.no_grad():
            heatmap, offsets, wh = self.model(image_tensor)
        
        # 处理预测结果
        detections = self._process_predictions(heatmap[0], offsets[0], wh[0], threshold)
        
        return image, detections
    
    def _process_predictions(self, heatmap, offsets, wh, threshold):
        """处理预测结果
        
        Args:
            heatmap: 预测的heatmap
            offsets: 预测的偏移量
            wh: 预测的宽高
            threshold: 检测阈值
            
        Returns:
            处理后的检测结果
        """
        detections = []
        
        # 计算缩放因子
        scale_x = IMAGE_SIZE[0] / HEATMAP_SIZE[1]
        scale_y = IMAGE_SIZE[1] / HEATMAP_SIZE[0]
        
        # 遍历每个类别
        for cls_idx in range(len(VOC_CLASSES)):
            cls_heatmap = heatmap[cls_idx].cpu().numpy()
            
            # 找到所有超过阈值的位置
            y_coords, x_coords = np.where(cls_heatmap > threshold)
            
            # 收集当前类别的检测结果
            cls_detections = []
            for y, x in zip(y_coords, x_coords):
                # 获取偏移量
                offset_x = offsets[0, y, x].cpu().item()
                offset_y = offsets[1, y, x].cpu().item()
                
                # 获取宽高（预测的是原图尺度的宽高）
                width = wh[0, y, x].cpu().item()
                height = wh[1, y, x].cpu().item()
                
                # 计算实际坐标
                center_x = (x + offset_x) * scale_x
                center_y = (y + offset_y) * scale_y
                
                # 使用预测的宽高计算边界框
                xmin = int(center_x - width / 2)
                ymin = int(center_y - height / 2)
                xmax = int(center_x + width / 2)
                ymax = int(center_y + height / 2)
                
                # 确保边界框在图像范围内
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(IMAGE_SIZE[0], xmax)
                ymax = min(IMAGE_SIZE[1], ymax)
                
                # 添加检测结果
                cls_detections.append({
                    'class': VOC_CLASSES[cls_idx],
                    'confidence': float(cls_heatmap[y, x]),
                    'bbox': [xmin, ymin, xmax, ymax]
                })
            
            # 对当前类别的检测结果按置信度排序
            cls_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # 对当前类别应用非极大值抑制（NMS）
            cls_detections = self._non_max_suppression(cls_detections, iou_threshold=IOU_THRESHOLD)
            
            # 限制每个类别的检测数量
            detections.extend(cls_detections[:10])  # 每个类别最多保留10个检测结果
        
        # 按置信度排序所有检测结果
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 限制总检测数量
        return detections[:MAX_DETECTIONS]  # 最多保留MAX_DETECTIONS个检测结果
    
    def _non_max_suppression(self, detections, iou_threshold=IOU_THRESHOLD):
        """非极大值抑制
        
        Args:
            detections: 检测结果列表
            iou_threshold: IoU阈值
            
        Returns:
            经过NMS处理的检测结果
        """
        if len(detections) == 0:
            return []
        
        # 提取边界框和置信度
        bboxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # 计算边界框面积
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        
        # 按置信度排序
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            # 选择置信度最高的边界框
            i = order[0]
            keep.append(detections[i])
            
            # 计算与其他边界框的IoU
            if order.size == 1:
                break
            
            # 计算交集
            xx1 = np.maximum(bboxes[i, 0], bboxes[order[1:], 0])
            yy1 = np.maximum(bboxes[i, 1], bboxes[order[1:], 1])
            xx2 = np.minimum(bboxes[i, 2], bboxes[order[1:], 2])
            yy2 = np.minimum(bboxes[i, 3], bboxes[order[1:], 3])
            
            # 计算交集面积
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            # 计算IoU，添加epsilon防止除零
            union = areas[i] + areas[order[1:]] - inter
            union = np.maximum(union, 1e-10)  # 确保分母不为零
            iou = inter / union
            
            # 保留IoU小于阈值的边界框
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def visualize(self, image, detections, output_path, box_color='r'):
        """可视化检测结果
        
        Args:
            image: 图像
            detections: 检测结果
            output_path: 输出路径
            box_color: 边界框颜色
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        # 绘制边界框
        for det in detections:
            xmin, ymin, xmax, ymax = det['bbox']
            confidence = det.get('confidence', 1.0)  # 真值标签没有置信度
            cls = det['class']
            
            # 绘制边界框
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                   linewidth=2, edgecolor=box_color, facecolor='none')
            plt.gca().add_patch(rect)
            
            # 添加类别和置信度
            if 'confidence' in det:
                plt.text(xmin, ymin - 10, f'{cls}: {confidence:.2f}', 
                        color=box_color, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
            else:
                plt.text(xmin, ymin - 10, cls, 
                        color=box_color, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title('Object Detection Results')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """主函数"""
    # 最佳模型路径
    best_model_path = MODEL_PATH.replace('.pth', '_best.pth')
    
    # 优先使用最佳模型
    if os.path.exists(best_model_path):
        model_path = best_model_path
        print(f"✅ Using best model: {best_model_path}")
    elif os.path.exists(MODEL_PATH):
        model_path = MODEL_PATH
        print(f"⚠️  Best model not found, using regular model: {MODEL_PATH}")
    else:
        print(f"❌ Error: No model found at {MODEL_PATH} or {best_model_path}")
        print("Please train the model first")
        return
    
    print("=" * 60)
    print("🚀 Starting inference...")
    print("=" * 60)
    
    # 初始化推理器
    runner = InferenceRunner(model_path)
    
    # 初始化预处理器
    preprocessor = Preprocessor(target_size=IMAGE_SIZE)
    
    # 从测试集中随机选择4张图像
    test_images = get_test_image_paths(num_images=4)
    
    print(f"\n📸 Selected {len(test_images)} images from test set:")
    for i, img in enumerate(test_images):
        print(f"   [{i+1}] {img}")
    
    # 创建两行四列的子图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 处理每张图像
    for i, image_name in enumerate(test_images):
        image_path = os.path.join(JPEGIMAGES_DIR, image_name)
        annotation_path = os.path.join(ANNOTATIONS_DIR, image_name.replace('.jpg', '.xml'))
        
        print(f"\n📷 [{i+1}/4] Processing: {image_name}")
        
        # 1. 预处理后的图像 + 真值标签（绿色box）
        preprocessed_image, boxes = preprocessor.preprocess_sample(image_path, annotation_path)
        axes[0, i].imshow(preprocessed_image)
        # 转换为检测结果格式
        gt_detections = []
        for cls, xmin, ymin, xmax, ymax in boxes:
            gt_detections.append({
                'class': cls,
                'bbox': [xmin, ymin, xmax, ymax]
            })
        # 可视化真值标签
        for det in gt_detections:
            xmin, ymin, xmax, ymax = det['bbox']
            cls = det['class']
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                   linewidth=2, edgecolor='g', facecolor='none')
            axes[0, i].add_patch(rect)
            axes[0, i].text(xmin, ymin - 10, cls, 
                          color='g', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        axes[0, i].set_title('Ground Truth')
        axes[0, i].axis('off')
        
        # 2. 预处理后的图像 + 预测结果（红色box + 类别）
        _, pred_detections = runner.predict(image_path)
        axes[1, i].imshow(preprocessed_image)
        for det in pred_detections:
            xmin, ymin, xmax, ymax = det['bbox']
            confidence = det['confidence']
            cls = det['class']
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                   linewidth=2, edgecolor='r', facecolor='none')
            axes[1, i].add_patch(rect)
            axes[1, i].text(xmin, ymin - 10, f'{cls}: {confidence:.2f}', 
                          color='r', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        axes[1, i].set_title('Prediction')
        axes[1, i].axis('off')
        
        # 打印预测信息
        print(f"   ✅ Ground Truth: {len(gt_detections)} objects")
        print(f"   🎯 Predictions: {len(pred_detections)} objects")
        for j, det in enumerate(pred_detections[:3]):  # 只显示前3个
            print(f"      [{j+1}] {det['class']}: {det['confidence']:.2f}")
        if len(pred_detections) > 3:
            print(f"      ... and {len(pred_detections) - 3} more")
    
    # 调整布局
    plt.tight_layout()
    
    # 保存结果
    output_path = os.path.join(OUTPUT_DIR, 'detection_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 60)
    print(f"✅ Detection results saved to: {output_path}")
    print("🎉 Inference completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()