"""测试模块

实现模型的测试流程，包括在验证集上评估模型性能。
参考CenterNet的多类别目标检测评估标准。
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import VOCDataset
from net import AnchorFreeDetector
from config import BATCH_SIZE, NUM_WORKERS, MODEL_PATH, OUTPUT_DIR, VOC_CLASSES, NUM_CLASSES, THRESHOLD, IOU_THRESHOLD, IMAGE_SIZE, HEATMAP_SIZE, QUICK_TEST, QUICK_TEST_BATCHES

def decode_detections(heatmap, offset, wh, threshold=THRESHOLD):
    """从预测结果中解码检测结果
    
    Args:
        heatmap: 预测的heatmap [1, NUM_CLASSES, H, W]
        offset: 预测的offset [1, 2, H, W]
        wh: 预测的wh [1, 2, H, W]
        threshold: 检测阈值
        
    Returns:
        boxes: 边界框列表
        scores: 置信度列表
        classes: 类别列表
    """
    boxes = []
    scores = []
    classes = []
    
    heatmap = heatmap[0]  # [NUM_CLASSES, H, W]
    offset = offset[0]  # [2, H, W]
    wh = wh[0]  # [2, H, W]
    
    # 计算缩放因子
    scale_x = IMAGE_SIZE[0] / HEATMAP_SIZE[1]
    scale_y = IMAGE_SIZE[1] / HEATMAP_SIZE[0]
    
    # 遍历每个类别
    for cls in range(heatmap.shape[0]):
        cls_heatmap = heatmap[cls].cpu().numpy()
        
        # 找到所有超过阈值的位置
        y_coords, x_coords = np.where(cls_heatmap > threshold)
        
        for y, x in zip(y_coords, x_coords):
            # 获取偏移量和宽高
            off_x = offset[0, y, x].cpu().item()
            off_y = offset[1, y, x].cpu().item()
            w = wh[0, y, x].cpu().item()
            h = wh[1, y, x].cpu().item()
            
            # 计算中心点（在原图尺度）
            cx = (x + off_x) * scale_x
            cy = (y + off_y) * scale_y
            
            # 计算边界框
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            # 过滤无效边界框
            if w <= 0 or h <= 0:
                continue
            if x1 < 0 or y1 < 0 or x2 > IMAGE_SIZE[0] or y2 > IMAGE_SIZE[1]:
                continue
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(cls_heatmap[y, x]))
            classes.append(cls)
    
    # 按置信度排序并限制最大检测数量
    if boxes:
        # 按置信度排序
        sorted_indices = np.argsort(scores)[::-1]
        boxes = [boxes[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        classes = [classes[i] for i in sorted_indices]
        
        # 限制最大检测数量
        from config import MAX_DETECTIONS
        if len(boxes) > MAX_DETECTIONS:
            boxes = boxes[:MAX_DETECTIONS]
            scores = scores[:MAX_DETECTIONS]
            classes = classes[:MAX_DETECTIONS]
    
    return boxes, scores, classes

def nms(boxes, scores, classes, iou_threshold=IOU_THRESHOLD):
    """非极大值抑制
    
    Args:
        boxes: 边界框列表
        scores: 置信度列表
        classes: 类别列表
        iou_threshold: IoU阈值
        
    Returns:
        过滤后的boxes, scores, classes
    """
    if len(boxes) == 0:
        return [], [], []
    
    # 转换为numpy数组
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    
    # 按类别分组处理
    keep_boxes = []
    keep_scores = []
    keep_classes = []
    
    for cls in np.unique(classes):
        # 获取当前类别的索引
        cls_mask = classes == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        
        # 按置信度排序
        order = cls_scores.argsort()[::-1]
        
        # 计算面积
        areas = (cls_boxes[:, 2] - cls_boxes[:, 0]) * (cls_boxes[:, 3] - cls_boxes[:, 1])
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # 计算IoU
            xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[order[1:], 0])
            yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[order[1:], 1])
            xx2 = np.minimum(cls_boxes[i, 2], cls_boxes[order[1:], 2])
            yy2 = np.minimum(cls_boxes[i, 3], cls_boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            union = areas[i] + areas[order[1:]] - inter
            union = np.maximum(union, 1e-10)
            iou = inter / union
            
            # 保留IoU小于阈值的
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        # 添加保留的检测结果
        keep_boxes.extend(cls_boxes[keep].tolist())
        keep_scores.extend(cls_scores[keep].tolist())
        keep_classes.extend([cls] * len(keep))
    
    return keep_boxes, keep_scores, keep_classes

def calculate_iou(box1, box2):
    """计算两个边界框的IoU
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU值
    """
    # 计算交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算并集
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_ap(recalls, precisions):
    """计算AP（Average Precision）
    
    使用11点插值法计算AP
    
    Args:
        recalls: 召回率列表
        precisions: 精确率列表
        
    Returns:
        AP值
    """
    # 在开头和结尾添加(0, 1)和(1, 0)
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])
    
    # 确保精确率单调递减
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # 计算11点插值
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max() / 11.0
    
    return ap

def evaluate_detections(all_predictions, all_targets, iou_threshold=IOU_THRESHOLD):
    """评估检测结果
    
    Args:
        all_predictions: 所有预测结果，格式为 {image_id: [boxes, scores, classes]}
        all_targets: 所有真值，格式为 {image_id: [boxes, classes]}
        iou_threshold: IoU阈值
        
    Returns:
        评估指标字典
    """
    # 初始化统计信息
    class_stats = {cls: {'tp': [], 'fp': [], 'fn': 0, 'scores': []} for cls in range(NUM_CLASSES)}
    
    # 遍历每张图像
    for image_id in all_predictions.keys():
        pred_boxes, pred_scores, pred_classes = all_predictions[image_id]
        target_boxes, target_classes = all_targets[image_id]
        
        # 如果没有预测和真值，跳过
        if len(pred_boxes) == 0 and len(target_boxes) == 0:
            continue
        
        # 如果没有预测，所有真值都是FN
        if len(pred_boxes) == 0:
            for cls in target_classes:
                class_stats[cls]['fn'] += 1
            continue
        
        # 如果没有真值，所有预测都是FP
        if len(target_boxes) == 0:
            for i, cls in enumerate(pred_classes):
                class_stats[cls]['fp'].append(1)
                class_stats[cls]['tp'].append(0)
                class_stats[cls]['scores'].append(pred_scores[i])
            continue
        
        # 匹配预测和真值
        matched = [False] * len(target_boxes)
        
        for i, (pred_box, pred_score, pred_cls) in enumerate(zip(pred_boxes, pred_scores, pred_classes)):
            # 找到同类别且IoU最大的真值
            best_iou = 0.0
            best_idx = -1
            
            for j, (target_box, target_cls) in enumerate(zip(target_boxes, target_classes)):
                if target_cls == pred_cls and not matched[j]:
                    iou = calculate_iou(pred_box, target_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
            
            # 判断是否为TP
            if best_iou >= iou_threshold and best_idx >= 0:
                matched[best_idx] = True
                class_stats[pred_cls]['tp'].append(1)
                class_stats[pred_cls]['fp'].append(0)
                class_stats[pred_cls]['scores'].append(pred_score)
            else:
                class_stats[pred_cls]['fp'].append(1)
                class_stats[pred_cls]['tp'].append(0)
                class_stats[pred_cls]['scores'].append(pred_score)
        
        # 统计未匹配的真值为FN
        for j, (target_box, target_cls) in enumerate(zip(target_boxes, target_classes)):
            if not matched[j]:
                class_stats[target_cls]['fn'] += 1
    
    # 计算每个类别的指标
    results = {}
    
    for cls in range(NUM_CLASSES):
        stats = class_stats[cls]
        
        if len(stats['tp']) == 0:
            # 如果没有预测，所有指标为0
            results[cls] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'ap': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': stats['fn']
            }
            continue
        
        # 按分数排序
        indices = np.argsort(stats['scores'])[::-1]
        tp = np.array(stats['tp'])[indices]
        fp = np.array(stats['fp'])[indices]
        scores = np.array(stats['scores'])[indices]
        
        # 累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算精确率和召回率
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recalls = tp_cumsum / (tp_cumsum[-1] + stats['fn'] + 1e-10)
        
        # 计算AP
        ap = calculate_ap(recalls, precisions)
        
        # 计算最终精确率和召回率
        final_precision = tp_cumsum[-1] / (tp_cumsum[-1] + fp_cumsum[-1] + 1e-10)
        final_recall = tp_cumsum[-1] / (tp_cumsum[-1] + stats['fn'] + 1e-10)
        
        # 计算F1
        f1 = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-10)
        
        results[cls] = {
            'precision': final_precision,
            'recall': final_recall,
            'f1': f1,
            'ap': ap,
            'tp': int(tp_cumsum[-1]),
            'fp': int(fp_cumsum[-1]),
            'fn': stats['fn']
        }
    
    return results

def test_model(model, dataloader, device):
    """测试模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        
    Returns:
        评估结果字典
    """
    # 设置模型为评估模式
    model.eval()
    
    # 存储所有预测和真值
    all_predictions = {}
    all_targets = {}
    
    print("\n" + "=" * 80)
    print("🔍 Running Evaluation...")
    print("=" * 80)
    
    # 不计算梯度
    with torch.no_grad():
        # 使用tqdm显示进度条
        pbar = tqdm(dataloader, desc='Evaluating')
        
        for batch_idx, (images, heatmaps, offsets, wh) in enumerate(pbar):
            # 移动数据到设备
            images = images.to(device)
            
            # 前向传播
            pred_heatmaps, pred_offsets, pred_wh = model(images)
            
            # 解码检测结果
            batch_size = images.shape[0]
            for i in range(batch_size):
                image_id = batch_idx * batch_size + i
                
                # 获取预测
                pred_boxes, pred_scores, pred_classes = decode_detections(
                    pred_heatmaps[i:i+1],
                    pred_offsets[i:i+1],
                    pred_wh[i:i+1],
                    threshold=THRESHOLD
                )
                
                # 应用NMS
                if len(pred_boxes) > 0:
                    pred_boxes, pred_scores, pred_classes = nms(
                        pred_boxes, pred_scores, pred_classes, IOU_THRESHOLD
                    )
                
                # 获取真值
                target_boxes, target_classes = extract_targets(
                    heatmaps[i], offsets[i], wh[i]
                )
                
                # 存储
                all_predictions[image_id] = [pred_boxes, pred_scores, pred_classes]
                all_targets[image_id] = [target_boxes, target_classes]
            
            # 快速测试：只运行指定数量的batch
            if QUICK_TEST and batch_idx + 1 >= QUICK_TEST_BATCHES:
                print(f"\n⏸️  Quick test mode: Stopping after {QUICK_TEST_BATCHES} batches")
                break
    
    # 计算不同IoU阈值下的AP
    print("\n📊 Calculating metrics...")
    
    results = {
        'AP50': evaluate_detections(all_predictions, all_targets, iou_threshold=0.5),
        'AP75': evaluate_detections(all_predictions, all_targets, iou_threshold=0.75),
    }
    
    # 计算mAP
    map_50 = np.mean([results['AP50'][cls]['ap'] for cls in range(NUM_CLASSES)])
    map_75 = np.mean([results['AP75'][cls]['ap'] for cls in range(NUM_CLASSES)])
    map_avg = (map_50 + map_75) / 2
    
    results['mAP50'] = map_50
    results['mAP75'] = map_75
    results['mAP'] = map_avg
    
    return results

def extract_targets(heatmap, offset, wh):
    """从真值中提取边界框
    
    Args:
        heatmap: 真值heatmap [NUM_CLASSES, H, W]
        offset: 真值offset [2, H, W]
        wh: 真值wh [2, H, W]
        
    Returns:
        boxes: 边界框列表
        classes: 类别列表
    """
    boxes = []
    classes = []
    
    # 计算缩放因子
    scale_x = IMAGE_SIZE[0] / HEATMAP_SIZE[1]
    scale_y = IMAGE_SIZE[1] / HEATMAP_SIZE[0]
    
    # 找到每个类别的中心点
    for cls in range(heatmap.shape[0]):
        # 使用较低的阈值查找中心点，提高检测率
        center_threshold = 0.5
        ys, xs = torch.where(heatmap[cls] >= center_threshold)
        
        for y, x in zip(ys, xs):
            # 获取偏移量和宽高
            off_x = offset[0, y, x].item()
            off_y = offset[1, y, x].item()
            w = wh[0, y, x].item()
            h = wh[1, y, x].item()
            
            # 过滤无效宽高
            if w <= 0 or h <= 0:
                continue
            
            # 计算中心点（在原图尺度）
            cx = (x + off_x) * scale_x
            cy = (y + off_y) * scale_y
            
            # 计算边界框
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            # 过滤无效边界框
            if x1 < 0 or y1 < 0 or x2 > IMAGE_SIZE[0] or y2 > IMAGE_SIZE[1]:
                continue
            
            # 确保每个中心点只被检测一次（去重）
            duplicate = False
            for existing_box in boxes:
                existing_cx = (existing_box[0] + existing_box[2]) / 2
                existing_cy = (existing_box[1] + existing_box[3]) / 2
                # 使用Python内置函数计算距离，避免NumPy 2.0警告
                distance = ((cx - existing_cx) ** 2 + (cy - existing_cy) ** 2) ** 0.5
                if distance < 10:  # 10像素内视为同一目标
                    duplicate = True
                    break
            
            if not duplicate:
                boxes.append([x1, y1, x2, y2])
                classes.append(cls)
    
    return boxes, classes

def save_results(results, output_path):
    """保存评估结果到txt文件
    
    Args:
        results: 评估结果字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入标题
        f.write("=" * 100 + "\n")
        f.write("🎯 Object Detection Evaluation Results (CenterNet Style)\n")
        f.write("=" * 100 + "\n\n")
        
        # 写入总体指标
        f.write("📊 Overall Metrics:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Metric':<20} {'Value':<20}\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'mAP@0.5':<20} {results['mAP50']:.4f}\n")
        f.write(f"{'mAP@0.75':<20} {results['mAP75']:.4f}\n")
        f.write(f"{'mAP (avg)':<20} {results['mAP']:.4f}\n")
        f.write("-" * 100 + "\n\n")
        
        # 写入每个类别的详细指标（AP50）
        f.write("📋 Per-Class Metrics (AP@0.5):\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'Class':<20} {'AP':<10} {'Precision':<12} {'Recall':<12} {'F1':<10} {'GT':<8} {'TP':<8} {'FP':<8} {'FN':<8}\n")
        f.write("-" * 110 + "\n")
        
        for cls in range(NUM_CLASSES):
            cls_name = VOC_CLASSES[cls]
            metrics = results['AP50'][cls]
            gt = metrics['tp'] + metrics['fn']  # GT = TP + FN
            f.write(f"{cls_name:<20} "
                   f"{metrics['ap']:<10.4f} "
                   f"{metrics['precision']:<12.4f} "
                   f"{metrics['recall']:<12.4f} "
                   f"{metrics['f1']:<10.4f} "
                   f"{gt:<8} "
                   f"{metrics['tp']:<8} "
                   f"{metrics['fp']:<8} "
                   f"{metrics['fn']:<8}\n")
        
        f.write("-" * 110 + "\n\n")
        
        # 写入每个类别的详细指标（AP75）
        f.write("📋 Per-Class Metrics (AP@0.75):\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'Class':<20} {'AP':<10} {'Precision':<12} {'Recall':<12} {'F1':<10} {'GT':<8} {'TP':<8} {'FP':<8} {'FN':<8}\n")
        f.write("-" * 110 + "\n")
        
        for cls in range(NUM_CLASSES):
            cls_name = VOC_CLASSES[cls]
            metrics = results['AP75'][cls]
            gt = metrics['tp'] + metrics['fn']  # GT = TP + FN
            f.write(f"{cls_name:<20} "
                   f"{metrics['ap']:<10.4f} "
                   f"{metrics['precision']:<12.4f} "
                   f"{metrics['recall']:<12.4f} "
                   f"{metrics['f1']:<10.4f} "
                   f"{gt:<8} "
                   f"{metrics['tp']:<8} "
                   f"{metrics['fp']:<8} "
                   f"{metrics['fn']:<8}\n")
        
        f.write("-" * 110 + "\n\n")
        
        # 写入说明
        f.write("📝 Metric Definitions:\n")
        f.write("-" * 110 + "\n")
        f.write("AP (Average Precision): PR曲线下的面积，使用11点插值法计算\n")
        f.write("mAP (mean AP): 所有类别AP的平均值\n")
        f.write("AP@0.5: IoU阈值为0.5时的AP\n")
        f.write("AP@0.75: IoU阈值为0.75时的AP\n")
        f.write("Precision: TP / (TP + FP)，预测为正例中真正为正例的比例\n")
        f.write("Recall: TP / (TP + FN)，真正为正例中被正确预测的比例\n")
        f.write("F1: 2 * Precision * Recall / (Precision + Recall)，精确率和召回率的调和平均\n")
        f.write("GT (Ground Truth): 真实标注的目标数量，GT = TP + FN\n")
        f.write("TP (True Positive): 正确检测的正例数量\n")
        f.write("FP (False Positive): 错误检测的正例数量\n")
        f.write("FN (False Negative): 漏检的正例数量\n")
        f.write("=" * 110 + "\n")

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🚀 Starting Test Pipeline")
    print("=" * 80)
    
    # 阶段1：设置设备
    print("\n📍 Stage 1/4: Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   ✅ Using device: {device}")
    
    # 阶段2：创建数据集
    print("\n📍 Stage 2/4: Creating dataset...")
    val_dataset = VOCDataset(split='val')
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    print(f"   ✅ Validation dataset size: {len(val_dataset)}")
    
    if QUICK_TEST:
        print(f"   ⚡ Quick test mode: Will run only {QUICK_TEST_BATCHES} batches")
    
    # 阶段3：加载模型
    print("\n📍 Stage 3/4: Loading model...")
    # 最佳模型路径
    best_model_path = MODEL_PATH.replace('.pth', '_best.pth')
    
    # 优先使用最佳模型
    if os.path.exists(best_model_path):
        model_path = best_model_path
        print(f"   ✅ Using best model: {best_model_path}")
    elif os.path.exists(MODEL_PATH):
        model_path = MODEL_PATH
        print(f"   ⚠️  Best model not found, using regular model: {MODEL_PATH}")
    else:
        print(f"   ❌ Error: No model found at {MODEL_PATH} or {best_model_path}")
        print("   Please train the model first")
        return
    
    model = AnchorFreeDetector()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"   ✅ Model loaded successfully from: {model_path}")
    
    # 阶段4：测试模型
    print("\n📍 Stage 4/4: Evaluating model...")
    results = test_model(model, val_loader, device)
    
    # 保存结果
    output_path = os.path.join(OUTPUT_DIR, 'evaluation_results.txt')
    save_results(results, output_path)
    
    # 打印总体指标
    print("\n" + "=" * 80)
    print("📊 Overall Results:")
    print("=" * 80)
    print(f"   mAP@0.5:  {results['mAP50']:.4f}")
    print(f"   mAP@0.75: {results['mAP75']:.4f}")
    print(f"   mAP:      {results['mAP']:.4f}")
    print("=" * 80)
    
    print(f"\n✅ Detailed results saved to: {output_path}")
    print("🎉 Testing completed!")

if __name__ == '__main__':
    main()
