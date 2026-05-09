# BEV 2D Box Interpolation Evaluation Report (Professional Edition)

## 1. Evaluation Overview

This report compares the performance of multiple BEV 2D Box interpolation algorithms using professional metrics.

## 2. Configuration

- IOU Threshold: 0.5
- Evaluation IOU Thresholds: [0.5, 0.75]
- Frame Interval: 1

## 3. Metrics Definition

### 3.1 IOU Metrics
| Metric | Description |
|--------|-------------|
| IOU@mean | Mean Intersection over Union |
| IOU@std | Standard deviation of IOU |
| IOU@med | Median IOU |
| IOU@90% | 90th percentile of IOU (90% of frames have IOU <= this value) |

### 3.2 Center and Pose Error Metrics (Unit: meters/radians)
| Metric | Description |
|--------|-------------|
| CE@mean(m) | Mean center L2 error in meters |
| CE@90%(m) | 90th percentile of center error |
| CorE@mean(m) | Mean corner distance error |
| YawE@mean(rad) | Mean absolute yaw angle error |

### 3.3 Trajectory Metrics
| Metric | Description |
|--------|-------------|
| ADE(m) | Average Displacement Error (average error per frame along the entire trajectory) |
| FDE(m) | Final Displacement Error (error at the last frame of the trajectory) |
| TrajLenRatio | Ratio of predicted trajectory length to ground truth |

### 3.4 Smoothness Metrics
| Metric | Description |
|--------|-------------|
| SpeedVar | Speed change variance (lower = smoother velocity) |
| AccelVar | Acceleration change variance |
| Jerk | Jerk (change of acceleration, lower = smoother motion) |

### 3.5 Detection Metrics
| Metric | Description |
|--------|-------------|
| Precision | Matching precision |
| Recall | Recall rate |
| mAP@0.5 | Mean Average Precision at IOU threshold 0.5 |

## 4. Evaluation Results

### 4.1 Full Metrics Comparison Table

| Method | IOU@mean | IOU@std | IOU@med | IOU@90% | CE@mean(m) | CE@90%(m) | CorE@mean(m) | YawE@mean(rad) | ADE(m) | FDE(m) | SpeedVar | AccelVar | Jerk | Precision | Recall | mAP@0.5 |
|--------|----------|---------|---------|---------|------------|-----------|--------------|----------------|--------|--------|----------|----------|------|-----------|--------|----------|
| linear | 0.9625 | 0.2013 | 0.9364 | 1.0461 | 0.0968 | 0.2470 | 0.0968 | 0.0000 | 0.0962 | 0.0155 | 0.0007 | 0.0007 | 0.0007 | 0.9927 | 0.9927 | 1.0000 |
| poly | 0.9129 | 0.2815 | 0.9809 | 1.0223 | 0.1726 | 0.7628 | 0.1808 | 0.0034 | 1.1186 | 2.4950 | 0.0000 | 0.0000 | 0.0000 | 0.6857 | 0.6857 | 0.8920 |
| kalman | 0.8454 | 0.1604 | 0.8605 | 0.9801 | 0.2503 | 0.6260 | 0.2503 | 0.0006 | 0.3589 | 0.3216 | 0.0577 | 0.1692 | 0.1692 | 0.9503 | 0.9503 | 0.9901 |
| spline | 0.9877 | 0.1593 | 1.0000 | 1.0314 | 0.0578 | 0.0815 | 0.0578 | 0.0000 | 0.0575 | 0.0234 | 0.0002 | 0.0000 | 0.0000 | 0.9942 | 0.9942 | 1.0000 |

## 5. Analysis & Conclusions
- **Best IOU**: spline (IOU=0.9877)
- **Best Center Error**: spline (CE=0.0578m)
- **Best ADE (Trajectory)**: spline (ADE=0.0575m)
- **Smoothest Velocity**: poly (SpeedVar=0.0000)
- **Smoothest Motion (Jerk)**: poly (Jerk=0.0000)
- **Best Precision**: spline (Precision=0.9942)
- **Best Recall**: spline (Recall=0.9942)

## 6. Method Recommendations
| Method | Use Case | Pros/Cons |
|--------|----------|----------|
| Linear | Uniform straight-line motion | Simple and fast, large error in curves/variable speed |
| Poly | Uniform acceleration, gentle turns | Better for variable speed than linear |
| Kalman | Smooth trajectories, noise suppression | Strong anti-noise, slightly complex |
| Spline | Non-linear motion, curves | Best fitting, requires more keyframes |

## 7. Output Files
- `output/box_result/`: Interpolated box results per method per frame
- `output/eval_metric/`: All metrics (IOU, error, ADE, smoothness, mAP)
- `output/vis/`: Visualization screenshots, video, charts
- `eval_report.md`: Summary report
