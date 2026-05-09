"""
BEV 2D Box Key Frame Interpolation Evaluation Pipeline

One-click execution: Preprocessing -> Multi-method Interpolation -> Evaluation -> Visualization
"""

import yaml
import json
import os
from typing import Dict, List

from preprocess.data_preprocessor import DataPreprocessor
from interp_method.linear_interp import run_linear_interp
from interp_method.poly_interp import run_poly_interp
from interp_method.kalman_filter import run_kalman_filter
from interp_method.spline_interp import run_spline_interp
from evaluation.evaluator import Evaluator
from visualization.visualizer import BEVVisualizer


def load_config(config_path: str) -> dict:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_results(results: List[Dict], output_path: str) -> None:
    """Save interpolation results"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {output_path}")


def save_eval_results(results: List[dict], output_path: str) -> None:
    """Save evaluation results"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved: {output_path}")


def generate_report(eval_results: List[dict], config: dict) -> None:
    """Generate evaluation report (Professional Edition)"""
    report_path = os.path.join(config["data"]["output_dir"], "eval_report.md")

    report_lines = [
        "# BEV 2D Box Interpolation Evaluation Report (Professional Edition)\n\n",
        "## 1. Evaluation Overview\n\n",
        "This report compares the performance of multiple BEV 2D Box interpolation algorithms using professional metrics.\n\n",
        "## 2. Configuration\n\n",
        f"- IOU Threshold: {config['evaluation']['iou_threshold']}\n",
        f"- Evaluation IOU Thresholds: {config['evaluation']['iou_thresholds_ap']}\n",
        f"- Frame Interval: {config['evaluation']['frame_interval']}\n",
        "\n## 3. Metrics Definition\n\n",
        "### 3.1 IOU Metrics\n",
        "| Metric | Description |\n",
        "|--------|-------------|\n",
        "| IOU@mean | Mean Intersection over Union |\n",
        "| IOU@std | Standard deviation of IOU |\n",
        "| IOU@med | Median IOU |\n",
        "| IOU@90% | 90th percentile of IOU (90% of frames have IOU <= this value) |\n",
        "\n### 3.2 Center and Pose Error Metrics (Unit: meters/radians)\n",
        "| Metric | Description |\n",
        "|--------|-------------|\n",
        "| CE@mean(m) | Mean center L2 error in meters |\n",
        "| CE@90%(m) | 90th percentile of center error |\n",
        "| CorE@mean(m) | Mean corner distance error |\n",
        "| YawE@mean(rad) | Mean absolute yaw angle error |\n",
        "\n### 3.3 Trajectory Metrics\n",
        "| Metric | Description |\n",
        "|--------|-------------|\n",
        "| ADE(m) | Average Displacement Error (average error per frame along the entire trajectory) |\n",
        "| FDE(m) | Final Displacement Error (error at the last frame of the trajectory) |\n",
        "| TrajLenRatio | Ratio of predicted trajectory length to ground truth |\n",
        "\n### 3.4 Smoothness Metrics\n",
        "| Metric | Description |\n",
        "|--------|-------------|\n",
        "| SpeedVar | Speed change variance (lower = smoother velocity) |\n",
        "| AccelVar | Acceleration change variance |\n",
        "| Jerk | Jerk (change of acceleration, lower = smoother motion) |\n",
        "\n### 3.5 Detection Metrics\n",
        "| Metric | Description |\n",
        "|--------|-------------|\n",
        "| Precision | Matching precision |\n",
        "| Recall | Recall rate |\n",
        "| mAP@0.5 | Mean Average Precision at IOU threshold 0.5 |\n",
        "\n## 4. Evaluation Results\n\n",
        "### 4.1 Full Metrics Comparison Table\n\n",
        "| Method | IOU@mean | IOU@std | IOU@med | IOU@90% | CE@mean(m) | CE@90%(m) | CorE@mean(m) | YawE@mean(rad) | ADE(m) | FDE(m) | SpeedVar | AccelVar | Jerk | Precision | Recall | mAP@0.5 |\n",
        "|--------|----------|---------|---------|---------|------------|-----------|--------------|----------------|--------|--------|----------|----------|------|-----------|--------|----------|\n"
    ]

    for result in eval_results:
        report_lines.append(
            f"| {result['method_name']} | "
            f"{result['iou_mean']:.4f} | "
            f"{result['iou_std']:.4f} | "
            f"{result['iou_median']:.4f} | "
            f"{result['iou_90_percentile']:.4f} | "
            f"{result['center_error_mean']:.4f} | "
            f"{result['center_error_90_percentile']:.4f} | "
            f"{result['corner_error_mean']:.4f} | "
            f"{result['yaw_error_mean']:.4f} | "
            f"{result['ade']:.4f} | "
            f"{result['fde']:.4f} | "
            f"{result['speed_variance']:.4f} | "
            f"{result['acceleration_variance']:.4f} | "
            f"{result['jerk']:.4f} | "
            f"{result['precision']:.4f} | "
            f"{result['recall']:.4f} | "
            f"{result['mAP'].get(0.5, 0):.4f} |\n"
        )

    report_lines.append("\n## 5. Analysis & Conclusions\n")

    best_iou = max(eval_results, key=lambda x: x['iou_mean'])
    best_center_error = min(eval_results, key=lambda x: x['center_error_mean'])
    best_ade = min(eval_results, key=lambda x: x['ade'])
    best_smooth_speed = min(eval_results, key=lambda x: x['speed_variance'])
    best_smooth_jerk = min(eval_results, key=lambda x: x['jerk'])
    best_precision = max(eval_results, key=lambda x: x['precision'])
    best_recall = max(eval_results, key=lambda x: x['recall'])

    report_lines.append(f"- **Best IOU**: {best_iou['method_name']} (IOU={best_iou['iou_mean']:.4f})\n")
    report_lines.append(f"- **Best Center Error**: {best_center_error['method_name']} (CE={best_center_error['center_error_mean']:.4f}m)\n")
    report_lines.append(f"- **Best ADE (Trajectory)**: {best_ade['method_name']} (ADE={best_ade['ade']:.4f}m)\n")
    report_lines.append(f"- **Smoothest Velocity**: {best_smooth_speed['method_name']} (SpeedVar={best_smooth_speed['speed_variance']:.4f})\n")
    report_lines.append(f"- **Smoothest Motion (Jerk)**: {best_smooth_jerk['method_name']} (Jerk={best_smooth_jerk['jerk']:.4f})\n")
    report_lines.append(f"- **Best Precision**: {best_precision['method_name']} (Precision={best_precision['precision']:.4f})\n")
    report_lines.append(f"- **Best Recall**: {best_recall['method_name']} (Recall={best_recall['recall']:.4f})\n")

    report_lines.append("\n## 6. Method Recommendations\n")
    report_lines.append("| Method | Use Case | Pros/Cons |\n")
    report_lines.append("|--------|----------|----------|\n")
    report_lines.append("| Linear | Uniform straight-line motion | Simple and fast, large error in curves/variable speed |\n")
    report_lines.append("| Poly | Uniform acceleration, gentle turns | Better for variable speed than linear |\n")
    report_lines.append("| Kalman | Smooth trajectories, noise suppression | Strong anti-noise, slightly complex |\n")
    report_lines.append("| Spline | Non-linear motion, curves | Best fitting, requires more keyframes |\n")

    report_lines.append("\n## 7. Output Files\n")
    report_lines.append("- `output/box_result/`: Interpolated box results per method per frame\n")
    report_lines.append("- `output/eval_metric/`: All metrics (IOU, error, ADE, smoothness, mAP)\n")
    report_lines.append("- `output/vis/`: Visualization screenshots, video, charts\n")
    report_lines.append("- `eval_report.md`: Summary report\n")

    with open(report_path, 'w') as f:
        f.writelines(report_lines)

    print(f"Report generated: {report_path}")


def main():
    """Main execution function"""
    print("=" * 70)
    print("  BEV 2D Box Key Frame Interpolation Evaluation Pipeline")
    print("=" * 70)

    config = load_config("config/config.yaml")
    print("Configuration loaded")

    print("\n[1/6] Data Preprocessing...")
    preprocessor = DataPreprocessor(config)
    preprocessor.run(
        key_frame_path=config["data"]["key_frame_box_path"],
        full_gt_path=config["data"]["full_gt_box_path"]
    )

    frame_range = preprocessor.get_frame_range()
    print(f"Frame range: {frame_range[0]} - {frame_range[1]}")

    print("\n[2/6] Running Interpolation Methods...")
    results_by_method = {}
    methods = config["interp_methods"]

    if "linear" in methods:
        linear_results = run_linear_interp(preprocessor.track_sequences, frame_range)
        results_by_method["linear"] = linear_results
        save_results(linear_results, os.path.join(config["data"]["output_dir"], "box_result", "linear_results.json"))

    if "poly" in methods:
        poly_results = run_poly_interp(preprocessor.track_sequences, frame_range)
        results_by_method["poly"] = poly_results
        save_results(poly_results, os.path.join(config["data"]["output_dir"], "box_result", "poly_results.json"))

    if "kalman" in methods:
        kalman_results = run_kalman_filter(preprocessor.track_sequences, frame_range)
        results_by_method["kalman"] = kalman_results
        save_results(kalman_results, os.path.join(config["data"]["output_dir"], "box_result", "kalman_results.json"))

    if "spline" in methods:
        spline_results = run_spline_interp(preprocessor.track_sequences, frame_range)
        results_by_method["spline"] = spline_results
        save_results(spline_results, os.path.join(config["data"]["output_dir"], "box_result", "spline_results.json"))

    print("\n[3/6] Evaluation...")
    evaluator = Evaluator(config)

    gt_boxes_dict = [box.to_dict() for box in preprocessor.full_gt_boxes]

    eval_results = evaluator.evaluate_all_methods(results_by_method, gt_boxes_dict)

    print("\n" + evaluator.format_results(eval_results))

    eval_results_dict = [r.to_dict() for r in eval_results]
    save_eval_results(eval_results_dict, os.path.join(config["data"]["output_dir"], "eval_metric", "eval_results.json"))

    print("\n[4/6] Visualization...")
    visualizer = BEVVisualizer(config)

    vis_boxes = results_by_method.copy()
    vis_boxes["gt"] = gt_boxes_dict

    vis_dir = os.path.join(config["data"]["output_dir"], "vis")
    os.makedirs(vis_dir, exist_ok=True)

    if config["visualization"]["output_video"]:
        video_path = os.path.join(vis_dir, "bev_interp_video.mp4")
        visualizer.visualize_sequence(vis_boxes, frame_range, video_path, show_velocity=True)

    metrics_path = os.path.join(vis_dir, "metrics_comparison.png")
    visualizer.plot_metrics(eval_results_dict, metrics_path)

    radar_path = os.path.join(vis_dir, "radar_chart.png")
    visualizer.plot_radar_chart(eval_results_dict, radar_path)

    print("\n[5/6] Generating Report...")
    generate_report(eval_results_dict, config)

    print("\n" + "=" * 70)
    print("  Pipeline Completed Successfully!")
    print("=" * 70)
    print(f"\nOutput Directory: {config['data']['output_dir']}")
    print(f"Report: {config['data']['output_dir']}eval_report.md")
    print("\nGenerated Files:")
    print("  - box_result/*.json              : Interpolation results per method")
    print("  - eval_metric/*.json             : Evaluation metrics (15+ professional indicators)")
    print("  - vis/bev_interp_video.mp4       : Visualization video with legend")
    print("  - vis/metrics_comparison.png     : Metrics comparison chart (2x3 subplots)")
    print("  - vis/metrics_comparison_smoothness.png : Smoothness metrics chart")
    print("  - vis/radar_chart.png            : Professional radar chart (7 dimensions)")
    print("  - eval_report.md                 : Complete evaluation report")


if __name__ == "__main__":
    main()
