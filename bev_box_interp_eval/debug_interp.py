
"""Debug脚本，检查插值方法开头几帧的结果"""

import json
import numpy as np

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

print("加载数据...")
gt_boxes = load_json('data/full_gt_boxes.json')
key_frame_boxes = load_json('data/key_frame_boxes.json')
linear = load_json('output/box_result/linear_results.json')
poly = load_json('output/box_result/poly_results.json')
kalman = load_json('output/box_result/kalman_results.json')
spline = load_json('output/box_result/spline_results.json')

def get_by_frame_and_track(boxes, frame_id, track_id):
    for box in boxes:
        if box['frame_id'] == frame_id and box['track_id'] == track_id:
            return box
    return None

print("\n=== 检查开头几帧 ===")
for frame_id in range(0, 30, 5):  # 检查0,5,10,15,20,25帧
    print(f"\n--- Frame {frame_id} ---")
    
    # 检查id1的车
    gt = get_by_frame_and_track(gt_boxes, frame_id, '1')
    l = get_by_frame_and_track(linear, frame_id, '1')
    p = get_by_frame_and_track(poly, frame_id, '1')
    k = get_by_frame_and_track(kalman, frame_id, '1')
    s = get_by_frame_and_track(spline, frame_id, '1')
    
    if gt:
        print(f"GT: {gt['center']}")
    if l:
        print(f"Linear: {l['center']}")
    if p:
        print(f"Poly: {p['center']}")
    if k:
        print(f"Kalman: {k['center']}")
    if s:
        print(f"Spline: {s['center']}")

print("\n=== 检查关键帧 ===")
for track_id in ['1']:
    print(f"\nTrack {track_id}:")
    key_frames = [f for f in key_frame_boxes if f['track_id'] == track_id]
    for f in key_frames[:5]:  # 前5个关键帧
        print(f"  Frame {f['frame_id']}: {f['center']}")

print("\n=== 检查所有方法的前几个结果 ===")
for method, data in [('linear', linear), ('poly', poly), ('kalman', kalman), ('spline', spline)]:
    print(f"\n{method}:")
    track1 = [f for f in data if f['track_id'] == '1']
    for f in track1[:10]:
        print(f"  Frame {f['frame_id']}: {f['center']}")
