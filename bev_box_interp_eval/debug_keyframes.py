
"""详细检查关键帧数据"""
import json
import numpy as np

data = json.load(open('data/key_frame_boxes.json'))

print("Track 1:")
for f in [x for x in data if x['track_id'] == '1'][:10]:
    print(f"  Frame {f['frame_id']}: center={f['center']}, speed={f['speed']:.2f}, yaw={f['yaw']:.4f}, vx={f['velocity'][0]:.4f}, vy={f['velocity'][1]:.4f}")

print("\nTrack 1 first 3 key frames:")
f0 = [x for x in data if x['track_id'] == '1' and x['frame_id'] == 0][0]
f10 = [x for x in data if x['track_id'] == '1' and x['frame_id'] == 10][0]

print(f"Frame 0: {f0['center']}")
print(f"Frame10: {f10['center']}")

dx = f10['center'][0] - f0['center'][0]
dy = f10['center'][1] - f0['center'][1]
print(f"Delta in 10 frames: dx={dx:.2f}, dy={dy:.2f}")
print(f"Speed per frame: dx/10={dx/10:.4f}, dy/10={dy/10:.4f}")
