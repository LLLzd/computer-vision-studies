"""
数据预处理模块

功能：
1. 加载关键帧标注和全帧真值数据
2. 按track_id拆分目标序列
3. 过滤异常框
4. 规整帧序列
"""

import json
import os
from typing import Dict, List, Tuple

from utils.data_format import BEVBox2D, TrackSequence


class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self, config: dict):
        self.config = config
        self.key_frame_boxes: List[BEVBox2D] = []
        self.full_gt_boxes: List[BEVBox2D] = []
        self.track_sequences: Dict[str, TrackSequence] = {}
    
    def load_key_frame_boxes(self, file_path: str) -> None:
        """加载关键帧标注Box"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"关键帧数据文件不存在: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.key_frame_boxes = [BEVBox2D.from_dict(box) for box in data]
        print(f"✅ 加载关键帧Box: {len(self.key_frame_boxes)} 个")
    
    def load_full_gt_boxes(self, file_path: str) -> None:
        """加载全帧真值Box"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"全帧真值数据文件不存在: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.full_gt_boxes = [BEVBox2D.from_dict(box) for box in data]
        print(f"✅ 加载全帧真值Box: {len(self.full_gt_boxes)} 个")
    
    def filter_abnormal_boxes(self, boxes: List[BEVBox2D]) -> List[BEVBox2D]:
        """
        过滤异常框
        
        过滤条件：
        1. 坐标越界（x1 >= x2 或 y1 >= y2）
        2. 宽高为0或负数
        3. 重复框（同一frame_id+track_id）
        """
        filtered = []
        seen = set()
        
        for box in boxes:
            # 检查坐标有效性
            if box.x1 >= box.x2 or box.y1 >= box.y2:
                continue
            
            # 检查宽高有效性
            if box.box_width <= 0 or box.box_height <= 0:
                continue
            
            # 检查重复
            key = (box.frame_id, box.track_id)
            if key in seen:
                continue
            seen.add(key)
            
            filtered.append(box)
        
        print(f"🔍 过滤异常框: {len(boxes)} -> {len(filtered)}")
        return filtered
    
    def build_track_sequences(self, boxes: List[BEVBox2D]) -> Dict[str, TrackSequence]:
        """
        按track_id构建目标跟踪序列
        
        返回：
            {track_id: TrackSequence}
        """
        sequences = {}
        
        for box in boxes:
            if box.track_id not in sequences:
                sequences[box.track_id] = TrackSequence(track_id=box.track_id, boxes=[])
            sequences[box.track_id].boxes.append(box)
        
        # 按frame_id排序
        for seq in sequences.values():
            seq.boxes.sort(key=lambda x: x.frame_id)
        
        print(f"📊 构建跟踪序列: {len(sequences)} 个目标")
        return sequences
    
    def split_key_frame_intervals(self, track_seq: TrackSequence) -> List[Tuple[int, int, List[BEVBox2D]]]:
        """
        切分关键帧区间
        
        返回：
            [(start_frame, end_frame, [key_frame_boxes]), ...]
        """
        key_frames = track_seq.get_key_frames()
        
        if len(key_frames) < 2:
            return []
        
        intervals = []
        for i in range(len(key_frames) - 1):
            start_frame = key_frames[i].frame_id
            end_frame = key_frames[i + 1].frame_id
            intervals.append((start_frame, end_frame, [key_frames[i], key_frames[i + 1]]))
        
        return intervals
    
    def get_frame_range(self) -> Tuple[int, int]:
        """获取全局帧范围"""
        all_frames = []
        for seq in self.track_sequences.values():
            start, end = seq.get_frame_range()
            all_frames.extend([start, end])
        
        if not all_frames:
            return (0, 0)
        
        return (min(all_frames), max(all_frames))
    
    def get_boxes_at_frame(self, frame_id: int) -> List[BEVBox2D]:
        """获取指定帧的所有Box"""
        boxes = []
        for seq in self.track_sequences.values():
            box = seq.get_box_at_frame(frame_id)
            if box:
                boxes.append(box)
        return boxes
    
    def run(self, key_frame_path: str, full_gt_path: str) -> None:
        """
        执行完整预处理流程
        
        参数：
            key_frame_path: 关键帧数据路径
            full_gt_path: 全帧真值数据路径
        """
        print("🚀 开始数据预处理...")
        
        # 加载数据
        self.load_key_frame_boxes(key_frame_path)
        self.load_full_gt_boxes(full_gt_path)
        
        # 过滤异常
        self.key_frame_boxes = self.filter_abnormal_boxes(self.key_frame_boxes)
        self.full_gt_boxes = self.filter_abnormal_boxes(self.full_gt_boxes)
        
        # 构建跟踪序列（基于关键帧）
        self.track_sequences = self.build_track_sequences(self.key_frame_boxes)
        
        print("✅ 数据预处理完成")
