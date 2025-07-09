#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查视频分辨率脚本
"""

import cv2
from pathlib import Path

def check_video_resolution(video_path):
    """检查单个视频的分辨率"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': total_frames
    }

def main():
    """检查所有视频的分辨率"""
    print("检查视频分辨率...")
    
    # 检查训练视频
    train_dir = Path("data/train")
    if train_dir.exists():
        print("\n训练视频:")
        video_files = list(train_dir.glob("*.avi"))
        
        resolutions = {}
        
        for video_file in video_files:  # 检查所有视频
            info = check_video_resolution(video_file)
            if info:
                resolution = f"{info['width']}x{info['height']}"
                if resolution not in resolutions:
                    resolutions[resolution] = []
                resolutions[resolution].append(video_file.name)
                
                print(f"  {video_file.name}: {resolution}, {info['fps']:.2f}fps, {info['total_frames']}帧")
        
        print(f"\n分辨率统计:")
        for res, files in resolutions.items():
            print(f"  {res}: {len(files)} 个视频")
    
    # 检查验证视频
    val_dir = Path("data/val")
    if val_dir.exists():
        print("\n验证视频:")
        video_files = list(val_dir.glob("*.avi"))
        
        for video_file in video_files:  # 检查所有视频
            info = check_video_resolution(video_file)
            if info:
                print(f"  {video_file.name}: {info['width']}x{info['height']}, {info['fps']:.2f}fps, {info['total_frames']}帧")

if __name__ == "__main__":
    main()
