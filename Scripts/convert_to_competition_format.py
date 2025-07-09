#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
转换为比赛格式脚本
将YOLO格式的标注转换为比赛要求的格式

比赛要求格式：
每列依次为帧号、目标ID、边界框左上角的X坐标、边界框左上角的Y坐标、
边界框的宽度、边界框的高度、目标类别(固定为1代表车辆)、-1、-1、-1

运行命令示例:
python Scripts/convert_to_competition_format.py --input_dir dataset/labels/train1 --output_dir results
"""

import pandas as pd
from pathlib import Path
import argparse
import re

def yolo_to_competition_format(yolo_line, img_width, img_height, frame_id, track_id):
    """
    将YOLO格式转换为比赛格式
    
    YOLO格式: class_id center_x center_y width height (归一化)
    比赛格式: frame_id track_id x y w h class_id -1 -1 -1 (像素坐标)
    """
    parts = yolo_line.strip().split()
    if len(parts) < 5:
        return None
    
    class_id, center_x, center_y, width, height = map(float, parts[:5])
    
    # 转换为像素坐标
    center_x_pixel = center_x * img_width
    center_y_pixel = center_y * img_height
    width_pixel = width * img_width
    height_pixel = height * img_height
    
    # 转换为左上角坐标
    x = center_x_pixel - width_pixel / 2
    y = center_y_pixel - height_pixel / 2
    
    # 比赛格式：目标类别固定为1代表车辆
    competition_class = 1
    
    return f"{frame_id},{track_id},{x:.2f},{y:.2f},{width_pixel:.2f},{height_pixel:.2f},{competition_class},-1,-1,-1"

def process_labels_directory(input_dir, output_dir, img_width=1920, img_height=1080):
    """处理标签目录"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 按视频分组处理
    video_groups = {}
    
    for label_file in input_path.glob("*.txt"):
        # 解析文件名：video_name_frame_xxxxxx.txt
        filename = label_file.stem
        
        # 提取视频名和帧号
        match = re.match(r'(.+)_frame_(\d+)', filename)
        if match:
            video_name = match.group(1)
            frame_id = int(match.group(2))
            
            if video_name not in video_groups:
                video_groups[video_name] = []
            
            # 读取YOLO标注
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                # 为每个目标分配一个临时ID（实际应该从跟踪结果获取）
                track_id = i + 1
                
                competition_line = yolo_to_competition_format(
                    line, img_width, img_height, frame_id, track_id
                )
                
                if competition_line:
                    video_groups[video_name].append(competition_line)
    
    # 为每个视频生成输出文件
    for video_name, lines in video_groups.items():
        output_file = output_path / f"{video_name}.txt"
        
        with open(output_file, 'w') as f:
            for line in sorted(lines):  # 按帧号排序
                f.write(line + '\n')
        
        print(f"生成 {output_file}: {len(lines)} 个检测结果")

def main():
    parser = argparse.ArgumentParser(description='转换YOLO格式为比赛格式')
    parser.add_argument('--input_dir', type=str, default='dataset/labels/train1',
                       help='YOLO标签目录')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='比赛格式输出目录')
    parser.add_argument('--img_width', type=int, default=1920,
                       help='图像宽度')
    parser.add_argument('--img_height', type=int, default=1080,
                       help='图像高度')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("转换YOLO格式为比赛格式")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"图像尺寸: {args.img_width}x{args.img_height}")
    print("=" * 60)
    
    process_labels_directory(
        args.input_dir, 
        args.output_dir, 
        args.img_width, 
        args.img_height
    )
    
    print("\n" + "=" * 60)
    print("转换完成！")
    print(f"结果保存在: {args.output_dir}")
    print("=" * 60)
    
    # 显示比赛格式说明
    print("\n比赛格式说明:")
    print("每行格式: 帧号,目标ID,左上角X,左上角Y,宽度,高度,类别(1),,-1,-1,-1")
    print("注意: 目标ID需要从实际跟踪结果获取，当前使用临时ID")

if __name__ == "__main__":
    main()
