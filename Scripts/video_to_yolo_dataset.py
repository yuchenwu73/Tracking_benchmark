#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频数据转换为YOLO格式数据集
将视频文件和CSV标注文件转换为图像和YOLO格式标注

运行命令示例:
# 基本用法 - data/train生成到VOCdevkit格式，data/val生成到test
python Scripts/video_to_yolo_dataset.py

# 指定输出目录
python Scripts/video_to_yolo_dataset.py --output_dir dataset

# 指定帧间隔
python Scripts/video_to_yolo_dataset.py --frame_interval 1

作者: Tracking Benchmark
功能:
1. 从视频中提取帧图像
2. 解析CSV标注文件
3. 转换为YOLO格式标注
4. data/train -> dataset/VOCdevkit/JPEGImages + dataset/VOCdevkit/txt (YOLO格式，不拆分)
5. data/val -> dataset/images/test (仅图像，无标注)
"""

import cv2
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil

def parse_csv_annotations(csv_path):
    """
    解析CSV标注文件

    实际CSV格式（基于数据检查）:
    帧号、目标ID、边界框左上角的X坐标、边界框左上角的Y坐标、边界框的宽度、边界框的高度、-1、-1、-1、-1

    注意：实际数据中没有类别列，所有目标默认为车辆类别

    参数:
        csv_path: CSV文件路径

    返回:
        dict: {frame_id: [(x, y, w, h, track_id), ...]}
    """
    print(f"解析标注文件: {csv_path}")

    # 读取CSV文件
    df = pd.read_csv(csv_path, header=None)
    print(f"CSV文件包含 {len(df)} 行数据，{len(df.columns)} 列")

    # 根据实际数据格式，CSV格式为: frame_id, track_id, x, y, width, height, -1, -1, -1, -1
    frame_col = 0  # 帧ID列
    track_col = 1  # 目标ID列
    x_col = 2      # x坐标列（左上角）
    y_col = 3      # y坐标列（左上角）
    w_col = 4      # 宽度列
    h_col = 5      # 高度列

    annotations = {}
    valid_count = 0
    invalid_count = 0

    for _, row in df.iterrows():
        frame_id = int(row[frame_col])
        track_id = int(row[track_col])
        x = float(row[x_col])
        y = float(row[y_col])
        w = float(row[w_col])
        h = float(row[h_col])

        # 过滤无效标注
        # 1. 过滤宽高小于等于0的无效框
        # 2. 过滤坐标异常的框（负坐标）
        # 3. 过滤track_id为0或负数的无效目标（根据MOT格式，track_id应该从1开始）
        if w <= 0 or h <= 0 or x < 0 or y < 0 or track_id <= 0:
            invalid_count += 1
            continue

        if frame_id not in annotations:
            annotations[frame_id] = []

        annotations[frame_id].append((x, y, w, h, track_id))
        valid_count += 1

    print(f"解析完成，共 {len(annotations)} 帧有标注")
    print(f"有效标注: {valid_count}, 无效标注: {invalid_count}")
    return annotations

def convert_to_yolo_format(bbox, img_width, img_height):
    """
    将边界框转换为YOLO格式

    输入: (x, y, w, h) - 左上角坐标和宽高
    输出: (center_x, center_y, width, height) - 归一化的中心坐标和宽高
    """
    x, y, w, h = bbox

    # 边界检查和修正
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))

    # 确保边界框不超出图像边界
    if x + w > img_width:
        w = img_width - x
    if y + h > img_height:
        h = img_height - y

    # 转换为中心坐标
    center_x = x + w / 2
    center_y = y + h / 2

    # 归一化
    center_x /= img_width
    center_y /= img_height
    w /= img_width
    h /= img_height

    # 确保归一化坐标在 [0, 1] 范围内
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return center_x, center_y, w, h

def process_video_and_annotations(video_path, csv_path, output_dir, video_name,
                                frame_interval=1):
    """
    处理单个视频和对应的标注文件
    提取视频的每一帧（按frame_interval间隔），为有标注的帧生成YOLO格式标签文件

    参数:
        video_path: 视频文件路径
        csv_path: CSV标注文件路径
        output_dir: 输出根目录
        video_name: 视频名称（用于文件命名）
        frame_interval: 帧间隔（每隔几帧提取一帧）
    """
    print(f"\n处理视频: {video_path}")

    # 使用VOCdevkit目录结构，但存储YOLO格式标注
    # dataset/VOCdevkit/JPEGImages 和 dataset/VOCdevkit/txt
    images_dir = Path(output_dir) / "VOCdevkit" / "JPEGImages"
    labels_dir = Path(output_dir) / "VOCdevkit" / "txt"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析标注文件
    annotations = parse_csv_annotations(csv_path)
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {width}x{height}, {fps:.2f}fps, {total_frames}帧")
    
    frame_count = 0
    saved_count = 0
    
    pbar = tqdm(total=total_frames, desc=f"处理 {video_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 按间隔提取帧
        if frame_count % frame_interval == 0:
            # 保存图像（每一帧都保存）
            img_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
            img_path = images_dir / img_filename
            cv2.imwrite(str(img_path), frame)

            # 如果该帧有标注，则保存YOLO格式标注
            if frame_count in annotations:
                label_filename = f"{video_name}_frame_{frame_count:06d}.txt"
                label_path = labels_dir / label_filename

                with open(label_path, 'w') as f:
                    for bbox_data in annotations[frame_count]:
                        x, y, w, h, _ = bbox_data  # track_id 不需要用到，用 _ 忽略

                        # 转换为YOLO格式
                        center_x, center_y, norm_w, norm_h = convert_to_yolo_format(
                            (x, y, w, h), width, height
                        )

                        # YOLO格式: class_id center_x center_y width height
                        # 根据比赛要求，所有目标都是车辆，类别ID为0（YOLO格式从0开始）
                        f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            saved_count += 1
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print(f"完成处理: 保存了 {saved_count} 帧图像和标注")
    return saved_count

def main():
    parser = argparse.ArgumentParser(description='将视频数据转换为数据集')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='dataset',
                       help='输出目录路径')
    parser.add_argument('--frame_interval', type=int, default=1,
                       help='帧间隔（每隔几帧提取一帧，默认1表示每帧都提取）')

    args = parser.parse_args()

    print("=" * 60)
    print("视频数据转换为数据集")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"帧间隔: {args.frame_interval}")
    print("输出结构:")
    print("  data/train -> dataset/VOCdevkit/JPEGImages + dataset/VOCdevkit/txt (YOLO格式)")
    print("  data/val   -> dataset/images/test (仅图像)")
    print("=" * 60)
    
    # 创建输出目录结构
    output_path = Path(args.output_dir)

    # 创建新的目录结构
    # data/train -> VOCdevkit格式
    voc_images_dir = output_path / "VOCdevkit" / "JPEGImages"
    voc_labels_dir = output_path / "VOCdevkit" / "txt"
    # data/val -> test格式
    test_images_dir = output_path / "images" / "test"

    # 清理并创建目录
    if output_path.exists():
        shutil.rmtree(output_path)

    voc_images_dir.mkdir(parents=True, exist_ok=True)
    voc_labels_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理训练数据 (data/train -> dataset/VOCdevkit/)
    train_data_dir = Path(args.data_dir) / "train"
    if train_data_dir.exists():
        print("\n处理训练数据 -> VOCdevkit格式...")
        video_files = list(train_data_dir.glob("*.avi"))

        print(f"找到 {len(video_files)} 个训练视频文件")

        # 处理所有训练视频，不进行拆分
        for video_file in video_files:
            csv_file = video_file.parent / (video_file.stem + '-gt.csv')

            if csv_file.exists():
                video_name = video_file.stem
                saved_frames = process_video_and_annotations(
                    video_file, csv_file, output_path, video_name,
                    args.frame_interval
                )
                print(f"训练数据 {video_name}: 保存 {saved_frames} 帧到VOCdevkit目录结构（YOLO格式标注）")
            else:
                print(f"警告: 未找到 {video_file} 对应的标注文件")
    else:
        print(f"警告: 训练数据目录不存在: {train_data_dir}")
    
    # 处理测试数据（data/val -> dataset/images/test，无标注）
    test_data_dir = Path(args.data_dir) / "val"

    if test_data_dir.exists():
        print("\n处理测试数据 -> images/test（仅图像，无标注）...")
        video_files = list(test_data_dir.glob("*.avi"))

        print(f"找到 {len(video_files)} 个测试视频文件")

        for video_file in video_files:
            video_name = video_file.stem
            saved_frames = extract_frames_only(
                video_file, output_path, video_name,
                args.frame_interval, "test"
            )
            print(f"测试数据 {video_name}: 保存 {saved_frames} 帧（仅图像）")
    else:
        print(f"警告: 测试数据目录不存在: {test_data_dir}")

    # 生成数据集配置文件
    create_dataset_yaml(output_path)

    print("\n" + "=" * 60)
    print("数据转换完成！")
    print(f"输出目录: {output_path}")
    print("目录结构:")
    print(f"  {output_path}/VOCdevkit/JPEGImages/ - 训练图像")
    print(f"  {output_path}/VOCdevkit/txt/ - 训练标注 (YOLO格式)")
    print(f"  {output_path}/images/test/ - 测试图像")
    print("=" * 60)

def extract_frames_only(video_path, output_dir, video_name, frame_interval=1, split="test"):
    """仅提取视频帧（用于测试集）"""
    print(f"提取视频帧: {video_path}")

    images_dir = Path(output_dir) / "images" / split
    images_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    saved_count = 0

    pbar = tqdm(total=total_frames, desc=f"提取 {video_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            img_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
            img_path = images_dir / img_filename
            cv2.imwrite(str(img_path), frame)
            saved_count += 1

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    return saved_count

def create_dataset_yaml(output_dir):
    """创建数据集配置文件"""
    yaml_content = f"""# 弱小目标检测跟踪数据集配置 - 车辆检测
path: {output_dir.absolute()}
train: VOCdevkit/JPEGImages  # 训练图像目录
val: VOCdevkit/JPEGImages    # 验证图像目录（与训练相同，后续手动拆分）
test: images/test            # 测试集（无标注）

# 类别定义
nc: 1
names:
  0: vehicle  # 车辆类别

# 数据集信息
dataset_info:
  description: "弱小目标检测跟踪数据集 - 卫星视频中的车辆检测"
  source: "线下决赛赛道2：弱小目标检测跟踪"
  format: "YOLO格式（使用VOCdevkit目录结构）"
  competition: "卫星视频车辆检测跟踪比赛"
  structure:
    train_images: "VOCdevkit/JPEGImages/ - 所有训练图像（未拆分）"
    train_labels: "VOCdevkit/txt/ - YOLO格式标注文件（class_id center_x center_y width height）"
    test_images: "images/test/ - 测试图像（无标注）"
  note: "虽然使用VOCdevkit目录名，但标注文件是YOLO格式，训练数据未拆分，需要后续手动划分训练/验证集"
"""

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"数据集配置文件已创建: {yaml_path}")
    print("注意: 虽然目录名为VOCdevkit，但标注文件是YOLO格式")
    print("注意: 训练数据未拆分，您需要后续手动划分训练/验证集")

if __name__ == "__main__":
    main()