#!/usr/bin/env python3
"""
无人机多目标跟踪系统 - 多线程批量处理版本
支持批量处理多个视频文件，每个视频生成一个结果文件

输出格式：
每行代表一帧中的一个物体，格式为：
帧号,目标ID,边界框左上角X坐标,边界框左上角Y坐标,边界框宽度,边界框高度,1,-1,-1,-1

主要特性：
- 批量处理多个视频文件
- 多线程异步帧读取，提升处理效率
- 每个视频生成独立的结果文件
- 支持命令行参数配置
- 优化的ID分配机制，减少ID浪费
- 改进的轨迹关联算法，提高跟踪稳定性

使用说明：
1. 将要处理的视频放入指定文件夹
2. 通过命令行参数配置运行选项
3. 运行脚本自动处理所有视频
4. 结果文件保存在指定输出文件夹中

命令行参数示例：
python3 uav_tracking.py --video_folder UAV --output_dir results --model_weights best.pt --sub_width 960 --sub_height 1080
"""

# 导入所需的Python库
import os
import threading
import queue
import torch
import time
import numpy as np
import cv2
import argparse
from ultralytics import YOLO
from tqdm import tqdm
# 导入YOLO工具类
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.engine.results import Results, Boxes

# 创建yaml_load函数的兼容版本
def yaml_load(file_path):
    """兼容的YAML加载函数"""
    yaml_instance = YAML()
    return yaml_instance.load(file_path)

# 导入跟踪器实现
from trackers.byte_tracker import BYTETracker
from trackers.bot_sort import BOTSORT

# 定义可用的跟踪器映射字典
TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}

def initialize_tracker(tracker_yaml: str, frame_rate: int = 30):
    """
    初始化目标跟踪器
    """
    # 加载并解析配置文件
    try:
        tracker_cfg = IterableSimpleNamespace(**yaml_load(check_yaml(tracker_yaml)))
    except Exception as e:
        print(f"配置文件加载失败: {e}")
        tracker_cfg = IterableSimpleNamespace(tracker_type="bytetrack")

    tracker_type = getattr(tracker_cfg, 'tracker_type', 'bytetrack')
    if tracker_type not in TRACKER_MAP:
        raise ValueError(f"不支持的跟踪器类型: {tracker_type}")
    return TRACKER_MAP[tracker_type](args=tracker_cfg, frame_rate=frame_rate)

def process_video(video_path, output_file_path, model, tracker_config, sub_width, sub_height):
    """
    处理单个视频文件并输出结果到指定文件
    
    参数:
        video_path: 视频文件路径
        output_file_path: 结果文件输出路径
        model: YOLO模型实例
        tracker_config: 跟踪器配置文件路径
        sub_width: 子图宽度
        sub_height: 子图高度
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算子图分割参数
    cols = max(1, width // sub_width)
    rows = max(1, height // sub_height)
    
    # 初始化目标跟踪器（每个视频使用独立的跟踪器）
    tracker = initialize_tracker(tracker_config, frame_rate=fps)
    
    # 帧队列和线程管理
    frame_queue = queue.Queue(maxsize=10)
    
    def read_frames():
        """多线程帧读取函数"""
        while True:
            ret, frame = cap.read()
            if not ret:
                frame_queue.put(None)  # 表示视频结束
                break
            frame_queue.put(frame)
    
    # 启动读取线程
    read_thread = threading.Thread(target=read_frames, daemon=True)
    read_thread.start()
    
    # 创建结果文件
    with open(output_file_path, "w") as output_file:
        # 主处理循环
        with tqdm(total=total_frames, desc=f"处理 {os.path.basename(video_path)}", unit="frame") as pbar:
            frame_id = 0  # 帧号计数器
            while True:
                # 获取帧
                frame = frame_queue.get()
                if frame is None:
                    break
                    
                frame_id += 1  # 帧号从1开始
                
                # 如果视频帧数为0，跳过处理
                if frame is None or frame.size == 0:
                    pbar.update(1)
                    continue
                
                # 初始化存储列表
                sub_frames = []
                detections = []
                
                # 图像分块处理
                for row in range(rows):
                    for col in range(cols):
                        x1 = col * sub_width
                        y1 = row * sub_height
                        x2 = min(x1 + sub_width, width)  # 确保不超出图像边界
                        y2 = min(y1 + sub_height, height)
                        
                        # 提取子图
                        sub_frame = frame[y1:y2, x1:x2]
                        # 检查子图是否有效
                        if sub_frame.size > 0:
                            sub_frames.append(sub_frame)
                
                # 如果没有有效的子图，跳过检测
                if not sub_frames:
                    pbar.update(1)
                    continue
                
                try:
                    # YOLO目标检测
                    results = model.predict(source=sub_frames, device=0, verbose=False)
                except Exception as e:
                    print(f"检测失败: {e}")
                    pbar.update(1)
                    continue
                
                # 目标跟踪处理
                for i, result in enumerate(results):
                    row = i // cols
                    col = i % cols
                    x_offset = col * sub_width
                    y_offset = row * sub_height
                    
                    # 检查结果是否有效
                    if result.boxes is None or result.boxes.data is None:
                        continue
                    
                    for box in result.boxes.data.cpu().numpy():
                        # 确保box有足够的数据
                        if len(box) < 6:
                            continue
                        x1, y1, x2, y2, conf, cls = box[:6]
                        # 添加偏移量得到原图坐标
                        x1 += x_offset
                        x2 += x_offset
                        y1 += y_offset
                        y2 += y_offset
                        detections.append([x1, y1, x2, y2, conf, cls])
                
                # 全局NMS去重处理
                if len(detections) > 0:
                    detections_array = np.array(detections)
                    from ultralytics.utils.ops import non_max_suppression
                    detections_tensor = torch.from_numpy(detections_array).float().unsqueeze(0)
                    nms_results = non_max_suppression(
                        detections_tensor,
                        conf_thres=0.25,
                        iou_thres=0.5,
                        max_det=300
                    )
                    
                    if nms_results and len(nms_results[0]) > 0:
                        detections = nms_results[0].cpu().numpy()
                    else:
                        detections = np.array([])
                else:
                    detections = np.array([])
                
                # 将检测结果转换为YOLO的Boxes格式
                if len(detections) > 0:
                    detections = Boxes(detections, frame.shape)
                else:
                    detections = Boxes(np.empty((0, 6)), frame.shape)
                
                # 更新目标跟踪器
                tracks = tracker.update(detections, frame)
                
                # 写入跟踪结果到文件
                if tracks is not None and len(tracks) > 0 and tracks.ndim == 2:
                    for track in tracks:
                        # 确保track有足够的数据
                        if len(track) < 5:
                            continue
                            
                        # 解析跟踪结果: [x1, y1, x2, y2, track_id, ...]
                        x1, y1, x2, y2 = track[:4]
                        track_id = int(track[4])
                        
                        # 计算宽度和高度
                        width_val = x2 - x1
                        height_val = y2 - y1
                        
                        # 写入结果: 帧号,目标ID,左上角X,左上角Y,宽度,高度,1,-1,-1,-1
                        line = f"{frame_id - 1},{track_id},{x1:.2f},{y1:.2f},{width_val:.2f},{height_val:.2f},1,-1,-1,-1\n"
                        output_file.write(line)
                
                # 更新进度条
                pbar.update(1)
    
    # 清理视频资源
    cap.release()
    print(f"✅ 完成处理: {os.path.basename(video_path)}")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='无人机多目标跟踪系统 - 批量处理视频')
    parser.add_argument('--video_folder', type=str, default='dataset1/test', 
                        help='包含视频文件的文件夹路径 (默认: UAV)')
    parser.add_argument('--output_dir', type=str, default='result', 
                        help='结果输出文件夹路径 (默认: result)')
    parser.add_argument('--model_weights', type=str, default='runs1/train/yolo116/weights/last.pt', 
                        help='YOLO模型权重文件路径 (默认: runs1/train/yolo116/weights/last.pt)')
    parser.add_argument('--tracker_config', type=str, default='cfg/bytetrack_improved.yaml', 
                        help='跟踪器配置文件路径 (默认: cfg/bytetrack_improved.yaml)')
    parser.add_argument('--sub_width', type=int, default=960, 
                        help='子图分割宽度 (默认: 960)')
    parser.add_argument('--sub_height', type=int, default=1080, 
                        help='子图分割高度 (默认: 1080)')
    parser.add_argument('--video_extensions', nargs='+', default=['.MOV', '.mp4', '.avi', '.mkv'], 
                        help='视频文件扩展名列表 (默认: .MOV .mp4 .avi .mkv)')
    
    args = parser.parse_args()
    
    # 打印参数配置
    print("\n" + "="*50)
    print("无人机多目标跟踪系统 - 批量处理模式")
    print("="*50)
    print(f"视频文件夹: {args.video_folder}")
    print(f"输出目录: {args.output_dir}")
    print(f"模型权重: {args.model_weights}")
    print(f"跟踪器配置: {args.tracker_config}")
    print(f"子图尺寸: {args.sub_width}x{args.sub_height}")
    print(f"支持的视频扩展名: {', '.join(args.video_extensions)}")
    print("="*50 + "\n")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化YOLO模型（只初始化一次，可重复使用）
    try:
        print("⏳ 正在加载YOLO模型...")
        model = YOLO(args.model_weights)
        print("✅ YOLO模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 获取视频文件夹中的所有视频文件
    video_files = []
    for file in os.listdir(args.video_folder):
        if any(file.lower().endswith(ext.lower()) for ext in args.video_extensions):
            video_files.append(file)
    
    if not video_files:
        print(f"在文件夹 {args.video_folder} 中未找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件待处理")
    
    # 处理所有视频文件
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(args.video_folder, video_file)
        # 生成输出文件名（保留原文件名，扩展名改为.txt）
        output_filename = os.path.splitext(video_file)[0] + ".txt"
        output_path = os.path.join(args.output_dir, output_filename)
        
        print(f"\n🔹 处理视频 [{i+1}/{len(video_files)}]: {video_file} -> {output_filename}")
        process_video(
            video_path=video_path,
            output_file_path=output_path,
            model=model,
            tracker_config=args.tracker_config,
            sub_width=args.sub_width,
            sub_height=args.sub_height
        )

    print("\n" + "="*50)
    print(f"🎉 所有视频处理完成！结果保存在: {args.output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()