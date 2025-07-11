# 运行命令示例：
# python tracker_comparison.py --input_dir data/val

import os
import argparse
import glob
import zipfile
import time
from ultralytics import YOLO
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm

def save_competition_results(tracking_results, video_name, tracker_name, output_dir="results"):
    """保存符合比赛要求的跟踪结果"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{video_name}_{tracker_name}.txt")
    
    with open(output_file, 'w') as f:
        for result in tracking_results:
            frame_id, track_id, x_center, y_center, width, height, class_id, conf1, conf2, conf3 = result
            # 将中心坐标转换为左上角坐标
            x_left = x_center - width / 2
            y_top = y_center - height / 2
            # 格式：帧号,目标ID,左上角X,左上角Y,宽度,高度,类别,-1,-1,-1
            f.write(f"{frame_id},{track_id},{x_left:.2f},{y_top:.2f},{width:.2f},{height:.2f},{class_id},{conf1},{conf2},{conf3}\n")
    
    print(f"比赛结果已保存到: {output_file}")
    return output_file

def process_video_with_tracker(video_path, model, tracker_config, output_dir):
    """
    使用指定跟踪器处理视频
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    tracker_name = os.path.splitext(os.path.basename(tracker_config))[0] if tracker_config else "default"
    
    print(f"\n🎬 处理视频: {video_path}")
    print(f"📊 使用跟踪器: {tracker_name}")
    
    competition_results = []
    
    try:
        # 根据是否指定跟踪器配置来调用
        if tracker_config:
            results = model.track(
                source=video_path,
                tracker=tracker_config,
                stream=True,
                verbose=False
            )
        else:
            # 使用默认跟踪器（BoT-SORT）
            results = model.track(
                source=video_path,
                stream=True,
                verbose=False
            )
        
        frame_id = -1  # 从-1开始，这样第一帧就是0
        start_time = time.time()
        
        for result in tqdm(results, desc=f"处理 {video_name} ({tracker_name})"):
            frame_id += 1
            
            # 使用官方推荐的检查方式
            if result.boxes and result.boxes.is_track:
                boxes = result.boxes.xywh.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    # 格式：帧号,目标ID,左上角X,左上角Y,宽度,高度,类别,-1,-1,-1
                    competition_results.append([frame_id, track_id, float(x), float(y), float(w), float(h), 1, -1, -1, -1])
        
        # 计算性能统计
        total_time = time.time() - start_time
        avg_fps = frame_id / total_time if total_time > 0 else 0
        
        print(f"✅ {tracker_name}: {len(competition_results)} 条记录, {avg_fps:.2f} FPS")
        
        # 保存结果
        if competition_results:
            save_competition_results(competition_results, video_name, tracker_name, output_dir)
        
        return {
            'tracker': tracker_name,
            'video': video_name,
            'tracks': len(competition_results),
            'fps': avg_fps,
            'time': total_time,
            'frames': frame_id
        }
        
    except Exception as e:
        print(f"❌ {tracker_name} 处理失败: {e}")
        return None

def compare_trackers(video_files, model, output_dir):
    """
    比较不同跟踪器的性能
    """
    print("\n🔍 跟踪器性能比较")
    print("=" * 60)
    
    # 可用的跟踪器配置
    trackers = [
        ("/data2/wuyuchen/Tracking_benchmark/cfg/botsort.yaml", "BoT-SORT (Default)"),
        ("/data2/wuyuchen/Tracking_benchmark/cfg/bytetrack.yaml", "ByteTrack")
    ]
    
    results = []
    
    for video_file in video_files:
        print(f"\n📹 测试视频: {os.path.basename(video_file)}")
        
        for tracker_config, tracker_desc in trackers:
            result = process_video_with_tracker(video_file, model, tracker_config, output_dir)
            if result:
                result['description'] = tracker_desc
                results.append(result)
    
    # 显示比较结果
    print(f"\n📊 性能比较结果")
    print("=" * 80)
    print(f"{'视频':<15} {'跟踪器':<15} {'轨迹数':<8} {'FPS':<8} {'时间(s)':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['video']:<15} {result['tracker']:<15} {result['tracks']:<8} "
              f"{result['fps']:<8.1f} {result['time']:<10.2f}")
    
    # 按跟踪器汇总
    tracker_summary = defaultdict(list)
    for result in results:
        tracker_summary[result['tracker']].append(result)
    
    print(f"\n📈 跟踪器汇总")
    print("=" * 60)
    for tracker, tracker_results in tracker_summary.items():
        avg_fps = sum(r['fps'] for r in tracker_results) / len(tracker_results)
        total_tracks = sum(r['tracks'] for r in tracker_results)
        print(f"{tracker:<15}: 平均FPS {avg_fps:.1f}, 总轨迹数 {total_tracks}")

def main():
    parser = argparse.ArgumentParser(description='跟踪器性能比较工具')
    parser.add_argument('--input_dir', type=str, required=True, help='输入视频目录')
    parser.add_argument('--output_dir', type=str, default='tracker_comparison', help='输出结果目录')
    parser.add_argument('--model_path', type=str, 
                       default='/data2/wuyuchen/Tracking_benchmark/runs/train/20250809_2327_yolo11m_imgsz1280_epoch300_bs8/weights/best.pt',
                       help='YOLO模型路径')
    parser.add_argument('--tracker', type=str, choices=['botsort', 'bytetrack', 'all'],
                       default='all', help='选择跟踪器')
    parser.add_argument('--video_extensions', nargs='+', 
                       default=['*.avi', '*.mp4', '*.mov', '*.MOV'], 
                       help='视频文件扩展名')
    
    args = parser.parse_args()
    
    print("🚀 YOLO跟踪器性能比较工具")
    print(f"📁 输入目录: {args.input_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🤖 模型路径: {args.model_path}")
    print(f"📊 跟踪器: {args.tracker}")
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"❌ 模型文件不存在: {args.model_path}")
        return
    
    # 初始化模型
    print("\n🔧 初始化YOLO模型...")
    try:
        model = YOLO(args.model_path)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 查找视频文件
    video_files = []
    for ext in args.video_extensions:
        pattern = os.path.join(args.input_dir, ext)
        video_files.extend(glob.glob(pattern))
    
    if not video_files:
        print(f"❌ 在目录 {args.input_dir} 中没有找到视频文件")
        return
    
    print(f"\n📹 找到 {len(video_files)} 个视频文件:")
    for video_file in video_files[:3]:  # 只显示前3个
        print(f"  - {video_file}")
    if len(video_files) > 3:
        print(f"  - ... 还有 {len(video_files) - 3} 个文件")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 根据选择执行比较
    if args.tracker == 'all':
        compare_trackers(video_files, model, args.output_dir)
    else:
        # 单个跟踪器测试
        tracker_map = {
            'botsort': '/data2/wuyuchen/Tracking_benchmark/cfg/botsort.yaml',
            'bytetrack': '/data2/wuyuchen/Tracking_benchmark/cfg/bytetrack.yaml'
        }
        
        tracker_config = tracker_map[args.tracker]
        
        print(f"\n🎯 使用 {args.tracker} 跟踪器处理所有视频")
        
        total_results = 0
        start_time = time.time()
        
        for video_file in video_files:
            result = process_video_with_tracker(video_file, model, tracker_config, args.output_dir)
            if result:
                total_results += result['tracks']
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 处理完成!")
        print(f"📊 总统计:")
        print(f"  - 处理视频数: {len(video_files)}")
        print(f"  - 总跟踪记录: {total_results}")
        print(f"  - 总处理时间: {total_time:.2f} 秒")

if __name__ == "__main__":
    main()
