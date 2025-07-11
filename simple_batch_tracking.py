# 运行命令示例：
# python simple_batch_tracking.py --input_dir data/test

import os
import argparse
import glob
import zipfile
import torch
import time
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm

def save_competition_results(tracking_results, video_name, output_dir="results"):
    """保存符合比赛要求的跟踪结果"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{video_name}.txt")
    
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

def create_competition_zip(results_dir="results", zip_name="results.zip"):
    """创建符合比赛要求的压缩包"""
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    arcname = os.path.join('results', file)
                    zipf.write(file_path, arcname)
    
    print(f"比赛压缩包已创建: {zip_name}")
    return zip_name

def process_video_simple(video_path, model, output_dir):
    """
    简化版视频处理函数，使用YOLO内置跟踪
    """
    print(f"\n🎬 处理视频: {video_path}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    competition_results = []

    try:
        # 使用YOLO内置跟踪功能 - 对齐官方最佳实践
        results = model.track(
            source=video_path,
            stream=True,  # 流式处理，节省内存
            verbose=False
            # 使用默认的BoT-SORT跟踪器，如官方文档所述
        )
        
        frame_id = -1  # 从-1开始，这样第一帧就是0
        for result in tqdm(results, desc=f"处理 {video_name}"):
            frame_id += 1

            # 使用官方推荐的检查方式
            if result.boxes and result.boxes.is_track:
                # 获取边界框和跟踪ID
                boxes = result.boxes.xywh.cpu().numpy()  # 中心坐标格式
                track_ids = result.boxes.id.int().cpu().tolist()

                # 保存每个跟踪目标（比赛格式：10个字段）
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    # 格式：帧号,目标ID,左上角X,左上角Y,宽度,高度,类别,-1,-1,-1
                    competition_results.append([frame_id, track_id, float(x), float(y), float(w), float(h), 1, -1, -1, -1])
        
        # 保存结果
        if competition_results:
            save_competition_results(competition_results, video_name, output_dir)
            print(f"✅ {video_name}: 保存了 {len(competition_results)} 条跟踪记录")
        else:
            print(f"⚠️ {video_name}: 没有跟踪结果")
            
    except Exception as e:
        print(f"❌ 处理视频 {video_name} 时出错: {e}")
        competition_results = []
    
    return competition_results

def main():
    parser = argparse.ArgumentParser(description='简化版批量视频跟踪处理')
    parser.add_argument('--input_dir', type=str, required=True, help='输入视频目录')
    parser.add_argument('--output_dir', type=str, default='results', help='输出结果目录')
    parser.add_argument('--timestamp', action='store_true', help='在输出目录名中添加时间戳')
    parser.add_argument('--model_path', type=str, 
                       default='/data2/wuyuchen/Tracking_benchmark/runs/train/20250809_2327_yolo11m_imgsz1280_epoch300_bs8/weights/best.pt',
                       help='YOLO模型路径')
    parser.add_argument('--video_extensions', nargs='+', 
                       default=['*.avi', '*.mp4', '*.mov', '*.MOV'], 
                       help='视频文件扩展名')
    
    args = parser.parse_args()
    
    print("🚀 简化版批量视频跟踪处理开始")
    print(f"📁 输入目录: {args.input_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🤖 模型路径: {args.model_path}")
    
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
    for video_file in video_files:
        print(f"  - {video_file}")
    
    # 创建输出目录（可选时间戳）
    if args.timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"{args.output_dir}_{timestamp}"
        print(f"📁 使用时间戳输出目录: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每个视频
    total_results = 0
    start_time = time.time()
    
    for video_file in video_files:
        results = process_video_simple(video_file, model, args.output_dir)
        total_results += len(results)
    
    # 创建压缩包
    print(f"\n📦 创建比赛提交压缩包...")
    zip_file = create_competition_zip(args.output_dir)
    
    # 显示总结
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🎉 批量处理完成!")
    print(f"📊 处理统计:")
    print(f"  - 处理视频数量: {len(video_files)}")
    print(f"  - 总跟踪记录: {total_results}")
    print(f"  - 总处理时间: {total_time:.2f} 秒")
    if len(video_files) > 0:
        print(f"  - 平均每视频: {total_time/len(video_files):.2f} 秒")
    print(f"📁 结果文件: {zip_file}")
    
    # 验证结果
    print(f"\n🔍 建议运行以下命令验证结果格式:")
    print(f"python test_competition_format.py --results_dir {args.output_dir} --zip_file {zip_file}")

if __name__ == "__main__":
    main()
