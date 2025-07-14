# 命令示例：
# python simple_batch_tracking.py --input_dir data/val --timestamp

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

def create_competition_zip(results_dir="results"):
    """创建符合比赛要求的压缩包，压缩包名与目录同名"""
    # 检查结果目录是否存在
    if not os.path.exists(results_dir):
        print(f"❌ 结果目录不存在: {results_dir}")
        return None

    # 生成与目录同名的压缩包
    dir_name = os.path.basename(results_dir.rstrip('/'))
    zip_name = f"{dir_name}.zip"

    # 统计文件数量
    txt_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))

    if not txt_files:
        print(f"⚠️  警告: 在 {results_dir} 中没有找到 .txt 文件")
        return None

    print(f"📁 找到 {len(txt_files)} 个结果文件")

    # 创建压缩包
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in txt_files:
            file_name = os.path.basename(file_path)
            # 压缩包内保持原始文件名结构
            arcname = os.path.join('results', file_name)
            zipf.write(file_path, arcname)
            print(f"  ✅ 添加文件: {file_name}")

    # 验证压缩包
    file_size = os.path.getsize(zip_name)
    print(f"📦 比赛压缩包已创建: {zip_name}")
    print(f"📊 压缩包大小: {file_size / 1024:.2f} KB")

    return zip_name

def verify_competition_zip(zip_file):
    """验证比赛压缩包的内容和格式"""
    if not os.path.exists(zip_file):
        print(f"❌ 压缩包不存在: {zip_file}")
        return False

    print(f"\n🔍 验证压缩包: {zip_file}")

    try:
        with zipfile.ZipFile(zip_file, 'r') as zipf:
            file_list = zipf.namelist()

            print(f"📋 压缩包内容 ({len(file_list)} 个文件):")

            valid_files = 0
            for file_name in file_list:
                print(f"  📄 {file_name}")

                # 检查文件路径格式
                if file_name.startswith('results/') and file_name.endswith('.txt'):
                    valid_files += 1

                    # 验证文件内容格式 (检查前几行)
                    try:
                        with zipf.open(file_name) as f:
                            lines = f.read().decode('utf-8').strip().split('\n')
                            if lines and lines[0]:  # 有内容
                                first_line = lines[0]
                                parts = first_line.split(',')
                                if len(parts) == 10:
                                    print(f"    ✅ 格式正确 ({len(lines)} 行数据)")
                                else:
                                    print(f"    ⚠️  格式可能有问题: {len(parts)} 个字段 (期望10个)")
                            else:
                                print(f"    ⚠️  文件为空")
                    except Exception as e:
                        print(f"    ❌ 读取文件出错: {e}")
                else:
                    print(f"    ⚠️  文件路径格式不正确")

            print(f"\n📊 验证结果:")
            print(f"  - 总文件数: {len(file_list)}")
            print(f"  - 有效文件数: {valid_files}")
            print(f"  - 压缩包大小: {os.path.getsize(zip_file) / 1024:.2f} KB")

            if valid_files == len(file_list) and valid_files > 0:
                print(f"  ✅ 压缩包验证通过!")
                return True
            else:
                print(f"  ⚠️  压缩包可能有问题")
                return False

    except Exception as e:
        print(f"❌ 验证压缩包时出错: {e}")
        return False

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
                       default='runs/train/20250712_1824_no_pretrain_yolo11x_imgsz1280_epoch300_bs4/weights/best.pt',
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

    # 验证压缩包
    if zip_file:
        verify_competition_zip(zip_file)
    
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
