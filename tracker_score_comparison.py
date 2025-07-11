# 运行命令示例：
# python tracker_score_comparison.py --comparison_dir tracker_comparison

import os
import argparse
import numpy as np
from collections import defaultdict
import pandas as pd

def load_tracking_results(file_path):
    """加载跟踪结果文件"""
    results = []
    if not os.path.exists(file_path):
        return results
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) >= 6:
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x_left = float(parts[2])
                y_top = float(parts[3])
                width = float(parts[4])
                height = float(parts[5])
                
                results.append({
                    'frame': frame_id,
                    'id': track_id,
                    'x': x_left + width/2,
                    'y': y_top + height/2,
                    'w': width,
                    'h': height,
                    'area': width * height
                })
    
    return results

def analyze_tracking_quality(results, video_name, tracker_name):
    """分析跟踪质量并计算质量分数"""
    if not results:
        return {
            'video': video_name,
            'tracker': tracker_name,
            'quality_score': 0.0,
            'total_detections': 0,
            'unique_ids': 0,
            'avg_track_length': 0.0,
            'avg_continuity': 0.0,
            'detections_per_frame': 0.0
        }
    
    # 基本统计
    total_detections = len(results)
    unique_ids = len(set(r['id'] for r in results))
    max_frame = max(r['frame'] for r in results)
    min_frame = min(r['frame'] for r in results)
    frame_span = max_frame - min_frame + 1
    
    # 轨迹长度分析
    track_lengths = defaultdict(int)
    track_frames = defaultdict(set)
    for r in results:
        track_lengths[r['id']] += 1
        track_frames[r['id']].add(r['frame'])
    
    # 计算轨迹连续性
    track_continuity = {}
    for track_id, frames in track_frames.items():
        frame_list = sorted(frames)
        if len(frame_list) > 1:
            expected_frames = frame_list[-1] - frame_list[0] + 1
            actual_frames = len(frame_list)
            continuity = actual_frames / expected_frames
        else:
            continuity = 1.0
        track_continuity[track_id] = continuity
    
    avg_continuity = np.mean(list(track_continuity.values())) if track_continuity else 0.0
    avg_track_length = np.mean(list(track_lengths.values())) if track_lengths else 0.0
    detections_per_frame = total_detections / frame_span if frame_span > 0 else 0
    
    # 质量评估 (简化版本，专注于比较)
    quality_factors = []
    
    # 1. 轨迹连续性 (30%)
    continuity_score = min(avg_continuity / 0.8, 1.0)  # 0.8为理想值
    quality_factors.append(('连续性', continuity_score, 0.3))
    
    # 2. 轨迹长度 (25%)
    length_score = min(avg_track_length / 20, 1.0)  # 20为理想值
    quality_factors.append(('长度', length_score, 0.25))
    
    # 3. 检测密度 (20%)
    ideal_density = 2.0  # 理想的每帧检测数
    if detections_per_frame <= ideal_density:
        density_score = detections_per_frame / ideal_density
    else:
        density_score = max(0.1, 1.0 - (detections_per_frame - ideal_density) / 10)
    quality_factors.append(('密度', density_score, 0.2))
    
    # 4. ID效率 (15%)
    id_efficiency = total_detections / unique_ids if unique_ids > 0 else 0
    efficiency_score = min(id_efficiency / 10, 1.0)  # 10为理想值
    quality_factors.append(('ID效率', efficiency_score, 0.15))
    
    # 5. 检测数量合理性 (10%)
    detection_score = min(total_detections / 100, 1.0)  # 100为基准
    quality_factors.append(('检测量', detection_score, 0.1))
    
    # 计算加权质量分数
    quality_score = sum(score * weight for _, score, weight in quality_factors)
    
    return {
        'video': video_name,
        'tracker': tracker_name,
        'quality_score': quality_score,
        'total_detections': total_detections,
        'unique_ids': unique_ids,
        'avg_track_length': avg_track_length,
        'avg_continuity': avg_continuity,
        'detections_per_frame': detections_per_frame,
        'quality_factors': quality_factors
    }

def estimate_competition_score(quality_score):
    """基于质量分数估算比赛得分"""
    # 经验公式：将质量分数映射到MOTA+IDF1范围
    estimated_mota = quality_score * 0.8
    estimated_idf1 = quality_score * 0.9
    estimated_score = (estimated_mota + estimated_idf1) / 2
    return estimated_score, estimated_mota, estimated_idf1

def main():
    parser = argparse.ArgumentParser(description='跟踪器得分比较工具')
    parser.add_argument('--comparison_dir', type=str, default='tracker_comparison', help='比较结果目录')
    parser.add_argument('--output_file', type=str, default='tracker_score_comparison.txt', help='比较报告输出文件')
    
    args = parser.parse_args()
    
    print("🏆 跟踪器得分比较分析")
    print("=" * 60)
    print(f"📁 比较目录: {args.comparison_dir}")
    
    if not os.path.exists(args.comparison_dir):
        print(f"❌ 比较目录不存在: {args.comparison_dir}")
        return
    
    # 获取所有结果文件
    result_files = [f for f in os.listdir(args.comparison_dir) if f.endswith('.txt')]
    
    if not result_files:
        print(f"❌ 在 {args.comparison_dir} 中没有找到txt文件")
        return
    
    # 解析文件名，提取视频名和跟踪器类型
    video_tracker_results = defaultdict(dict)
    
    for result_file in result_files:
        # 文件名格式: video_tracker.txt
        name_parts = result_file[:-4].split('_')
        if len(name_parts) >= 2:
            video_name = '_'.join(name_parts[:-1])
            tracker_name = name_parts[-1]
            
            file_path = os.path.join(args.comparison_dir, result_file)
            results = load_tracking_results(file_path)
            analysis = analyze_tracking_quality(results, video_name, tracker_name)
            
            video_tracker_results[video_name][tracker_name] = analysis
    
    if not video_tracker_results:
        print("❌ 没有找到有效的比较结果")
        return
    
    print(f"📹 找到 {len(video_tracker_results)} 个视频的比较结果")
    
    # 收集所有跟踪器类型
    all_trackers = set()
    for video_results in video_tracker_results.values():
        all_trackers.update(video_results.keys())
    
    all_trackers = sorted(all_trackers)
    print(f"🔧 跟踪器类型: {', '.join(all_trackers)}")
    
    # 详细比较每个视频
    print(f"\n📊 各视频详细比较:")
    print("=" * 80)
    
    tracker_totals = defaultdict(list)
    
    for video_name in sorted(video_tracker_results.keys()):
        print(f"\n📹 {video_name}:")
        print(f"{'跟踪器':<12} {'质量分数':<8} {'预估得分':<8} {'检测数':<8} {'轨迹数':<8} {'平均长度':<8} {'连续性':<8}")
        print("-" * 80)
        
        video_results = video_tracker_results[video_name]
        video_scores = {}
        
        for tracker in all_trackers:
            if tracker in video_results:
                analysis = video_results[tracker]
                quality_score = analysis['quality_score']
                estimated_score, _, _ = estimate_competition_score(quality_score)
                
                print(f"{tracker:<12} {quality_score:<8.3f} {estimated_score:<8.3f} "
                      f"{analysis['total_detections']:<8} {analysis['unique_ids']:<8} "
                      f"{analysis['avg_track_length']:<8.1f} {analysis['avg_continuity']:<8.3f}")
                
                video_scores[tracker] = estimated_score
                tracker_totals[tracker].append(estimated_score)
            else:
                print(f"{tracker:<12} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
        
        # 显示该视频的最佳跟踪器
        if video_scores:
            best_tracker = max(video_scores.keys(), key=lambda x: video_scores[x])
            best_score = video_scores[best_tracker]
            print(f"🏆 最佳: {best_tracker} (得分: {best_score:.3f})")
    
    # 总体比较
    print(f"\n📈 总体比较结果:")
    print("=" * 60)
    
    tracker_summary = {}
    for tracker in all_trackers:
        if tracker_totals[tracker]:
            avg_score = np.mean(tracker_totals[tracker])
            std_score = np.std(tracker_totals[tracker])
            min_score = np.min(tracker_totals[tracker])
            max_score = np.max(tracker_totals[tracker])
            video_count = len(tracker_totals[tracker])
            
            tracker_summary[tracker] = {
                'avg_score': avg_score,
                'std_score': std_score,
                'min_score': min_score,
                'max_score': max_score,
                'video_count': video_count
            }
    
    # 按平均得分排序
    sorted_trackers = sorted(tracker_summary.keys(), key=lambda x: tracker_summary[x]['avg_score'], reverse=True)
    
    print(f"{'排名':<4} {'跟踪器':<12} {'平均得分':<10} {'标准差':<8} {'范围':<15} {'视频数':<6}")
    print("-" * 60)
    
    for i, tracker in enumerate(sorted_trackers, 1):
        summary = tracker_summary[tracker]
        range_str = f"{summary['min_score']:.3f}-{summary['max_score']:.3f}"
        
        if i == 1:
            rank_icon = "🥇"
        elif i == 2:
            rank_icon = "🥈"
        elif i == 3:
            rank_icon = "🥉"
        else:
            rank_icon = f"{i}."
        
        print(f"{rank_icon:<4} {tracker:<12} {summary['avg_score']:<10.3f} "
              f"{summary['std_score']:<8.3f} {range_str:<15} {summary['video_count']:<6}")
    
    # 推荐
    if sorted_trackers:
        best_tracker = sorted_trackers[0]
        best_score = tracker_summary[best_tracker]['avg_score']
        
        print(f"\n🎯 推荐结果:")
        print(f"最佳跟踪器: {best_tracker}")
        print(f"平均得分: {best_score:.3f}")
        
        if best_score >= 0.7:
            grade = "🥇 优秀"
        elif best_score >= 0.5:
            grade = "🥈 良好"
        elif best_score >= 0.3:
            grade = "🥉 一般"
        else:
            grade = "📉 需要改进"
        
        print(f"预估等级: {grade}")
        
        # 性能差异分析
        if len(sorted_trackers) > 1:
            second_best = sorted_trackers[1]
            score_diff = tracker_summary[best_tracker]['avg_score'] - tracker_summary[second_best]['avg_score']
            
            if score_diff > 0.05:
                print(f"💡 {best_tracker} 明显优于 {second_best} (差距: {score_diff:.3f})")
            elif score_diff > 0.02:
                print(f"💡 {best_tracker} 略优于 {second_best} (差距: {score_diff:.3f})")
            else:
                print(f"💡 {best_tracker} 与 {second_best} 性能接近 (差距: {score_diff:.3f})")
    
    # 保存详细报告
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write("跟踪器得分比较报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("总体排名:\n")
        for i, tracker in enumerate(sorted_trackers, 1):
            summary = tracker_summary[tracker]
            f.write(f"{i}. {tracker}: {summary['avg_score']:.3f} ± {summary['std_score']:.3f}\n")
        
        f.write(f"\n各视频详细结果:\n")
        for video_name in sorted(video_tracker_results.keys()):
            f.write(f"\n{video_name}:\n")
            video_results = video_tracker_results[video_name]
            for tracker in all_trackers:
                if tracker in video_results:
                    analysis = video_results[tracker]
                    estimated_score, _, _ = estimate_competition_score(analysis['quality_score'])
                    f.write(f"  {tracker}: {estimated_score:.3f}\n")
        
        if sorted_trackers:
            f.write(f"\n推荐: {sorted_trackers[0]} (平均得分: {tracker_summary[sorted_trackers[0]]['avg_score']:.3f})\n")
    
    print(f"\n📄 详细报告已保存到: {args.output_file}")

if __name__ == "__main__":
    main()
