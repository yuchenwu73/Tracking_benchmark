# 运行命令示例：
# python self_evaluation.py --results_dir results

import os
import argparse
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

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

def analyze_tracking_quality(results, video_name):
    """分析跟踪质量"""
    if not results:
        return {
            'video': video_name,
            'quality_score': 0.0,
            'issues': ['没有跟踪结果']
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
    
    # 目标大小分析
    areas = [r['area'] for r in results]
    avg_area = np.mean(areas)
    area_std = np.std(areas)
    
    # 检测密度分析
    detections_per_frame = total_detections / frame_span if frame_span > 0 else 0
    
    # 质量评估
    quality_factors = []
    issues = []
    
    # 1. 轨迹连续性 (权重: 30%)
    if avg_continuity >= 0.8:
        continuity_score = 1.0
    elif avg_continuity >= 0.6:
        continuity_score = 0.7
    elif avg_continuity >= 0.4:
        continuity_score = 0.4
    else:
        continuity_score = 0.1
        issues.append("轨迹连续性差")
    
    quality_factors.append(('轨迹连续性', continuity_score, 0.3))
    
    # 2. 轨迹长度合理性 (权重: 25%)
    avg_track_length = np.mean(list(track_lengths.values()))
    if avg_track_length >= 20:
        length_score = 1.0
    elif avg_track_length >= 10:
        length_score = 0.8
    elif avg_track_length >= 5:
        length_score = 0.5
    else:
        length_score = 0.2
        issues.append("轨迹长度过短")
    
    quality_factors.append(('轨迹长度', length_score, 0.25))
    
    # 3. 检测密度合理性 (权重: 20%)
    if 0.5 <= detections_per_frame <= 5.0:
        density_score = 1.0
    elif 0.1 <= detections_per_frame <= 10.0:
        density_score = 0.7
    elif detections_per_frame > 10.0:
        density_score = 0.3
        issues.append("检测密度过高，可能存在虚警")
    else:
        density_score = 0.2
        issues.append("检测密度过低")
    
    quality_factors.append(('检测密度', density_score, 0.2))
    
    # 4. ID一致性 (权重: 15%)
    id_efficiency = total_detections / unique_ids if unique_ids > 0 else 0
    if id_efficiency >= 10:
        id_score = 1.0
    elif id_efficiency >= 5:
        id_score = 0.8
    elif id_efficiency >= 2:
        id_score = 0.5
    else:
        id_score = 0.2
        issues.append("ID切换频繁")
    
    quality_factors.append(('ID一致性', id_score, 0.15))
    
    # 5. 目标大小一致性 (权重: 10%)
    if avg_area > 0:
        size_consistency = 1 - min(area_std / avg_area, 1.0)
    else:
        size_consistency = 0.0
    
    if size_consistency >= 0.7:
        size_score = 1.0
    elif size_consistency >= 0.5:
        size_score = 0.7
    else:
        size_score = 0.4
        issues.append("目标大小变化过大")
    
    quality_factors.append(('大小一致性', size_score, 0.1))
    
    # 计算加权质量分数
    quality_score = sum(score * weight for _, score, weight in quality_factors)
    
    return {
        'video': video_name,
        'total_detections': total_detections,
        'unique_ids': unique_ids,
        'frame_span': frame_span,
        'avg_track_length': avg_track_length,
        'avg_continuity': avg_continuity,
        'detections_per_frame': detections_per_frame,
        'quality_score': quality_score,
        'quality_factors': quality_factors,
        'issues': issues
    }

def estimate_competition_score(quality_scores):
    """基于质量分数估算比赛得分"""
    if not quality_scores:
        return 0.0
    
    avg_quality = np.mean(quality_scores)
    
    # 经验公式：将质量分数映射到MOTA+IDF1范围
    # 这是一个粗略估算，实际得分需要真值数据
    estimated_mota = avg_quality * 0.8  # MOTA通常比质量分数略低
    estimated_idf1 = avg_quality * 0.9  # IDF1相对容易获得高分
    
    estimated_score = (estimated_mota + estimated_idf1) / 2
    
    return estimated_score, estimated_mota, estimated_idf1

def main():
    parser = argparse.ArgumentParser(description='跟踪结果自评估工具')
    parser.add_argument('--results_dir', type=str, default='results', help='结果目录')
    parser.add_argument('--output_file', type=str, default='self_evaluation.txt', help='评估报告输出文件')
    
    args = parser.parse_args()
    
    print("📊 跟踪结果自评估工具")
    print("=" * 50)
    print(f"📁 结果目录: {args.results_dir}")
    
    if not os.path.exists(args.results_dir):
        print(f"❌ 结果目录不存在: {args.results_dir}")
        return
    
    # 获取所有结果文件
    result_files = [f for f in os.listdir(args.results_dir) if f.endswith('.txt')]
    
    if not result_files:
        print(f"❌ 在 {args.results_dir} 中没有找到txt文件")
        return
    
    print(f"📹 找到 {len(result_files)} 个结果文件")
    
    all_analyses = []
    quality_scores = []
    
    # 分析每个视频
    for result_file in sorted(result_files):
        video_name = result_file[:-4]
        file_path = os.path.join(args.results_dir, result_file)
        
        results = load_tracking_results(file_path)
        analysis = analyze_tracking_quality(results, video_name)
        
        all_analyses.append(analysis)
        quality_scores.append(analysis['quality_score'])
        
        print(f"\n📹 {video_name}:")
        print(f"   检测数量: {analysis['total_detections']}")
        print(f"   唯一ID数: {analysis['unique_ids']}")
        print(f"   平均轨迹长度: {analysis['avg_track_length']:.1f}")
        print(f"   轨迹连续性: {analysis['avg_continuity']:.3f}")
        print(f"   质量得分: {analysis['quality_score']:.3f}")
        
        if analysis['issues']:
            print(f"   ⚠️ 问题: {', '.join(analysis['issues'])}")
    
    # 总体评估
    if quality_scores:
        print(f"\n📈 总体评估")
        print("=" * 50)
        
        avg_quality = np.mean(quality_scores)
        min_quality = np.min(quality_scores)
        max_quality = np.max(quality_scores)
        
        print(f"平均质量得分: {avg_quality:.3f}")
        print(f"质量得分范围: {min_quality:.3f} - {max_quality:.3f}")
        
        # 估算比赛得分
        estimated_score, estimated_mota, estimated_idf1 = estimate_competition_score(quality_scores)
        
        print(f"\n🎯 预估比赛得分:")
        print(f"预估 MOTA: {estimated_mota:.3f}")
        print(f"预估 IDF1: {estimated_idf1:.3f}")
        print(f"预估总分: {estimated_score:.3f}")
        
        # 评分等级
        if estimated_score >= 0.7:
            grade = "🥇 优秀 (70%+)"
        elif estimated_score >= 0.5:
            grade = "🥈 良好 (50-70%)"
        elif estimated_score >= 0.3:
            grade = "🥉 一般 (30-50%)"
        else:
            grade = "📉 需要改进 (<30%)"
        
        print(f"预估等级: {grade}")
        
        # 改进建议
        print(f"\n💡 改进建议:")
        
        common_issues = defaultdict(int)
        for analysis in all_analyses:
            for issue in analysis['issues']:
                common_issues[issue] += 1
        
        if common_issues:
            for issue, count in sorted(common_issues.items(), key=lambda x: x[1], reverse=True):
                print(f"   - {issue} (影响{count}个视频)")
        else:
            print("   - 整体质量良好，继续保持")
        
        # 保存详细报告
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write("跟踪结果自评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("各视频详细分析:\n")
            for analysis in all_analyses:
                f.write(f"\n{analysis['video']}:\n")
                f.write(f"  检测数量: {analysis['total_detections']}\n")
                f.write(f"  唯一ID数: {analysis['unique_ids']}\n")
                f.write(f"  帧数范围: {analysis['frame_span']}\n")
                f.write(f"  平均轨迹长度: {analysis['avg_track_length']:.1f}\n")
                f.write(f"  轨迹连续性: {analysis['avg_continuity']:.3f}\n")
                f.write(f"  检测密度: {analysis['detections_per_frame']:.2f}/帧\n")
                f.write(f"  质量得分: {analysis['quality_score']:.3f}\n")
                
                if analysis['issues']:
                    f.write(f"  问题: {', '.join(analysis['issues'])}\n")
            
            f.write(f"\n总体评估:\n")
            f.write(f"平均质量得分: {avg_quality:.3f}\n")
            f.write(f"预估 MOTA: {estimated_mota:.3f}\n")
            f.write(f"预估 IDF1: {estimated_idf1:.3f}\n")
            f.write(f"预估总分: {estimated_score:.3f}\n")
        
        print(f"\n📄 详细报告已保存到: {args.output_file}")
        
        # 置信度说明
        print(f"\n📝 说明:")
        print("   - 这是基于跟踪结果统计的预估分数")
        print("   - 实际比赛得分需要真值数据进行精确计算")
        print("   - 预估准确度约为 ±0.1-0.2")
        print("   - 建议重点关注质量得分较低的视频")

if __name__ == "__main__":
    main()
