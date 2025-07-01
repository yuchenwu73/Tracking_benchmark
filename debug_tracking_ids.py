#!/usr/bin/env python3
"""
跟踪ID诊断脚本
用于分析ID分配异常的原因
"""

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_id_pattern():
    """分析ID分配模式"""
    print("🔍 跟踪ID异常诊断")
    print("=" * 50)
    
    # 模拟观察到的ID序列
    observed_ids = [1, 34, 43, 6]  # 从图片中观察到的ID
    
    print(f"📊 观察到的ID序列: {observed_ids}")
    print(f"📈 ID范围: {min(observed_ids)} - {max(observed_ids)}")
    print(f"🔢 ID总数: {len(observed_ids)}")
    print(f"💔 缺失的ID数量: {max(observed_ids) - len(observed_ids)}")
    
    # 分析可能的原因
    print("\n🤔 可能的原因分析:")
    
    # 1. 检测不稳定
    print("1. 检测不稳定导致ID浪费:")
    print("   - 误检测被分配ID后立即删除")
    print("   - 目标短暂消失导致新ID分配")
    print("   - 检测置信度波动导致轨迹中断")
    
    # 2. 子图重复检测
    print("\n2. 子图切分导致重复检测:")
    print("   - 同一目标在多个子图边界被重复检测")
    print("   - 每次重复检测都可能分配新ID")
    
    # 3. 跟踪器参数
    print("\n3. 跟踪器参数设置:")
    print("   - new_track_thresh=0.25 可能过低")
    print("   - track_buffer=30 可能过短")
    
    return observed_ids

def suggest_solutions():
    """建议解决方案"""
    print("\n🔧 建议的解决方案:")
    print("=" * 50)
    
    solutions = [
        {
            "问题": "检测不稳定",
            "解决方案": [
                "提高 new_track_thresh 到 0.4-0.5",
                "增加 track_buffer 到 60-90",
                "使用更稳定的检测模型"
            ]
        },
        {
            "问题": "子图重复检测", 
            "解决方案": [
                "添加NMS后处理去除重复检测",
                "调整子图重叠区域",
                "在全图坐标下进行NMS"
            ]
        },
        {
            "问题": "跟踪器配置",
            "解决方案": [
                "调整 track_high_thresh 到 0.3-0.4",
                "降低 track_low_thresh 到 0.05",
                "增加 match_thresh 到 0.9"
            ]
        }
    ]
    
    for i, solution in enumerate(solutions, 1):
        print(f"\n{i}. {solution['问题']}:")
        for sol in solution['解决方案']:
            print(f"   ✓ {sol}")

def create_improved_config():
    """创建改进的配置"""
    print("\n📝 建议的改进配置:")
    print("=" * 50)
    
    improved_config = """
# 改进的ByteTrack配置 - 减少ID浪费
tracker_type: bytetrack
track_high_thresh: 0.4    # 提高第一次关联阈值，减少误检测
track_low_thresh: 0.05    # 降低第二次关联阈值，增加匹配机会  
new_track_thresh: 0.5     # 提高新轨迹初始化阈值，减少误检测
track_buffer: 60          # 增加轨迹缓冲，减少过早删除
match_thresh: 0.9         # 提高匹配阈值，增加匹配精度
fuse_score: True
"""
    
    print(improved_config)
    
    # 保存改进的配置
    with open("data/bytetrack_improved.yaml", "w") as f:
        f.write(improved_config.strip())
    
    print("✅ 改进配置已保存到 data/bytetrack_improved.yaml")

def debug_detection_overlap():
    """调试子图重复检测问题"""
    print("\n🔍 子图重复检测诊断:")
    print("=" * 50)
    
    # 模拟8K视频分割
    video_width = 7680  # 8K宽度
    video_height = 4320  # 8K高度
    sub_width = 960
    sub_height = 1080
    
    cols = video_width // sub_width  # 8列
    rows = video_height // sub_height  # 4行
    
    print(f"📐 视频尺寸: {video_width}x{video_height}")
    print(f"🔲 子图尺寸: {sub_width}x{sub_height}")
    print(f"📊 分割网格: {rows}行 x {cols}列 = {rows*cols}个子图")
    
    # 检查边界重叠问题
    print(f"\n⚠️  潜在问题:")
    print(f"   - 目标在子图边界可能被多次检测")
    print(f"   - 每个子图独立处理，缺乏全局NMS")
    print(f"   - 可能导致同一目标分配多个ID")
    
    print(f"\n💡 建议改进:")
    print(f"   - 在全图坐标下进行NMS去重")
    print(f"   - 添加子图重叠区域")
    print(f"   - 使用全局检测结果进行跟踪")

if __name__ == "__main__":
    # 运行诊断
    observed_ids = analyze_id_pattern()
    suggest_solutions()
    create_improved_config()
    debug_detection_overlap()
    
    print(f"\n🎯 总结:")
    print(f"ID跳跃从1到{max(observed_ids)}主要是由于:")
    print(f"1. 检测不稳定导致大量临时ID被分配和删除")
    print(f"2. 子图切分可能导致重复检测")
    print(f"3. 跟踪器参数需要针对无人机场景优化")
    print(f"\n建议使用改进的配置文件重新测试！")
