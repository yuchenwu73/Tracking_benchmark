import motmetrics as mm
import numpy as np
import os
from glob import glob
import pandas as pd

def evaluate_single_video(gt_path, pred_path):
    """评估单个视频的跟踪结果"""
    gt = mm.io.loadtxt(gt_path, fmt="mot15-2D")
    pred = mm.io.loadtxt(pred_path, fmt="mot15-2D")
    
    acc = mm.utils.compare_to_groundtruth(gt, pred, 'iou', distth=0.5)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'idf1', 'precision', 'recall', 'mostly_tracked', 'mostly_lost'], name='acc')
    
    mota = summary['mota'].values[0]
    idf1 = summary['idf1'].values[0]
    summary['score'] = (mota + idf1) / 2
    
    return summary

def batch_evaluate(gt_dir, pred_dir, output_csv='results.csv'):
    """
    批量评估多个视频的跟踪结果
    
    参数:
        gt_dir: 存放真实标注的目录
        pred_dir: 存放预测结果的目录
        output_csv: 结果输出文件路径
    """
    # 获取视频序列列表
    gt_files = sorted(glob(os.path.join(gt_dir, '*.txt')))
    video_names = [os.path.splitext(os.path.basename(f))[0] for f in gt_files]
    
    all_results = []
    
    print(f"{'视频名称':<20} {'MOTA':>6} {'IDF1':>6} {'Score':>6} {'Precision':>9} {'Recall':>7}")
    print("-"*70)
    
    for name in video_names:
        gt_path = os.path.join(gt_dir, f"{name}.txt")
        pred_path = os.path.join(pred_dir, f"{name}.txt")
        
        if not os.path.exists(pred_path):
            print(f"警告: {pred_path} 不存在，跳过")
            continue
            
        try:
            res = evaluate_single_video(gt_path, pred_path)
            row = {
                'video': name,
                'mota': res['mota'].values[0],
                'idf1': res['idf1'].values[0],
                'score': res['score'],
                'precision': res['precision'].values[0],
                'recall': res['recall'].values[0],
                'mostly_tracked': res['mostly_tracked'].values[0],
                'mostly_lost': res['mostly_lost'].values[0]
            }
            all_results.append(row)
            
            print(f"{name:<20} {row['mota']:>6.3f} {row['idf1']:>6.3f} {row['score']:>6.3f} "
                  f"{row['precision']:>9.3f} {row['recall']:>7.3f}")
        except Exception as e:
            print(f"评估 {name} 时出错: {str(e)}")
            continue
    
    # 计算平均值
    if all_results:
        df = pd.DataFrame(all_results)
        mean_results = {
            'video': '平均',
            'mota': df['mota'].mean(),
            'idf1': df['idf1'].mean(),
            'score': df['score'].mean(),
            'precision': df['precision'].mean(),
            'recall': df['recall'].mean(),
            'mostly_tracked': df['mostly_tracked'].mean(),
            'mostly_lost': df['mostly_lost'].mean()
        }
        
        print("-"*70)
        print(f"{'平均':<20} {mean_results['mota']:>6.3f} {mean_results['idf1']:>6.3f} "
              f"{mean_results['score']:>6.3f} {mean_results['precision']:>9.3f} "
              f"{mean_results['recall']:>7.3f}")
        
        # 保存结果到CSV
        df.loc['mean'] = mean_results
        df.to_csv(output_csv, index=False)
        print(f"\n结果已保存到 {output_csv}")
    else:
        print("没有有效的评估结果")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='批量评估跟踪结果')
    parser.add_argument('--gt_dir', required=True, help='真实标注目录')
    parser.add_argument('--pred_dir', required=True, help='预测结果目录')
    parser.add_argument('--output', default='evaluate.csv', help='输出CSV文件路径')
    parser.add_argument('--pattern',default='multi',help='单文件single或者多文件multi')
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("输出结果评估")
    print("="*50)

    print(f"视频文件夹: {args.gt_dir}")
    print(f"输出目录: {args.pred_dir}")
    print(f"结果CSV: {args.output}")
    print(f"评估模式: {args.pattern}")

    print("\n" + "="*100)
    if args.pattern == 'multi':
        batch_evaluate(args.gt_dir, args.pred_dir, args.output)
    elif args.pattern == 'single':
        print(evaluate_single_video(args.gt_dir, args.pred_dir))

    print("="*100)