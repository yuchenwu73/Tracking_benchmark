#!/usr/bin/env python3
"""
快速测试不同loss缩放策略的效果
每个策略只训练少量epoch来快速验证

使用方法:
python quick_loss_test.py --device 6
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import datetime
from ultralytics import YOLO
import torch
import time

def quick_test_strategy(model_name, strategy_name, box_weight, cls_weight, dfl_weight, device, epochs=10):
    """快速测试一个策略"""
    
    print(f"\n{'='*60}")
    print(f"测试策略: {strategy_name}")
    print(f"Loss权重: box={box_weight}, cls={cls_weight}, dfl={dfl_weight}")
    print(f"训练轮数: {epochs}")
    print(f"{'='*60}")
    
    # 加载模型
    model = YOLO(f'{model_name}.pt')
    
    # 生成测试名称
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    name = f"test_{timestamp}_{strategy_name}"
    
    start_time = time.time()
    
    try:
        # 训练
        results = model.train(
            data='dataset/data.yaml',
            epochs=epochs,
            imgsz=640,  # 使用较小尺寸加快测试
            batch=8,    # 较大批次加快测试
            device=device,
            workers=4,
            lr0=0.001,
            
            # 测试的loss权重
            box=box_weight,
            cls=cls_weight,
            dfl=dfl_weight,
            
            optimizer='SGD',
            patience=epochs,  # 不早停
            save_period=epochs,  # 只在最后保存
            cache=False,  # 不缓存加快启动
            amp=True,
            project='runs/test',
            name=name,
            exist_ok=True,
            verbose=False  # 减少输出
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 获取最终的loss值
        if hasattr(results, 'results_dict'):
            final_loss = results.results_dict.get('train/loss', 'N/A')
        else:
            final_loss = 'N/A'
        
        print(f"✅ {strategy_name} 完成")
        print(f"   耗时: {duration:.1f}秒")
        print(f"   最终loss: {final_loss}")
        
        return True, final_loss, duration
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ {strategy_name} 失败: {str(e)}")
        print(f"   耗时: {duration:.1f}秒")
        return False, None, duration

def main():
    parser = argparse.ArgumentParser(description='快速测试loss缩放策略')
    parser.add_argument('--device', type=str, default='6', help='设备ID')
    parser.add_argument('--model', type=str, default='yolo11n', help='测试用模型 (建议用小模型)')
    parser.add_argument('--epochs', type=int, default=10, help='每个策略的测试轮数')
    
    args = parser.parse_args()
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"使用设备: CUDA:{args.device}")
    else:
        print("使用设备: CPU")
        args.device = 'cpu'
    
    # 定义测试策略
    test_strategies = [
        # (策略名称, box权重, cls权重, dfl权重)
        ('default', 7.5, 0.5, 1.5),
        ('scale_10x', 75, 5, 15),
        ('scale_100x', 750, 50, 150),
        ('scale_1000x', 7500, 500, 1500),
        ('box_focus', 750, 50, 75),
        ('cls_focus', 375, 500, 75),
        ('balanced_high', 375, 25, 75),
    ]
    
    print(f"开始快速测试 {len(test_strategies)} 个loss缩放策略")
    print(f"模型: {args.model}")
    print(f"每个策略训练: {args.epochs} epochs")
    print(f"图像尺寸: 640 (加快测试)")
    
    # 记录结果
    results = []
    total_start_time = time.time()
    
    for strategy_name, box_weight, cls_weight, dfl_weight in test_strategies:
        success, final_loss, duration = quick_test_strategy(
            args.model, strategy_name, box_weight, cls_weight, dfl_weight, 
            args.device, args.epochs
        )
        
        results.append({
            'strategy': strategy_name,
            'success': success,
            'final_loss': final_loss,
            'duration': duration,
            'box_weight': box_weight,
            'cls_weight': cls_weight,
            'dfl_weight': dfl_weight
        })
    
    total_duration = time.time() - total_start_time
    
    # 打印总结
    print(f"\n{'='*80}")
    print("测试结果总结")
    print(f"{'='*80}")
    print(f"总耗时: {total_duration:.1f}秒")
    print()
    
    print(f"{'策略':<15} {'状态':<6} {'最终Loss':<12} {'耗时(秒)':<8} {'权重(box/cls/dfl)'}")
    print("-" * 80)
    
    successful_strategies = []
    
    for result in results:
        status = "✅成功" if result['success'] else "❌失败"
        loss_str = f"{result['final_loss']}" if result['final_loss'] != 'N/A' else 'N/A'
        weights_str = f"{result['box_weight']}/{result['cls_weight']}/{result['dfl_weight']}"
        
        print(f"{result['strategy']:<15} {status:<6} {loss_str:<12} {result['duration']:<8.1f} {weights_str}")
        
        if result['success']:
            successful_strategies.append(result['strategy'])
    
    print(f"\n成功的策略 ({len(successful_strategies)}/{len(test_strategies)}): {', '.join(successful_strategies)}")
    
    # 推荐策略
    if successful_strategies:
        print(f"\n推荐用于正式训练的策略:")
        for strategy in successful_strategies[:3]:  # 推荐前3个成功的策略
            print(f"  python train_fix_loss.py --strategy {strategy} --model yolo11m --device {args.device}")
    else:
        print(f"\n⚠️  所有策略都失败了，建议检查数据集或降低缩放倍数")
    
    print(f"\n完整训练命令示例:")
    print(f"python train_fix_loss.py --strategy scale_100x --model yolo11m --device {args.device} --epochs 300")

if __name__ == '__main__':
    main()
