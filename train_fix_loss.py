#!/usr/bin/env python3
"""
专门用于修复loss过小问题的YOLO训练脚本
支持多种loss缩放策略

使用示例:
# 策略1: 100倍缩放
python train_fix_loss.py --strategy scale_100x --model yolo11m --device 6

# 策略2: 1000倍缩放  
python train_fix_loss.py --strategy scale_1000x --model yolo11m --device 6

# 策略3: 自定义缩放
python train_fix_loss.py --strategy custom --box 750 --cls 50 --dfl 150 --model yolo11m --device 6

# 策略4: 渐进式缩放
python train_fix_loss.py --strategy progressive --model yolo11m --device 6
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import datetime
from ultralytics import YOLO
import torch

def get_loss_weights(strategy, custom_box=None, custom_cls=None, custom_dfl=None):
    """根据策略获取loss权重"""
    
    base_box, base_cls, base_dfl = 7.5, 0.5, 1.5
    
    if strategy == 'default':
        return base_box, base_cls, base_dfl
    
    elif strategy == 'scale_10x':
        return base_box * 10, base_cls * 10, base_dfl * 10
    
    elif strategy == 'scale_100x':
        return base_box * 100, base_cls * 100, base_dfl * 100
    
    elif strategy == 'scale_1000x':
        return base_box * 1000, base_cls * 1000, base_dfl * 1000
    
    elif strategy == 'scale_10000x':
        return base_box * 10000, base_cls * 10000, base_dfl * 10000
    
    elif strategy == 'box_focus':
        # 重点关注边界框回归
        return base_box * 100, base_cls * 10, base_dfl * 50
    
    elif strategy == 'cls_focus':
        # 重点关注分类
        return base_box * 50, base_cls * 100, base_dfl * 50
    
    elif strategy == 'balanced_high':
        # 平衡但都较高
        return base_box * 50, base_cls * 50, base_dfl * 50
    
    elif strategy == 'custom':
        # 自定义权重
        box_weight = custom_box if custom_box is not None else base_box
        cls_weight = custom_cls if custom_cls is not None else base_cls
        dfl_weight = custom_dfl if custom_dfl is not None else base_dfl
        return box_weight, cls_weight, dfl_weight
    
    else:
        raise ValueError(f"未知策略: {strategy}")

def train_with_strategy(args):
    """使用指定策略训练"""
    
    # 获取loss权重
    if args.strategy == 'progressive':
        return train_progressive(args)
    else:
        box_weight, cls_weight, dfl_weight = get_loss_weights(
            args.strategy, args.box, args.cls, args.dfl
        )
        return train_single_strategy(args, box_weight, cls_weight, dfl_weight)

def train_single_strategy(args, box_weight, cls_weight, dfl_weight):
    """单一策略训练"""
    
    # 生成实验名称
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    name = f"{timestamp}_{args.model}_{args.strategy}_box{box_weight}_cls{cls_weight}_dfl{dfl_weight}"
    
    # 打印配置
    print("=" * 80)
    print(f"训练策略: {args.strategy}")
    print(f"模型: {args.model}")
    print(f"设备: {args.device}")
    print(f"图像尺寸: {args.imgsz}")
    print(f"批次大小: {args.batch}")
    print(f"训练轮数: {args.epochs}")
    print("-" * 80)
    print("Loss权重配置:")
    print(f"Box Loss: {box_weight} (缩放倍数: {box_weight/7.5:.1f}x)")
    print(f"Cls Loss: {cls_weight} (缩放倍数: {cls_weight/0.5:.1f}x)")
    print(f"DFL Loss: {dfl_weight} (缩放倍数: {dfl_weight/1.5:.1f}x)")
    print("=" * 80)
    
    # 加载模型
    model = YOLO(f'{args.model}.pt')
    
    # 训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        
        # 关键: 手动设置的loss权重
        box=box_weight,
        cls=cls_weight,
        dfl=dfl_weight,
        
        # 其他参数
        optimizer='SGD',
        patience=args.patience,
        save_period=10,
        cache=True,
        close_mosaic=10,
        amp=True,
        project='runs/train',
        name=name,
        exist_ok=True,
        verbose=True
    )
    
    return results

def train_progressive(args):
    """渐进式训练: 从小缩放到大缩放"""
    
    strategies = [
        ('scale_10x', 50),      # 前50个epoch用10倍缩放
        ('scale_100x', 100),    # 接下来100个epoch用100倍缩放  
        ('scale_1000x', 150),   # 最后150个epoch用1000倍缩放
    ]
    
    model = YOLO(f'{args.model}.pt')
    
    for i, (strategy, epochs) in enumerate(strategies):
        box_weight, cls_weight, dfl_weight = get_loss_weights(strategy)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        name = f"{timestamp}_{args.model}_progressive_stage{i+1}_{strategy}"
        
        print(f"\n{'='*60}")
        print(f"渐进式训练 - 阶段 {i+1}/3")
        print(f"策略: {strategy}")
        print(f"轮数: {epochs}")
        print(f"Loss权重: box={box_weight}, cls={cls_weight}, dfl={dfl_weight}")
        print(f"{'='*60}")
        
        # 训练当前阶段
        results = model.train(
            data=args.data,
            epochs=epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            lr0=args.lr0,
            
            box=box_weight,
            cls=cls_weight,
            dfl=dfl_weight,
            
            optimizer='SGD',
            patience=args.patience,
            save_period=10,
            cache=True,
            close_mosaic=10,
            amp=True,
            project='runs/train',
            name=name,
            exist_ok=True,
            verbose=True,
            resume=i > 0  # 从第二阶段开始恢复训练
        )
    
    return results

def main():
    parser = argparse.ArgumentParser(description='修复YOLO训练loss过小问题')
    
    # 策略选择
    parser.add_argument('--strategy', type=str, default='scale_100x',
                       choices=['default', 'scale_10x', 'scale_100x', 'scale_1000x', 'scale_10000x',
                               'box_focus', 'cls_focus', 'balanced_high', 'custom', 'progressive'],
                       help='Loss缩放策略')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='yolo11m',
                       choices=['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'],
                       help='模型大小')
    parser.add_argument('--data', type=str, default='dataset/data.yaml', help='数据集配置')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=1280, help='图像尺寸')
    parser.add_argument('--batch', type=int, default=4, help='批次大小')
    parser.add_argument('--device', type=str, default='6', help='设备ID')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--lr0', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--patience', type=int, default=50, help='早停耐心值')
    
    # 自定义loss权重 (仅当strategy='custom'时使用)
    parser.add_argument('--box', type=float, help='自定义box loss权重')
    parser.add_argument('--cls', type=float, help='自定义cls loss权重')
    parser.add_argument('--dfl', type=float, help='自定义dfl loss权重')
    
    args = parser.parse_args()
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"CUDA可用, 使用设备: {args.device}")
    else:
        print("CUDA不可用, 使用CPU")
        args.device = 'cpu'
    
    # 开始训练
    results = train_with_strategy(args)
    
    print("\n训练完成!")
    return results

if __name__ == '__main__':
    main()
