#!/usr/bin/env python3
"""
一键修复loss过小问题的简单脚本

直接运行即可，会自动使用合适的loss缩放参数
"""

import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
import datetime

def main():
    print("🚀 YOLO训练 - 使用默认权重")
    print("=" * 50)

    # 使用默认权重参数
    model_name = 'yolo11m'
    device = '6'

    # 默认权重 (不缩放)
    box_weight = 7.5    # 默认值
    cls_weight = 0.5    # 默认值
    dfl_weight = 1.5    # 默认值

    print(f"模型: {model_name}")
    print(f"设备: CUDA:{device}")
    print(f"Loss权重: 使用默认值")
    print(f"  - Box Loss: {box_weight} (默认)")
    print(f"  - Cls Loss: {cls_weight} (默认)")
    print(f"  - DFL Loss: {dfl_weight} (默认)")
    print("=" * 50)
    
    # 生成实验名称
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    name = f"{timestamp}_default_weights_{model_name}"

    # 加载模型
    print("加载模型...")
    model = YOLO(f'{model_name}.pt')

    # 开始训练
    print("开始训练...")
    print("💡 如果需要调整loss权重，可以尝试:")
    print("   python train.py --model yolo11m --device 6 --box 75 --cls 5 --dfl 15")
    print()
    
    results = model.train(
        data='dataset/data.yaml',
        epochs=300,
        imgsz=1280,
        batch=4,
        device=device,
        workers=8,
        lr0=0.001,
        
        # 使用默认的loss权重
        box=box_weight,
        cls=cls_weight,
        dfl=dfl_weight,
        
        # 其他参数
        optimizer='SGD',
        patience=50,
        save_period=10,
        cache=True,
        close_mosaic=10,
        amp=True,
        project='runs/train',
        name=name,
        exist_ok=True,
        verbose=True
    )
    
    print("✅ 训练完成!")
    print(f"结果保存在: runs/train/{name}")
    
    return results

if __name__ == '__main__':
    main()
