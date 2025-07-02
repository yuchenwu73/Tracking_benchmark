# 无人机目标跟踪基准测试系统

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org)
[![YOLO](https://img.shields.io/badge/YOLO-v11-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目简介

基于 YOLO11 的无人机目标检测与多目标跟踪系统，支持模型训练、验证和性能基准测试。

## 主要功能

- **目标检测**: YOLO11 无人机检测模型
- **多目标跟踪**: ByteTrack、BoT-SORT 等跟踪算法
- **模型训练**: 支持自定义数据集训练
- **性能验证**: 模型精度和速度评估
- **基准测试**: 多算法性能对比

## 项目结构

```
Tracking_benchmark/
├── train.py                      # 模型训练
├── val.py                        # 模型验证
├── benchmark.py                  # 性能基准测试
├── uav_tracking_single_thread.py # 单线程跟踪
├── uav_tracking_multi_thread.py  # 多线程跟踪
├── yolo11.yaml                   # YOLO11模型配置
├── dataset/                      # 训练数据集
│   ├── data.yaml                 # 数据集配置
│   ├── images/                   # 训练图片
│   └── labels/                   # 标注文件
├── cfg/                          # 跟踪算法配置
│   ├── bytetrack.yaml
│   ├── bytetrack_improved.yaml
│   └── botsort.yaml
├── tracker/                      # 跟踪算法实现
│   ├── bytetrack_tracker.py
│   ├── botsort_tracker.py
│   ├── base_tracker.py
│   └── utils/
└── Scripts/                      # 辅助工具
    ├── detect.py                 # 单图检测
    ├── get_FPS.py               # 性能测试
    └── val.py                   # 验证脚本
```

## 环境要求

- Python 3.9+
- PyTorch 2.2.2 (CUDA 12.1)
- Ultralytics 8.3.0+
- OpenCV 4.10.0+

## 快速开始

### 安装依赖
```bash
git clone https://github.com/yuchenwu73/Tracking_benchmark.git
cd Tracking_benchmark
conda create -n tracking python=3.9
conda activate tracking
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 模型训练
```bash
# 训练 YOLO11 模型
python train.py

# 验证模型性能
python val.py
```

### 目标跟踪
```bash
# 单线程跟踪（调试用）
python uav_tracking_single_thread.py

# 多线程跟踪（生产用）
python uav_tracking_multi_thread.py
```

### 性能测试
```bash
# 运行基准测试
python benchmark.py
```

## 支持的跟踪算法

- **ByteTrack**: 高性能多目标跟踪
- **BoT-SORT**: 改进的跟踪算法

## 配置文件

### 数据集配置 (dataset/data.yaml)
```yaml
path: dataset
train: images
val: images
names:
  0: drone
```

### 跟踪算法配置 (cfg/)
- `bytetrack.yaml`: ByteTrack 参数
- `bytetrack_improved.yaml`: 改进版 ByteTrack
- `botsort.yaml`: BoT-SORT 参数

## 性能基准测试

`benchmark.py` 提供完整的性能评估：

- **模型验证**: 精度指标 (mAP, Precision, Recall)
- **速度测试**: FPS 和推理时间
- **格式对比**: PyTorch vs ONNX vs TorchScript
- **跟踪算法对比**: 多种算法性能比较

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件


