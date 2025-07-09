# 弱小目标检测跟踪基准测试系统

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org)
[![YOLO](https://img.shields.io/badge/YOLO-v11-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目简介

基于 YOLO11 的弱小目标检测与多目标跟踪系统，专门针对卫星视频中的车辆检测跟踪任务进行优化。支持完整的数据处理、模型训练、验证和性能基准测试流程。


## 项目结构

```
Tracking_benchmark/
├── train.py                      # 模型训练（小目标优化）
├── val.py                        # 模型验证
├── benchmark.py                  # 性能基准测试
├── uav_tracking_single_thread.py # 单线程跟踪
├── uav_tracking_multi_thread.py  # 多线程跟踪
├── yolo11.yaml                   # YOLO11模型配置
├── data/                         # 原始数据
│   ├── train/                    # 训练视频和标注
│   │   ├── *.avi                 # 视频文件
│   │   └── *-gt.csv             # CSV标注文件
│   └── val/                      # 测试视频
│       └── *.avi                 # 测试视频文件
├── dataset/                      # 处理后的数据集
│   ├── VOCdevkit/               # 训练数据存储目录（YOLO格式）
│   │   ├── JPEGImages/          # 所有训练图像（未拆分）
│   │   └── txt/                 # YOLO格式标注文件
│   ├── images/                  # 标准YOLO目录结构
│   │   ├── train/               # 训练图像（手动拆分后）
│   │   ├── val/                 # 验证图像（手动拆分后）
│   │   └── test/                # 测试图像（无标注）
│   ├── labels/                  # 标准YOLO标注目录
│   │   ├── train/               # 训练标注（手动拆分后）
│   │   └── val/                 # 验证标注（手动拆分后）
│   └── data.yaml                # 数据集配置
├── cfg/                          # 跟踪算法配置
│   ├── bytetrack.yaml
│   ├── bytetrack_improved.yaml  # 优化的ByteTrack配置
│   └── botsort.yaml
├── tracker/                      # 跟踪算法实现
│   ├── bytetrack_tracker.py
│   ├── botsort_tracker.py
│   ├── base_tracker.py
│   └── utils/
└── Scripts/                      # 数据处理和分析工具
    ├── video_to_yolo_dataset.py  # 视频转数据集
    ├── check_video_resolution.py # 检查视频分辨率
    ├── convert_to_competition_format.py # 转换为竞赛格式
    ├── detect.py                # 单图检测
    ├── get_FPS.py               # 性能测试
    ├── heatmap.py               # 热力图生成
    ├── main_profile.py          # 模型性能分析
    └── val.py                   # 验证脚本
```

## 环境要求

- Python 3.9+
- PyTorch 2.2.2 (CUDA 12.1)
- Ultralytics 8.3.0+
- OpenCV 4.10.0+

## 快速开始

### 1. 环境安装
```bash
git clone https://github.com/yuchenwu73/Tracking_benchmark.git
cd Tracking_benchmark
conda create -n tracking python=3.9
conda activate tracking
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. 数据准备
将原始数据放入 `data/` 目录：
```
data/
├── train/          # 训练数据
│   ├── 1-2.avi
│   ├── 1-2-gt.csv
│   ├── 2-7.avi
│   ├── 2-7-gt.csv
│   └── ...
└── val/            # 测试数据
    ├── 5-1.avi
    ├── 14-2.avi
    └── ...
```

### 3. 数据转换
```bash
# 将视频数据转换为YOLO训练格式
python Scripts/video_to_yolo_dataset.py

# 82划分数据集
python dataset/split_data.py
```

### 4. 模型训练

```bash
# 1. 配置好数据集路径 dataset/data.yaml
path: /to/your/path
train: images/train  # 训练图像目录
val: images/val    # 验证图像目录
test: images/test            # 测试集（无标注）

# 类别定义
nc: 1
names:
  0: vehicle  # 车辆类别


# 2. 训练小目标检测模型
python train.py

# 3. 验证模型性能
python val.py
```

### 5. 目标跟踪
```bash
# 单线程跟踪
python uav_tracking_single_thread.py

# 多线程跟踪（推荐）
python uav_tracking_multi_thread.py
```

### 6. 性能评估
```bash
# 运行完整基准测试
python benchmark.py
```


## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件


