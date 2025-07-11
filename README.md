# 卫星视频小目标检测跟踪系统

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org)
[![YOLO](https://img.shields.io/badge/YOLO-v11-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 项目简介

基于 YOLO11 的卫星视频小目标检测与多目标跟踪系统，专门针对卫星视频中的车辆检测跟踪任务进行优化。使用官方 Ultralytics 跟踪器（BoT-SORT/ByteTrack），支持ReID功能，输出符合比赛要求的10字段MOT格式结果。

**评估指标**: score = (MOTA + IDF1) / 2，权重各50%

## 🚀 快速开始

### 一键运行（推荐）
```bash
# 1. 批量处理所有验证视频（推荐）
python simple_batch_tracking.py --input_dir data/val --output_dir results

# 2. 实时跟踪单个视频（支持GUI）
python uav_tracking_multi_thread.py data/val/16-4.avi

# 3. 跟踪器性能比较
python tracker_comparison.py --input_dir data/val --tracker all

# 4. 结果格式验证
python test_competition_format.py --results_dir results

# 5. 质量评估
python self_evaluation.py --results_dir results
```

### 🎯 核心特性
- ✅ **官方跟踪器**: 使用 Ultralytics BoT-SORT（默认，启用ReID）和 ByteTrack
- ✅ **小目标优化**: 专门针对卫星视频中的小目标车辆检测优化
- ✅ **实时处理**: 支持 8K 视频实时跟踪（~25 FPS）
- ✅ **GUI 界面**: 可视化跟踪结果，支持跟踪器切换（按 T 键）
- ✅ **批量处理**: 自动处理多个视频文件
- ✅ **比赛格式**: 输出符合比赛要求的10字段MOT格式结果
- ✅ **质量评估**: 基于MOTA+IDF1的质量评估系统


## 📁 项目结构

```
Tracking_benchmark/
├── 🎯 核心跟踪脚本
│   ├── simple_batch_tracking.py     # 批量跟踪（推荐）
│   ├── uav_tracking_multi_thread.py # GUI可视化跟踪
│   └── tracker_comparison.py        # 跟踪器性能比较
├── 🔬 评估工具
│   ├── self_evaluation.py           # 质量评估（MOTA+IDF1）
│   ├── test_competition_format.py   # 格式验证
│   ├── benchmark.py                 # 性能基准测试
│   └── tracker_score_comparison.py  # 跟踪器得分比较
├── 🤖 模型训练验证
│   ├── train.py                     # 模型训练
│   ├── val.py                       # 模型验证
│   └── yolo11.yaml                  # YOLO11模型配置
├── 📊 数据和配置
│   ├── data/                        # 原始数据
│   │   ├── train/                   # 训练视频和标注
│   │   └── val/                     # 验证视频（10个.avi文件）
│   ├── dataset/                     # 处理后的数据集
│   │   ├── VOCdevkit/              # 训练数据（YOLO格式）
│   │   ├── images/                 # 图像目录
│   │   ├── labels/                 # 标注目录
│   │   └── data.yaml               # 数据集配置
│   └── cfg/                         # 跟踪器配置
│       ├── botsort.yaml            # BoT-SORT配置（启用ReID）
│       └── bytetrack.yaml          # ByteTrack配置
├── 🔧 工具脚本
│   └── Scripts/                     # 数据处理工具
│       ├── check_video_resolution.py
│       ├── detect.py
│       ├── get_FPS.py
│       ├── heatmap.py
│       └── main_profile.py
├── 📈 输出结果
│   ├── results/                     # 跟踪结果文件（10个.txt）
│   ├── tracker_comparison/          # 跟踪器比较结果
│   ├── runs/train/                  # 训练输出
│   └── logs/                        # 日志文件
└── 🏗️ 系统文件
    ├── trackers/                    # 官方跟踪器实现
    ├── requirements.txt             # 依赖包
    └── README.md                    # 项目说明
```

## 🛠️ 环境要求

- Python 3.9+
- PyTorch 2.2.2 (CUDA 12.1)
- Ultralytics 8.3.0+
- OpenCV 4.10.0+

## ⚙️ 安装配置

### 1. 环境安装
```bash
git clone https://github.com/yuchenwu73/Tracking_benchmark.git
cd Tracking_benchmark
conda create -n track python=3.9
conda activate track
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. 数据准备
项目已包含验证数据，位于 `data/val/` 目录：
```
data/val/
├── 14-2.avi    # 卫星视频1
├── 15-1.avi    # 卫星视频2
├── 16-1.avi    # 卫星视频3
├── 16-4.avi    # 卫星视频4
├── 22-1.avi    # 卫星视频5
├── 22-2.avi    # 卫星视频6
├── 22-3.avi    # 卫星视频7
├── 25-1.avi    # 卫星视频8
├── 25-2.avi    # 卫星视频9
└── 5-1.avi     # 卫星视频10
```

### 3. 模型训练（可选）
```bash
python train.py

# 验证模型性能
python val.py
```

## 🎯 使用指南

### 批量跟踪（推荐）
```bash
# 批量处理所有验证视频
python simple_batch_tracking.py --input_dir data/val --output_dir results
```

### 实时跟踪（GUI）
```bash
# 多线程跟踪，支持GUI和跟踪器切换
python uav_tracking_multi_thread.py data/val/16-4.avi

# GUI 控制键：
# T: 切换跟踪器 (BoT-SORT ↔ ByteTrack)
# Q: 退出程序
# F: 全屏切换
# Space: 暂停/继续
# R: 重置窗口
```

### 跟踪器比较
```bash
# 比较所有跟踪器性能
python tracker_comparison.py --input_dir data/val --tracker all

# 测试单个跟踪器
python tracker_comparison.py --input_dir data/val --tracker botsort
python tracker_comparison.py --input_dir data/val --tracker bytetrack
```

### 质量评估
```bash
# 格式验证
python test_competition_format.py --results_dir results

# 质量评估（MOTA+IDF1）
python self_evaluation.py --results_dir results

python benchmark.py
```


## 📊 输出格式

### 比赛标准格式
跟踪结果采用比赛要求的标准格式（10个字段）：
```
帧号,目标ID,边界框左上角X,边界框左上角Y,边界框宽度,边界框高度,目标类别,-1,-1,-1
```

**字段说明：**
- 帧号：从0开始的帧序号
- 目标ID：跟踪目标的唯一标识符
- 目标类别：固定为1（代表车辆）
- 最后三个字段：固定为-1

**示例：**
```
0,1,712.96,195.25,14.36,13.99,1,-1,-1,-1
0,2,997.47,437.26,7.26,6.31,1,-1,-1,-1
1,1,714.12,196.08,14.28,13.87,1,-1,-1,-1
```

### 文件结构
```
results/
├── 16-4.txt          # 视频跟踪结果
├── 14-2.txt
├── ...
└── results.zip       # 压缩包（用于提交）
```

## 🔧 跟踪器配置

项目使用官方 Ultralytics 跟踪器：

### BoT-SORT (默认，推荐)
- **配置文件**: `cfg/botsort.yaml`
- **特点**: 高精度，支持 ReID 和全局运动补偿
- **ReID**: 已启用 (`with_reid: True`)
- **适用**: 精度要求高的场景

### ByteTrack
- **配置文件**: `cfg/bytetrack.yaml`
- **特点**: 高效率，速度快
- **适用**: 实时性要求高的场景

### ReID 参数说明
- `proximity_thresh: 0.5` - ReID匹配的最小IoU阈值
- `appearance_thresh: 0.8` - 外观相似度阈值
- `with_reid: True` - 启用重识别功能



## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件


