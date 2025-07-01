# 🚁 无人机多目标跟踪系统

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org)
[![YOLO](https://img.shields.io/badge/YOLO-v11-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 项目简介

本项目是一个专门针对**8K高分辨率视频**中无人机目标的**多目标检测与跟踪系统**。系统基于**YOLO11**目标检测模型和先进的多目标跟踪算法（**ByteTrack**），能够在高分辨率视频中准确检测和跟踪小型无人机目标。

## ✨ 主要特性

- 🎯 **高精度检测**: 基于YOLO11模型的无人机目标检测
- 🔄 **智能跟踪**: 改进的ByteTrack算法，减少ID跳跃，支持多目标同时跟踪
- 📹 **8K视频支持**: 专门优化处理8K高分辨率视频，采用子图切分策略
- ⚡ **高效处理**: 提供单线程和多线程两个版本，适应不同使用场景
- 📊 **实时监控**: 显示帧数、进度、速度、目标数量等性能信息
- 🎨 **可视化跟踪**: 绿色轨迹线显示运动路径，黄色ID显示在轨迹起始点
- 🎮 **交互控制**: Q退出、R重置、F全屏、Space暂停等快捷键操作

## 📁 项目结构

```
Tracking_benchmark/
├── 📄 README.md                      # 项目说明文档
├── 📋 requirements.txt               # Python依赖列表
├── 🧠 last.pt                       # 训练好的YOLO11模型
├── 🧠 last.onnx                     # ONNX格式模型
├── 🚀 uav_tracking_single_thread.py # 单线程跟踪脚本（推荐调试）
├── 🚀 uav_tracking_multi_thread.py  # 多线程跟踪脚本（推荐生产）
├── 📊 benchmark.py                  # 性能基准测试
├── 🏋️ train.py                      # 模型训练脚本
├── 📁 data/                         # 数据和配置目录
│   ├── 🔧 bytetrack.yaml           # ByteTrack配置
│   ├── 🔧 bytetrack_improved.yaml  # 改进的ByteTrack配置
│   ├── 🔧 botsort.yaml             # BoT-SORT配置
│   ├── 🔧 uav.yaml                 # 数据集配置
│   ├── 📝 classes.txt              # 类别标签
│   ├── 🖼️ images/                   # 图像数据
│   └── 🏷️ labels/                   # 标注数据
├── 📁 tracker/                     # 跟踪算法模块
│   ├── 🎯 bytetrack_tracker.py     # ByteTrack算法实现
│   ├── 🎯 botsort_tracker.py       # BoT-SORT算法实现
│   ├── 🎯 base_tracker.py          # 基础跟踪器
│   └── 🛠️ utils/                    # 工具函数
├── 📁 Scripts/                     # 辅助脚本
│   ├── 🔍 detect.py                # 单张图片检测
│   ├── ⚡ get_FPS.py               # 性能测试
│   └── 🌡️ heatmap.py               # 热力图生成
├── 📁 UAV/                         # 测试视频目录
│   ├── 🎥 无人机/                   # 无人机视频文件
│   └── 📊 vis/                     # 可视化结果
├── 📁 tracking_output_single_thread/ # 单线程输出结果
└── 📁 tracking_output_multi_thread/  # 多线程输出结果
```

## 💻 环境要求

### 🖥️ 系统要求
- **Python**: 3.9+
- **CUDA**: 12.1+ (推荐GPU加速)
- **GPU显存**: 8GB+ (处理8K视频)
- **操作系统**: Windows/Linux/macOS
- **显示**: 支持GUI显示（MobaXterm等）

### 📦 核心依赖
- **PyTorch**: 2.2.2 (CUDA 12.1)
- **OpenCV**: 4.10.0+ (视频处理)
- **Ultralytics**: 8.3.0+ (YOLO11)
- **NumPy**: 1.24.0+ (数值计算)

## 🚀 安装指南

### 1️⃣ 克隆项目
```bash
git clone https://github.com/yuchenwu73/Tracking_benchmark.git
cd Tracking_benchmark
```

### 2️⃣ 安装PyTorch (CUDA版本)
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

### 3️⃣ 安装其他依赖
```bash
pip install -r requirements.txt
```

### 4️⃣ 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 5️⃣ 准备模型文件
确保项目根目录下有训练好的模型文件：
- `last.pt` - PyTorch模型（没有去训练，best.pt也可以）
- `last.onnx` - ONNX模型（可选）

## 🎬 快速开始

### 1️⃣ 准备视频文件
将你的无人机视频文件放在合适的位置，支持格式：
- `.MOV`, `.mp4`, `.avi`, `.mkv` 等

### 2️⃣ 修改视频路径
在运行前，需要修改脚本中的视频路径：
```python
# 在 uav_tracking_single_thread.py 或 uav_tracking_multi_thread.py 中修改
video_path = "UAV/无人机/你的视频文件.MOV"
```

### 3️⃣ 选择运行模式
```bash
# 🔧 单线程版本（推荐调试使用）
python uav_tracking_single_thread.py

# 🚀 多线程版本（推荐生产使用）
python uav_tracking_multi_thread.py

# 📊 性能基准测试
python benchmark.py
```

### 4️⃣ 交互控制
运行后可使用以下快捷键：
- **Q**: 退出程序
- **R**: 重置跟踪
- **F**: 切换全屏
- **Space**: 暂停/继续
- **鼠标点击**: 显示坐标

## 📋 详细使用说明

### 🔧 单线程版本 (推荐调试)
```bash
python uav_tracking_single_thread.py
```

**特点：**
- ✅ 支持8K视频处理
- ✅ 自动子图切分 (960x1080)
- ✅ 实时性能监控
- ✅ 轨迹可视化
- ✅ 输出目录：`tracking_output_single_thread/`

**适用场景：**
- 🔍 调试开发
- 📁 小文件处理
- 🎯 精确控制

### 🚀 多线程版本 (推荐生产)
```bash
python uav_tracking_multi_thread.py
```

**特点：**
- ⚡ 异步帧读取，减少I/O等待
- 📦 队列缓冲机制，提升处理效率
- 🎥 适合处理大型8K视频文件
- 📈 更高的处理吞吐量
- 📁 输出目录：`tracking_output_multi_thread/`

**适用场景：**
- 🏭 生产环境
- 📹 大文件处理
- ⚡ 追求性能

### 📊 性能基准测试
```bash
python benchmark.py
```

**功能：**
- 📈 多种跟踪算法性能对比
- 📋 详细的性能统计报告
- ⏱️ FPS和处理时间分析
- 💾 结果保存到日志文件

## 🎯 跟踪结果说明

### 📺 界面显示元素
1. **🟡 黄色ID数字**: 目标跟踪ID，显示在轨迹起始点
2. **🟢 绿色轨迹线**: 显示每个目标的历史运动路径
3. **📊 实时信息**: 帧数、进度、速度、目标数量
4. **🎮 操作提示**: 快捷键说明

### 📈 性能信息
- **Frame**: 当前帧/总帧数
- **Progress**: 处理进度百分比
- **Speed**: 实时处理速度 (FPS)
- **Objects**: 当前检测到的目标数量

## 🔧 配置说明

### 📁 输出目录
- `tracking_output_single_thread/` - 单线程版本输出
- `tracking_output_multi_thread/` - 多线程版本输出
- 每30帧保存一张跟踪结果图片
- 最后一帧保存为 `final_frame.jpg`

### ⚙️ 跟踪参数
可在 `data/bytetrack_improved.yaml` 中调整：
```yaml
track_thresh: 0.6      # 跟踪阈值
track_buffer: 30       # 跟踪缓冲帧数
match_thresh: 0.8      # 匹配阈值
frame_rate: 30         # 视频帧率
```

## 🐛 常见问题 & 性能说明

### Q: 为什么ID会跳跃？
A: 当目标过小或过大导致检测失败时，会出现轨迹断裂，同一目标可能获得多个ID。

### Q: 如何提高跟踪性能？
A:
1. 使用多线程版本 (~2.5-3.0 FPS vs 单线程 ~1.5-2.0 FPS)
2. 调整 `track_thresh` 参数 (推荐0.6)
3. 确保GPU内存充足 (推荐8GB+)
4. 支持视频格式：MP4、MOV、AVI、MKV等OpenCV兼容格式



## 🙏 致谢

感谢以下开源项目的贡献：
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO11目标检测
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - 多目标跟踪算法
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - 改进的跟踪算法

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📧 联系方式

如有问题或建议，请提交 [Issue](https://github.com/your-username/Tracking_benchmark/issues)

---

⭐ 如果这个项目对你有帮助，请给个星标支持！


