<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# **使用 Ultralytics YOLO 进行多目标跟踪**

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418637-1d6250fd-1515-4c10-a844-a32818ae6d46.png" alt="Ultralytics YOLO trackers visualization">

[目标跟踪](https://www.ultralytics.com/glossary/object-tracking)是[视频分析](https://en.wikipedia.org/wiki/Video_content_analysis)的一个关键方面，涉及识别视频帧内对象的位置和类别，并在其移动时为每个检测到的对象分配一个唯一的ID。此功能支持广泛的应用，从监控和安全系统到[实时](https://www.ultralytics.com/glossary/real-time-inference)体育分析和自动驾驶汽车导航。在我们的[跟踪文档页面](https://docs.ultralytics.com/modes/track/)上了解有关跟踪的更多信息。

-----

## 🎯 **为什么选择 Ultralytics YOLO 进行目标跟踪？**

Ultralytics YOLO 跟踪器提供的输出与标准[目标检测](https://docs.ultralytics.com/tasks/detect/)一致，但增加了持久的对象ID。这简化了在视频流中跟踪对象和执行后续分析的过程。以下是为什么 Ultralytics YOLO 是您目标跟踪需求的绝佳选择：

  * **效率高：** 在不牺牲准确性的情况下实时处理视频流。
  * **灵活性强：** 支持多种强大的跟踪算法和配置。
  * **易于使用：** 提供简单的 [Python API](https://docs.ultralytics.com/usage/python/) 和 [CLI](https://docs.ultralytics.com/usage/cli/) 选项，可快速集成和部署。
  * **可定制性：** 可轻松与[自定义训练的 YOLO 模型](https://docs.ultralytics.com/modes/train/)集成，从而能够在专业的、特定领域的应用中进行部署。

**观看：** 使用 Ultralytics YOLOv8 进行目标检测和跟踪。

[](https://www.google.com/search?q=%5Bhttps://www.youtube.com/watch%3Fv%3DhHyHmOtmEgs%5D\(https://www.youtube.com/watch%3Fv%3DhHyHmOtmEgs\))

-----

## ✨ **功能一览**

Ultralytics YOLO 扩展了其强大的目标检测功能，以提供强大而通用的目标跟踪：

  * **实时跟踪：** 在高帧率视频中无缝跟踪对象。
  * **多跟踪器支持：** 从一系列已建立的跟踪算法中进行选择。
  * **可定制的跟踪器配置：** 通过调整各种参数，使跟踪算法适应特定要求。

-----

## 🛠️ **可用跟踪器**

Ultralytics YOLO 支持以下跟踪算法。通过传递相关的 YAML 配置文件来启用它们，例如 `tracker=tracker_type.yaml`：

  * **BoT-SORT：** 使用 [`botsort.yaml`](https://www.google.com/search?q=%5Bhttps://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/botsort.yaml%5D\(https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/botsort.yaml\)) 来启用此跟踪器。基于 [BoT-SORT 论文](https://arxiv.org/abs/2206.14651) 及其官方[代码实现](https://github.com/NirAharon/BoT-SORT)。
  * **ByteTrack：** 使用 [`bytetrack.yaml`](https://www.google.com/search?q=%5Bhttps://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml%5D\(https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml\)) 来启用此跟踪器。基于 [ByteTrack 论文](https://arxiv.org/abs/2110.06864) 及其官方[代码实现](https://github.com/FoundationVision/ByteTrack)。

默认跟踪器是 **BoT-SORT**。

-----

## ⚙️ **用法**

要对视频流运行跟踪器，请使用经过训练的检测、分割或姿态模型，例如 [Ultralytics YOLOv8n](https://www.google.com/search?q=https://docs.ultralytics.com/models/yolo8/)、YOLOv8n-seg 或 YOLOv8n-pose。

```python
# Python
from ultralytics import YOLO

# 加载官方或自定义模型
model = YOLO("yolov8n.pt")  # 加载官方检测模型
# model = YOLO("yolov8n-seg.pt")  # 加载官方分割模型
# model = YOLO("yolov8n-pose.pt")  # 加载官方姿态模型
# model = YOLO("path/to/best.pt")  # 加载自定义训练的模型

# 使用模型执行跟踪
results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # 使用默认跟踪器进行跟踪
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # 使用 ByteTrack 跟踪器进行跟踪
```

```bash
# CLI
# 使用命令行界面通过各种模型执行跟踪
yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" # 官方检测模型
# yolo track model=yolov8n-seg.pt source="https://youtu.be/LNwODJXcvt4"  # 官方分割模型
# yolo track model=yolov8n-pose.pt source="https://youtu.be/LNwODJXcvt4" # 官方姿态模型
# yolo track model=path/to/best.pt source="https://youtu.be/LNwODJXcvt4" # 自定义训练的模型

# 使用 ByteTrack 跟踪器进行跟踪
# yolo track model=path/to/best.pt tracker="bytetrack.yaml"
```

如上所示，当在视频或流媒体源上运行时，所有[检测](https://docs.ultralytics.com/tasks/detect/)、[分割](https://docs.ultralytics.com/tasks/segment/)和[姿态](https://docs.ultralytics.com/tasks/pose/)模型都可用于跟踪。

-----

## 🔧 **配置**

### **跟踪参数**

跟踪配置与预测模式共享属性，例如 `conf`（置信度阈值）、`iou`（[交并比](https://www.ultralytics.com/glossary/intersection-over-union-iou)阈值）和 `show`（显示结果）。有关其他配置，请参阅[预测模式文档](https://docs.ultralytics.com/modes/predict/)。

```python
# Python
from ultralytics import YOLO

# 配置跟踪参数并运行跟踪器
model = YOLO("yolov8n.pt")
results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
```

```bash
# CLI
# 配置跟踪参数并使用命令行界面运行跟踪器
yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.3 iou=0.5 show
```

### **跟踪器选择**

Ultralytics 允许您使用修改后的跟踪器配置文件。从 [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) 创建跟踪器配置文件的副本（例如，`custom_tracker.yaml`）并根据您的需要调整任何配置（`tracker_type` 除外）。

```python
# Python
from ultralytics import YOLO

# 加载模型并使用自定义配置文件运行跟踪器
model = YOLO("yolov8n.pt")
results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker="custom_tracker.yaml")
```

```bash
# CLI
# 加载模型并使用命令行界面通过自定义配置文件运行跟踪器
yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
```

有关跟踪参数的完整列表，请参阅存储库中的[跟踪配置文件](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)。

-----

## 🐍 **Python 示例**

### **持久化跟踪循环**

此 Python 脚本使用 [OpenCV (`cv2`)](https://www.google.com/search?q=%5Bhttps://opencv.org/%5D\(https://opencv.org/\)) 和 Ultralytics YOLOv8 在视频帧上执行目标跟踪。确保您已安装必要的包（`opencv-python` 和 `ultralytics`）。[`persist=True`](https://www.google.com/search?q=%5Bhttps://docs.ultralytics.com/modes/predict/%23tracking%5D\(https://docs.ultralytics.com/modes/predict/%23tracking\)) 参数表示当前帧是序列中的下一帧，允许跟踪器保持前一帧的跟踪连续性。

```python
# Python
import cv2

from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO("yolov8n.pt")

# 打开视频文件
video_path = "path/to/video.mp4"
cap = cv2.VideoCapture(video_path)

# 循环遍历视频帧
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()

    if success:
        # 在帧上运行 YOLOv8 跟踪，并在帧之间保持跟踪
        results = model.track(frame, persist=True)

        # 在帧上可视化结果
        annotated_frame = results[0].plot()

        # 显示带注释的帧
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # 如果按下 'q' 则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频结束则中断循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()
```

请注意使用 `model.track(frame)` 而不是 `model(frame)`，这专门启用了目标跟踪。此脚本处理每个视频帧，可视化跟踪结果并显示它们。按“q”退出循环。

### **随时间绘制轨迹**

在连续帧上可视化对象轨迹可以为了解视频中的运动模式提供有价值的见解。Ultralytics YOLOv8 使绘制这些轨迹变得高效。

以下示例演示了如何使用 YOLOv8 的跟踪功能来绘制检测到的对象的运动。该脚本打开一个视频，逐帧读取，并使用基于 [PyTorch](https://pytorch.org/) 构建的 YOLO 模型来识别和跟踪对象。通过存储检测到的[边界框](https://www.ultralytics.com/glossary/bounding-box)的中心点并连接它们，我们可以使用 [NumPy](https://numpy.org/) 进行数值运算来绘制表示被跟踪对象路径的线条。

```python
# Python
from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO("yolov8n.pt")

# 打开视频文件
video_path = "path/to/video.mp4"
cap = cv2.VideoCapture(video_path)

# 存储跟踪历史
track_history = defaultdict(lambda: [])

# 循环遍历视频帧
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()

    if success:
        # 在帧上运行 YOLOv8 跟踪，并在帧之间保持跟踪
        result = model.track(frame, persist=True)[0]

        # 获取框和跟踪 ID
        if result.boxes and result.boxes.is_track:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()

            # 在帧上可视化结果
            frame = result.plot()

            # 绘制轨迹
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y 中心点
                if len(track) > 30:  # 保留 30 帧的 30 个轨迹
                    track.pop(0)

                # 绘制跟踪线
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # 显示带注释的帧
        cv2.imshow("YOLOv8 Tracking", frame)

        # 如果按下 'q' 则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频结束则中断循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()
```

### **多线程跟踪**

多线程跟踪允许同时在多个视频流上运行目标跟踪，这对于处理来自多个摄像头的输入的系统非常有利，通过并发处理提高效率。

此 Python 脚本利用 Python 的 [`threading`](https://www.google.com/search?q=%5Bhttps://docs.python.org/3/library/threading.html%5D\(https://docs.python.org/3/library/threading.html\)) 模块进行并发跟踪器执行。每个线程管理单个视频文件的跟踪。

`run_tracker_in_thread` 函数接受视频文件路径、模型和唯一窗口索引等参数。它包含主跟踪循环，读取帧，运行跟踪器，并在专用窗口中显示结果。

此示例使用两个模型 `yolov8n.pt` 和 `yolov8n-seg.pt`，分别在 `video_file1` 和 `video_file2` 中跟踪对象。

在 `threading.Thread` 中设置 `daemon=True` 可确保线程在主程序完成时退出。线程使用 `start()` 启动，主线程使用 `join()` 等待它们完成。

最后，`cv2.destroyAllWindows()` 在线程完成后关闭所有 OpenCV 窗口。

```python
# Python
import threading

import cv2

from ultralytics import YOLO

# 定义模型名称和视频源
MODEL_NAMES = ["yolov8n.pt", "yolov8n-seg.pt"]
SOURCES = ["path/to/video.mp4", "0"]  # 本地视频，0 表示网络摄像头


def run_tracker_in_thread(model_name, filename):
    """
    在自己的线程中运行 YOLO 跟踪器以进行并发处理。

    Args:
        model_name (str): YOLOv8 模型对象。
        filename (str): 视频文件的路径或网络摄像头/外部摄像头源的标识符。
    """
    model = YOLO(model_name)
    results = model.track(filename, save=True, stream=True)
    for r in results:
        pass


# 使用 for 循环创建并启动跟踪器线程
tracker_threads = []
for video_file, model_name in zip(SOURCES, MODEL_NAMES):
    thread = threading.Thread(target=run_tracker_in_thread, args=(model_name, video_file), daemon=True)
    tracker_threads.append(thread)
    thread.start()

# 等待所有跟踪器线程完成
for thread in tracker_threads:
    thread.join()

# 清理并关闭窗口
cv2.destroyAllWindows()
```

通过创建遵循相同模式的其他线程，可以轻松扩展此设置以处理更多视频流。在我们的[关于目标跟踪的博客文章](https://www.ultralytics.com/blog/object-detection-and-tracking-with-ultralytics-yolov8)中探索更多应用。

-----

## 🤝 **贡献新的跟踪器**

您是否在多目标跟踪方面经验丰富，并已使用 Ultralytics YOLO 实现或改编了算法？我们鼓励您为我们的 [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) 中的跟踪器部分做出贡献！您的贡献可以帮助扩展 Ultralytics [生态系统](https://docs.ultralytics.com/)中可用的跟踪解决方案。

要做出贡献，请查看我们的[贡献指南](https://docs.ultralytics.com/help/contributing/)，了解有关提交[拉取请求 (PR)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) 的说明 🛠️。我们期待您的贡献！

让我们共同努力，增强 Ultralytics YOLO 的跟踪能力，并为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)和[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)社区提供更强大的工具 🙏！