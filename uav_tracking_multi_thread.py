#!/usr/bin/env python3
"""
无人机多目标跟踪系统 - 官方API + 优化GUI版本
支持8K视频实时处理，具备完整的轨迹可视化和性能监控功能

主要特性：
- 使用YOLO官方内置跟踪器（BoT-SORT/ByteTrack）
- 智能轨迹绘制：黄色ID显示在轨迹起始点，绿色轨迹线
- 详细的性能统计和监控
- 可调整窗口大小和全屏显示
- 支持暂停/继续播放
- 实时跟踪器切换功能

核心算法：
- YOLO官方内置跟踪器（默认BoT-SORT）
- 官方推荐的persist=True逐帧处理
- 完全对齐官方最佳实践

使用说明：
- Q: 退出程序
- R: 重置窗口大小
- F: 切换全屏
- Space: 暂停/继续
- T: 切换跟踪器类型

- 鼠标左键: 显示坐标

"""

# 导入所需的Python库
import os  # 用于文件和路径操作
import threading  # 多线程支持
import queue  # 线程间通信队列
import torch  # PyTorch深度学习框架
import time  # 用于性能计时和延迟控制
from ultralytics import YOLO  # YOLO目标检测模型
from collections import defaultdict  # 用于轨迹历史记录的默认字典
import cv2  # OpenCV图像处理库
import numpy as np  # 数值计算库
from tqdm import tqdm  # 进度条显示


# 使用YOLO官方内置跟踪器，无需导入自定义跟踪器

# 定义可用的跟踪器类型
AVAILABLE_TRACKERS = ["botsort", "bytetrack"]

def get_tracker_config(tracker_type):
    """
    获取跟踪器配置

    参数:
        tracker_type: 跟踪器类型 ("botsort", "bytetrack")

    返回:
        跟踪器配置字符串或None（None表示使用默认BoT-SORT）
    """
    if tracker_type == "bytetrack":
        return "/data2/wuyuchen/Tracking_benchmark/cfg/bytetrack.yaml"
    elif tracker_type == "botsort":
        return "/data2/wuyuchen/Tracking_benchmark/cfg/botsort.yaml"
    else:
        return None  # 使用默认跟踪器（BoT-SORT）

def mouse_callback(event, x, y, flags, param):
    """
    鼠标回调函数 - 点击显示坐标位置
    """
    _ = flags, param  # 忽略未使用的参数
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse click at: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"Right click at: ({x}, {y})")



def draw_tracks(frame, track_history):
    """
    在图像上绘制目标轨迹和ID标识

    轨迹可视化说明：
    - 绿色线条：目标的历史移动轨迹
    - 黄色数字：目标的唯一ID，显示在轨迹起始点
    - ID数字可能不连续（如1,3,9,30），这是正常现象，表示中间的ID已被删除

    参数说明：
        frame: 当前视频帧
        track_history: 包含所有目标轨迹历史的字典
                      格式: {track_id: [(x1,y1), (x2,y2), ...]}

    返回值：
        绘制完轨迹的图像帧
    """
    # 遍历每个目标的轨迹历史
    for track_id, track in track_history.items():
        # 绘制轨迹线：如果轨迹点数大于1，用绿色线条连接所有历史位置
        if len(track) > 1:
            points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=4)

        # 绘制目标ID：在轨迹起始点显示黄色ID数字
        if len(track) > 0:
            start_point = tuple(np.array(track[0]).astype(int))
            cv2.putText(
                frame,
                str(track_id),  # 显示轨迹ID（可能不连续，如1,3,9,30等）
                start_point,    # 显示位置：轨迹的第一个点（起始位置）
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,            # 字体大小（从3减小到1.2，更合适的大小）
                (0, 255, 255),  # 黄色 (BGR格式: 蓝=0, 绿=255, 红=255)
                2,              # 字体粗细（从3减小到2）
                cv2.LINE_AA     # 抗锯齿
            )
    return frame

# 初始化YOLO模型
# 使用预训练的权重文件'best.pt'
model = YOLO("/data2/wuyuchen/Tracking_benchmark/runs/train/20250809_2327_yolo11m_imgsz1280_epoch300_bs8/weights/best.pt")

# 初始化跟踪器类型
# 使用官方YOLO内置跟踪器，更稳定可靠
current_tracker = "botsort"  # 可选: "botsort" (默认), "bytetrack"
print(f"🔧 使用跟踪器: {current_tracker}")

# 全局变量用于跟踪器切换
tracker_config = get_tracker_config(current_tracker)

# 打开视频文件
# 注意：确保视频路径正确且可访问
video_path = "/data2/wuyuchen/Tracking_benchmark/data/train/1-2.avi"  # 使用比赛验证集视频
cap = cv2.VideoCapture(video_path)

# 获取视频文件名（不含扩展名）用于保存比赛结果
video_name = os.path.splitext(os.path.basename(video_path))[0]

# 获取视频基本信息
fps = cap.get(cv2.CAP_PROP_FPS)  # 视频帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数

# 设置子图分割参数
# 由于原视频分辨率较高，需要分块处理以提高效率
sub_width = 960  # 子图宽度
sub_height = 1080  # 子图高度
cols = width // sub_width  # 水平分割数
rows = height // sub_height  # 垂直分割数

# 初始化轨迹历史记录
# 使用defaultdict自动创建新目标的轨迹列表
track_history = defaultdict(lambda: [])


frame_processing_times = []  # 存储每帧的处理时间

# 创建可调整大小的窗口
window_name = "YOLO目标跟踪系统(多线程版) - 按Q退出"

# 检查OpenCV GUI支持并创建窗口
gui_available = True
try:
    # 检查DISPLAY环境变量
    import os
    if not os.environ.get('DISPLAY'):
        print("⚠️ 没有DISPLAY环境变量，将使用无GUI模式")
        gui_available = False
    else:
        # 尝试创建窗口
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 允许调整窗口大小并保持比例
        print("✅ OpenCV GUI支持正常")
except Exception as e:
    print(f"❌ OpenCV GUI错误: {e}")
    print("� 尝试解决方案:")
    print("   1. 检查DISPLAY环境变量: echo $DISPLAY")
    print("   2. 在MobaXterm中启用X11转发")
    print("   3. 重新安装opencv-python: pip uninstall opencv-python && pip install opencv-python")
    print("   4. 或安装带GUI支持的版本: pip install opencv-contrib-python")
    gui_available = False
    print("⚠️  将在无GUI模式下运行，仅显示处理进度")

if gui_available:
    print("�🖥️  Window Controls:")
    print("   - Drag window edges to resize")
    print("   - Click top-right buttons to maximize/minimize/close")
    print("   - Press Q to exit")
    print("   - Press R to reset window size")
    print("   - Press F to toggle fullscreen")
    print("   - Press SPACE to pause/resume")
    print("   - Press T to switch tracker")
    print("   - Press S to save current frame")
    print("   - Left click to show coordinates")
else:
    print("🖥️  无GUI模式:")
    print("   - 处理结果将保存到输出目录")
    print("   - 按Ctrl+C停止处理")

# 初始化性能统计
frame_processing_times = []

# 初始化窗口控制变量
is_fullscreen = False
is_paused = False
window_initialized = False
mouse_callback_set = False

# 帧队列和线程管理
frame_queue = queue.Queue(maxsize=10)

def read_frames():
    """
    多线程帧读取函数
    在后台线程中持续读取视频帧，提高处理效率
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)  # 表示视频结束
            break
        frame_queue.put(frame)

# 启动读取线程
read_thread = threading.Thread(target=read_frames, daemon=True)
read_thread.start()

# 主处理循环
# 使用tqdm显示处理进度
with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
    while True:
        # 开始计时
        frame_start_time = time.time()

        # 初始化各阶段计时器
        timing = {
            "read_frame": 0,  # 帧读取时间
            "predict": 0,     # 目标检测时间
            "track": 0,       # 目标跟踪时间
            "draw": 0,        # 绘制显示时间
        }

        # 从队列中获取帧
        read_start = time.time()
        frame = frame_queue.get()
        if frame is None:  # 如果读取结束，退出循环
            break
        timing["read_frame"] = time.time() - read_start

        # 初始化存储列表
        sub_frames = []  # 存储分割后的子图
        detections = []  # 存储检测结果

        # 图像分块处理
        # 将大分辨率图像分割成小块进行处理
        for row in range(rows):
            for col in range(cols):
                # 计算当前子图的坐标
                x1 = col * sub_width
                y1 = row * sub_height
                x2 = x1 + sub_width
                y2 = y1 + sub_height

                # 提取子图
                sub_frame = frame[y1:y2, x1:x2]
                sub_frames.append(sub_frame)

        # YOLO目标检测
        predict_start = time.time()
        # 使用GPU进行批量检测
        results = model.predict(source=sub_frames, device=0, verbose=False)
        timing["predict"] = time.time() - predict_start

        # 目标跟踪处理
        track_start = time.time()
        # 处理检测结果，将子图坐标映射回原图坐标
        for i, result in enumerate(results):
            row = i // cols
            col = i % cols
            x_offset = col * sub_width
            y_offset = row * sub_height

            # 坐标转换
            for box in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box[:6]
                # 添加偏移量得到原图坐标
                x1 += x_offset
                x2 += x_offset
                y1 += y_offset
                y2 += y_offset
                detections.append([x1, y1, x2, y2, conf, cls])

        # 🔧 添加全局NMS去重处理，避免重复检测导致ID浪费
        if len(detections) > 0:
            detections_array = np.array(detections)
            # 使用YOLO的NMS函数进行去重
            from ultralytics.utils.ops import non_max_suppression
            import torch

            # 转换为tensor格式 [x1, y1, x2, y2, conf, cls]
            detections_tensor = torch.from_numpy(detections_array).float().unsqueeze(0)

            # 应用NMS去重，IoU阈值设为0.5以去除重复检测
            nms_results = non_max_suppression(
                detections_tensor,
                conf_thres=0.25,  # 置信度阈值
                iou_thres=0.5,    # IoU阈值，去除重复检测
                max_det=300       # 最大检测数量
            )[0]

            if nms_results is not None and len(nms_results) > 0:
                detections = nms_results.cpu().numpy()
            else:
                detections = np.array([])
        else:
            detections = np.array([])

        # 使用官方YOLO跟踪API
        track_start = time.time()

        # 根据当前跟踪器类型进行跟踪
        if tracker_config:
            results = model.track(frame, persist=True, tracker=tracker_config)
        else:
            results = model.track(frame, persist=True)  # 使用默认BoT-SORT

        timing["track"] = time.time() - track_start

        # 绘制检测和跟踪结果
        draw_start = time.time()

        # 检查是否有有效的跟踪结果
        if results[0].boxes and results[0].boxes.is_track:
            # 绘制检测框（调整线条粗细为更细的边界框，减小字体大小）
            anno_frame = results[0].plot(img=frame, line_width=3, font_size=0.8)

            # 更新和绘制轨迹
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # 记录目标中心点



            annotated_frame = draw_tracks(anno_frame, track_history)
        else:
            # 没有跟踪结果时，只绘制原始帧和现有轨迹
            annotated_frame = draw_tracks(frame, track_history)
            track_ids = []  # 空的跟踪ID列表

        # 调整图像大小用于显示
        resized_image = cv2.resize(annotated_frame, (1920, 1080))
        timing["draw"] = time.time() - draw_start

        # 记录本帧处理时间
        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time
        frame_processing_times.append({"total": frame_time, **timing})

        # 在图像上添加信息文字（使用英文避免字体问题）- 无背景框
        info_text = [
            f"Frame: {pbar.n}/{total_frames}",
            f"Progress: {(pbar.n/total_frames)*100:.1f}%",
            f"Speed: {1/frame_time:.1f} FPS" if frame_time > 0 else "Speed: Calculating...",
            f"Objects: {len(track_ids) if 'track_ids' in locals() else 0}",
            f"Tracker: {current_tracker.upper()}{' (Default)' if current_tracker == 'botsort' else ''} | Yellow ID = Track Start Point",
            f"Q:Exit | R:Reset | F:Fullscreen | Space:Pause | Click for coordinates"
        ]

        # 直接绘制信息文字（无背景框）
        for i, text in enumerate(info_text):
            cv2.putText(resized_image, text, (20, 35 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 显示处理后的图像（仅在GUI可用时）
        if gui_available:
            try:
                cv2.imshow(window_name, resized_image)

                # 初始化窗口设置（只在第一次显示时执行）
                if not window_initialized:
                    try:
                        # 等待窗口完全创建
                        cv2.waitKey(1)
                        cv2.resizeWindow(window_name, 1280, 720)  # 设置默认窗口大小为720p
                        cv2.moveWindow(window_name, 100, 100)  # 设置窗口初始位置
                        window_initialized = True
                        print("✅ Window size and position set")
                    except cv2.error as e:
                        print(f"⚠️ Window setup failed: {e}")
                        window_initialized = True  # 避免重复尝试
            except cv2.error as e:
                print(f"⚠️ 图像显示失败: {e}")
                gui_available = False
                print("⚠️ 切换到无GUI模式")

        # 延迟设置鼠标回调（在窗口稳定后且GUI可用时）
        if gui_available and window_initialized and not mouse_callback_set and pbar.n > 10:
            try:
                # 检查窗口是否存在
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.setMouseCallback(window_name, mouse_callback)
                    mouse_callback_set = True
                    print("✅ Mouse callback set")
                else:
                    # 窗口还不可见，继续等待
                    pass
            except cv2.error as e:
                print(f"⚠️ 鼠标回调设置失败: {e}")
                mouse_callback_set = True  # 避免重复尝试



        # 更新进度条
        pbar.update(1)

        # 键盘事件处理（仅在GUI可用时）
        if gui_available:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # 按Q退出
                break
            elif key == ord("r"):  # 按R重置窗口大小
                try:
                    cv2.resizeWindow(window_name, 1280, 720)
                    cv2.moveWindow(window_name, 100, 100)
                    print("✅ Window reset to default size")
                except cv2.error as e:
                    print(f"⚠️ Window reset failed: {e}")
            elif key == ord("f"):  # 按F切换全屏
                try:
                    if not is_fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        is_fullscreen = True
                        print("✅ Fullscreen mode enabled")
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        is_fullscreen = False
                        print("✅ Windowed mode enabled")
                except cv2.error as e:
                    print(f"⚠️ Fullscreen toggle failed: {e}")
            elif key == ord(" "):  # 按空格暂停/继续
                is_paused = not is_paused
                if is_paused:
                    print("⏸️ Paused - Press SPACE to resume")
                    while is_paused:
                        key = cv2.waitKey(30) & 0xFF
                        if key == ord(" "):
                            is_paused = False
                            print("▶️ Resumed")
                        elif key == ord("q"):
                            break
            elif key == ord("t"):  # 按T切换跟踪器
                # 切换跟踪器类型
                current_idx = AVAILABLE_TRACKERS.index(current_tracker)
                current_tracker = AVAILABLE_TRACKERS[(current_idx + 1) % len(AVAILABLE_TRACKERS)]
                tracker_config = get_tracker_config(current_tracker)
                print(f"🔄 切换到跟踪器: {current_tracker}")


# 清理资源和显示统计信息
print("\n🔄 正在清理资源...")



# 释放视频捕获对象和窗口
cap.release()
if gui_available:
    cv2.destroyAllWindows()
print("✅ 资源清理完成")

# 显示跟踪统计
print(f"✅ 跟踪完成，共处理 {len(track_history)} 个目标轨迹")

# 计算并显示性能统计
if frame_processing_times:
    print(f"\n📊 处理完成！共处理 {len(frame_processing_times)} 帧")

    # 计算平均处理时间（跳过第一帧，因为可能包含初始化开销）
    if len(frame_processing_times) > 1:
        valid_times = frame_processing_times[1:]
        average_stats = {}
        for key in valid_times[0].keys():
            average_stats[key] = sum(times[key] for times in valid_times) / len(valid_times)

        print("\n⏱️ 平均帧处理时间:")
        for key, value in average_stats.items():
            if key != "total":
                percentage = (value / average_stats['total']) * 100 if average_stats['total'] > 0 else 0
                print(f"  {key}: {value:.4f}s ({percentage:.2f}%)")
        print(f"总计: {average_stats['total']:.4f}s")

        # 计算平均FPS
        if average_stats['total'] > 0:
            avg_fps = 1 / average_stats['total']
            print(f"平均处理速度: {avg_fps:.2f} FPS")
    else:
        print("处理帧数不足，无法计算统计信息")
else:
    print("未处理任何帧")

print("\n🎉 程序执行完成！")
