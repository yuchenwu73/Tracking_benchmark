#!/usr/bin/env python3
"""
无人机多目标跟踪系统 - 单线程版本
支持8K视频处理，具备完整的轨迹可视化和性能监控功能

主要特性：
- 单线程同步处理，适合调试和小文件处理
- 子图分割处理，支持超高分辨率视频
- 智能轨迹绘制：黄色ID显示在轨迹起始点，绿色轨迹线
- 优化的ID分配机制，减少ID浪费
- 详细的性能统计和监控
- 可调整窗口大小和全屏显示
- 支持暂停/继续播放

核心算法：
- YOLO目标检测 + ByteTrack多目标跟踪
- 改进的轨迹关联算法，提高跟踪稳定性
- 智能ID管理，减少因检测失败导致的ID跳跃

使用说明：
- Q: 退出程序
- R: 重置窗口大小
- F: 切换全屏
- Space: 暂停/继续
- 鼠标左键: 显示坐标

适用场景：
- 调试和开发阶段
- 小文件快速处理
- 算法参数调优

"""

# 导入所需的Python库
import os  # 用于文件和路径操作
import torch  # PyTorch深度学习框架
import time  # 用于性能计时和延迟控制
from ultralytics import YOLO  # YOLO目标检测模型
from collections import defaultdict  # 用于轨迹历史记录的默认字典
import cv2  # OpenCV图像处理库
import numpy as np  # 数值计算库
from tqdm import tqdm  # 进度条显示
# 导入YOLO工具类 - 修复版本兼容性问题
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.engine.results import Results, Boxes

# 创建yaml_load函数的兼容版本
def yaml_load(file_path):
    """兼容的YAML加载函数"""
    yaml_instance = YAML()
    return yaml_instance.load(file_path)
# 导入跟踪器实现
from tracker.bytetrack_tracker import BYTETracker
from tracker.botsort_tracker import BOTSORT

# 定义可用的跟踪器映射字典
# 支持两种跟踪器：ByteTrack和BoT-SORT
TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}

def initialize_tracker(tracker_yaml: str, frame_rate: int = 30):
    """
    初始化目标跟踪器

    参数说明：
        tracker_yaml: 跟踪器配置文件的路径，包含跟踪器的参数设置
        frame_rate: 视频帧率，用于跟踪器的时间计算，默认30fps

    返回值：
        初始化好的跟踪器实例
    """
    # 加载并解析配置文件
    try:
        tracker_cfg = IterableSimpleNamespace(**yaml_load(check_yaml(tracker_yaml)))
    except Exception as e:
        print(f"配置文件加载失败: {e}")
        # 使用默认配置
        tracker_cfg = IterableSimpleNamespace(tracker_type="bytetrack")

    tracker_type = getattr(tracker_cfg, 'tracker_type', 'bytetrack')
    # 检查跟踪器类型是否支持
    if tracker_type not in TRACKER_MAP:
        raise ValueError(f"不支持的跟踪器类型: {tracker_type}")
    return TRACKER_MAP[tracker_type](args=tracker_cfg, frame_rate=frame_rate)

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
            cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)

        # 绘制目标ID：在轨迹起始点显示黄色ID数字
        if len(track) > 0:
            start_point = tuple(np.array(track[0]).astype(int))
            cv2.putText(
                frame,
                str(track_id),  # 显示轨迹ID（可能不连续，如1,3,9,30等）
                start_point,    # 显示位置：轨迹的第一个点（起始位置）
                cv2.FONT_HERSHEY_SIMPLEX,
                3,              # 字体大小
                (0, 255, 255),  # 黄色 (BGR格式: 蓝=0, 绿=255, 红=255)
                3,              # 字体粗细
                cv2.LINE_AA     # 抗锯齿
            )
    return frame

# 初始化YOLO模型
# 使用预训练的权重文件'last.pt'
model = YOLO("/data2/wuyuchen/Tracking_benchmark/runs/train/20250629_yolo11/weights/best.pt")

# 初始化目标跟踪器
# 使用改进的ByteTrack配置，优化ID分配机制，减少ID浪费
# 配置特点：提高新轨迹创建阈值，增加轨迹缓冲时间，改善关联精度
tracker = initialize_tracker("cfg/bytetrack_improved.yaml")
# tracker = initialize_tracker("data/bytetrack.yaml")  # 原始配置（ID浪费较多）
# tracker = initialize_tracker("data/botsort.yaml")    # BoT-SORT（需要ReID特征）

# 打开视频文件
# 注意：确保视频路径正确且可访问
video_path = "UAV/0.MOV"
cap = cv2.VideoCapture(video_path)

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

# 创建可调整大小的窗口
window_name = "YOLO目标跟踪系统(单线程版) - 按Q退出"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 允许调整窗口大小并保持比例

print("🖥️  Window Controls:")
print("   - Drag window edges to resize")
print("   - Click top-right buttons to maximize/minimize/close")
print("   - Press Q to exit")
print("   - Press R to reset window size")
print("   - Press F to toggle fullscreen")
print("   - Press SPACE to pause/resume")
print("   - Left click to show coordinates")

# 初始化性能统计
frame_processing_times = []

# 初始化窗口控制变量
is_fullscreen = False
is_paused = False
window_initialized = False
mouse_callback_set = False

# 主处理循环
# 使用tqdm显示处理进度
with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
    while cap.isOpened():
        # 开始计时
        frame_start_time = time.time()

        # 初始化各阶段计时器
        timing = {
            "read_frame": 0,  # 帧读取时间
            "predict": 0,     # 目标检测时间
            "track": 0,       # 目标跟踪时间
            "draw": 0,        # 绘制显示时间
        }

        # 读取视频帧
        read_start = time.time()
        ret, frame = cap.read()
        if not ret:  # 如果读取失败，退出循环
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

        # 将检测结果转换为YOLO的Boxes格式
        if len(detections) > 0:
            detections = Boxes(detections, frame.shape)
        else:
            detections = Boxes(np.empty((0, 6)), frame.shape)

        # 更新目标跟踪器
        tracks = tracker.update(detections, frame)
        timing["track"] = time.time() - track_start

        # 绘制检测和跟踪结果
        draw_start = time.time()

        # 检查是否有有效的跟踪结果
        if len(tracks) > 0 and tracks.ndim == 2:
            # 有跟踪结果时的处理
            all_detections = Results(frame, path="", names=model.predictor.model.names, boxes=torch.as_tensor(tracks[:, :-1]))

            # 绘制检测框
            anno_frame = all_detections.plot(img=frame, line_width=5, font_size=5)

            # 更新和绘制轨迹
            boxes = all_detections.boxes.xywh.cpu()
            track_ids = all_detections.boxes.id.int().cpu().tolist()
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
            f"Tracker: ByteTrack (Improved) | Yellow ID = Track Start Point",
            "Q:Exit | R:Reset | F:Fullscreen | Space:Pause | Click for coordinates"
        ]

        # 直接绘制信息文字（无背景框）
        for i, text in enumerate(info_text):
            cv2.putText(resized_image, text, (20, 35 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 显示处理后的图像
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

        # 延迟设置鼠标回调（在窗口稳定后）
        if window_initialized and not mouse_callback_set and pbar.n > 10:
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

        # 保存处理后的图像（每30帧保存一次）
        if pbar.n % 30 == 0:  # 每30帧保存一张
            output_dir = "tracking_output_single_thread"
            output_path = f"{output_dir}/frame_{pbar.n:06d}.jpg"
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(output_path, resized_image)
            print(f"保存图片: {output_path}")

        # 更新进度条
        pbar.update(1)

        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):  # 退出
            print("用户按Q键退出程序")
            break
        elif key == ord("r") or key == ord("R"):  # 重置窗口大小
            cv2.resizeWindow(window_name, 1280, 720)
            cv2.moveWindow(window_name, 100, 100)
            is_fullscreen = False
            print("窗口大小已重置")
        elif key == ord("f") or key == ord("F"):  # 切换全屏
            try:
                if is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 1280, 720)
                    is_fullscreen = False
                    print("退出全屏模式")
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    is_fullscreen = True
                    print("进入全屏模式")
            except cv2.error as e:
                print(f"全屏切换失败: {e}")
        elif key == ord(" "):  # 空格键暂停/继续
            is_paused = not is_paused
            if is_paused:
                print("⏸️  视频已暂停，按空格键继续")
                while is_paused:
                    pause_key = cv2.waitKey(30) & 0xFF
                    if pause_key == ord(" "):
                        is_paused = False
                        print("▶️  继续播放")
                    elif pause_key == ord("q") or pause_key == ord("Q"):
                        print("用户在暂停状态下按Q键退出")
                        break
                if pause_key == ord("q") or pause_key == ord("Q"):
                    break

# 清理资源
# 保存最后一帧结果
if 'resized_image' in locals():
    output_dir = "tracking_output_single_thread"
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f"{output_dir}/final_frame.jpg", resized_image)
    print(f"保存最终帧: {output_dir}/final_frame.jpg")
# 释放视频捕获对象
cap.release()
# 关闭所有窗口
cv2.destroyAllWindows()
print("视频处理完成！")

# 计算并显示性能统计信息
# 去除第一帧（通常会有额外开销）
average_stats = {key: sum(times[key] for times in frame_processing_times[1:]) / len(frame_processing_times[1:]) for key in frame_processing_times[0]}

# 打印详细的性能统计
print("平均帧处理时间统计:")
for key, value in average_stats.items():
    if key != "total":
        print(f"  {key}: {value:.4f}秒 ({(value / average_stats['total']) * 100:.2f}%)")
print(f"总计: {average_stats['total']:.4f}秒")
