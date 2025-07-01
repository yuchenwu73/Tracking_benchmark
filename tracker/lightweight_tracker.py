# Ultralytics YOLO 🚀, AGPL-3.0 license

# 导入所需的Python库
from functools import partial  # 用于创建偏函数
from pathlib import Path  # 用于处理文件路径
import torch  # PyTorch深度学习框架
from ultralytics.utils import IterableSimpleNamespace, yaml_load  # YOLO工具类
from ultralytics.utils.checks import check_yaml  # YAML配置检查工具
from .botsort_tracker import BOTSORT  # BoT-SORT跟踪器
from .bytetrack_tracker import BYTETracker  # ByteTrack跟踪器

# 定义跟踪器类型映射字典，支持ByteTrack和BoT-SORT两种跟踪器
TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}


def on_predict_start(predictor: object, persist: bool = False) -> None:
    """
    在预测开始时初始化目标跟踪器
    
    参数说明:
        predictor: 预测器对象
        persist: 是否保持已存在的跟踪器状态
    
    异常:
        AssertionError: 当跟踪器类型不是'bytetrack'或'botsort'时抛出
    """
    # 如果已经存在跟踪器且需要保持状态，则直接返回
    if hasattr(predictor, "trackers") and persist:
        return

    # 加载并检查跟踪器配置文件
    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))

    # 验证跟踪器类型是否支持
    if cfg.tracker_type not in {"bytetrack", "botsort"}:
        raise AssertionError(f"目前仅支持'bytetrack'和'botsort'跟踪器，但收到了'{cfg.tracker_type}'")

    # 初始化跟踪器列表
    trackers = []
    for _ in range(predictor.dataset.bs):
        # 根据配置创建对应类型的跟踪器实例
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
        # 非流式模式下只需要一个跟踪器
        if predictor.dataset.mode != "stream":
            break
    predictor.trackers = trackers
    # 初始化视频路径列表，用于判断是否需要重置跟踪器
    predictor.vid_path = [None] * predictor.dataset.bs


def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    """
    在预测后处理阶段结束时更新目标跟踪结果
    
    参数说明:
        predictor: 包含预测结果的预测器对象
        persist: 是否保持跟踪器状态
    """
    # 获取当前批次的路径和原始图像
    path, im0s = predictor.batch[:2]

    # 判断任务类型和数据模式
    is_obb = predictor.args.task == "obb"  # 是否为有向边界框任务
    is_stream = predictor.dataset.mode == "stream"  # 是否为流式处理模式
    
    # 遍历每个图像进行处理
    for i in range(len(im0s)):
        # 获取对应的跟踪器
        tracker = predictor.trackers[i if is_stream else 0]
        # 构建视频保存路径
        vid_path = predictor.save_dir / Path(path[i]).name
        
        # 如果视频路径发生变化且不保持状态，则重置跟踪器
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        # 获取检测结果并转换为numpy数组
        det = (predictor.results[i].obb if is_obb else predictor.results[i].boxes).cpu().numpy()
        if len(det) == 0:
            continue
            
        # 使用跟踪器更新目标状态
        tracks = tracker.update(det, im0s[i])
        if len(tracks) == 0:
            continue
            
        # 更新预测结果
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = predictor.results[i][idx]
        
        # 更新边界框信息
        update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
        predictor.results[i].update(**update_args)


def register_tracker(model: object, persist: bool) -> None:
    """
    为模型注册目标跟踪回调函数
    
    参数说明:
        model: 需要注册跟踪功能的模型对象
        persist: 是否保持跟踪器状态
    """
    # 注册预测开始和后处理结束时的回调函数
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))
    model.add_callback("on_predict_postprocess_end", partial(on_predict_postprocess_end, persist=persist))
