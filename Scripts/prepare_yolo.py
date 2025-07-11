import os
import cv2
from tqdm import tqdm

# 路径配置
VIDEO_DIR = 'videos'
LABEL_DIR = 'labels'
OUT_IMG = 'images_yolo'
OUT_LABEL = 'labels_yolo'

# 创建输出目录
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LABEL, exist_ok=True)

def extract_video_frames(video_path, prefix):
    frames = []
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fname = f"{prefix}_{idx:06d}.jpg"
        cv2.imwrite(os.path.join(OUT_IMG, fname), frame)
        frames.append(fname)
        idx += 1
    cap.release()
    return frames

def parse_txt_label(txt_path, frame_prefix, frame_count, video_width, video_height):
    """
    标签格式：
    <frame_id> <object_id> <x> <y> <w> <h> ...
    转换为 YOLO 格式
    """
    frame_annos = {}

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            frame_id = int(parts[0])
            x, y, w, h = map(int, parts[2:6])

            xc = (x + w / 2) / video_width
            yc = (y + h / 2) / video_height
            wn = w / video_width
            hn = h / video_height
            yolo_line = f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"

            if frame_id not in frame_annos:
                frame_annos[frame_id] = []
            frame_annos[frame_id].append(yolo_line)

    for i in range(frame_count):
        if i not in frame_annos:
            continue
        label_path = os.path.join(OUT_LABEL, f"{frame_prefix}_{i:06d}.txt")
        with open(label_path, 'w') as f:
            f.write('\n'.join(frame_annos[i]))

def process_video_and_label(video_file, label_file):
    name = os.path.splitext(os.path.basename(video_file))[0]

    cap = cv2.VideoCapture(video_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frames = extract_video_frames(video_file, name)
    parse_txt_label(label_file, name, len(frames), width, height)

# 主流程：匹配 video 和 label 文件
video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')])

for video_name in tqdm(video_files, desc="处理视频"):
    name = os.path.splitext(video_name)[0]
    video_path = os.path.join(VIDEO_DIR, video_name)

    # 假设标签是同名加 -gt.txt，比如 1-2.avi 对应 1-2-gt.txt
    label_name = f"{name}.txt"
    label_path = os.path.join(LABEL_DIR, label_name)

    if not os.path.exists(label_path):
        print(f"标签文件不存在：{label_path}")
        continue

    process_video_and_label(video_path, label_path)

print("视频帧提取与YOLO格式标签转换完成！")
