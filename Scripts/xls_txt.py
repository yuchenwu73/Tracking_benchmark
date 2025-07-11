import os
import shutil
import pandas as pd

# 原始目录（含 .avi 和 .csv）
SOURCE_DIR = r"C:\Users\gmmdld\Desktop\train"

# 输出目录
VIDEO_OUT = r"D:\yolov5-5.0-1\videos"
LABEL_OUT = r"D:\yolov5-5.0-1\labels"

os.makedirs(VIDEO_OUT, exist_ok=True)
os.makedirs(LABEL_OUT, exist_ok=True)

# 遍历源文件夹
for file in os.listdir(SOURCE_DIR):
    full_path = os.path.join(SOURCE_DIR, file)

    # 复制视频
    if file.endswith(".avi"):
        shutil.copy(full_path, os.path.join(VIDEO_OUT, file))

    # 处理标签文件
    elif file.endswith(".csv") and "-gt" in file:
        base_name = file.replace("-gt.csv", "")
        txt_filename = base_name + ".txt"
        txt_path = os.path.join(LABEL_OUT, txt_filename)

        # 读取 CSV（尝试以 tab 分隔）
        df = pd.read_csv(full_path, header=None, sep='\t')

        with open(txt_path, 'w') as f:
            for _, row in df.iterrows():
                if len(row) < 6:
                    continue
                frame_id, object_id, x, y, w, h = row[:6]
                f.write(f"{int(frame_id)}\t{int(object_id)}\t{int(x)}\t{int(y)}\t{int(w)}\t{int(h)}\t-1\t-1\t-1\t-1\n")

print("CSV 转换完成，视频和标签已分开保存！")
