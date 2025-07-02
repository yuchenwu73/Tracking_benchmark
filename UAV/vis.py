import cv2
import os

def save_frame_with_resizes(video_path, output_dir, start_second=0, scales=(0.5, 0.25, 0.1)):
    """
    从视频中读取一帧并保存原始尺寸和不同缩放比例的图片。

    :param video_path: 视频文件路径
    :param output_dir: 输出目录
    :param start_second: 从指定秒数开始读取帧
    :param scales: 缩放尺寸列表，例如[(width1, height1), (width2, height2)]
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(start_second * fps)

    # 设置视频到目标帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    success, frame = cap.read()
    if not success:
        print(f"无法读取视频中从第 {start_second} 秒开始的帧")
        cap.release()
        return

    # 保存原始帧P
    original_frame_path = os.path.join(output_dir, f"frame_original.png")
    cv2.imwrite(original_frame_path, frame)
    print(f"原始尺寸图片已保存: {original_frame_path}")

    # 保存缩放后的图片
    for idx, (width, height) in enumerate(scales):
        resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        scaled_frame_path = os.path.join(output_dir, f"frame_scaled_{idx+1}.png")
        cv2.imwrite(scaled_frame_path, resized_frame)
        print(f"缩放尺寸 {width}x{height} 图片已保存: {scaled_frame_path}")

    # 释放视频资源
    cap.release()

if __name__ == "__main__":
    video_file = "无人机/1.MOV"  # 替换为8K视频文件的路径
    output_folder = "vis/"          # 替换为保存图片的输出目录
    scales_list = [(3840, 2160), (1920, 1080), (1280, 720)]  # 替换为所需的缩放尺寸
    save_frame_with_resizes(video_file, output_folder, start_second=10, scales=scales_list)
