# 1. 运行uav_tracking.py输出结果
# 2. 运行score.py输出评估结果

# video_folder: 输入视频文件夹
# output_dir: 输出结果文件夹
# model_weights: 模型权重文件
# gt_dir: 真实标注文件夹
# pred_dir: 预测结果文件夹
# pattern: 评估模式 multi代表有多个视频，single代表单个视频

python tracking.py --video_folder test --output_dir evaluate/result --model_weights runs/train/20250809_2327_yolo11m_imgsz1280_epoch300_bs8/weights/best.pt
python score.py --gt_dir data/train/1-2-gt.csv --pred_dir evaluate/result/1-2.txt --pattern single