import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/20250809_2327_yolo11m_imgsz1280_epoch300_bs8/weights/best.pt') # select your model.pt path
    model.predict(source='data/images/frame_0110_sub_01.jpg',
                  imgsz=1088,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )