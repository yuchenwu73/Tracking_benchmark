import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/data2/wuyuchen/Tracking_benchmark/last.pt') # select your model.pt path
    model.predict(source='data/images/frame_0110_sub_01.jpg',
                  imgsz=1088,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )