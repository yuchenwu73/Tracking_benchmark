import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/data2/wuyuchen/Tracking_benchmark/last.pt')
    model.val(data='/data2/wuyuchen/Tracking_benchmark/data/uav.yaml',
              imgsz=1088,
              batch=16,
              verbose = True,
              project='runs/val',
              name='exp'
              )