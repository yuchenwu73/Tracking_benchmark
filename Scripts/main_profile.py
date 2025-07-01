import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # choose your yaml file
    model = YOLO('/ultralytics-20240309/yolov8-GHGNetV2-Attn-LSCD.yaml')
    model.info(detailed=True)
    model.profile(imgsz=[640, 640])
    model.fuse()