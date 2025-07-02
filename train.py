import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolo11.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=True,
                imgsz=1088,  # 与验证时保持一致，更好检测小目标无人机
                epochs=300,
                batch=32,  # 由于使用imgsz=1088，减少batch size以避免GPU内存不足
                close_mosaic=10,
                workers=8,
                device='7',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='yolo11',
                )