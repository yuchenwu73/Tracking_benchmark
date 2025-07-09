import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# nohup python train1.py > logs/yolo11m_imgsz1280_epoch300_bs8.log 2>&1 &


if __name__ == '__main__':
    model = YOLO('yolo11m.yaml')
    model.load('yolo11m.pt') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=True,
                imgsz=1280,
                epochs=300,
                batch=8,
                close_mosaic=10,
                workers=8,
                device='6',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='yolo11m_imgsz1280_epoch300_bs8',
                )