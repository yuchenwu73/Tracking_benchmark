import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# nohup python train2.py > logs/no_pretrain_yolo11m_imgsz1280_epoch300_bs8.log 2>&1 &


if __name__ == '__main__':
    model = YOLO('yolo11m.yaml')
    model.load('yolo11m.pt') # pretrain weights
    model.train(data='dataset/data.yaml',
                cache=True,
                imgsz=1280, 
                epochs=300,
                batch=8,
                close_mosaic=10,
                workers=8,
                device='5',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='20250709_1445_no_pretrain_yolo11m_imgsz1280_epoch300_bs8',
                )