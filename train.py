import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# nohup python train.py > logs/yolo11m_imgsz640_epoch300_bs32.log 2>&1 &


if __name__ == '__main__':
    model = YOLO('yolo11m.yaml')
    model.load('yolo11m.pt') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=True,
                imgsz=640,  # 可以调整为32的倍数，例如1280等
                epochs=300,
                batch=32,   # 爆显存了要调整
                close_mosaic=10,
                workers=8,
                device='7',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='yolo11m_imgsz640_epoch300_bs32',
                )

