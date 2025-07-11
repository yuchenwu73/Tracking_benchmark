import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# nohup python train.py > logs/20250711_1457_no_pretrain_yolo11m_imgsz1280_epoch300_bs8.log 2>&1 &

if __name__ == '__main__':
    model = YOLO('yolo11m.yaml')
    # model.load('yolo11m.pt') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=True,
                imgsz=1280,  # 可以调整为32的倍数，例如1280等
                epochs=300, 
                batch=8,   # 爆显存了要调整
                close_mosaic=10,
                workers=8,
                device='7',
                optimizer='SGD', # using SGD

                # === Loss权重参数 (如果loss过小可以调整) ===
                box=75,     # 边界框回归loss权重 (默认7.5, 可改成75/750/7500)
                cls=5,     # 分类loss权重 (默认0.5, 可改成5/50/500)
                dfl=15,     # DFL loss权重 (默认1.5, 可改成15/150/1500)

                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='20250711_1457_no_pretrain_yolo11m_imgsz1280_epoch300_bs8',
                )

