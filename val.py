# from ultralytics.utils.benchmarks import benchmark
from ultralytics import YOLO

if __name__ == '__main__':
    # Benchmark on GPU
    # benchmark(model="last.pt", data="data/uav.yaml", imgsz=1088, half=False, device=0,verbose=True)
    
    # # Validate on CPU
    # model = YOLO("last.pt")
    # model.val(data="data/uav.yaml", imgsz=1088, verbose=True, device="cpu")
    
    # Validate on GPU
    model = YOLO('/data2/wuyuchen/Tracking_benchmark/runs/train/yolo113/weights/best.pt')
    model.val(data='dataset/data.yaml',
              imgsz=1088,
              batch=16,
              verbose = True,
              project='runs/val',
              name='exp'
              )