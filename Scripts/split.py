import os
import random
import shutil

IMG_DIR = 'images_yolo'
LABEL_DIR = 'labels_yolo'

OUT_BASE = 'mydata_yolo'
train_ratio = 0.8

img_list = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])
random.seed(42)
random.shuffle(img_list)

train_num = int(len(img_list) * train_ratio)
train_list = img_list[:train_num]
val_list = img_list[train_num:]

for split, file_list in [('train', train_list), ('val', val_list)]:
    os.makedirs(os.path.join(OUT_BASE, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUT_BASE, 'labels', split), exist_ok=True)
    for fname in file_list:
        label_name = fname.replace('.jpg', '.txt')
        label_path = os.path.join(LABEL_DIR, label_name)

        # 如果没有标签就跳过（不复制这张图片）
        if not os.path.exists(label_path):
            continue

        shutil.copy(os.path.join(IMG_DIR, fname), os.path.join(OUT_BASE, 'images', split, fname))
        shutil.copy(label_path, os.path.join(OUT_BASE, 'labels', split, label_name))
