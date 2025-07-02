# 📁 Dataset Directory

## 📖 目录说明

此目录用于存放YOLO训练数据集，包含图片和对应的标注文件。

## 📂 目录结构

```
dataset/
├── README.md           # 本说明文件
├── images/             # 图片目录
│   ├── .gitkeep       # 保持目录结构
│   ├── train_001.jpg  # 训练图片（被.gitignore忽略）
│   ├── train_002.jpg  # 训练图片（被.gitignore忽略）
│   └── ...
└── labels/             # 标签目录
    ├── .gitkeep       # 保持目录结构
    ├── train_001.txt  # YOLO格式标注（被.gitignore忽略）
    ├── train_002.txt  # YOLO格式标注（被.gitignore忽略）
    └── ...
```

## 🚀 使用方法

### 1️⃣ 复制数据集
```bash
# 从ultralytics目录复制数据集
cp -r ultralytics-20240309/dataset .

# 或者手动创建数据集
mkdir -p dataset/images dataset/labels
```

### 2️⃣ 数据格式要求

**图片格式：**
- 支持：`.jpg`, `.png`, `.jpeg`, `.bmp`, `.tiff`
- 推荐：`.jpg` 格式，减少存储空间

**标注格式（YOLO）：**
```
class_id center_x center_y width height
```
- `class_id`: 类别ID（0开始）
- `center_x, center_y`: 目标中心点坐标（归一化0-1）
- `width, height`: 目标宽高（归一化0-1）

### 3️⃣ 配置文件
确保 `data/uav.yaml` 中的路径指向此目录：
```yaml
path: dataset
train: images
val: images
```

## ⚠️ 重要说明

- 📁 **目录结构保留**: 使用`.gitkeep`文件保持空目录在GitHub上可见
- 🚫 **文件被忽略**: 图片和标注文件被`.gitignore`忽略，不会上传到GitHub
- 💾 **本地使用**: 数据集文件只在本地存在，用于训练和测试
- 🔄 **数据同步**: 团队成员需要单独准备数据集文件

## 📊 数据集统计

训练完成后，可以在此记录数据集信息：
- 图片总数：___ 张
- 训练集：___ 张
- 验证集：___ 张
- 类别数量：___ 个
- 标注框总数：___ 个
