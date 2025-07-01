#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集路径更新工具 - 适用于Linux系统
更新train.txt和val.txt中的图像路径
"""

import os
import re

def update_dataset_paths(txt_file):
    """
    更新YOLO数据集txt文件中的图像路径
    
    Args:
        txt_file (str): 要更新的txt文件路径
    """
    print(f"正在更新文件: {txt_file}")

    # 读取原始文件
    with open(txt_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    updated_lines = []
    updated_count = 0

    for line in lines:
        original_path = line.strip()
        if not original_path:  # 跳过空行
            continue

        # 使用正则表达式提取文件名
        # 匹配形如 frame_xxxx_sub_xx.jpg 的文件名
        match = re.search(r'(frame_\d+_sub_\d+\.jpg)$', original_path)

        if match:
            filename = match.group(1)
            # 构建更新后的路径
            new_path = f"data/images/{filename}"

            # 验证文件是否存在
            if os.path.exists(new_path):
                updated_lines.append(new_path)
                if original_path != new_path:
                    updated_count += 1
            else:
                print(f"警告: 文件不存在 - {new_path}")
                updated_lines.append(new_path)  # 仍然添加，但给出警告
        else:
            print(f"警告: 无法解析路径格式 - {original_path}")
            updated_lines.append(original_path)  # 保持原样

    # 写回更新后的路径
    with open(txt_file, 'w', encoding='utf-8') as file:
        for path in updated_lines:
            file.write(path + '\n')

    print(f"更新完成: {txt_file}")
    print(f"总行数: {len(lines)}, 更新行数: {updated_count}")
    print(f"最终有效路径数: {len(updated_lines)}")
    return updated_count

def verify_paths(txt_file):
    """
    验证txt文件中的路径是否都存在

    Args:
        txt_file (str): 要验证的txt文件路径
    """
    print(f"\n验证文件: {txt_file}")

    with open(txt_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    missing_files = []
    existing_files = 0

    for line in lines:
        path = line.strip()
        if not path:
            continue

        if os.path.exists(path):
            existing_files += 1
        else:
            missing_files.append(path)

    print(f"存在的文件: {existing_files}")
    print(f"缺失的文件: {len(missing_files)}")

    if missing_files:
        print("缺失的文件列表（前10个）:")
        for missing_file in missing_files[:10]:
            print(f"  - {missing_file}")
        if len(missing_files) > 10:
            print(f"  ... 还有 {len(missing_files) - 10} 个文件")

    return len(missing_files) == 0

def main():
    """主函数"""
    print("=" * 60)
    print("数据集路径更新工具")
    print("=" * 60)

    # 检查当前工作目录
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")

    # 定义文件路径
    train_file = 'data/train.txt'
    val_file = 'data/val.txt'

    # 检查文件是否存在
    for file_path in [train_file, val_file]:
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 - {file_path}")
            return

    # 检查data/images目录是否存在
    images_dir = 'data/images'
    if not os.path.exists(images_dir):
        print(f"错误: 图像目录不存在 - {images_dir}")
        return

    print(f"图像目录存在: {images_dir}")

    # 更新路径
    print("\n开始更新路径...")
    train_updated = update_dataset_paths(train_file)
    val_updated = update_dataset_paths(val_file)

    print(f"\n更新总结:")
    print(f"训练集更新: {train_updated} 行")
    print(f"验证集更新: {val_updated} 行")

    # 验证更新结果
    print("\n验证更新结果...")
    train_ok = verify_paths(train_file)
    val_ok = verify_paths(val_file)

    if train_ok and val_ok:
        print("\n✅ 所有路径更新成功！")
    else:
        print("\n⚠️  部分路径可能仍有问题，请检查上述警告信息")

    print("\n路径更新完成！")

if __name__ == "__main__":
    main()
