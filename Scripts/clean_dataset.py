#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理数据集目录脚本 - 保留.gitkeep文件
运行命令: python Scripts/clean_dataset.py
"""

import os
import shutil
from pathlib import Path
import argparse

def clean_directory_keep_gitkeep(directory, dry_run=False):
    """
    删除目录内容但保留.gitkeep文件
    
    参数:
        directory: 要清理的目录路径
        dry_run: 是否只显示将要删除的文件，不实际删除
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"目录不存在: {directory}")
        return
    
    if not dir_path.is_dir():
        print(f"路径不是目录: {directory}")
        return
    
    deleted_files = 0
    deleted_dirs = 0
    
    print(f"{'[预览模式] ' if dry_run else ''}清理目录: {directory}")
    
    for item in dir_path.iterdir():
        if item.name == '.gitkeep':
            print(f"  保留: {item.name}")
            continue
        
        if dry_run:
            if item.is_file():
                print(f"  将删除文件: {item.name}")
                deleted_files += 1
            elif item.is_dir():
                print(f"  将删除目录: {item.name}/")
                deleted_dirs += 1
        else:
            try:
                if item.is_file():
                    item.unlink()
                    print(f"  ✅ 删除文件: {item.name}")
                    deleted_files += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    print(f"  ✅ 删除目录: {item.name}/")
                    deleted_dirs += 1
            except Exception as e:
                print(f"  ❌ 删除失败 {item.name}: {e}")
    
    print(f"\n{'预计' if dry_run else '实际'}删除: {deleted_files} 个文件, {deleted_dirs} 个目录")
    if not dry_run:
        print("✅ 清理完成，.gitkeep 文件已保留")

def clean_multiple_directories(directories, dry_run=False):
    """清理多个目录"""
    for directory in directories:
        clean_directory_keep_gitkeep(directory, dry_run)
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='清理数据集目录，保留.gitkeep文件')
    parser.add_argument('directories', nargs='*', default=['dataset'],
                       help='要清理的目录列表（默认: dataset）')
    parser.add_argument('--dry-run', action='store_true',
                       help='预览模式，只显示将要删除的文件，不实际删除')
    parser.add_argument('--all', action='store_true',
                       help='清理所有常见的数据目录')
    
    args = parser.parse_args()
    
    if args.all:
        # 清理所有常见的数据目录
        common_dirs = [
            'dataset',
            'dataset/images/train',
            'dataset/images/val', 
            'dataset/images/test',
            'dataset/labels/train',
            'dataset/labels/val',
            'dataset/VOCdevkit/JPEGImages',
            'dataset/VOCdevkit/txt',
            'runs/train',
            'runs/val',
            'runs/detect'
        ]
        # 只清理存在的目录
        existing_dirs = [d for d in common_dirs if Path(d).exists()]
        if existing_dirs:
            print("将清理以下目录:")
            for d in existing_dirs:
                print(f"  - {d}")
            
            if not args.dry_run:
                confirm = input("\n确认清理这些目录? (y/N): ")
                if confirm.lower() != 'y':
                    print("操作已取消")
                    return
            
            clean_multiple_directories(existing_dirs, args.dry_run)
        else:
            print("没有找到需要清理的目录")
    else:
        clean_multiple_directories(args.directories, args.dry_run)

if __name__ == "__main__":
    main()
