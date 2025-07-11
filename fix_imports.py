#!/usr/bin/env python3
"""
修复导入问题的脚本
"""

import os
import sys

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 创建一个简单的utils模块
utils_dir = os.path.join(current_dir, 'utils')
if not os.path.exists(utils_dir):
    os.makedirs(utils_dir)

# 创建简单的LOGGER
logger_content = '''
import logging

# 创建简单的logger
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建格式器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 添加处理器到logger
if not LOGGER.handlers:
    LOGGER.addHandler(console_handler)
'''

with open(os.path.join(utils_dir, '__init__.py'), 'w') as f:
    f.write(logger_content)

# 创建简单的ops模块
ops_content = '''
import numpy as np

def xywh2ltwh(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (left, top, width, height) format.
    
    Args:
        x (np.ndarray): Input bounding box coordinates in (x, y, width, height) format.
        
    Returns:
        np.ndarray: Bounding box coordinates in (left, top, width, height) format.
    """
    y = x.clone() if isinstance(x, type(x)) and hasattr(x, 'clone') else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # left = center_x - width/2
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top = center_y - height/2
    return y
'''

with open(os.path.join(utils_dir, 'ops.py'), 'w') as f:
    f.write(ops_content)

print("✅ 导入修复完成!")
print("现在可以运行: python TestEvaluate.sh")
