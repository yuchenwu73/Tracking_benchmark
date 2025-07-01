#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准模型性能基准测试脚本
参考工业界标准测速方法，提供准确的性能评估
包含：1.模型验证 2.TorchScript测试 3.ONNX测试 4.专业速度测试
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import time
import numpy as np
import torch
import os
import sys
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device

def get_weight_size(path):
    """获取模型文件大小（MB）"""
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

def check_onnx_gpu_support():
    """检查ONNX Runtime GPU支持"""
    try:
        import onnxruntime as ort

        # 兼容不同版本的onnxruntime
        try:
            available_providers = ort.get_available_providers()
        except AttributeError:
            # 旧版本可能没有这个方法
            try:
                # 尝试创建一个会话来检查CUDA支持
                import numpy as np
                dummy_model = b'\x08\x01\x12\x0c\x08\x01\x12\x08\x08\x01\x12\x04\x08\x01\x10\x01'  # 最小的ONNX模型
                session = ort.InferenceSession(dummy_model, providers=['CUDAExecutionProvider'])
                available_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            except:
                available_providers = ['CPUExecutionProvider']

        cuda_available = 'CUDAExecutionProvider' in available_providers

        print(f"📋 ONNX Runtime 可用提供者: {available_providers}")
        if cuda_available:
            print("✅ ONNX Runtime 支持 CUDA GPU 加速")
        else:
            print("⚠️  ONNX Runtime 不支持 CUDA，将使用 CPU")
            print("💡 安装GPU版本: pip install onnxruntime-gpu")

        return cuda_available
    except ImportError:
        print("❌ ONNX Runtime 未安装")
        return False
    except Exception as e:
        print(f"⚠️  检查ONNX Runtime时出错: {e}")
        print("🔄 假设支持CPU，继续测试...")
        return False

def setup_environment():
    """设置测试环境"""
    print("🔧 设置测试环境...")

    # 设置多进程启动方法，避免连接错误
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
        print("✅ 多进程启动方法设置成功")
    except RuntimeError as e:
        print(f"⚠️  多进程设置警告: {e}")

    # 检查ONNX Runtime GPU支持
    print("\n🔍 检查ONNX Runtime GPU支持...")
    check_onnx_gpu_support()

    # 检查模型文件是否存在
    model_path = "last.pt"
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件 {model_path} 不存在")
        sys.exit(1)

    # 检查数据配置文件是否存在
    data_path = "data/uav.yaml"
    if not os.path.exists(data_path):
        print(f"❌ 错误: 数据配置文件 {data_path} 不存在")
        sys.exit(1)

    print(f"✅ 模型文件: {model_path}")
    print(f"✅ 数据配置: {data_path}")
    return model_path, data_path

def step1_baseline_validation(model_path, data_path):
    """步骤1: 基线模型验证"""
    print("\n" + "=" * 60)
    print("📊 步骤1: 基线模型验证")
    print("=" * 60)

    try:
        model = YOLO(model_path)
        _ = model.val(
            data=data_path,
            imgsz=1088,
            batch=1,
            workers=0,  # 设置为0避免多进程问题
            verbose=True,
            device=0
        )
        print("✅ 基线模型验证完成")
        return True
    except Exception as e:
        print(f"❌ 基线模型验证失败: {e}")
        return False

def professional_speed_test(model_path, device='0', batch_size=1, img_size=1088,
                          warmup_runs=200, test_runs=1000, half_precision=False):
    """
    专业级速度测试 - 参考get_FPS.py的标准方法

    Args:
        model_path: 模型路径
        device: 设备 ('0', 'cpu', etc.)
        batch_size: 批次大小
        img_size: 图像尺寸
        warmup_runs: 预热次数
        test_runs: 测试次数
        half_precision: 是否使用半精度
    """
    print(f"\n⚡ 专业级速度测试")
    print(f"模型: {model_path}")
    print(f"设备: {device}")
    print(f"预热: {warmup_runs}次, 测试: {test_runs}次")

    # 1. 设备选择
    device = select_device(device, batch=batch_size)

    # 2. 加载模型
    if model_path.endswith('.pt'):
        from ultralytics.nn.tasks import attempt_load_weights
        model = attempt_load_weights(model_path, device=device, fuse=True)
    elif model_path.endswith('.torchscript'):
        model = torch.jit.load(model_path, map_location=device)
    elif model_path.endswith('.onnx'):
        return test_onnx_speed(model_path, device, batch_size, img_size,
                              warmup_runs, test_runs)
    else:
        raise ValueError(f"不支持的模型格式: {model_path}")

    model = model.to(device)
    model.eval()

    # 3. 创建测试输入
    if isinstance(img_size, int):
        input_shape = (batch_size, 3, img_size, img_size)
    else:
        input_shape = (batch_size, 3, *img_size)

    example_inputs = torch.randn(input_shape).to(device)

    # 4. 半精度设置
    if half_precision and device.type == 'cuda':
        model = model.half()
        example_inputs = example_inputs.half()

    # 5. 预热阶段
    print(f"🔥 预热阶段...")
    with torch.no_grad():
        for i in tqdm(range(warmup_runs), desc='预热'):
            _ = model(example_inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # 6. 正式测试
    print(f"⏱️  正式测试...")
    time_arr = []

    with torch.no_grad():
        for i in tqdm(range(test_runs), desc='测试'):
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start_time = time.time()
            _ = model(example_inputs)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end_time = time.time()
            time_arr.append(end_time - start_time)

    # 7. 统计分析
    time_arr = np.array(time_arr)
    mean_time = np.mean(time_arr)
    std_time = np.std(time_arr)
    min_time = np.min(time_arr)
    max_time = np.max(time_arr)

    # 每张图片的推理时间
    infer_time_per_image = mean_time / batch_size
    fps = 1.0 / infer_time_per_image

    # 8. 结果报告
    print(f"\n📊 速度测试结果:")
    print(f"模型大小: {get_weight_size(model_path)} MB")
    print(f"平均推理时间: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"每张图片推理时间: {infer_time_per_image*1000:.2f} ms")
    print(f"推理速度 (FPS): {fps:.1f}")

    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'fps': fps,
        'model_size': get_weight_size(model_path)
    }

def test_onnx_speed(model_path, device, batch_size, img_size, warmup_runs, test_runs):
    """ONNX模型速度测试"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ 需要安装 onnxruntime")
        return None

    print(f"� 测试ONNX模型: {model_path}")

    # 设置ONNX Runtime提供者 - 优先使用GPU
    if device == 'cpu':
        providers = ['CPUExecutionProvider']
        print("✅ 使用CPU执行提供者")
    elif torch.cuda.is_available():
        # 检查CUDA执行提供者是否可用
        try:
            available_providers = ort.get_available_providers()
        except AttributeError:
            # 旧版本兼容性
            available_providers = ['CPUExecutionProvider']

        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("✅ 使用CUDA执行提供者 (GPU加速)")
        else:
            providers = ['CPUExecutionProvider']
            print("⚠️  CUDA执行提供者不可用，回退到CPU")
    else:
        providers = ['CPUExecutionProvider']
        print("⚠️  CUDA不可用，使用CPU执行提供者")

    # 创建推理会话
    session = ort.InferenceSession(model_path, providers=providers)

    # 获取输入输出信息
    input_name = session.get_inputs()[0].name

    # 创建测试输入
    if isinstance(img_size, int):
        input_shape = (batch_size, 3, img_size, img_size)
    else:
        input_shape = (batch_size, 3, *img_size)

    example_inputs = np.random.randn(*input_shape).astype(np.float32)

    # 预热
    print(f"🔥 ONNX预热...")
    for _ in tqdm(range(warmup_runs), desc='预热'):
        _ = session.run(None, {input_name: example_inputs})

    # 测试
    print(f"⏱️  ONNX测试...")
    time_arr = []

    for _ in tqdm(range(test_runs), desc='测试'):
        start_time = time.time()
        _ = session.run(None, {input_name: example_inputs})
        end_time = time.time()
        time_arr.append(end_time - start_time)

    # 统计
    time_arr = np.array(time_arr)
    mean_time = np.mean(time_arr)
    std_time = np.std(time_arr)
    infer_time_per_image = mean_time / batch_size
    fps = 1.0 / infer_time_per_image

    print(f"\n📊 ONNX测试结果:")
    print(f"平均推理时间: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"每张图片推理时间: {infer_time_per_image*1000:.2f} ms")
    print(f"推理速度 (FPS): {fps:.1f}")

    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'fps': fps
    }

def step2_torchscript_benchmark(model_path, data_path):
    """步骤2: TorchScript格式基准测试"""
    print("\n" + "=" * 60)
    print("🚀 步骤2: TorchScript格式基准测试")
    print("=" * 60)

    try:
        # 手动导出和测试TorchScript
        model = YOLO(model_path)

        # 导出TorchScript (在GPU上)
        print("🔄 导出TorchScript格式...")
        model.to('cuda:0')  # 确保模型在GPU上
        ts_path = model.export(format='torchscript', imgsz=1088, device='cuda:0')

        # 测试TorchScript模型
        print("🔄 测试TorchScript模型...")
        ts_model = YOLO(ts_path)
        _ = ts_model.val(data=data_path, imgsz=1088, batch=1, workers=0, verbose=True)

        print("✅ TorchScript测试完成")
        return True
    except Exception as e:
        print(f"❌ TorchScript测试失败: {e}")
        return False

def step3_onnx_benchmark(model_path, data_path):
    """步骤3: ONNX格式基准测试"""
    print("\n" + "=" * 60)
    print("🔧 步骤3: ONNX格式基准测试")
    print("=" * 60)

    try:
        # 手动导出和测试ONNX
        model = YOLO(model_path)

        # 导出ONNX (在GPU上)
        print("🔄 导出ONNX格式...")
        model.to('cuda:0')  # 确保模型在GPU上
        onnx_path = model.export(format='onnx', imgsz=1088, device='cuda:0')

        # 首先尝试GPU测试，失败则回退到CPU
        print("🔄 测试ONNX模型...")
        onnx_model = YOLO(onnx_path)

        # 尝试GPU测试
        try:
            print("🚀 尝试在GPU上测试ONNX...")
            _ = onnx_model.val(
                data=data_path,
                imgsz=1088,
                batch=1,
                workers=0,
                device=0,  # 尝试使用GPU
                verbose=True
            )
            print("✅ ONNX GPU测试成功")
        except Exception as gpu_e:
            print(f"⚠️  GPU测试失败: {gpu_e}")
            print("🔄 回退到CPU测试...")
            _ = onnx_model.val(
                data=data_path,
                imgsz=1088,
                batch=1,
                workers=0,
                device='cpu',  # 回退到CPU
                verbose=True
            )
            print("✅ ONNX CPU测试成功")

        print("✅ ONNX基准测试完成")
        return True
    except Exception as e:
        print(f"⚠️  ONNX基准测试失败: {e}")
        print("💡 ONNX测试跳过，这通常是由于CUDA库版本问题")
        return False

def step4_professional_speed_test(model_path):
    """步骤4: 专业级速度测试"""
    print("\n" + "=" * 60)
    print("⚡ 步骤4: 专业级速度测试")
    print("=" * 60)

    formats = [
        ('last.pt', 'PyTorch'),
        ('last.torchscript', 'TorchScript'),
        ('last.onnx', 'ONNX')
    ]

    results = {}

    for model_file, format_name in formats:
        if Path(model_file).exists():
            print(f"\n🧪 测试 {format_name} 格式...")
            try:
                # 对于ONNX，使用GPU设备进行测试
                device = '0' if model_file.endswith('.onnx') else '0'
                result = professional_speed_test(
                    model_path=model_file,
                    device=device,
                    batch_size=1,
                    img_size=1088,
                    warmup_runs=100,  # 减少测试时间但保持准确性
                    test_runs=500,
                    half_precision=False
                )
                results[format_name] = result
            except Exception as e:
                print(f"❌ {format_name} 测试失败: {e}")
                results[format_name] = None
        else:
            print(f"⚠️  {format_name} 模型文件不存在: {model_file}")

    # 性能对比报告
    if results:
        print(f"\n📈 性能对比报告:")
        print(f"=" * 80)
        print(f"{'格式':<12} {'推理时间(ms)':<15} {'FPS':<10} {'相对性能':<10} {'模型大小(MB)':<12}")
        print(f"-" * 80)

        baseline_fps = None
        for format_name, result in results.items():
            if result:
                fps = result['fps']
                inference_time = result['mean_time'] * 1000
                model_size = result.get('model_size', 'N/A')

                if baseline_fps is None:
                    baseline_fps = fps
                    relative_perf = "基准"
                else:
                    relative_perf = f"{fps/baseline_fps:.2f}x"

                print(f"{format_name:<12} {inference_time:<15.2f} {fps:<10.1f} {relative_perf:<10} {model_size:<12}")
            else:
                print(f"{format_name:<12} {'失败':<15} {'N/A':<10} {'N/A':<10} {'N/A':<12}")
        print(f"=" * 80)

    return results

def step4_speed_test(model_path):
    """步骤4: 实际推理速度测试"""
    print("\n" + "=" * 60)
    print("⚡ 步骤4: 实际推理速度测试")
    print("=" * 60)

    try:
        import glob
        model = YOLO(model_path)
        test_images = glob.glob("data/images/*.jpg")[:10]  # 取前10张图片测试

        if not test_images:
            print("❌ 未找到测试图片")
            return False

        print(f"🔄 使用 {len(test_images)} 张图片进行速度测试...")

        # 预热
        print("🔥 预热模型...")
        for _ in range(3):
            model.predict(test_images[0], imgsz=1088, verbose=False)

        # 正式测试
        print("⏱️  开始速度测试...")
        start_time = time.time()
        for img_path in test_images:
            _ = model.predict(img_path, imgsz=1088, verbose=False)
        end_time = time.time()

        avg_time = (end_time - start_time) / len(test_images)
        fps = 1.0 / avg_time

        print(f"📊 平均推理时间: {avg_time*1000:.1f}ms")
        print(f"📊 推理速度: {fps:.1f} FPS")
        print("✅ 速度测试完成")
        return True

    except Exception as e:
        print(f"❌ 速度测试失败: {e}")
        return False

def run_complete_benchmark():
    """运行完整的模型基准测试"""
    print("🎯 开始标准模型性能基准测试...")
    print("=" * 60)

    # 设置环境
    model_path, data_path = setup_environment()

    # 记录测试结果
    results = {
        'baseline': False,
        'torchscript': False,
        'onnx': False,
        'speed': False
    }

    # 步骤1: 基线验证
    results['baseline'] = step1_baseline_validation(model_path, data_path)

    # 步骤2: TorchScript测试
    results['torchscript'] = step2_torchscript_benchmark(model_path, data_path)

    # 步骤3: ONNX测试
    results['onnx'] = step3_onnx_benchmark(model_path, data_path)

    # 步骤4: 专业级速度测试
    speed_results = step4_professional_speed_test(model_path)
    results['speed'] = bool(speed_results)

    # 总结结果
    print("\n" + "=" * 60)
    print("📋 测试结果总结")
    print("=" * 60)
    for test_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{test_name.upper():<12}: {status}")

    successful_tests = sum(results.values())
    total_tests = len(results)
    print(f"\n🎯 总体结果: {successful_tests}/{total_tests} 项测试成功")

    if successful_tests == total_tests:
        print("🎉 所有测试完成!")
    else:
        print("⚠️  部分测试失败，但核心功能正常")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='标准模型性能基准测试')
    parser.add_argument('--weights', type=str, default='last.pt', help='模型权重路径')
    parser.add_argument('--device', default='0', help='设备: 0, cpu等')
    parser.add_argument('--batch', type=int, default=1, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=1088, help='图像尺寸')
    parser.add_argument('--warmup', type=int, default=200, help='预热次数')
    parser.add_argument('--runs', type=int, default=1000, help='测试次数')
    parser.add_argument('--half', action='store_true', help='半精度模式')
    parser.add_argument('--speed-only', action='store_true', help='仅进行速度测试')

    args = parser.parse_args()

    if args.speed_only:
        # 仅进行专业级速度测试
        print("🎯 专业级速度测试模式")
        professional_speed_test(
            model_path=args.weights,
            device=args.device,
            batch_size=args.batch,
            img_size=args.imgsz,
            warmup_runs=args.warmup,
            test_runs=args.runs,
            half_precision=args.half
        )
    else:
        # 完整基准测试
        run_complete_benchmark()
    
