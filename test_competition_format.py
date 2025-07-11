# 运行命令示例：
# python test_competition_format.py --results_dir results

import os
import argparse
import zipfile
import re

def validate_txt_format(file_path):
    """
    验证txt文件格式是否符合比赛要求
    
    参数:
        file_path: txt文件路径
    
    返回:
        (is_valid, error_messages)
    """
    errors = []
    line_count = 0
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                line_count += 1
                
                # 检查字段数量
                fields = line.split(',')
                if len(fields) != 10:
                    errors.append(f"第{line_num}行: 字段数量错误，期望10个，实际{len(fields)}个")
                    continue
                
                try:
                    # 验证数据类型
                    frame_id = int(fields[0])
                    track_id = int(fields[1])
                    x_left = float(fields[2])
                    y_top = float(fields[3])
                    width = float(fields[4])
                    height = float(fields[5])
                    cls = int(fields[6])
                    field8 = fields[7]  # 第8个字段，必须是-1
                    field9 = fields[8]  # 第9个字段，必须是-1
                    field10 = fields[9]  # 第10个字段，必须是-1
                    
                    # 检查数值范围
                    if frame_id < 0:
                        errors.append(f"第{line_num}行: 帧号不能为负数")
                    if track_id <= 0:
                        errors.append(f"第{line_num}行: 目标ID必须为正数")
                    if width <= 0 or height <= 0:
                        errors.append(f"第{line_num}行: 边界框宽高必须为正数")
                    if cls != 1:
                        errors.append(f"第{line_num}行: 目标类别必须为1")
                    if field8 != "-1":
                        errors.append(f"第{line_num}行: 第8个字段必须为-1")
                    if field9 != "-1" or field10 != "-1":
                        errors.append(f"第{line_num}行: 第9、10个字段必须为-1")
                        
                except ValueError as e:
                    errors.append(f"第{line_num}行: 数据类型错误 - {str(e)}")
                    
    except Exception as e:
        errors.append(f"文件读取错误: {str(e)}")
    
    return len(errors) == 0, errors, line_count

def validate_zip_structure(zip_path):
    """
    验证压缩包结构是否符合要求
    
    参数:
        zip_path: 压缩包路径
    
    返回:
        (is_valid, error_messages, file_list)
    """
    errors = []
    file_list = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            file_list = zipf.namelist()
            
            # 检查是否有results目录
            has_results_dir = any(name.startswith('results/') for name in file_list)
            if not has_results_dir:
                errors.append("压缩包中缺少results/目录")
            
            # 检查txt文件
            txt_files = [name for name in file_list if name.endswith('.txt') and name.startswith('results/')]
            if not txt_files:
                errors.append("压缩包中没有找到txt文件")
            
            # 检查文件命名格式
            for txt_file in txt_files:
                filename = os.path.basename(txt_file)
                if not re.match(r'^[\w\-]+\.txt$', filename):
                    errors.append(f"文件名格式不规范: {filename}")
                    
    except Exception as e:
        errors.append(f"压缩包读取错误: {str(e)}")
    
    return len(errors) == 0, errors, file_list

def main():
    parser = argparse.ArgumentParser(description='验证比赛结果格式')
    parser.add_argument('--results_dir', type=str, default='results', help='结果目录路径')
    parser.add_argument('--zip_file', type=str, default='results.zip', help='压缩包路径')
    
    args = parser.parse_args()
    
    print("🔍 比赛结果格式验证工具")
    print("=" * 50)
    
    # 验证结果目录
    if os.path.exists(args.results_dir):
        print(f"\n📁 验证结果目录: {args.results_dir}")
        
        txt_files = [f for f in os.listdir(args.results_dir) if f.endswith('.txt')]
        if not txt_files:
            print("❌ 结果目录中没有找到txt文件")
        else:
            print(f"✅ 找到 {len(txt_files)} 个txt文件")
            
            total_lines = 0
            valid_files = 0
            
            for txt_file in txt_files:
                file_path = os.path.join(args.results_dir, txt_file)
                is_valid, errors, line_count = validate_txt_format(file_path)
                total_lines += line_count
                
                if is_valid:
                    print(f"  ✅ {txt_file}: {line_count} 条记录，格式正确")
                    valid_files += 1
                else:
                    print(f"  ❌ {txt_file}: 格式错误")
                    for error in errors[:5]:  # 只显示前5个错误
                        print(f"     - {error}")
                    if len(errors) > 5:
                        print(f"     - ... 还有 {len(errors) - 5} 个错误")
            
            print(f"\n📊 统计信息:")
            print(f"  - 有效文件: {valid_files}/{len(txt_files)}")
            print(f"  - 总跟踪记录: {total_lines}")
    else:
        print(f"❌ 结果目录不存在: {args.results_dir}")
    
    # 验证压缩包
    if os.path.exists(args.zip_file):
        print(f"\n📦 验证压缩包: {args.zip_file}")
        
        is_valid, errors, file_list = validate_zip_structure(args.zip_file)
        
        if is_valid:
            print("✅ 压缩包结构正确")
            txt_files_in_zip = [f for f in file_list if f.endswith('.txt')]
            print(f"  - 包含 {len(txt_files_in_zip)} 个txt文件")
            for txt_file in txt_files_in_zip:
                print(f"    • {txt_file}")
        else:
            print("❌ 压缩包结构错误:")
            for error in errors:
                print(f"  - {error}")
    else:
        print(f"❌ 压缩包不存在: {args.zip_file}")
    
    # 生成示例数据
    print(f"\n📝 生成示例数据格式:")
    print("=" * 30)
    print("# 正确的数据格式示例:")
    print("0,1,712.96,195.25,14.36,13.99,1,-1,-1,-1")
    print("0,2,997.47,437.26,7.26,6.31,1,-1,-1,-1")
    print("1,1,715.32,198.11,14.28,13.87,1,-1,-1,-1")
    print("1,3,284.91,792.55,13.09,15.51,1,-1,-1,-1")
    print()
    print("字段说明:")
    print("帧号,目标ID,左上角X,左上角Y,宽度,高度,类别(1),-1,-1,-1")
    
    # 检查常见错误
    print(f"\n⚠️ 常见错误检查:")
    print("=" * 30)
    
    common_issues = [
        "坐标格式错误（应为左上角坐标，不是中心坐标）",
        "目标类别不是1",
        "第8、9、10个字段不是-1",
        "字段数量不是10个",
        "压缩包缺少results/目录",
        "文件名格式不规范"
    ]
    
    for i, issue in enumerate(common_issues, 1):
        print(f"{i}. {issue}")
    
    print(f"\n✨ 验证完成！")
    print("如果发现错误，请根据提示修正后重新生成结果。")

if __name__ == "__main__":
    main()
