#!/usr/bin/env python3
"""
简化版文件匹配脚本
直接用于你的 data1 和 data2 匹配分析
"""

import re
from typing import List, Tuple, Dict, Set


def extract_matching_key(file_path: str) -> str:
    """
    从文件路径提取匹配键
    例如: imgfeats/prhmr_b2nips/scene/filename.pt -> imgfeats/scene/base_filename
    """
    parts = file_path.split('/')
    if len(parts) < 4:
        return file_path
    
    # 提取第1级、第3级目录和文件名
    level1 = parts[0]  # imgfeats
    level3 = parts[2]  # 场景目录
    filename = parts[3]  # 文件名
    
    # 提取基础文件名（去除后缀）
    base_filename = filename.replace('.pt', '')
    
    # 去除可能的后缀模式
    # 模式1: _-a, _-b 等
    base_filename = re.sub(r'_-[a-zA-Z]+$', '', base_filename)
    
    # 模式2: _数字 (如 _2401)
    base_filename = re.sub(r'_\d+$', '', base_filename)
    
    return f"{level1}/{level3}/{base_filename}"


def quick_match_analysis(names1: List[str], names2: List[str]) -> None:
    """
    快速匹配分析并打印结果
    """
    print("开始分析文件匹配...")
    
    # 创建匹配键映射
    keys1_to_path = {}
    keys2_to_path = {}
    
    for path in names1:
        key = extract_matching_key(path)
        keys1_to_path[key] = path
    
    for path in names2:
        key = extract_matching_key(path)
        keys2_to_path[key] = path
    
    # 分析匹配情况
    keys1 = set(keys1_to_path.keys())
    keys2 = set(keys2_to_path.keys())
    
    matched = keys1 & keys2
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    
    # 打印结果
    print("\n" + "="*60)
    print("匹配分析结果")
    print("="*60)
    print(f"Data1 总文件数: {len(names1)}")
    print(f"Data2 总文件数: {len(names2)}")
    print(f"匹配的文件数: {len(matched)}")
    print(f"仅在 Data1 中: {len(only_in_1)}")
    print(f"仅在 Data2 中: {len(only_in_2)}")
    
    if len(names1) > 0:
        print(f"Data1 匹配率: {len(matched)/len(names1)*100:.2f}%")
    if len(names2) > 0:
        print(f"Data2 匹配率: {len(matched)/len(names2)*100:.2f}%")
    
    # 显示一些示例
    print(f"\n匹配示例 (前5个):")
    for i, key in enumerate(list(matched)[:5]):
        print(f"{i+1}. 匹配键: {key}")
        print(f"   Data1: {keys1_to_path[key]}")
        print(f"   Data2: {keys2_to_path[key]}")
        print()
    
    if only_in_1:
        print(f"仅在 Data1 中的示例 (前5个):")
        for i, key in enumerate(list(only_in_1)[:5]):
            print(f"{i+1}. {keys1_to_path[key]}")
    
    if only_in_2:
        print(f"\n仅在 Data2 中的示例 (前5个):")
        for i, key in enumerate(list(only_in_2)[:5]):
            print(f"{i+1}. {keys2_to_path[key]}")
    
    return {
        'matched_count': len(matched),
        'only_in_1_count': len(only_in_1),
        'only_in_2_count': len(only_in_2),
        'matched_pairs': [(keys1_to_path[key], keys2_to_path[key]) for key in matched],
        'only_in_1': [keys1_to_path[key] for key in only_in_1],
        'only_in_2': [keys2_to_path[key] for key in only_in_2]
    }


# 使用示例和测试
if __name__ == "__main__":
    # 测试数据
    test_names1 = [
        'imgfeats/prhmr_b2nips/20240416_1_171_yogastudio_orbit_timeofday/seq_000000-moyo_Boat_Pose_or_Paripurna_Navasana.pt',
        'imgfeats/prhmr_b2nips/20250219_3-4_250_middleeast_vcam_approach/seq_000249-us_2072_XS.pt'
    ]
    
    test_names2 = [
        'imgfeats/3db/20240416_1_171_yogastudio_orbit_timeofday/seq_000000-moyo_Boat_Pose_or_Paripurna_Navasana_-a.pt',
        'imgfeats/3db/20250219_3-4_250_middleeast_vcam_approach/seq_000249-us_2072_XS_2401.pt'
    ]
    
    print("测试匹配脚本...")
    results = quick_match_analysis(test_names1, test_names2)
    
    print("\n" + "="*60)
    print("使用方法:")
    print("="*60)
    print("# 在你的代码中使用:")
    print("names1 = list(data1.keys())")
    print("names2 = list(data2.keys())")
    print("results = quick_match_analysis(names1, names2)")
