"""
将3DB BEDLAM2特征文件转换到新的目录结构

该脚本基于smplpose_v3_3db.pt文件，提取对应的3DB特征并保存到新目录。
原始文件包含: {'feature', 'lh_feature', 'rh_feature'}
新文件只包含: feature tensor (T, 1024)

使用方法:
    # 转换完整数据集
    python scripts/utils/convert_3db_features.py --smplpose /path/to/smplpose_v3_3db.pt
    
    # 转换debug版本（1024样本）
    python scripts/utils/convert_3db_features.py --smplpose /path/to/smplpose_v3_3db_random1024.pt
"""

from pathlib import Path
import torch
from tqdm import tqdm
from time import time
import numpy as np

import sys
sys.path.append(".")

from core.utils.pylogger import Log


def convert_3db_feature_file(src_path, dst_path):
    """
    转换单个3DB特征文件
    
    Args:
        src_path: 源文件路径 (包含 {'feature', 'lh_feature', 'rh_feature'})
        dst_path: 目标文件路径 (只保存 feature tensor)
    
    Returns:
        feature_shape: 转换后的feature形状
    """
    # 加载原始文件
    data = torch.load(src_path)
    
    # 提取feature字段
    if isinstance(data, dict) and 'feature' in data:
        feature = data['feature']
    else:
        raise ValueError(f"Expected dict with 'feature' key, got {type(data)} with keys {data.keys() if isinstance(data, dict) else 'N/A'}")
    
    # 确保目标目录存在
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存feature tensor
    torch.save(feature, dst_path)
    
    return feature.shape


def convert_3db_features(
    smplpose_file,
    src_dir="/animation-cfs-nj/gaigeying/dataset_opensource/BEDLAM2/3DB_COS/sam3db_pose_feat",
    dst_dir="/animation-cfs-nj/gaigeying/dataset_opensource/wham_parsed_data/ft_local/BEDLAM2_3DB/imgfeats/3db",
):
    """
    批量转换3DB特征文件，基于smplpose_v3文件确定要转换哪些文件
    
    Args:
        smplpose_file: smplpose_v3_3db.pt 或 smplpose_v3_3db_random1024.pt 文件路径
        src_dir: 源目录 (sam3db_pose_feat)
        dst_dir: 目标目录 (imgfeats/3db)
    """
    src_root = Path(src_dir)
    dst_root = Path(dst_dir)
    
    Log.info(f"[3DB-Feature-Convert] Starting conversion")
    Log.info(f"[3DB-Feature-Convert] Reference: {smplpose_file}")
    Log.info(f"[3DB-Feature-Convert] Source: {src_root}")
    Log.info(f"[3DB-Feature-Convert] Destination: {dst_root}")
    tic = time()
    
    # 加载smplpose文件，获取需要转换的文件列表
    Log.info(f"[3DB-Feature-Convert] Loading smplpose file...")
    smplpose_data = torch.load(smplpose_file)
    Log.info(f"[3DB-Feature-Convert] Found {len(smplpose_data)} sequences in smplpose file")
    
    # 从键名中提取文件路径: 'imgfeats/3db/folder/file.pt' -> 'folder/file.pt'
    files_to_convert = []
    for key in smplpose_data.keys():
        if key.startswith('imgfeats/3db/'):
            rel_path = key.replace('imgfeats/3db/', '')
            src_file = src_root / rel_path
            if src_file.exists():
                files_to_convert.append((src_file, rel_path))
            else:
                Log.warn(f"[3DB-Feature-Convert] Source file not found: {src_file}")
    
    Log.info(f"[3DB-Feature-Convert] Will convert {len(files_to_convert)} files")
    
    # 转换文件
    success_count = 0
    error_files = []
    
    for src_file, rel_path in tqdm(files_to_convert, desc="Converting features"):
        try:
            dst_file = dst_root / rel_path
            convert_3db_feature_file(src_file, dst_file)
            success_count += 1
        except Exception as e:
            error_files.append((str(rel_path), str(e)))
            Log.error(f"Error converting {rel_path}: {e}")
    
    # 报告统计信息
    elapsed = time() - tic
    Log.info(f"[3DB-Feature-Convert] Conversion completed in {elapsed:.2f}s")
    Log.info(f"[3DB-Feature-Convert] Successfully converted: {success_count}/{len(files_to_convert)} files")
    
    if error_files:
        Log.warn(f"[3DB-Feature-Convert] Failed: {len(error_files)} files")
        for rel_path, error in error_files[:5]:
            Log.warn(f"  {rel_path}: {error}")
    
    # 验证：随机检查几个文件
    if success_count > 0:
        Log.info(f"[3DB-Feature-Convert] Verifying converted files...")
        sample_files = list(dst_root.glob("**/*.pt"))[:3]
        for sample_file in sample_files:
            feature = torch.load(sample_file)
            rel_path = sample_file.relative_to(dst_root)
            Log.info(f"  {rel_path}: shape={feature.shape}, dtype={feature.dtype}")
    
    Log.info(f"[3DB-Feature-Convert] Done!")

if __name__ == "__main__":
    """
    使用示例:
        # 基于完整数据集的smplpose文件转换
        python scripts/utils/convert_3db_features.py --smplpose /path/to/smplpose_v3_3db.pt
        
        # 基于debug版本的smplpose文件转换（1024样本）
        python scripts/utils/convert_3db_features.py --smplpose /path/to/smplpose_v3_3db_random1024.pt
        
        # 自定义源目录和目标目录
        python scripts/utils/convert_3db_features.py --smplpose /path/to/smplpose.pt --src_dir /path/to/src --dst_dir /path/to/dst
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert 3DB BEDLAM2 feature files based on smplpose_v3 file")
    parser.add_argument(
        "--smplpose",
        type=str,
        required=True,
        help="Path to smplpose_v3_3db.pt or smplpose_v3_3db_random1024.pt file"
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default="/animation-cfs-nj/gaigeying/dataset_opensource/BEDLAM2/3DB_COS/sam3db_pose_feat",
        help="Source directory containing original 3DB features"
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="/animation-cfs-nj/gaigeying/dataset_opensource/wham_parsed_data/ft_local/BEDLAM2_3DB/imgfeats/3db",
        help="Destination directory for converted features"
    )
    
    args = parser.parse_args()
    
    convert_3db_features(
        smplpose_file=args.smplpose,
        src_dir=args.src_dir,
        dst_dir=args.dst_dir
    )
