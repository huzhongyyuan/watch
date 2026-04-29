"""
生成3DB BEDLAM2版本的smplpose_v3.pt文件

该脚本从本地3DB数据生成与HMR2版本格式兼容的smplpose_v3.pt文件。
主要功能：
1. 从seq_info/*.npz读取标签数据
2. 从sam3db_pose_feat/*.pt读取特征数据
3. 转换数据格式并生成skeleton
4. 保存为smplpose_v3.pt格式

使用方法：
    python scripts/utils/generate_3db_bedlam2_smplpose_v3.py
"""

from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from time import time

import sys
sys.path.append(".")

from core.utils.smplx_utils import make_smplx
from core.utils.pylogger import Log


def convert_seq_info_to_motion_dict(seq_info_path, smplx_model, device='cuda'):
    """
    将seq_info中的npz数据转换为motion_dict格式
    
    Args:
        seq_info_path: seq_info/*.npz文件路径
        smplx_model: SMPLX模型实例
        device: 计算设备
        
    Returns:
        motion_dict: 包含所有必要字段的字典，如果length<=1则返回None
    """
    data = np.load(seq_info_path, allow_pickle=True)
    
    length = data['pose_world'].shape[0]
    
    if length <= 1:
        return None
    
    motion_dict = {
        'poses_world': torch.from_numpy(data['pose_world']).float(),  # (T, 69)
        'trans_world': torch.from_numpy(data['trans_world']).float(),  # (T, 3)
        'trans_cam': torch.from_numpy(data['trans_cam']).float(),  # (T, 3)
        'cam_ext': torch.from_numpy(data['cam_ext']).float(),  # (T, 4, 4)
        'cam_int': torch.from_numpy(data['cam_int']).float(),  # (T, 3, 3)
        'global_orient_incam': torch.from_numpy(data['pose_cam'][:, :3]).float(),  # (T, 3)
        'betas': torch.from_numpy(data['shape'][0]).float(),  # (16,)
    }
    
    # 计算skeleton: 使用smplx模型和betas
    with torch.no_grad():
        betas_input = motion_dict['betas'][None].to(device)  # (1, 16)
        smplx_output = smplx_model(betas=betas_input)
        skeleton = smplx_output.joints[0].cpu()  # (127, 3)
    motion_dict['skeleton'] = skeleton
    
    # valid_start_end: (0, length)
    motion_dict['valid_start_end'] = [0, length]
    
    # bbx_xys: 从center和scale计算
    # center: (T, 2), scale: (T,) -> bbx_xys: (T, 3) [center_x, center_y, size]
    center = torch.from_numpy(data['center']).float()  # (T, 2)
    scale = torch.from_numpy(data['scale']).float()  # (T,)
    size = scale * 200  # 根据bedlam2.py中的处理方法
    motion_dict['bbx_xys'] = torch.cat([center, size.unsqueeze(-1)], dim=-1)  # (T, 3)
    
    # 保留valid_mask（如果存在）
    if 'valid_mask' in data:
        motion_dict['valid_mask'] = torch.from_numpy(data['valid_mask']).bool()
    
    return motion_dict


def generate_3db_smplpose_v3(
    root_dir="/animation-cfs-nj/gaigeying/dataset_opensource/BEDLAM2/3DB_COS",
    output_path=None,
    device='cuda',
    debug=False,
    seed=42
):
    """
    生成3DB版本的smplpose_v3.pt文件
    
    Args:
        root_dir: 3DB数据根目录
        output_path: 输出文件路径，默认为root_dir/smplpose_v3_3db.pt
        device: 计算设备
        debug: 是否启用debug模式（只处理1024个样本）
        seed: debug模式下的随机种子
    """
    root = Path(root_dir)
    seq_info_dir = root / "seq_info"
    feature_dir = root / "sam3db_pose_feat"
    
    if output_path is None:
        if debug:
            output_path = root / "smplpose_v3_3db_random1024.pt"
        else:
            output_path = root / "smplpose_v3_3db.pt"
    
    mode_str = "DEBUG (random1024)" if debug else "FULL"
    Log.info(f"[3DB-BEDLAM2] Starting generation from {root_dir} [{mode_str}]")
    Log.info(f"[3DB-BEDLAM2] Output will be saved to {output_path}")
    tic = time()
    
    # 初始化SMPLX模型（使用与BEDLAM2相同的配置）
    Log.info(f"[3DB-BEDLAM2] Initializing SMPLX model...")
    smplx_model = make_smplx('smplx-circle', gender='neutral').to(device)
    smplx_model.eval()
    
    # 扫描所有feature文件（以feature为准）
    Log.info(f"[3DB-BEDLAM2] Scanning feature files...")
    feature_files = sorted(list(feature_dir.glob("**/*.pt")))
    Log.info(f"[3DB-BEDLAM2] Found {len(feature_files)} feature files")
    
    if len(feature_files) == 0:
        raise ValueError(f"No feature files found in {feature_dir}")
    
    # Debug模式：随机采样1024个文件
    if debug:
        np.random.seed(seed)
        if len(feature_files) > 1024:
            feature_files = list(np.random.choice(feature_files, 1024, replace=False))
            Log.info(f"[3DB-BEDLAM2] Debug mode: randomly sampled 1024 files (seed={seed})")
        else:
            Log.warn(f"[3DB-BEDLAM2] Debug mode: only {len(feature_files)} files available, using all")
    
    # 构建motion_files字典
    motion_files = {}
    missing_seq_info = []
    error_files = []
    
    for feat_file in tqdm(feature_files, desc="Processing sequences"):
        # 计算相对路径
        rel_path = feat_file.relative_to(feature_dir)
        
        # 构建seq_info路径（.pt -> .npz）
        seq_info_path = seq_info_dir / rel_path.parent / (rel_path.stem + '.npz')
        
        # 检查seq_info文件是否存在
        if not seq_info_path.exists():
            missing_seq_info.append(str(rel_path))
            continue
        
        try:
            # 转换数据
            motion_dict = convert_seq_info_to_motion_dict(seq_info_path, smplx_model, device)
            
            # 跳过长度<=1的序列
            if motion_dict is None:
                continue
            
            # 构建新的键名：'imgfeats/3db/文件夹名/文件名.pt'
            key = f"imgfeats/3db/{rel_path.parent.name}/{rel_path.name}"
            
            # 存储到motion_files
            motion_files[key] = motion_dict
            
        except Exception as e:
            error_files.append((str(rel_path), str(e)))
            Log.error(f"Error processing {rel_path}: {e}")
            continue
    
    # 报告统计信息
    Log.info(f"[3DB-BEDLAM2] Successfully processed: {len(motion_files)} sequences")
    if missing_seq_info:
        Log.warn(f"[3DB-BEDLAM2] Missing seq_info files: {len(missing_seq_info)}")
        Log.warn(f"[3DB-BEDLAM2] First 5 missing: {missing_seq_info[:5]}")
    if error_files:
        Log.warn(f"[3DB-BEDLAM2] Error processing files: {len(error_files)}")
        Log.warn(f"[3DB-BEDLAM2] First 5 errors: {error_files[:5]}")
    
    # 保存到磁盘
    Log.info(f"[3DB-BEDLAM2] Saving to {output_path}...")
    torch.save(motion_files, output_path)
    
    elapsed = time() - tic
    Log.info(f"[3DB-BEDLAM2] Generation completed in {elapsed:.2f}s")
    Log.info(f"[3DB-BEDLAM2] Total sequences: {len(motion_files)}")
    
    # 验证：加载并检查
    Log.info(f"[3DB-BEDLAM2] Verifying saved file...")
    loaded = torch.load(output_path)
    Log.info(f"[3DB-BEDLAM2] Verification passed. Keys: {len(loaded)}")
    
    # 打印示例键和数据结构
    if len(loaded) > 0:
        sample_key = list(loaded.keys())[0]
        sample_data = loaded[sample_key]
        Log.info(f"[3DB-BEDLAM2] Sample key: {sample_key}")
        Log.info(f"[3DB-BEDLAM2] Sample data keys: {list(sample_data.keys())}")
        Log.info(f"[3DB-BEDLAM2] Sample shapes:")
        for k, v in sample_data.items():
            if isinstance(v, torch.Tensor):
                Log.info(f"  {k}: {v.shape}")
            elif isinstance(v, list):
                Log.info(f"  {k}: {v}")
    
    return motion_files


if __name__ == "__main__":
    """
        # 完整数据集
        python scripts/utils/generate_3db_bedlam2_smplpose_v3.py

        # Debug模式（1024样本）
        python scripts/utils/generate_3db_bedlam2_smplpose_v3.py --debug

        # 自定义随机种子
        python scripts/utils/generate_3db_bedlam2_smplpose_v3.py --debug --seed 123
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate 3DB BEDLAM2 smplpose_v3.pt")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/animation-cfs-nj/gaigeying/dataset_opensource/BEDLAM2/3DB_COS",
        help="Root directory of 3DB BEDLAM2 data"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for smplpose_v3.pt (default: root_dir/smplpose_v3_3db.pt or smplpose_v3_3db_random1024.pt in debug mode)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for SMPLX computation (cuda or cpu)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: only process 1024 random samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for debug mode sampling (default: 42)"
    )
    
    args = parser.parse_args()
    
    motion_files = generate_3db_smplpose_v3(
        root_dir=args.root_dir,
        output_path=args.output_path,
        device=args.device,
        debug=args.debug,
        seed=args.seed
    )
