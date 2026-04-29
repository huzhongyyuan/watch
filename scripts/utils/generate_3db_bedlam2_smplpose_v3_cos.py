"""
生成3DB BEDLAM2版本的smplpose_v3.pt文件

该脚本从*COS*直接完成3DB数据生成与HMR2版本格式兼容的smplpose_v3.pt文件。
主要功能：
1. 从COS的seq_info/*.npz读取标签数据
2. 从COS的sam3db_pose_feat/*.pt读取特征数据
3. 转换数据格式并生成skeleton
4. 保存为smplpose_v3.pt格式

使用方法：
    python scripts/utils/generate_3db_bedlam2_smplpose_v3.py
"""

import json
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from time import time

from gameai_cossign import GameaiCosSign

import sys
sys.path.append(".")

from core.utils.smplx_utils import make_smplx
from core.utils.pylogger import Log


FEAT_NAME = "sam3db_pose_feat"
client = GameaiCosSign(os.getenv('COS_USERNAME'), os.getenv('COS_PASSWORD'))


def convert_seq_info_to_motion_dict(npz_cos_url, smplx_model, device='cuda', scene_name=None, npz_name=None, download_root=None):
    """
    将seq_info中的npz数据转换为motion_dict格式
    
    Args:
        npz_cos_url: seq_info/*.npz文件的COS URL
        smplx_model: SMPLX模型实例
        device: 计算设备
        download_root: 下载缓存目录，可选
        
    Returns:
        motion_dict: 包含所有必要字段的字典，如果length<=1则返回None
    """

    data = np.load(client.download_file_bytes(npz_cos_url), allow_pickle=True)
    if download_root is not None:
        os.makedirs(os.path.join(download_root, scene_name), exist_ok=True)
        np.savez(os.path.join(download_root, scene_name, npz_name), **data)

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
    metadata_dir,
    cos_bucket,
    output_path,
    download_root=None,
    device='cuda',
    debug=False,
    seed=42
):
    """
    生成3DB版本的smplpose_v3.pt文件
    
    Args:
        metadata_dir: metadata文件目录
        cos_bucket: COS存储桶名称
        output_path: 输出文件路径
        device: 计算设备
        debug: 是否启用debug模式（只处理1024个样本）
        seed: debug模式下的随机种子
    """
    metadata_dir = Path(metadata_dir)
    metadata_dict = {}
    for file in metadata_dir.glob("*.json"):
        with open(file, 'r') as f:
            metadata_dict[file.stem] = json.load(f)

    mode_str = "DEBUG (random1024)" if debug else "FULL"
    Log.info(f"[3DB-BEDLAM2] Starting generation from COS [{mode_str}]")
    Log.info(f"[3DB-BEDLAM2] Output will be saved to {output_path}")
    tic = time()

    # 初始化SMPLX模型（使用与BEDLAM2相同的配置）
    Log.info(f"[3DB-BEDLAM2] Initializing SMPLX model...")
    smplx_model = make_smplx('smplx-circle', gender='neutral').to(device)
    smplx_model.eval()

    # 扫描所有feature文件（以feature为准）
    Log.info(f"[3DB-BEDLAM2] Scanning feature files...")
    feature_files = []
    for scene_name in metadata_dict:
        for pt_name in metadata_dict[scene_name][FEAT_NAME]:
            npz_name = pt_name.replace('.pt', '.npz')
            pt_cos_url = metadata_dict[scene_name][FEAT_NAME][pt_name]
            npz_cos_url = metadata_dict[scene_name]["seq_info"][npz_name]
            feature_files.append((scene_name, pt_name, pt_cos_url, npz_name, npz_cos_url))
    feature_files = sorted(feature_files)
    Log.info(f"[3DB-BEDLAM2] Found {len(feature_files)} feature files")

    if len(feature_files) == 0:
        raise ValueError(f"No feature files found in COS of {metadata_dir}")

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

    for scene_name, pt_name, pt_cos_url, npz_name, npz_cos_url in tqdm(feature_files, desc="Processing sequences"):

        try:
            # # 转换数据
            # npz_download_root = os.path.join(download_root, "seq_info") if download_root is not None else None
            # # npz_download_root = None
            # # if os.path.exists(os.path.join(npz_download_root, scene_name, npz_name)):
            # #     continue
            # motion_dict = convert_seq_info_to_motion_dict(npz_cos_url, smplx_model, device, scene_name, npz_name, npz_download_root)

            # # 跳过长度<=1的序列
            # if motion_dict is None:
            #     continue

            # # 构建新的键名：'imgfeats/3db/文件夹名/文件名.pt'
            # key = f"imgfeats/3db/{scene_name}/{pt_name}"

            # # 存储到motion_files
            # motion_files[key] = motion_dict

            # 缓存特征
            if download_root is not None:
                pt_download_root = os.path.join(download_root, "imgfeats", "3db_all", scene_name)
                os.makedirs(pt_download_root, exist_ok=True)
                feat_pt = torch.load(client.download_file_bytes(pt_cos_url))
                torch.save(feat_pt, os.path.join(pt_download_root, pt_name))

        except Exception as e:
            error_files.append((scene_name, pt_name, str(e)))
            Log.error(f"Error processing {scene_name}/{pt_name}: {e}")
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
        default="/animation-cfs-nj/gaigeying/dataset_opensource/BEDLAM2/3DB_COS_gvhmr",
        help="Root directory of 3DB BEDLAM2 data"
    )
    parser.add_argument(
        '--cos_bucket',
        type=str,
        default='dcc-skinning-1258344700',
        help='Bucket name of COS.'
    )
    parser.add_argument(
        '--metadata_dir',
        type=str,
        default='/animation-cfs-nj/xingyanliu/data/motion/bedlam2/preprocess/metadata_gvhmr',
        help='Path to json file storing upload COS paths of all files.'
    )
    parser.add_argument(
        "--download_root",
        type=str,
        default=None,
        help="Root directory for downloading files (default: None)"
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
        metadata_dir=args.metadata_dir,
        cos_bucket=args.cos_bucket,
        output_path=args.output_path,
        download_root=args.download_root,
        device=args.device,
        debug=args.debug,
        seed=args.seed
    )
