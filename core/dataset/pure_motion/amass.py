import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from pathlib import Path
from core.utils.pylogger import Log
from core.configs import MainStore, builds

from .base_dataset import BaseDataset
from .utils import *
from core.utils.geo.hmr_global import get_tgtcoord_rootparam
from core.utils.wis3d_utils import make_wis3d, add_motion_as_lines, convert_motion_as_line_mesh


class AmassDataset(BaseDataset):
    def __init__(
        self,
        motion_frames=120,
        l_factor=1.5,  # speed augmentation
        skip_moyo=True,  # not contained in the ICCV19 released version
        cam_augmentation="v11",
        random1024=False,  # DEBUG
        limit_size=None,
    ):
        self.root = Path("inputs/AMASS/hmr4d_support")
        self.motion_frames = motion_frames
        self.l_factor = l_factor
        self.random1024 = random1024
        self.skip_moyo = skip_moyo
        self.dataset_name = "AMASS"
        super().__init__(cam_augmentation, limit_size)

    def _load_dataset(self):
        filename = self.root / "smplxpose_v2.pth"
        Log.info(f"[{self.dataset_name}] Loading from {filename} ...")
        tic = Log.time()
        if self.random1024:  # Debug, faster loading
            try:
                Log.info(f"[{self.dataset_name}] Loading 1024 samples for debugging ...")
                self.motion_files = torch.load(self.root / "smplxpose_v2_random1024.pth")
            except:
                Log.info(f"[{self.dataset_name}] Not found! Saving 1024 samples for debugging ...")
                self.motion_files = torch.load(filename)
                keys = list(self.motion_files.keys())
                keys = np.random.choice(keys, 1024, replace=False)
                self.motion_files = {k: self.motion_files[k] for k in keys}
                torch.save(self.motion_files, self.root / "smplxpose_v2_random1024.pth")
        else:
            self.motion_files = torch.load(filename)
        self.seqs = list(self.motion_files.keys())
        Log.info(f"[{self.dataset_name}] {len(self.seqs)} sequences. Elapsed: {Log.time() - tic:.2f}s")

    def _get_idx2meta(self):
        # We expect to see the entire sequence during one epoch,
        # so each sequence will be sampled max(SeqLength // MotionFrames, 1) times
        seq_lengths = []
        self.idx2meta = []

        # Skip too-long idle-prefix
        motion_start_id = {}
        for vid in self.motion_files:
            if self.skip_moyo and "moyo_smplxn" in vid:
                continue
            seq_length = self.motion_files[vid]["pose"].shape[0]
            start_id = motion_start_id[vid] if vid in motion_start_id else 0
            seq_length = seq_length - start_id
            if seq_length < 25:  # Skip clips that are too short
                continue
            num_samples = max(seq_length // self.motion_frames, 1)
            seq_lengths.append(seq_length)
            self.idx2meta.extend([(vid, start_id)] * num_samples)
        hours = sum(seq_lengths) / 30 / 3600
        Log.info(f"[{self.dataset_name}] has {hours:.1f} hours motion -> Resampled to {len(self.idx2meta)} samples.")

    def _load_data(self, idx):
        """
        - Load original data
        - Augmentation: speed-augmentation to L frames
        """
        # Load original data
        mid, start_id = self.idx2meta[idx]
        raw_data = self.motion_files[mid]
        raw_len = raw_data["pose"].shape[0] - start_id
        data = {
            "body_pose": raw_data["pose"][start_id:, 3:],  # (F, 63)
            "betas": raw_data["beta"].repeat(raw_len, 1),  # (10)
            "global_orient": raw_data["pose"][start_id:, :3],  # (F, 3)
            "transl": raw_data["trans"][start_id:],  # (F, 3)
        }

        # Get {tgt_len} frames from data
        # Random select a subset with speed augmentation  [start, end)
        tgt_len = self.motion_frames
        raw_subset_len = np.random.randint(int(tgt_len / self.l_factor), int(tgt_len * self.l_factor))
        if raw_subset_len <= raw_len:
            start = np.random.randint(0, raw_len - raw_subset_len + 1)
            end = start + raw_subset_len
        else:  # interpolation will use all possible frames (results in a slow motion)
            start = 0
            end = raw_len
        data = {k: v[start:end] for k, v in data.items()}

        # Interpolation (vec + r6d)
        data_interpolated = interpolate_smpl_params(data, tgt_len)

        # AZ -> AY
        data_interpolated["global_orient"], data_interpolated["transl"], _ = get_tgtcoord_rootparam(
            data_interpolated["global_orient"],
            data_interpolated["transl"],
            tsf="az->ay",
        )

        data_interpolated["data_name"] = "amass"
        return data_interpolated


class MIBDataset(BaseDataset):
    def __init__(
        self,
        motion_frames=120,
        l_factor=1.5,  # speed augmentation
        cam_augmentation="v11",
        random1024=False,  # DEBUG
        limit_size=None,
        dataset_name="MIB80H"
    ):
        self.root = Path("inputs/AMASS/hmr4d_support")
        self.motion_frames = motion_frames
        self.l_factor = l_factor
        self.random1024 = random1024
        self.dataset_name = dataset_name
        super().__init__(cam_augmentation, limit_size)

    def _load_dataset(self):
        filename = self.root / f"{self.dataset_name}.pth"
        Log.info(f"[{self.dataset_name}] Loading from {filename} ...")
        tic = Log.time()
        if self.random1024:  # Debug, faster loading
            try:
                Log.info(f"[{self.dataset_name}] Loading 1024 samples for debugging ...")
                self.motion_files = torch.load(self.root / "MIB80H_random1024.pth")
            except:
                Log.info(f"[{self.dataset_name}] Not found! Saving 1024 samples for debugging ...")
                self.motion_files = torch.load(filename)
                keys = list(self.motion_files.keys())
                keys = np.random.choice(keys, 1024, replace=False)
                self.motion_files = {k: self.motion_files[k] for k in keys}
                torch.save(self.motion_files, self.root / "MIB80H_random1024.pth")
        else:
            self.motion_files = torch.load(filename)
        self.seqs = list(self.motion_files.keys())
        Log.info(f"[{self.dataset_name}] {len(self.seqs)} sequences. Elapsed: {Log.time() - tic:.2f}s")

    def _get_idx2meta(self):
        # 每个序列将被采样 max(SeqLength // MotionFrames, 1) 次
        seq_lengths = []
        self.idx2meta = []

        for vid in self.motion_files:
            seq_length = self.motion_files[vid]["pose"].shape[0]
            if seq_length < 25:  # 跳过过短的片段
                continue
            num_samples = max(seq_length // self.motion_frames, 1)
            seq_lengths.append(seq_length)
            self.idx2meta.extend([(vid, 0)] * num_samples)  # 无起始帧跳过
        hours = sum(seq_lengths) / 30 / 3600
        Log.info(f"[{self.dataset_name}] has {hours:.1f} hours motion -> Resampled to {len(self.idx2meta)} samples.")

    def _load_data(self, idx):
        """
        - 加载原始数据
        - 增强：速度增强到 L 帧
        """
        # 加载原始数据
        mid, start_id = self.idx2meta[idx]
        raw_data = self.motion_files[mid]
        raw_len = raw_data["pose"].shape[0] - start_id
        data = {
            "body_pose": raw_data["pose"][start_id:, 3:],  # (F, 63)
            "betas": raw_data["beta"].repeat(raw_len, 1),  # (10)
            "global_orient": raw_data["pose"][start_id:, :3],  # (F, 3)
            "transl": raw_data["trans"][start_id:],  # (F, 3)
        }

        # 从数据中获取 {tgt_len} 帧
        tgt_len = self.motion_frames
        raw_subset_len = np.random.randint(int(tgt_len / self.l_factor), int(tgt_len * self.l_factor))
        if raw_subset_len <= raw_len:
            start = np.random.randint(0, raw_len - raw_subset_len + 1)
            end = start + raw_subset_len
        else:  # 插值将使用所有可能的帧（产生慢动作）
            start = 0
            end = raw_len
        data = {k: v[start:end] for k, v in data.items()}

        # 插值 (vec + r6d)
        data_interpolated = interpolate_smpl_params(data, tgt_len)

        # 注意：移除了坐标系转换（MIB 数据集已经是 y-down）
        data_interpolated["data_name"] = "mib"
        return data_interpolated


# 注册 Hydra 对象
group_name = "train_datasets/pure_motion_amass"
MainStore.store(name="v11", node=builds(AmassDataset, cam_augmentation="v11"), group=group_name)

# 添加 MIBDataset 注册
group_name_mib = "train_datasets/pure_motion_mib"
MainStore.store(name="v11", node=builds(MIBDataset, cam_augmentation="v11"), group=group_name_mib)

# 添加 MIBDataset 700h (实572h) 注册
mib_700h_00 = "train_datasets/pure_motion_mib_700h_00"
MainStore.store(name="v11", node=builds(MIBDataset, cam_augmentation="v11", dataset_name="mibdataset_700h_split00"), group=mib_700h_00)
mib_700h_01 = "train_datasets/pure_motion_mib_700h_01"
MainStore.store(name="v11", node=builds(MIBDataset, cam_augmentation="v11", dataset_name="mibdataset_700h_split01"), group=mib_700h_01)
mib_700h_02 = "train_datasets/pure_motion_mib_700h_02"
MainStore.store(name="v11", node=builds(MIBDataset, cam_augmentation="v11", dataset_name="mibdataset_700h_split02"), group=mib_700h_02)
mib_700h_03 = "train_datasets/pure_motion_mib_700h_03"
MainStore.store(name="v11", node=builds(MIBDataset, cam_augmentation="v11", dataset_name="mibdataset_700h_split03"), group=mib_700h_03)
mib_700h_04 = "train_datasets/pure_motion_mib_700h_04"
MainStore.store(name="v11", node=builds(MIBDataset, cam_augmentation="v11", dataset_name="mibdataset_700h_split04"), group=mib_700h_04)
mib_700h_05 = "train_datasets/pure_motion_mib_700h_05"
MainStore.store(name="v11", node=builds(MIBDataset, cam_augmentation="v11", dataset_name="mibdataset_700h_split05"), group=mib_700h_05)
mib_700h_06 = "train_datasets/pure_motion_mib_700h_06"
MainStore.store(name="v11", node=builds(MIBDataset, cam_augmentation="v11", dataset_name="mibdataset_700h_split06"), group=mib_700h_06)
mib_700h_07 = "train_datasets/pure_motion_mib_700h_07"
MainStore.store(name="v11", node=builds(MIBDataset, cam_augmentation="v11", dataset_name="mibdataset_700h_split07"), group=mib_700h_07)
mib_700h_08 = "train_datasets/pure_motion_mib_700h_08"
MainStore.store(name="v11", node=builds(MIBDataset, cam_augmentation="v11", dataset_name="mibdataset_700h_split08"), group=mib_700h_08)
mib_700h_09 = "train_datasets/pure_motion_mib_700h_09"
MainStore.store(name="v11", node=builds(MIBDataset, cam_augmentation="v11", dataset_name="mibdataset_700h_split09"), group=mib_700h_09)
mib_700h_10 = "train_datasets/pure_motion_mib_700h_10"
MainStore.store(name="v11", node=builds(MIBDataset, cam_augmentation="v11", dataset_name="mibdataset_700h_split10"), group=mib_700h_10)
