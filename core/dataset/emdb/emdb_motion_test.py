from pathlib import Path
import numpy as np
import torch
from torch.utils import data
from core.utils.pylogger import Log
from core.utils.wis3d_utils import make_wis3d, add_motion_as_lines

from core.utils.geo_transform import compute_cam_angvel
from core.utils.geo_transform import extract_rotation_components_ydown, generate_transform, generate_inverse_transform
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_rotation_6d
from core.utils.geo.hmr_cam import estimate_K, resize_K
from core.utils.geo.flip_utils import flip_kp2d_coco17

from .utils import EMDB1_NAMES, EMDB2_NAMES

VID_PRESETS = {1: EMDB1_NAMES, 2: EMDB2_NAMES}


from core.configs import MainStore, builds


class EmdbSmplFullSeqDataset(data.Dataset):
    def __init__(self, split=1, flip_test=False):
        """
        split: 1 for EMDB-1, 2 for EMDB-2
        flip_test: if True, extra flip data will be returned
        """
        super().__init__()
        self.dataset_name = "EMDB"
        self.split = split
        self.dataset_id = f"EMDB_{split}"
        Log.info(f"[{self.dataset_name}] Full sequence, split={split}")

        # Load evaluation protocol from WHAM labels
        tic = Log.time()
        self.emdb_dir = Path("inputs/EMDB/hmr4d_support")
        # 'name', 'gender', 'smpl_params', 'mask', 'K_fullimg', 'T_w2c', 'bbx_xys', 'kp2d', 'features'
        self.labels = torch.load(self.emdb_dir / "emdb_vit_v4.pt")
        self.cam_traj = torch.load(self.emdb_dir / "emdb_dpvo_traj.pt")  # estimated with DPVO

        # Setup dataset index
        self.idx2meta = []
        for vid in VID_PRESETS[split]:
            seq_length = len(self.labels[vid]["mask"])
            self.idx2meta.append((vid, 0, seq_length))  # start=0, end=seq_length
        Log.info(f"[{self.dataset_name}] {len(self.idx2meta)} sequences. Elapsed: {Log.time() - tic:.2f}s")

        # If flip_test is enabled, we will return extra data for flipped test
        self.flip_test = flip_test
        if self.flip_test:
            Log.info(f"[{self.dataset_name}] Flip test enabled")

    def __len__(self):
        return len(self.idx2meta)

    def _load_data(self, idx):
        data = {}

        # [vid, start, end]
        vid, start, end = self.idx2meta[idx]
        length = end - start
        meta = {"dataset_id": self.dataset_id, "vid": vid, "vid-start-end": (start, end)}
        data.update({"meta": meta, "length": length})

        label = self.labels[vid]

        # smpl_params in world
        gender = label["gender"]
        smpl_params = label["smpl_params"]
        mask = label["mask"]
        data.update({"smpl_params": smpl_params, "gender": gender, "mask": mask})

        # camera
        # K_fullimg = label["K_fullimg"]  # We use estimated K
        width_height = (1440, 1920) if vid != "P0_09_outdoor_walk" else (720, 960)
        K_fullimg = estimate_K(*width_height)
        T_w2c = label["T_w2c"]
        data.update({"K_fullimg": K_fullimg, "T_w2c": T_w2c})

        # R_w2c -> cam_angvel
        use_DPVO = False
        if use_DPVO:
            traj = self.cam_traj[data["meta"]["vid"]]  # (L, 7)
            R_w2c = quaternion_to_matrix(traj[:, [6, 3, 4, 5]]).mT  # (L, 3, 3)
        else:  # GT
            R_w2c = data["T_w2c"][:, :3, :3]  # (L, 3, 3)
        data["cam_angvel"] = compute_cam_angvel(R_w2c)  # (L, 6)
        # T_c2w consists of camera orientation and translation.
        # Note that dot(R_w2c) is not the angular velocity of the camera.
        T_c2w = generate_inverse_transform(data["T_w2c"])
        cam_orient, cam_transl = T_c2w[:, :3, :3], T_c2w[:, :3, 3]
        r_yaw, R_rest = extract_rotation_components_ydown(cam_orient)
        data['cam_yaw'], data['cam_roll_pitch'] = matrix_to_rotation_6d(r_yaw), matrix_to_rotation_6d(R_rest)
        data['cam_angvel_r_yaw'] = compute_cam_angvel(r_yaw)
        data['cam_transl'] = cam_transl
        # # normalize cam_transl then get local trajectory
        # data['cam_transl_local'] = torch.einsum(
        #     '...ij,...j->...i', r_yaw[0].transpose(-1, -2),
        #     # (cam_transl - torch.mean(cam_transl, dim=-2)) / (torch.norm(torch.std(cam_transl, dim=-2), dim=-1) + 1e-4)
        #     (cam_transl - cam_transl[..., 0, :])
        # )
        from core.utils.geo.hmr_global import get_local_vel_mat
        data['cam_transl_local'] = get_local_vel_mat(cam_transl, cam_orient)

        # image bbx, features
        bbx_xys = label["bbx_xys"]
        f_imgseq = label["features"]
        kp2d = label["kp2d"]
        data.update({"bbx_xys": bbx_xys, "f_imgseq": f_imgseq, "kp2d": kp2d})

        # to render a video
        video_path = self.emdb_dir / f"videos/{vid}.mp4"
        frame_id = torch.where(mask)[0].long()
        resize_factor = 0.5
        width_height_render = torch.tensor(width_height) * resize_factor
        K_render = resize_K(K_fullimg, resize_factor)
        bbx_xys_render = bbx_xys * resize_factor
        data["meta_render"] = {
            "split": self.split,
            "name": vid,
            "video_path": str(video_path),
            "resize_factor": resize_factor,
            "frame_id": frame_id,
            "width_height": width_height_render.int(),
            "K": K_render,
            "bbx_xys": bbx_xys_render,
            "R_cam_type": "DPVO" if use_DPVO else "GtGyro",
        }

        # if enable flip_test
        if self.flip_test:
            imgfeat_dir = self.emdb_dir / "imgfeats/emdb_flip"
            f_img_dict = torch.load(imgfeat_dir / f"{vid}.pt")

            flipped_bbx_xys = f_img_dict["bbx_xys"].float()  # (L, 3)
            flipped_features = f_img_dict["features"].float()  # (L, 1024)
            width = width_height[0]
            flipped_kp2d = flip_kp2d_coco17(kp2d, width)  # (L, 17, 3)

            R_flip_x = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()
            flipped_R_w2c = R_flip_x @ R_w2c.clone()
            R_flip_x_4x4 = torch.eye(4).float()
            R_flip_x_4x4[:3, :3] = R_flip_x
            flipped_T_w2c = R_flip_x_4x4 @ data["T_w2c"]
            flipped_T_c2w = generate_inverse_transform(flipped_T_w2c)
            flipped_cam_orient, flipped_cam_transl = flipped_T_c2w[:, :3, :3], flipped_T_c2w[:, :3, 3]
            flipped_r_yaw, flipped_R_rest = extract_rotation_components_ydown(flipped_cam_orient)
            # # normalize cam_transl then get local trajectory
            # flipped_cam_transl_local = torch.einsum(
            #     '...ij,...j->...i', flipped_r_yaw[0].transpose(-1, -2),
            #     # (flipped_cam_transl - torch.mean(flipped_cam_transl, dim=-2)) / (torch.norm(torch.std(flipped_cam_transl, dim=-2), dim=-1) + 1e-4)
            #     (flipped_cam_transl - flipped_cam_transl[..., 0, :])
            # )
            flipped_cam_local_vel = get_local_vel_mat(flipped_cam_transl, flipped_cam_orient)

            data_flip = {
                "bbx_xys": flipped_bbx_xys,
                "f_imgseq": flipped_features,
                "kp2d": flipped_kp2d,
                "cam_angvel": compute_cam_angvel(flipped_R_w2c),
                "cam_angvel_r_yaw": compute_cam_angvel(flipped_r_yaw),
                # "cam_transl_local": flipped_cam_transl_local,
                "cam_transl_local": flipped_cam_local_vel,
            }
            data["flip_test"] = data_flip

        return data

    def _process_data(self, data):
        length = data["length"]
        data["K_fullimg"] = data["K_fullimg"][None].repeat(length, 1, 1)
        return data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data)
        return data


class EmdbSmpl3dbFullSeqDataset(EmdbSmplFullSeqDataset):
    """EMDB dataset using 3db features instead of vit features
    
    Note: 3db version does not support flip_test as flip features are not available yet.
    """
    def __init__(self, split=1, flip_test=False):
        if flip_test:
            Log.warn(f"[EMDB-3DB] flip_test is not supported for 3db version, setting to False")
            flip_test = False
        
        # Call parent init but override the labels loading
        self.dataset_name = "EMDB-3DB"
        self.split = split
        self.dataset_id = f"EMDB_{split}"
        Log.info(f"[{self.dataset_name}] Full sequence, split={split}")

        # Load evaluation protocol - using 3db features
        tic = Log.time()
        self.emdb_dir = Path("inputs/EMDB/hmr4d_support")
        # 'name', 'gender', 'smpl_params', 'mask', 'K_fullimg', 'T_w2c', 'bbx_xys', 'kp2d', 'features'
        self.labels = torch.load(self.emdb_dir / "emdb_3db_v1.pt")
        self.cam_traj = torch.load(self.emdb_dir / "emdb_dpvo_traj.pt")  # estimated with DPVO

        # Setup dataset index
        self.idx2meta = []
        for vid in VID_PRESETS[split]:
            seq_length = len(self.labels[vid]["mask"])
            self.idx2meta.append((vid, 0, seq_length))  # start=0, end=seq_length
        Log.info(f"[{self.dataset_name}] {len(self.idx2meta)} sequences. Elapsed: {Log.time() - tic:.2f}s")

        # flip_test is not supported for 3db version
        self.flip_test = False


# EMDB-1 and EMDB-2
MainStore.store(
    name="v1",
    node=builds(EmdbSmplFullSeqDataset, populate_full_signature=True),
    group="test_datasets/emdb1",
)
MainStore.store(
    name="v1_fliptest",
    node=builds(EmdbSmplFullSeqDataset, flip_test=True, populate_full_signature=True),
    group="test_datasets/emdb1",
)
MainStore.store(
    name="v1",
    node=builds(EmdbSmplFullSeqDataset, split=2, populate_full_signature=True),
    group="test_datasets/emdb2",
)
MainStore.store(
    name="v1_fliptest",
    node=builds(EmdbSmplFullSeqDataset, split=2, flip_test=True, populate_full_signature=True),
    group="test_datasets/emdb2",
)

# EMDB-1-3DB and EMDB-2-3DB (using 3db features, no flip_test support)
MainStore.store(
    name="v1",
    node=builds(EmdbSmpl3dbFullSeqDataset, populate_full_signature=True),
    group="test_datasets/emdb1_3db",
)
MainStore.store(
    name="v1",
    node=builds(EmdbSmpl3dbFullSeqDataset, split=2, populate_full_signature=True),
    group="test_datasets/emdb2_3db",
)
