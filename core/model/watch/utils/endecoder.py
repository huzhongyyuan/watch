import torch
import torch.nn as nn
from pytorch3d.transforms import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from core.configs import MainStore, builds
from core.utils.geo.augment_noisy_pose import gaussian_augment
import core.utils.matrix as matrix
from core.utils.pylogger import Log
from core.utils.geo_transform import extract_rotation_components_ydown, compute_cam_angvel
from core.utils.geo.hmr_global import get_local_transl_vel, get_local_vel_mat, rollout_local_transl_vel
from core.utils.smplx_utils import make_smplx
from . import stats_compose


class EnDecoder(nn.Module):
    def __init__(self,
                 s_name_hmr="DEFAULT_01", s_name_human="DEFAULT_01", s_name_cam="DEFAULT_01",
                 noise_pose_k=10):
        super().__init__()
        # Load mean, std
        stats_hmr = getattr(stats_compose, s_name_hmr)
        Log.info(f"[EnDecoder] Use {s_name_hmr} for HMR statistics!")
        self.register_buffer("mean_hmr", torch.tensor(stats_hmr["mean"]).float(), False)
        self.register_buffer("std_hmr", torch.tensor(stats_hmr["std"]).float(), False)
        stats_human = getattr(stats_compose, s_name_human)
        Log.info(f"[EnDecoder] Use {s_name_human} for human trajectory statistics!")
        self.register_buffer("mean_human", torch.tensor(stats_human["mean"]).float(), False)
        self.register_buffer("std_human", torch.tensor(stats_human["std"]).float(), False)
        stats_cam = getattr(stats_compose, s_name_cam)
        Log.info(f"[EnDecoder] Use {s_name_cam} for camera trajectory statistics!")
        self.register_buffer("mean_cam", torch.tensor(stats_cam["mean"]).float(), False)
        self.register_buffer("std_cam", torch.tensor(stats_cam["std"]).float(), False)

        # option
        self.noise_pose_k = noise_pose_k

        # smpl
        self.smplx_model = make_smplx("supermotion_v437coco17")
        parents = self.smplx_model.parents[:22]
        self.register_buffer("parents_tensor", parents, False)
        self.parents = parents.tolist()

    def get_noisyobs(self, data, return_type="r6d"):
        """
        Noisy observation contains local pose with noise
        Args:
            data (dict):
                body_pose: (B, L, J*3) or (B, L, J, 3)
        Returns:
            noisy_bosy_pose: (B, L, J, 6) or (B, L, J, 3) or (B, L, 3, 3) depends on return_type
        """
        body_pose = data["body_pose"]  # (B, L, 63)
        B, L, _ = body_pose.shape
        body_pose = body_pose.reshape(B, L, -1, 3)

        # (B, L, J, C)
        return_mapping = {"R": 0, "r6d": 1, "aa": 2}
        return_id = return_mapping[return_type]
        noisy_bosy_pose = gaussian_augment(body_pose, self.noise_pose_k, to_R=True)[return_id]
        return noisy_bosy_pose

    def normalize_body_pose_r6d(self, body_pose_r6d):
        """body_pose_r6d: (B, L, {J*6}/{J, 6}) ->  (B, L, J*6)"""
        B, L = body_pose_r6d.shape[:2]
        body_pose_r6d = body_pose_r6d.reshape(B, L, -1)
        if self.mean_hmr.shape[-1] == 1:  # no mean, std provided
            return body_pose_r6d
        body_pose_r6d = (body_pose_r6d - self.mean_hmr[:126]) / self.std_hmr[:126]  # (B, L, C)
        return body_pose_r6d

    def fk_v2(self, body_pose, betas, global_orient=None, transl=None, get_intermediate=False):
        """
        Use smplx params to get joints without forwarding full smplx model.
        Args:
            body_pose: (B, L, 63)
            betas: (B, L, 10)
            global_orient: (B, L, 3)
        Returns:
            joints: (B, L, 22, 3)
        """
        B, L = body_pose.shape[:2]
        if global_orient is None:
            global_orient = torch.zeros((B, L, 3), device=body_pose.device)
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(aa)  # (B, L, 22, 3, 3)

        skeleton = self.smplx_model.get_skeleton(betas)[..., :22, :]  # (B, L, 22, 3)
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)

        if transl is not None:
            local_skeleton[..., 0, :] += transl  # B, L, 22, 3

        mat = matrix.get_TRS(rotmat, local_skeleton)  # B, L, 22, 4, 4
        fk_mat = matrix.forward_kinematics(mat, self.parents)  # B, L, 22, 4, 4
        joints = matrix.get_position(fk_mat)  # B, L, 22, 3
        if not get_intermediate:
            return joints
        else:
            return joints, mat, fk_mat

    def get_local_pos(self, betas):
        skeleton = self.smplx_model.get_skeleton(betas)[..., :22, :]  # (B, L, 22, 3)
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)
        return local_skeleton

    def encode_hmr(self, inputs):
        """Encode HMR params for WATCH HMR head.
        Returns: {
            body_pose_r6d,  # (B, L, (J-1)*6) -> 0:126
            betas, # (B, L, 10) -> 126:136
            global_orient_r6d,  # (B, L, 6) -> 136:142  incam
            }
        """
        B, L = inputs["smpl_params_c"]["body_pose"].shape[:2]
        # cam
        smpl_params_c = inputs["smpl_params_c"]
        body_pose = smpl_params_c["body_pose"].reshape(B, L, 21, 3)
        body_pose_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).flatten(-2)
        betas = smpl_params_c["betas"]
        global_orient_mat = axis_angle_to_matrix(smpl_params_c["global_orient"])
        global_orient_r6d = matrix_to_rotation_6d(global_orient_mat)

        # returns
        x = torch.cat([body_pose_r6d, betas, global_orient_r6d], dim=-1)
        x_norm = (x - self.mean_hmr) / (self.std_hmr + 1e-4)
        return x_norm

    def encode_human_move(self, inputs):
        """
        Encode human movement params for WATCH human move head.
        """
        smpl_params_w = inputs["smpl_params_w"]
        global_orient_w = smpl_params_w["global_orient"]
        human_rot_yaw, human_roll_pitch = extract_rotation_components_ydown(axis_angle_to_matrix(global_orient_w))
        human_angvel_r_yaw = compute_cam_angvel(human_rot_yaw)
        human_local_vel = get_local_transl_vel(smpl_params_w["transl"], global_orient_w)

        x = torch.cat([human_local_vel, matrix_to_rotation_6d(human_roll_pitch), human_angvel_r_yaw], dim=-1)
        x_norm = (x - self.mean_human) / (self.std_human + 1e-4)
        return x_norm

    def encode_cam_move(self, inputs):
        """
        Encode camera movement params for WATCH camera move head.
        """
        cam_roll_pitch = inputs["cam_roll_pitch"]
        cam_angvel_r_yaw = inputs["cam_angvel_r_yaw"]
        cam_local_vel = get_local_vel_mat(
            inputs["cam_transl"],
            torch.einsum("...ij,...jk->...ik",
                         rotation_6d_to_matrix(inputs["cam_yaw"]),
                         rotation_6d_to_matrix(inputs["cam_roll_pitch"])),
        )

        x = torch.cat([cam_local_vel, cam_roll_pitch, cam_angvel_r_yaw], dim=-1)
        x_norm = (x - self.mean_cam) / (self.std_cam + 1e-4)
        return x_norm

    def encode_translw(self, inputs):
        """
        definition: {
                body_pose_r6d,  # (B, L, (J-1)*6) -> 0:126
                betas, # (B, L, 10) -> 126:136
                global_orient_r6d,  # (B, L, 6) -> 136:142  incam
                global_orient_gv_r6d: # (B, L, 6) -> 142:148  gv
                local_transl_vel,  # (B, L, 3) -> 148:151, smpl-coord
            }
        """
        # local_transl_vel
        smpl_params_w = inputs["smpl_params_w"]
        local_transl_vel = get_local_transl_vel(smpl_params_w["transl"], smpl_params_w["global_orient"])

        # returns
        x = local_transl_vel
        x_norm = (x - self.mean_human[-3:]) / self.std_human[-3:]
        return x_norm

    def decode_human_move(self, x_norm):
        """
        Human movement decoder
        Args:
            x_norm: (B, L, 15),
             consists of human local transl vel (B, L, 3), human roll_pitch (B, L, 6), human yaw angvel (B, L, 6)
        Returns:
            output (dict): human_local_vel (B, L, 3), human_roll_pitch (B, L, 6), human_angvel_rz (B, L, 6)
        """
        x = (x_norm * self.std_human) + self.mean_human

        human_local_vel = x[:, :, :3]
        human_roll_pitch = x[:, :, 3:9]
        human_angvel_r_yaw = x[:, :, 9:15]

        output = {
            "human_local_vel": human_local_vel,
            "human_roll_pitch": human_roll_pitch,
            "human_angvel_r_yaw": human_angvel_r_yaw,
        }
        return output

    def decode_cam_move(self, x_norm):
        """
        Camera movement decoder
        Args:
            x_norm: (B, L, 15),
             consists of cam local transl vel (B, L, 3), cam roll_pitch (B, L, 6), cam yaw angvel (B, L, 6)
        Returns:
            output (dict): cam_local_vel (B, L, 3), cam_roll_pitch (B, L, 6), cam_angvel_rz (B, L, 6)
        """
        x = (x_norm * self.std_cam) + self.mean_cam
        cam_local_vel = x[:, :, :3]
        cam_roll_pitch = x[:, :, 3:9]
        cam_angvel_r_yaw = x[:, :, 9:15]

        output = {
            "cam_local_vel": cam_local_vel,
            "cam_roll_pitch": cam_roll_pitch,
            "cam_angvel_r_yaw": cam_angvel_r_yaw,
        }
        return output

    def decode_hmr(self, x_norm):
        """
        In camera HMR params decoder, predicted x into params.
        Args:
            x_norm: (B, L, 142), consists of body_pose_r6d (B, L, 126), betas (B, L, 10), global_orient_r6d (B, L, 6)
        Returns:
            output (dict): body_pose (B, L, 63), betas (B, L, 10), global_orient (B, L, 3)
        """
        B, L, C = x_norm.shape
        x = (x_norm * self.std_hmr) + self.mean_hmr

        body_pose_r6d = x[:, :, :126]
        betas = x[:, :, 126:136]
        global_orient_r6d = x[:, :, 136:142]

        body_pose = matrix_to_axis_angle(rotation_6d_to_matrix(body_pose_r6d.reshape(B, L, -1, 6)))
        body_pose = body_pose.flatten(-2)
        global_orient_c = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_r6d))

        output = {
            "body_pose": body_pose,
            "betas": betas,
            "global_orient": global_orient_c,
        }

        return output


class EnDecoderV4(EnDecoder):
    def encode_human_move(self, inputs):
        """
        Encode human movement params for WATCH human move head.
        """
        smpl_params_w = inputs["smpl_params_w"]
        global_orient_w = smpl_params_w["global_orient"]
        human_local_vel = get_local_transl_vel(smpl_params_w["transl"], global_orient_w)

        x = torch.cat([human_local_vel], dim=-1)
        x_norm = (x - self.mean_human) / (self.std_human + 1e-4)
        return x_norm

    def encode_cam_move(self, inputs):
        """
        Encode camera movement params for WATCH camera move head.
        """
        cam_roll_pitch = inputs["cam_roll_pitch"]
        cam_local_vel = get_local_vel_mat(
            inputs["cam_transl"],
            torch.einsum("...ij,...jk->...ik",
                         rotation_6d_to_matrix(inputs["cam_yaw"]),
                         rotation_6d_to_matrix(inputs["cam_roll_pitch"])),
        )

        x = torch.cat([cam_local_vel, cam_roll_pitch], dim=-1)
        x_norm = (x - self.mean_cam) / (self.std_cam + 1e-4)
        return x_norm

    def decode_human_move(self, x_norm):
        """
        Human movement decoder
        Args:
            x_norm: (B, L, 3), human local transl vel
        Returns:
            output (dict): human_local_vel (B, L, 3)
        """
        x = (x_norm * self.std_human) + self.mean_human

        return {"human_local_vel": x}

    def decode_cam_move(self, x_norm):
        """
        Camera movement decoder
        Args:
            x_norm: (B, L, 9),
             consists of cam local transl vel (B, L, 3), cam roll_pitch (B, L, 6)
        Returns:
            output (dict): cam_local_vel (B, L, 3), cam_roll_pitch (B, L, 6)
        """
        x = (x_norm * self.std_cam) + self.mean_cam
        cam_local_vel = x[:, :, :3]
        cam_roll_pitch = x[:, :, 3:9]

        output = {
            "cam_local_vel": cam_local_vel,
            "cam_roll_pitch": cam_roll_pitch,
        }
        return output


group_name = "endecoder/watch"
cfg_base = builds(EnDecoder, populate_full_signature=True)
# MainStore.store(name="v1_no_stdmean", node=cfg_base, group=group_name)
# MainStore.store(name="v1", node=cfg_base(stats_name="MM_V1"), group=group_name)
# MainStore.store(
#     name="v1_amass_local_bedlam_cam",
#     node=cfg_base(stats_name="MM_V1_AMASS_LOCAL_BEDLAM_CAM"),
#     group=group_name,
# )
#
# MainStore.store(name="v2", node=cfg_base(stats_name="MM_V2"), group=group_name)
# MainStore.store(name="v2_1", node=cfg_base(stats_name="MM_V2_1"), group=group_name)

MainStore.store(
    name="v3_1_amass_human_bedlam_cam",
    node=cfg_base(
        s_name_hmr="MM_V3_AMASS_LOCAL_BEDLAM_HMR",
        s_name_human="MM_V3_AMASS_HUMAN_MOVE",
        s_name_cam="MM_V3_AMASS_CAM_MOVE"
    ),
    group=group_name,
)

cfg_base_v4 = builds(EnDecoderV4, populate_full_signature=True)
MainStore.store(
    name="v4_1_amass_human_bedlam_cam",
    node=cfg_base_v4(
        s_name_hmr="MM_V3_AMASS_LOCAL_BEDLAM_HMR",
        s_name_human="MM_V4_AMASS_HUMAN_MOVE",
        s_name_cam="MM_V4_AMASS_CAM_MOVE"
    ),
    group=group_name,
)
