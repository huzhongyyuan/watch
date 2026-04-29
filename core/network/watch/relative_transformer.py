import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from core.configs import MainStore, builds

from core.network.base_arch.transformer.encoder_rope import EncoderRoPEBlock, CrossAttentionBlock
from core.network.base_arch.transformer.layer import zero_module

from core.utils.net_utils import length_to_mask
from timm.models.vision_transformer import Mlp


class NetworkEncoderRoPE(nn.Module):
    def __init__(
        self,
        max_len=120,  # max sequence length
        cliffcam_dim=3, cam_angvel_dim=6, cam_traj_dim=3, imgseq_dim=1024,  # condition
        # cliffcam_dim=3, cam_angvel_dim=6, cam_traj_dim=0, imgseq_dim=1024,  # condition
        latent_dim=512, num_layers=12, num_heads=8, mlp_ratio=4.0,  # intermediate
        # head output dim
        dim_hmr=142,  # 21*6 body_pose + 10 betas + 6 global_orient_in_cam
        dim_weak_cam=3,  # CLIFF camera params for weak projection
        dim_cam_move=15,  # 6 cam_roll_pitch_pose + 6 angle_vel_yaw + 3 cam_local_vel
        dim_human_move=15,  # 6 human_roll_pitch_pose + 6 angle_vel_yaw + 3 human_local_vel
        dim_static_conf=6,  # static configuration parameters
        dropout=0.1, avgbeta=True,
    ):
        super().__init__()
        self.max_len = max_len

        # condition
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.imgseq_dim = imgseq_dim
        self.cam_traj_dim = cam_traj_dim

        # intermediate
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # ===== build model ===== #
        # Input (Kp2d)
        # Main token: map d_obs 2 to 32
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(17, 32), requires_grad=True)
        self.embed_noisyobs = Mlp(
            17 * 32, hidden_features=self.latent_dim * 2, out_features=self.latent_dim, drop=dropout
        )

        self._build_condition_embedder()

        # Transformer
        self.transformer_blocks = nn.ModuleList([
            EncoderRoPEBlock(self.latent_dim, self.num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(self.num_layers)
        ])

        # self.head_hmr = Mlp(self.latent_dim, out_features=dim_hmr)
        # self.head_cam_move = dim_cam_move > 0
        # if self.head_cam_move:
        #     self.head_cam_move = Mlp(self.latent_dim, out_features=dim_cam_move)
        # self.head_human_move = dim_human_move > 0
        # if self.head_human_move:
        #     self.head_human_move = Mlp(self.latent_dim, out_features=dim_human_move)
        self.head_hmr = Mlp(self.latent_dim, out_features=dim_hmr+dim_cam_move+dim_human_move)
        self.head_cam_move, self.head_human_move = False, False

        self.head_weak_cam = dim_weak_cam > 0  # keep extra_output for easy-loading old ckpt
        if self.head_weak_cam:
            self.head_weak_cam = Mlp(self.latent_dim, out_features=dim_weak_cam)
            self.register_buffer("pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False)
            self.register_buffer("pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False)

        self.head_static_conf = dim_static_conf > 0
        if self.head_static_conf:
            self.head_static_conf = Mlp(self.latent_dim, out_features=dim_static_conf)

        self.avgbeta = avgbeta

    def _build_condition_embedder(self):
        latent_dim = self.latent_dim
        dropout = self.dropout
        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(self.cliffcam_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )
        if self.cam_angvel_dim > 0:
            self.cam_angvel_embedder = nn.Sequential(
                nn.Linear(self.cam_angvel_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
        if self.imgseq_dim > 0:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(self.imgseq_dim),
                zero_module(nn.Linear(self.imgseq_dim, latent_dim)),
            )
        if self.cam_traj_dim > 0:
            self.cam_traj_embedder = nn.Sequential(
                nn.Linear(self.cam_traj_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )

    def forward(self, length, obs=None, f_cliffcam=None, f_cam_angvel=None, f_cam_transl=None, f_imgseq=None):
        """
        Args:
            obs: (B, L, C), nosiy pose observation
            length: (B), valid length of x, if None then use x.shape[2]
            f_imgseq: (B, L, C)
            f_cliffcam: (B, L, 3), CLIFF-Cam parameters (bbx-detection in the full-image)
            f_cam_angvel: (B, L, 6), Camera angular velocity
            f_cam_transl: (B, L, C), Camera translation, normalized or scaled
        """
        B, L, J, C = obs.shape
        assert J == 17 and C == 3

        # Main token from observation (2D pose)
        obs = obs.clone()
        visible_mask = obs[..., [2]] > 0.5  # (B, L, J, 1)
        obs[~visible_mask[..., 0]] = 0  # set low-conf to all zeros
        f_obs = self.learned_pos_linear(obs[..., :2])  # (B, L, J, 32)
        f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
        x = self.embed_noisyobs(f_obs.view(B, L, -1))  # (B, L, J*32) -> (B, L, C)

        # Condition
        f_to_add = [self.cliffcam_embedder(f_cliffcam)]
        if hasattr(self, "cam_angvel_embedder"):
            f_to_add.append(self.cam_angvel_embedder(f_cam_angvel))
        if f_imgseq is not None and hasattr(self, "imgseq_embedder"):
            f_to_add.append(self.imgseq_embedder(f_imgseq))
        if f_cam_transl is not None and hasattr(self, "cam_traj_embedder"):
            f_to_add.append(self.cam_traj_embedder(f_cam_transl))

        for f_delta in f_to_add:
            x = x + f_delta

        # Setup length and make padding mask
        assert B == length.size(0)
        pmask = ~length_to_mask(length, L)  # (B, L)

        if L > self.max_len:
            attnmask = torch.ones((L, L), device=x.device, dtype=torch.bool)
            for i in range(L):
                min_ind, max_ind = max(0, i - self.max_len // 2), min(L, i + self.max_len // 2)
                min_ind, max_ind = min(L - self.max_len, min_ind), max(self.max_len, max_ind)
                attnmask[i, min_ind:max_ind] = False
        else:
            attnmask = None

        # Transformer
        for block in self.transformer_blocks:
            x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)

        # Body_in_camera head
        pred_hmr = self.head_hmr(x)  # (B, L, C)
        if self.avgbeta:
            betas = (pred_hmr[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None]  # (B, C)
            betas = repeat(betas, "b c -> b l c", l=L)
            pred_hmr = torch.cat([pred_hmr[..., :126], betas, pred_hmr[..., 136:]], dim=-1)

        # Camera movement head
        pred_cam_move = None
        if self.head_cam_move:
            pred_cam_move = self.head_cam_move(x)
            # pred_cam_move = pred_cam_move * self.pred_cam_move_std + self.pred_cam_move_mean

        # Human movement head
        pred_human_move = None
        if self.head_human_move:
            pred_human_move = self.head_human_move(x)
            # pred_human_move = pred_human_move * self.pred_human_move_std + self.pred_human_move_mean

        # CLIFF-Cam Head
        pred_weak_cam = None
        if self.head_weak_cam:
            pred_weak_cam = self.head_weak_cam(x)
            pred_weak_cam = pred_weak_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_weak_cam[..., 0], 0.25)  # min_clamp s to 0.25 (prevent negative prediction)

        static_conf_logits = None
        if self.head_static_conf:
            static_conf_logits = self.head_static_conf(x)  # (B, L, C')

        output = {
            "pred_context": x,
            # "pred_hmr": pred_hmr,
            # "pred_cam_move": pred_cam_move,
            # "pred_human_move": pred_human_move,
            "pred_hmr": pred_hmr[..., :142],
            "pred_cam_move": pred_hmr[..., 142:157],
            "pred_human_move": pred_hmr[..., 157:],
            "pred_weak_cam": pred_weak_cam,
            "static_conf_logits": static_conf_logits,
        }
        return output


class NetworkEncoderRoPENew(nn.Module):
    def __init__(
        self,
        max_len=120,  # max sequence length
        cliffcam_dim=3, cam_angvel_dim=6, cam_traj_dim=3, imgseq_dim=1024,  # condition v3
        # cliffcam_dim=3, cam_angvel_dim=6, cam_traj_dim=0, imgseq_dim=1024,  # condition v2
        latent_dim=512, num_layers=12, num_heads=8, mlp_ratio=4.0,  # intermediate
        # head output dim
        dim_hmr=142,  # 21*6 body_pose + 10 betas + 6 global_orient_in_cam
        dim_weak_cam=3,  # CLIFF camera params for weak projection
        dim_cam_move=9,  # 6 cam_roll_pitch_pose + 3 cam_local_vel
        dim_human_move=3,  # 3 human_local_vel
        dim_static_conf=6,  # static configuration parameters
        dropout=0.1, avgbeta=True,
    ):
        super().__init__()
        self.max_len = max_len

        # condition
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.imgseq_dim = imgseq_dim
        self.cam_traj_dim = cam_traj_dim

        # intermediate
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # ===== build model ===== #
        # Input (Kp2d)
        # Main token: map d_obs 2 to 32
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(17, 32), requires_grad=True)
        self.embed_noisyobs = Mlp(
            17 * 32, hidden_features=self.latent_dim * 2, out_features=self.latent_dim, drop=dropout
        )

        self._build_condition_embedder()

        # Transformer
        self.transformer_blocks = nn.ModuleList([
            EncoderRoPEBlock(self.latent_dim, self.num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(self.num_layers)
        ])

        self.head_hmr = Mlp(self.latent_dim, out_features=dim_hmr+dim_cam_move+dim_human_move)

        self.head_weak_cam = dim_weak_cam > 0  # keep extra_output for easy-loading old ckpt
        if self.head_weak_cam:
            self.head_weak_cam = Mlp(self.latent_dim, out_features=dim_weak_cam)
            self.register_buffer("pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False)
            self.register_buffer("pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False)

        self.head_static_conf = dim_static_conf > 0
        if self.head_static_conf:
            self.head_static_conf = Mlp(self.latent_dim, out_features=dim_static_conf)

        self.avgbeta = avgbeta

    def _build_condition_embedder(self):
        latent_dim = self.latent_dim
        dropout = self.dropout
        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(self.cliffcam_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )
        if self.cam_angvel_dim > 0:
            self.cam_angvel_embedder = nn.Sequential(
                nn.Linear(self.cam_angvel_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
        if self.imgseq_dim > 0:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(self.imgseq_dim),
                zero_module(nn.Linear(self.imgseq_dim, latent_dim)),
            )
        if self.cam_traj_dim > 0:
            self.cam_traj_embedder = nn.Sequential(
                nn.Linear(self.cam_traj_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )

    def forward(self, length, obs=None, f_cliffcam=None, f_cam_angvel=None, f_cam_transl=None, f_imgseq=None):
        """
        Args:
            obs: (B, L, C), nosiy pose observation
            length: (B), valid length of x, if None then use x.shape[2]
            f_imgseq: (B, L, C)
            f_cliffcam: (B, L, 3), CLIFF-Cam parameters (bbx-detection in the full-image)
            f_cam_angvel: (B, L, 6), Camera angular velocity
            f_cam_transl: (B, L, C), Camera translation, normalized or scaled
        """
        B, L, J, C = obs.shape
        assert J == 17 and C == 3

        # Main token from observation (2D pose)
        obs = obs.clone()
        visible_mask = obs[..., [2]] > 0.5  # (B, L, J, 1)
        obs[~visible_mask[..., 0]] = 0  # set low-conf to all zeros
        f_obs = self.learned_pos_linear(obs[..., :2])  # (B, L, J, 32)
        f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
        x = self.embed_noisyobs(f_obs.view(B, L, -1))  # (B, L, J*32) -> (B, L, C)

        # Condition
        f_to_add = [self.cliffcam_embedder(f_cliffcam)]
        if hasattr(self, "cam_angvel_embedder"):
            f_to_add.append(self.cam_angvel_embedder(f_cam_angvel))
        if f_imgseq is not None and hasattr(self, "imgseq_embedder"):
            f_to_add.append(self.imgseq_embedder(f_imgseq))
        if f_cam_transl is not None and hasattr(self, "cam_traj_embedder"):
            f_to_add.append(self.cam_traj_embedder(f_cam_transl))

        for f_delta in f_to_add:
            x = x + f_delta

        # Setup length and make padding mask
        assert B == length.size(0)
        pmask = ~length_to_mask(length, L)  # (B, L)

        if L > self.max_len:
            attnmask = torch.ones((L, L), device=x.device, dtype=torch.bool)
            for i in range(L):
                min_ind, max_ind = max(0, i - self.max_len // 2), min(L, i + self.max_len // 2)
                min_ind, max_ind = min(L - self.max_len, min_ind), max(self.max_len, max_ind)
                attnmask[i, min_ind:max_ind] = False
        else:
            attnmask = None

        # Transformer
        for block in self.transformer_blocks:
            x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)

        pred_hmr = self.head_hmr(x)  # (B, L, C)
        if self.avgbeta:
            betas = (pred_hmr[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None]  # (B, C)
            betas = repeat(betas, "b c -> b l c", l=L)
            pred_hmr = torch.cat([pred_hmr[..., :126], betas, pred_hmr[..., 136:]], dim=-1)

        # CLIFF-Cam Head
        pred_weak_cam = None
        if self.head_weak_cam:
            pred_weak_cam = self.head_weak_cam(x)
            pred_weak_cam = pred_weak_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_weak_cam[..., 0], 0.25)  # min_clamp s to 0.25 (prevent negative prediction)

        static_conf_logits = None
        if self.head_static_conf:
            static_conf_logits = self.head_static_conf(x)  # (B, L, C')

        output = {
            "pred_context": x,
            "pred_hmr": pred_hmr[..., :142],
            "pred_cam_move": pred_hmr[..., 142:151],
            "pred_human_move": pred_hmr[..., 151:],
            "pred_weak_cam": pred_weak_cam,
            "static_conf_logits": static_conf_logits,
        }
        return output


class NetworkEncoderRoPEFusion(nn.Module):
    """
    Extended version of NetworkEncoderRoPE with configurable condition fusion mechanism.
    Supports: "add" (original), "gate" (gated fusion), "adaln" (AdaLN conditioning).

    - fusion_mode="add": Exact same behavior as NetworkEncoderRoPE (backward compatible).
    - fusion_mode="gate": Learned gating on aggregated condition before adding to main token.
    - fusion_mode="adaln": Condition is projected to per-layer scale/shift for AdaLN modulation.
    """
    def __init__(
        self,
        max_len=120,  # max sequence length
        cliffcam_dim=3, cam_angvel_dim=6, cam_traj_dim=3, imgseq_dim=1024,  # condition
        latent_dim=512, num_layers=12, num_heads=8, mlp_ratio=4.0,  # intermediate
        # head output dim
        dim_hmr=142,  # 21*6 body_pose + 10 betas + 6 global_orient_in_cam
        dim_weak_cam=3,  # CLIFF camera params for weak projection
        dim_cam_move=15,  # 6 cam_roll_pitch_pose + 6 angle_vel_yaw + 3 cam_local_vel
        dim_human_move=15,  # 6 human_roll_pitch_pose + 6 angle_vel_yaw + 3 human_local_vel
        dim_static_conf=6,  # static configuration parameters
        dropout=0.1, avgbeta=True,
        fusion_mode="add",  # "add" | "gate" | "adaln"
    ):
        super().__init__()
        self.max_len = max_len
        self.fusion_mode = fusion_mode

        # condition
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.imgseq_dim = imgseq_dim
        self.cam_traj_dim = cam_traj_dim

        # intermediate
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # ===== build model ===== #
        # Input (Kp2d)
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(17, 32), requires_grad=True)
        self.embed_noisyobs = Mlp(
            17 * 32, hidden_features=self.latent_dim * 2, out_features=self.latent_dim, drop=dropout
        )

        self._build_condition_embedder()
        self._build_fusion_module()

        # Transformer
        self.transformer_blocks = nn.ModuleList([
            EncoderRoPEBlock(self.latent_dim, self.num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(self.num_layers)
        ])

        # Heads (same structure as NetworkEncoderRoPE)
        self.head_hmr = Mlp(self.latent_dim, out_features=dim_hmr + dim_cam_move + dim_human_move)
        self.head_cam_move, self.head_human_move = False, False

        self.head_weak_cam = dim_weak_cam > 0
        if self.head_weak_cam:
            self.head_weak_cam = Mlp(self.latent_dim, out_features=dim_weak_cam)
            self.register_buffer("pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False)
            self.register_buffer("pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False)

        self.head_static_conf = dim_static_conf > 0
        if self.head_static_conf:
            self.head_static_conf = Mlp(self.latent_dim, out_features=dim_static_conf)

        self.avgbeta = avgbeta

    def _build_condition_embedder(self):
        latent_dim = self.latent_dim
        dropout = self.dropout
        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(self.cliffcam_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )
        if self.cam_angvel_dim > 0:
            self.cam_angvel_embedder = nn.Sequential(
                nn.Linear(self.cam_angvel_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
        if self.imgseq_dim > 0:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(self.imgseq_dim),
                zero_module(nn.Linear(self.imgseq_dim, latent_dim)),
            )
        if self.cam_traj_dim > 0:
            self.cam_traj_embedder = nn.Sequential(
                nn.Linear(self.cam_traj_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )

    def _build_fusion_module(self):
        """Build the condition fusion module according to self.fusion_mode."""
        d = self.latent_dim
        N = self.num_layers
        if self.fusion_mode == "add":
            pass  # No extra modules needed
        elif self.fusion_mode == "gate":
            # Gated fusion: gate = sigmoid(proj(cond)), output = x + gate * cond
            self.gate_proj = nn.Sequential(
                nn.Linear(d, d),
                nn.Sigmoid(),
            )
        elif self.fusion_mode == "adaln":
            # AdaLN: condition -> per-layer (scale_msa, shift_msa, scale_mlp, shift_mlp)
            # Output dim = num_layers * 4 * latent_dim
            self.adaln_proj = nn.Sequential(
                nn.SiLU(),
                zero_module(nn.Linear(d, N * 4 * d)),
            )
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}. Choose from 'add', 'gate', 'adaln'.")

    def _embed_conditions(self, f_cliffcam, f_cam_angvel, f_cam_transl, f_imgseq):
        """Embed all conditions and return a list of condition tensors."""
        f_to_add = [self.cliffcam_embedder(f_cliffcam)]
        if hasattr(self, "cam_angvel_embedder"):
            f_to_add.append(self.cam_angvel_embedder(f_cam_angvel))
        if f_imgseq is not None and hasattr(self, "imgseq_embedder"):
            f_to_add.append(self.imgseq_embedder(f_imgseq))
        if f_cam_transl is not None and hasattr(self, "cam_traj_embedder"):
            f_to_add.append(self.cam_traj_embedder(f_cam_transl))
        return f_to_add

    def forward(self, length, obs=None, f_cliffcam=None, f_cam_angvel=None, f_cam_transl=None, f_imgseq=None):
        """
        Args:
            obs: (B, L, J, 3), noisy pose observation
            length: (B), valid length of x
            f_imgseq: (B, L, C)
            f_cliffcam: (B, L, 3), CLIFF-Cam parameters
            f_cam_angvel: (B, L, 6), Camera angular velocity
            f_cam_transl: (B, L, C), Camera translation
        """
        B, L, J, C = obs.shape
        assert J == 17 and C == 3

        # Main token from observation (2D pose)
        obs = obs.clone()
        visible_mask = obs[..., [2]] > 0.5  # (B, L, J, 1)
        obs[~visible_mask[..., 0]] = 0
        f_obs = self.learned_pos_linear(obs[..., :2])  # (B, L, J, 32)
        f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
        x = self.embed_noisyobs(f_obs.view(B, L, -1))  # (B, L, C)

        # Embed conditions
        f_to_add = self._embed_conditions(f_cliffcam, f_cam_angvel, f_cam_transl, f_imgseq)

        # Apply fusion
        adaln_params = None
        if self.fusion_mode == "add":
            for f_delta in f_to_add:
                x = x + f_delta
        elif self.fusion_mode == "gate":
            cond = sum(f_to_add)
            gate = self.gate_proj(cond)
            x = x + gate * cond
        elif self.fusion_mode == "adaln":
            cond = sum(f_to_add)
            x = x + cond  # still add cond as input (like a residual)
            adaln_params = self.adaln_proj(cond)  # (B, L, N*4*D)

        # Setup length and make padding mask
        assert B == length.size(0)
        pmask = ~length_to_mask(length, L)  # (B, L)

        if L > self.max_len:
            attnmask = torch.ones((L, L), device=x.device, dtype=torch.bool)
            for i in range(L):
                min_ind, max_ind = max(0, i - self.max_len // 2), min(L, i + self.max_len // 2)
                min_ind, max_ind = min(L - self.max_len, min_ind), max(self.max_len, max_ind)
                attnmask[i, min_ind:max_ind] = False
        else:
            attnmask = None

        # Transformer
        d = self.latent_dim
        for i, block in enumerate(self.transformer_blocks):
            if adaln_params is not None:
                # Extract per-layer AdaLN params: (scale_msa, shift_msa, scale_mlp, shift_mlp)
                offset = i * 4 * d
                scale_msa = adaln_params[..., offset:offset + d]
                shift_msa = adaln_params[..., offset + d:offset + 2 * d]
                scale_mlp = adaln_params[..., offset + 2 * d:offset + 3 * d]
                shift_mlp = adaln_params[..., offset + 3 * d:offset + 4 * d]
                x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask,
                          adaln_scale_msa=scale_msa, adaln_shift_msa=shift_msa,
                          adaln_scale_mlp=scale_mlp, adaln_shift_mlp=shift_mlp)
            else:
                x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)

        # Heads
        pred_hmr = self.head_hmr(x)  # (B, L, C)
        if self.avgbeta:
            betas = (pred_hmr[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None]
            betas = repeat(betas, "b c -> b l c", l=L)
            pred_hmr = torch.cat([pred_hmr[..., :126], betas, pred_hmr[..., 136:]], dim=-1)

        pred_cam_move = None
        if self.head_cam_move:
            pred_cam_move = self.head_cam_move(x)
        pred_human_move = None
        if self.head_human_move:
            pred_human_move = self.head_human_move(x)

        pred_weak_cam = None
        if self.head_weak_cam:
            pred_weak_cam = self.head_weak_cam(x)
            pred_weak_cam = pred_weak_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_weak_cam[..., 0], 0.25)

        static_conf_logits = None
        if self.head_static_conf:
            static_conf_logits = self.head_static_conf(x)

        output = {
            "pred_context": x,
            "pred_hmr": pred_hmr[..., :142],
            "pred_cam_move": pred_hmr[..., 142:157],
            "pred_human_move": pred_hmr[..., 157:],
            "pred_weak_cam": pred_weak_cam,
            "static_conf_logits": static_conf_logits,
        }
        return output


class NetworkEncoderRoPECamCross(nn.Module):
    """
    Improved camera feature input for WATCH Transformer.

    Key improvements over NetworkEncoderRoPE:
    1. Joint camera embedding: concat all camera geometric features -> single MLP
       (captures cross-feature correlations, e.g. static camera has both angvel~I and transl~0)
    2. Cross-attention: body tokens cross-attend to camera context every K layers
       (selective use of camera info, adaptive to static/dynamic scenarios)
    3. f_imgseq (human appearance): keeps additive fusion (naturally aligned with body)

    cam_input_mode:
      "separate" : original separate embedders + sum (backward compatible with NetworkEncoderRoPE)
      "joint"    : concat(cliffcam, angvel, transl) -> joint MLP

    cam_fusion_mode:
      "add"       : all conditions added at input (identical to NetworkEncoderRoPE when cam_input_mode="separate")
      "cross_attn": camera -> cross-attention only; f_imgseq -> addition
      "add_cross" : camera -> added at input + cross-attention at deeper layers
    """
    def __init__(
        self,
        max_len=120,  # max sequence length
        cliffcam_dim=3, cam_angvel_dim=6, cam_traj_dim=3, imgseq_dim=1024,  # condition
        latent_dim=512, num_layers=12, num_heads=8, mlp_ratio=4.0,  # intermediate
        # head output dim
        dim_hmr=142,  # 21*6 body_pose + 10 betas + 6 global_orient_in_cam
        dim_weak_cam=3,  # CLIFF camera params for weak projection
        dim_cam_move=15,  # 6 cam_roll_pitch_pose + 6 angle_vel_yaw + 3 cam_local_vel
        dim_human_move=15,  # 6 human_roll_pitch_pose + 6 angle_vel_yaw + 3 human_local_vel
        dim_static_conf=6,  # static configuration parameters
        dropout=0.1, avgbeta=True,
        # ---- New parameters for camera input improvement ----
        cam_input_mode="joint",        # "separate" | "joint"
        cam_fusion_mode="cross_attn",  # "add" | "cross_attn" | "add_cross"
        cross_attn_interval=4,         # add cross-attn every K layers
        cross_attn_heads=4,            # number of heads for cross-attn
    ):
        super().__init__()
        self.max_len = max_len
        self.cam_input_mode = cam_input_mode
        self.cam_fusion_mode = cam_fusion_mode
        self.cross_attn_interval = cross_attn_interval

        # condition dims
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.imgseq_dim = imgseq_dim
        self.cam_traj_dim = cam_traj_dim

        # intermediate
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # ===== Build model ===== #
        # Input (Kp2d) — identical to NetworkEncoderRoPE
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(17, 32), requires_grad=True)
        self.embed_noisyobs = Mlp(
            17 * 32, hidden_features=latent_dim * 2, out_features=latent_dim, drop=dropout
        )

        # Camera condition embedders
        self._build_camera_embedder()

        # Image feature embedder (always additive, unchanged)
        if imgseq_dim > 0:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(imgseq_dim),
                zero_module(nn.Linear(imgseq_dim, latent_dim)),
            )

        # Cross-attention layers (only when needed)
        if cam_fusion_mode in ("cross_attn", "add_cross"):
            num_cross_attn = num_layers // cross_attn_interval
            self.cross_attn_layers = nn.ModuleList([
                CrossAttentionBlock(latent_dim, num_heads=cross_attn_heads, dropout=dropout)
                for _ in range(num_cross_attn)
            ])
            # Layer indices after which cross-attention is applied
            self.cross_attn_layer_indices = set(
                (i + 1) * cross_attn_interval - 1 for i in range(num_cross_attn)
            )

        # Transformer (self-attention blocks)
        self.transformer_blocks = nn.ModuleList([
            EncoderRoPEBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Heads — identical to NetworkEncoderRoPE
        self.head_hmr = Mlp(latent_dim, out_features=dim_hmr + dim_cam_move + dim_human_move)
        self.head_cam_move, self.head_human_move = False, False

        self.head_weak_cam = dim_weak_cam > 0
        if self.head_weak_cam:
            self.head_weak_cam = Mlp(latent_dim, out_features=dim_weak_cam)
            self.register_buffer("pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False)
            self.register_buffer("pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False)

        self.head_static_conf = dim_static_conf > 0
        if self.head_static_conf:
            self.head_static_conf = Mlp(latent_dim, out_features=dim_static_conf)

        self.avgbeta = avgbeta

    def _build_camera_embedder(self):
        """Build camera feature embedder based on cam_input_mode."""
        d = self.latent_dim
        dp = self.dropout

        if self.cam_input_mode == "separate":
            # Original: separate embedders for each camera feature
            self.cliffcam_embedder = nn.Sequential(
                nn.Linear(self.cliffcam_dim, d), nn.SiLU(), nn.Dropout(dp),
                zero_module(nn.Linear(d, d)),
            )
            if self.cam_angvel_dim > 0:
                self.cam_angvel_embedder = nn.Sequential(
                    nn.Linear(self.cam_angvel_dim, d), nn.SiLU(), nn.Dropout(dp),
                    zero_module(nn.Linear(d, d)),
                )
            if self.cam_traj_dim > 0:
                self.cam_traj_embedder = nn.Sequential(
                    nn.Linear(self.cam_traj_dim, d), nn.SiLU(), nn.Dropout(dp),
                    zero_module(nn.Linear(d, d)),
                )

        elif self.cam_input_mode == "joint":
            # New: concat all camera features -> joint embedding
            joint_dim = self.cliffcam_dim + self.cam_angvel_dim + self.cam_traj_dim  # default: 3+6+3=12
            self.joint_cam_embedder = nn.Sequential(
                nn.Linear(joint_dim, d),
                nn.SiLU(),
                nn.Dropout(dp),
                nn.Linear(d, d),
                nn.SiLU(),
                nn.Dropout(dp),
                zero_module(nn.Linear(d, d)),
            )
        else:
            raise ValueError(f"Unknown cam_input_mode: {self.cam_input_mode}")

    def _embed_camera(self, f_cliffcam, f_cam_angvel, f_cam_transl):
        """Embed camera features into a single tensor (B, L, D)."""
        if self.cam_input_mode == "separate":
            cam_embeds = [self.cliffcam_embedder(f_cliffcam)]
            if hasattr(self, "cam_angvel_embedder") and f_cam_angvel is not None:
                cam_embeds.append(self.cam_angvel_embedder(f_cam_angvel))
            if hasattr(self, "cam_traj_embedder") and f_cam_transl is not None:
                cam_embeds.append(self.cam_traj_embedder(f_cam_transl))
            return sum(cam_embeds)

        elif self.cam_input_mode == "joint":
            parts = [f_cliffcam]
            if self.cam_angvel_dim > 0:
                if f_cam_angvel is not None:
                    parts.append(f_cam_angvel)
                else:
                    parts.append(torch.zeros(*f_cliffcam.shape[:-1], self.cam_angvel_dim,
                                             device=f_cliffcam.device, dtype=f_cliffcam.dtype))
            if self.cam_traj_dim > 0:
                if f_cam_transl is not None:
                    parts.append(f_cam_transl)
                else:
                    parts.append(torch.zeros(*f_cliffcam.shape[:-1], self.cam_traj_dim,
                                             device=f_cliffcam.device, dtype=f_cliffcam.dtype))
            cam_concat = torch.cat(parts, dim=-1)  # (B, L, joint_dim)
            return self.joint_cam_embedder(cam_concat)

    def forward(self, length, obs=None, f_cliffcam=None, f_cam_angvel=None, f_cam_transl=None, f_imgseq=None):
        """
        Same interface as NetworkEncoderRoPE.forward() for drop-in replacement.
        """
        B, L, J, C = obs.shape
        assert J == 17 and C == 3

        # Main token from observation (2D pose) — identical to original
        obs = obs.clone()
        visible_mask = obs[..., [2]] > 0.5  # (B, L, J, 1)
        obs[~visible_mask[..., 0]] = 0
        f_obs = self.learned_pos_linear(obs[..., :2])  # (B, L, J, 32)
        f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
        x = self.embed_noisyobs(f_obs.view(B, L, -1))  # (B, L, D)

        # Camera embedding
        cam_embed = self._embed_camera(f_cliffcam, f_cam_angvel, f_cam_transl)

        # Image feature embedding (always additive — human-centric, naturally body-aligned)
        if f_imgseq is not None and hasattr(self, "imgseq_embedder"):
            x = x + self.imgseq_embedder(f_imgseq)

        # Apply camera fusion
        cam_context = None  # will be set if cross-attention is used
        if self.cam_fusion_mode == "add":
            # Original behavior: add camera embedding to main token
            x = x + cam_embed
        elif self.cam_fusion_mode == "cross_attn":
            # Camera only through cross-attention (not added at input)
            cam_context = cam_embed
        elif self.cam_fusion_mode == "add_cross":
            # Camera added at input + cross-attention at deeper layers
            x = x + cam_embed
            cam_context = cam_embed
        else:
            raise ValueError(f"Unknown cam_fusion_mode: {self.cam_fusion_mode}")

        # Setup length and padding mask
        assert B == length.size(0)
        pmask = ~length_to_mask(length, L)  # (B, L), True=padding

        if L > self.max_len:
            attnmask = torch.ones((L, L), device=x.device, dtype=torch.bool)
            for i in range(L):
                min_ind, max_ind = max(0, i - self.max_len // 2), min(L, i + self.max_len // 2)
                min_ind, max_ind = min(L - self.max_len, min_ind), max(self.max_len, max_ind)
                attnmask[i, min_ind:max_ind] = False
        else:
            attnmask = None

        # Transformer with optional cross-attention
        cross_attn_idx = 0
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)

            # Cross-attention to camera context after specific layers
            if cam_context is not None and hasattr(self, 'cross_attn_layers'):
                if i in self.cross_attn_layer_indices:
                    x = self.cross_attn_layers[cross_attn_idx](x, cam_context, key_padding_mask=pmask)
                    cross_attn_idx += 1

        # ===== Heads — identical to NetworkEncoderRoPE ===== #
        pred_hmr = self.head_hmr(x)  # (B, L, dim_hmr + dim_cam_move + dim_human_move)
        if self.avgbeta:
            betas = (pred_hmr[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None]
            betas = repeat(betas, "b c -> b l c", l=L)
            pred_hmr = torch.cat([pred_hmr[..., :126], betas, pred_hmr[..., 136:]], dim=-1)

        pred_cam_move = None
        if self.head_cam_move:
            pred_cam_move = self.head_cam_move(x)
        pred_human_move = None
        if self.head_human_move:
            pred_human_move = self.head_human_move(x)

        pred_weak_cam = None
        if self.head_weak_cam:
            pred_weak_cam = self.head_weak_cam(x)
            pred_weak_cam = pred_weak_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_weak_cam[..., 0], 0.25)

        static_conf_logits = None
        if self.head_static_conf:
            static_conf_logits = self.head_static_conf(x)

        output = {
            "pred_context": x,
            "pred_hmr": pred_hmr[..., :142],
            "pred_cam_move": pred_hmr[..., 142:157],
            "pred_human_move": pred_hmr[..., 157:],
            "pred_weak_cam": pred_weak_cam,
            "static_conf_logits": static_conf_logits,
        }
        return output


# Add to MainStore
group_name = "network/watch"
MainStore.store(
    name="relative_transformer",
    node=builds(NetworkEncoderRoPE, populate_full_signature=True),
    group=group_name,
)
MainStore.store(
    name="relative_transformer_v4",
    node=builds(NetworkEncoderRoPENew, populate_full_signature=True),
    group=group_name,
)
# New: configurable fusion mode variants
MainStore.store(
    name="relative_transformer_gate",
    node=builds(NetworkEncoderRoPEFusion, populate_full_signature=True, fusion_mode="gate"),
    group=group_name,
)
MainStore.store(
    name="relative_transformer_adaln",
    node=builds(NetworkEncoderRoPEFusion, populate_full_signature=True, fusion_mode="adaln"),
    group=group_name,
)
MainStore.store(
    name="relative_transformer_fusion",
    node=builds(NetworkEncoderRoPEFusion, populate_full_signature=True),
    group=group_name,
)
# Camera cross-attention variants
MainStore.store(
    name="relative_transformer_cam_cross",
    node=builds(NetworkEncoderRoPECamCross, populate_full_signature=True,
                cam_input_mode="joint", cam_fusion_mode="cross_attn"),
    group=group_name,
)
MainStore.store(
    name="relative_transformer_cam_add_cross",
    node=builds(NetworkEncoderRoPECamCross, populate_full_signature=True,
                cam_input_mode="joint", cam_fusion_mode="add_cross"),
    group=group_name,
)
MainStore.store(
    name="relative_transformer_cam_joint_add",
    node=builds(NetworkEncoderRoPECamCross, populate_full_signature=True,
                cam_input_mode="joint", cam_fusion_mode="add"),
    group=group_name,
)