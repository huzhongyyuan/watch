"""
Dual-Stream Backbone for WATCH

Key idea: separate Body Stream (handles human pose) from Camera Stream
(handles cam motion). Cross-attention every K layers lets body query camera.

Why: Single backbone (current WATCH/GVHMR) has shared weights, gradient
dominated by HMR loss (×500 weight) → cam predictions get crumbs.
Dual stream gives camera dedicated capacity.

Architecture:
    Body Stream:    [B_layer x 12]
                    ↑↓ Cross-attn every 3 layers
    Camera Stream:  [C_layer x 6]

    Body input: kp2d obs + cliffcam + image features
    Camera input: cam_angvel + cam_transl + scene features (optional)

This is a NEW network class, not modifying NetworkEncoderRoPE.

Drop-in replacement: register as `network/watch/dual_stream`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp

from core.network.base_arch.transformer.encoder_rope import EncoderRoPEBlock
from core.network.base_arch.transformer.layer import zero_module
from core.utils.net_utils import length_to_mask
from core.configs import MainStore, builds


class CrossAttentionBlock(nn.Module):
    """Cross-attention from query stream to context stream.

    Query (body) attends to context (camera).
    Output projection is zero-init for stable residual training.
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(embed_dim, eps=1e-6)

        self.to_q = nn.Linear(embed_dim, embed_dim)
        self.to_kv = nn.Linear(embed_dim, embed_dim * 2)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Zero-init for stable residual
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # Learnable gate (start at 0)
        self.gate = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x, context, key_padding_mask=None):
        """
        Args:
            x: (B, L_q, D) query (body)
            context: (B, L_kv, D) key-value (camera)
            key_padding_mask: (B, L_kv) True for padding

        Returns:
            x + gate * cross_attn(x, context)
        """
        B, L_q, D = x.shape
        L_kv = context.shape[1]

        q = self.to_q(self.norm_q(x))
        kv = self.to_kv(self.norm_kv(context))
        k, v = kv.chunk(2, dim=-1)

        q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L_q, D)
        out = self.proj(out)

        return x + self.gate * out


class DualStreamEncoder(nn.Module):
    """Dual-stream Transformer encoder for WATCH.

    Drop-in replacement for NetworkEncoderRoPE.
    Same forward signature, returns same dict format.

    Args:
        body_layers: depth of body stream
        cam_layers: depth of camera stream
        cross_attn_interval: insert cross-attn every N body layers
        ...other args same as NetworkEncoderRoPE...
    """

    def __init__(
        self,
        max_len=120,
        cliffcam_dim=3, cam_angvel_dim=6, cam_traj_dim=3, imgseq_dim=1024,
        latent_dim=512, body_layers=12, cam_layers=6, num_heads=8, mlp_ratio=4.0,
        cross_attn_interval=3,
        dim_hmr=142, dim_weak_cam=3,
        dim_cam_move=15, dim_human_move=15,
        dim_static_conf=6,
        dropout=0.1, avgbeta=True,
    ):
        super().__init__()
        self.max_len = max_len
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.imgseq_dim = imgseq_dim
        self.cam_traj_dim = cam_traj_dim
        self.latent_dim = latent_dim
        self.body_layers = body_layers
        self.cam_layers = cam_layers
        self.cross_attn_interval = cross_attn_interval
        self.num_heads = num_heads
        self.dropout = dropout
        self.avgbeta = avgbeta

        # ===== Body stream input ===== #
        # Main token: 2D keypoints
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(17, 32), requires_grad=True)
        self.embed_noisyobs = Mlp(
            17 * 32, hidden_features=latent_dim * 2, out_features=latent_dim, drop=dropout
        )

        # Body conditions (image features + cliffcam)
        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(cliffcam_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )
        if imgseq_dim > 0:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(imgseq_dim),
                zero_module(nn.Linear(imgseq_dim, latent_dim)),
            )

        # ===== Camera stream input ===== #
        # Camera-specific tokens: cam_angvel + cam_transl
        self.cam_token_init = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        if cam_angvel_dim > 0:
            self.cam_angvel_embedder = nn.Sequential(
                nn.Linear(cam_angvel_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, latent_dim),
            )
        if cam_traj_dim > 0:
            self.cam_traj_embedder = nn.Sequential(
                nn.Linear(cam_traj_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, latent_dim),
            )

        # ===== Body stream blocks ===== #
        self.body_blocks = nn.ModuleList([
            EncoderRoPEBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(body_layers)
        ])

        # ===== Camera stream blocks ===== #
        self.cam_blocks = nn.ModuleList([
            EncoderRoPEBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(cam_layers)
        ])

        # ===== Cross-attention layers ===== #
        # Insert at end of body block i where (i+1) % interval == 0
        num_cross = body_layers // cross_attn_interval
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(latent_dim, num_heads=num_heads // 2, dropout=dropout)
            for _ in range(num_cross)
        ])

        # ===== Output heads ===== #
        # Body output: HMR
        self.head_hmr = Mlp(latent_dim, out_features=dim_hmr)

        # Camera output: cam_move (cam transl_vel + roll_pitch + angvel)
        self.head_cam_move = Mlp(latent_dim, out_features=dim_cam_move)

        # Human move output (from body stream, but separate head)
        self.head_human_move = Mlp(latent_dim, out_features=dim_human_move)

        # Auxiliary heads
        self.head_weak_cam = dim_weak_cam > 0
        if self.head_weak_cam:
            self.head_weak_cam = Mlp(latent_dim, out_features=dim_weak_cam)
            self.register_buffer("pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False)
            self.register_buffer("pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False)

        self.head_static_conf = dim_static_conf > 0
        if self.head_static_conf:
            self.head_static_conf = Mlp(latent_dim, out_features=dim_static_conf)

    def forward(self, length, obs=None, f_cliffcam=None, f_cam_angvel=None,
                f_cam_transl=None, f_imgseq=None):
        """
        Args:
            length: (B,)
            obs: (B, L, J=17, 3) noisy 2D pose with visibility
            f_cliffcam: (B, L, 3)
            f_cam_angvel: (B, L, 6)
            f_cam_transl: (B, L, 3)
            f_imgseq: (B, L, 1024)
        """
        B, L = obs.shape[:2]
        device = obs.device

        # Padding mask
        pmask = ~length_to_mask(length, max_len=L)  # (B, L) True = padding

        # ===== Body stream tokens ===== #
        obs = obs.clone()
        visible_mask = obs[..., [2]] > 0.5
        obs[~visible_mask[..., 0]] = 0
        f_obs = self.learned_pos_linear(obs[..., :2])
        f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
        body_x = self.embed_noisyobs(f_obs.view(B, L, -1))  # (B, L, D)

        if f_cliffcam is not None:
            body_x = body_x + self.cliffcam_embedder(f_cliffcam)
        if f_imgseq is not None and self.imgseq_dim > 0:
            body_x = body_x + self.imgseq_embedder(f_imgseq)

        # ===== Camera stream tokens ===== #
        cam_x = self.cam_token_init.expand(B, L, -1)
        if f_cam_angvel is not None and self.cam_angvel_dim > 0:
            cam_x = cam_x + self.cam_angvel_embedder(f_cam_angvel)
        if f_cam_transl is not None and self.cam_traj_dim > 0:
            cam_x = cam_x + self.cam_traj_embedder(f_cam_transl)

        # ===== Forward both streams with cross-attention ===== #
        cam_block_idx = 0
        cross_idx = 0
        for body_idx, body_block in enumerate(self.body_blocks):
            # Body self-attention
            body_x = body_block(body_x, attn_mask=None, tgt_key_padding_mask=pmask)

            # Camera self-attention (interleaved, fewer layers)
            if cam_block_idx < self.cam_layers:
                # Step camera stream proportionally
                steps = max(1, self.body_layers // self.cam_layers)
                if (body_idx + 1) % steps == 0 and cam_block_idx < self.cam_layers:
                    cam_x = self.cam_blocks[cam_block_idx](
                        cam_x, attn_mask=None, tgt_key_padding_mask=pmask
                    )
                    cam_block_idx += 1

            # Cross-attention from body to camera (every K layers)
            if (body_idx + 1) % self.cross_attn_interval == 0 and cross_idx < len(self.cross_attn_blocks):
                body_x = self.cross_attn_blocks[cross_idx](body_x, cam_x, key_padding_mask=pmask)
                cross_idx += 1

        # ===== Output heads ===== #
        # Body stream produces hmr + human_move
        pred_hmr = self.head_hmr(body_x)  # (B, L, 142)
        if self.avgbeta:
            betas = (pred_hmr[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None]
            betas = repeat(betas, "b c -> b l c", l=L)
            pred_hmr = torch.cat([pred_hmr[..., :126], betas, pred_hmr[..., 136:]], dim=-1)

        pred_human_move = self.head_human_move(body_x)  # (B, L, 15)

        # Camera stream produces cam_move
        pred_cam_move = self.head_cam_move(cam_x)  # (B, L, 15)

        # Auxiliary outputs (from body stream)
        pred_weak_cam = None
        if self.head_weak_cam:
            pred_weak_cam = self.head_weak_cam(body_x)
            pred_weak_cam = pred_weak_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_weak_cam[..., 0], 0.25)

        static_conf_logits = None
        if self.head_static_conf:
            static_conf_logits = self.head_static_conf(body_x)

        return {
            "pred_context": body_x,
            "pred_cam_context": cam_x,
            "pred_hmr": pred_hmr,
            "pred_cam_move": pred_cam_move,
            "pred_human_move": pred_human_move,
            "pred_weak_cam": pred_weak_cam,
            "static_conf_logits": static_conf_logits,
        }


# Register
MainStore.store(
    name="dual_stream",
    node=builds(DualStreamEncoder, populate_full_signature=True),
    group="network/watch",
)
