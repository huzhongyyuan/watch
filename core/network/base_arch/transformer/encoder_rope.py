import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import Mlp
from typing import Optional, Tuple
from einops import einsum, rearrange, repeat
from core.network.base_arch.embeddings.rotary_embedding import ROPE


class RoPEAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.rope = ROPE(self.head_dim, max_seq_len=4096)

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, L, C)
        # attn_mask: (L, L)
        # key_padding_mask: (B, L)
        B, L, _ = x.shape
        xq, xk, xv = self.query(x), self.key(x), self.value(x)

        xq = xq.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xk = xk.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xv = xv.reshape(B, L, self.num_heads, -1).transpose(1, 2)

        xq = self.rope.rotate_queries_or_keys(xq)  # B, N, L, C
        xk = self.rope.rotate_queries_or_keys(xk)  # B, N, L, C

        attn_score = einsum(xq, xk, "b n i c, b n j c -> b n i j") / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_mask = attn_mask.reshape(1, 1, L, L).expand(B, self.num_heads, -1, -1)
            attn_score = attn_score.masked_fill(attn_mask, float("-inf"))
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.reshape(B, 1, 1, L).expand(-1, self.num_heads, L, -1)
            attn_score = attn_score.masked_fill(key_padding_mask, float("-inf"))

        attn_score = torch.softmax(attn_score, dim=-1)
        attn_score = self.dropout(attn_score)
        output = einsum(attn_score, xv, "b n i j, b n j c -> b n i c")  # B, N, L, C
        output = output.transpose(1, 2).reshape(B, L, -1)  # B, L, C
        output = self.proj(output)  # B, L, C
        return output


class EncoderRoPEBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = RoPEAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)

        self.gate_msa = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate_mlp = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Zero-out adaLN modulation layers
        nn.init.constant_(self.gate_msa, 0)
        nn.init.constant_(self.gate_mlp, 0)

    def forward(self, x, attn_mask=None, tgt_key_padding_mask=None,
                adaln_scale_msa=None, adaln_shift_msa=None,
                adaln_scale_mlp=None, adaln_shift_mlp=None):
        """
        Args:
            x: (B, L, C)
            attn_mask: (L, L) or None
            tgt_key_padding_mask: (B, L) or None
            adaln_scale_msa: (B, L, C) or None — AdaLN scale for self-attention branch
            adaln_shift_msa: (B, L, C) or None — AdaLN shift for self-attention branch
            adaln_scale_mlp: (B, L, C) or None — AdaLN scale for MLP branch
            adaln_shift_mlp: (B, L, C) or None — AdaLN shift for MLP branch
        Note:
            When all adaln_* are None, behavior is identical to the original implementation.
        """
        # Self-attention branch
        x_normed = self.norm1(x)
        if adaln_scale_msa is not None:
            x_normed = x_normed * (1 + adaln_scale_msa) + adaln_shift_msa
        x = x + self.gate_msa * self._sa_block(
            x_normed, attn_mask=attn_mask, key_padding_mask=tgt_key_padding_mask
        )
        # MLP branch
        x_normed = self.norm2(x)
        if adaln_scale_mlp is not None:
            x_normed = x_normed * (1 + adaln_scale_mlp) + adaln_shift_mlp
        x = x + self.gate_mlp * self.mlp(x_normed)
        return x

    def _sa_block(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, L, C)
        x = self.attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return x


class CrossAttentionBlock(nn.Module):
    """
    Lightweight cross-attention block for camera conditioning.
    Body tokens (query) selectively attend to camera context tokens (key/value).
    Zero-initialized output projection + learnable gate for stable residual training.
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

        # Zero-init output projection for stable residual learning
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # Learnable gate (init to 0, so cross-attn starts as identity)
        self.gate = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x, context, key_padding_mask=None):
        """
        Args:
            x: (B, L, D) body tokens (query)
            context: (B, L, D) camera context tokens (key, value)
            key_padding_mask: (B, L) True for padding positions
        Returns:
            x + gate * cross_attn(x, context)
        """
        B, L, D = x.shape

        q = self.to_q(self.norm_q(x))        # (B, L, D)
        kv = self.to_kv(self.norm_kv(context))  # (B, L, 2D)
        k, v = kv.chunk(2, dim=-1)

        # Reshape for multi-head attention
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, L, L)

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # (B, 1, 1, L)
                float('-inf')
            )

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, L, d)
        out = out.transpose(1, 2).reshape(B, L, D)  # (B, L, D)
        out = self.proj(out)

        return x + self.gate * out
