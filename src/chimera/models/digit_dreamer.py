"""DigitDreamer: a small SD3-style MM-DiT (multimodal diffusion transformer).

Two token streams are processed jointly: the *image* stream (patchified spatial
latents) and a *conditioning* stream (class-label tokens). Each stream keeps its
own LayerNorm/adaLN modulation, QKV projections, output projection and MLP, but
the two are concatenated for a single joint self-attention so information flows
between them. A global conditioning vector ``c`` (timestep + class embedding)
drives adaLN-Zero modulation throughout, matching the DiT / SD3 recipe.

The network predicts the rectified-flow velocity for the image latent; the
conditioning stream is discarded at the output.

Reuses the ``MLP`` primitive and the ``F.scaled_dot_product_attention`` idiom
from ``chimera.models.gpt`` (here non-causal, no RoPE / KV-cache).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpt import MLP


def modulate(x, shift, scale):
    """adaLN modulation: ``x * (1 + scale) + shift`` (scale/shift are (B, D))."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embed a scalar timestep in ``[0, 1]`` with a sinusoidal + MLP head."""

    def __init__(self, dim: int, freq_dim: int = 256, max_period: float = 10000.0):
        super().__init__()
        self.freq_dim = freq_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def _sinusoidal(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float()[:, None] * freqs[None]  # (B, half)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, freq_dim)

    def forward(self, t):
        emb = self._sinusoidal(t).to(self.mlp[0].weight.dtype)
        return self.mlp(emb)


class JointAttention(nn.Module):
    """Multi-head self-attention over the concatenation of two streams.

    Each stream owns its QKV and output projections; attention is computed on the
    concatenated ``[cond ; image]`` sequence so the streams attend to each other.
    """

    def __init__(self, dim: int, n_head: int):
        super().__init__()
        assert dim % n_head == 0, "dim must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = dim // n_head

        self.qkv_img = nn.Linear(dim, 3 * dim)
        self.qkv_cond = nn.Linear(dim, 3 * dim)
        self.proj_img = nn.Linear(dim, dim)
        self.proj_cond = nn.Linear(dim, dim)

    def _split_heads(self, qkv, B, T):
        # (B, T, 3*dim) -> 3 x (B, n_head, T, head_dim)
        q, k, v = qkv.view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        return q, k, v

    def forward(self, img, cond):
        B, Ti, _ = img.shape
        Tc = cond.shape[1]

        qi, ki, vi = self._split_heads(self.qkv_img(img), B, Ti)
        qc, kc, vc = self._split_heads(self.qkv_cond(cond), B, Tc)

        # Joint attention over [cond ; image] along the sequence dimension.
        q = torch.cat([qc, qi], dim=2)
        k = torch.cat([kc, ki], dim=2)
        v = torch.cat([vc, vi], dim=2)

        out = F.scaled_dot_product_attention(q, k, v)  # non-causal
        out = out.transpose(1, 2).reshape(B, Tc + Ti, self.n_head * self.head_dim)

        cond_out, img_out = out[:, :Tc], out[:, Tc:]
        return self.proj_img(img_out), self.proj_cond(cond_out)


class DigitDreamerBlock(nn.Module):
    """One dual-stream MM-DiT block: joint attention + per-stream MLP, adaLN-Zero."""

    def __init__(self, dim: int, n_head: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1_img = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm1_cond = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = JointAttention(dim, n_head)
        self.norm2_img = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_cond = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp_img = MLP(dim, hidden)
        self.mlp_cond = MLP(dim, hidden)

        # adaLN-Zero: regress 6 modulation vectors per stream from c.
        self.ada_img = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.ada_cond = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, img, cond, c):
        (
            shift_msa_i,
            scale_msa_i,
            gate_msa_i,
            shift_mlp_i,
            scale_mlp_i,
            gate_mlp_i,
        ) = self.ada_img(c).chunk(6, dim=-1)
        (
            shift_msa_c,
            scale_msa_c,
            gate_msa_c,
            shift_mlp_c,
            scale_mlp_c,
            gate_mlp_c,
        ) = self.ada_cond(c).chunk(6, dim=-1)

        attn_img, attn_cond = self.attn(
            modulate(self.norm1_img(img), shift_msa_i, scale_msa_i),
            modulate(self.norm1_cond(cond), shift_msa_c, scale_msa_c),
        )
        img = img + gate_msa_i.unsqueeze(1) * attn_img
        cond = cond + gate_msa_c.unsqueeze(1) * attn_cond

        img = img + gate_mlp_i.unsqueeze(1) * self.mlp_img(
            modulate(self.norm2_img(img), shift_mlp_i, scale_mlp_i)
        )
        cond = cond + gate_mlp_c.unsqueeze(1) * self.mlp_cond(
            modulate(self.norm2_cond(cond), shift_mlp_c, scale_mlp_c)
        )
        return img, cond


class FinalLayer(nn.Module):
    """adaLN-Zero + linear projection of the image stream to output channels."""

    def __init__(self, dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.ada(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class DigitDreamer(nn.Module):
    """MM-DiT velocity network over a spatial latent with class conditioning.

    Args:
        latent_channels: channels of the input latent grid.
        latent_size: spatial size of the (square) latent grid.
        patch_size: patch size for tokenizing the latent (1 keeps every cell).
        dim: transformer width.
        depth: number of MM-DiT blocks.
        n_head: attention heads.
        n_classes: number of real classes (an extra null class is added for CFG).
        n_cond_tokens: number of tokens in the conditioning stream.
        class_dropout_prob: prob. of dropping the label to null during training.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: int = 7,
        patch_size: int = 1,
        dim: int = 256,
        depth: int = 6,
        n_head: int = 4,
        n_classes: int = 10,
        n_cond_tokens: int = 4,
        class_dropout_prob: float = 0.1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        assert latent_size % patch_size == 0, "latent_size must be divisible by patch"
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.null_class = n_classes  # index of the CFG null token
        self.class_dropout_prob = class_dropout_prob

        self.grid = latent_size // patch_size
        self.n_patches = self.grid * self.grid

        # Image stream: patch-embed via a stride-p conv, plus learned pos emb.
        self.patch_embed = nn.Conv2d(
            latent_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, dim))

        # Conditioning stream: class embedding expanded to learned cond tokens.
        self.class_embed = nn.Embedding(n_classes + 1, dim)
        self.cond_tokens = nn.Parameter(torch.zeros(1, n_cond_tokens, dim))

        # Global conditioning vector: timestep + class embedding.
        self.t_embed = TimestepEmbedder(dim)

        self.blocks = nn.ModuleList(
            [DigitDreamerBlock(dim, n_head, mlp_ratio) for _ in range(depth)]
        )
        self.final = FinalLayer(dim, patch_size, latent_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_basic)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cond_tokens, std=0.02)
        nn.init.normal_(self.class_embed.weight, std=0.02)

        # Timestep MLP init.
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # adaLN-Zero: zero the modulation outputs so blocks start as identity.
        for block in self.blocks:
            nn.init.zeros_(block.ada_img[-1].weight)
            nn.init.zeros_(block.ada_img[-1].bias)
            nn.init.zeros_(block.ada_cond[-1].weight)
            nn.init.zeros_(block.ada_cond[-1].bias)
        nn.init.zeros_(self.final.ada[-1].weight)
        nn.init.zeros_(self.final.ada[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def unpatchify(self, x):
        # x: (B, n_patches, p*p*C) -> (B, C, H, W)
        B = x.shape[0]
        p, C, g = self.patch_size, self.latent_channels, self.grid
        x = x.reshape(B, g, g, p, p, C)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(B, C, g * p, g * p)

    def forward(self, z, t, y):
        """z: (B, C, H, W) noisy latent; t: (B,) in [0,1]; y: (B,) class ids."""
        if self.training and self.class_dropout_prob > 0:
            drop = torch.rand(y.shape[0], device=y.device) < self.class_dropout_prob
            y = torch.where(drop, torch.full_like(y, self.null_class), y)

        # Image tokens.
        img = self.patch_embed(z).flatten(2).transpose(1, 2)  # (B, n_patches, dim)
        img = img + self.pos_embed

        # Conditioning tokens + global vector.
        class_emb = self.class_embed(y)  # (B, dim)
        cond = self.cond_tokens + class_emb.unsqueeze(1)  # (B, n_cond, dim)
        c = self.t_embed(t) + class_emb  # (B, dim)

        for block in self.blocks:
            img, cond = block(img, cond, c)

        out = self.final(img, c)  # (B, n_patches, p*p*C)
        return self.unpatchify(out)
