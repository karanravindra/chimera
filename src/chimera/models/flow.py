import torch
from torch import nn

from chimera.nn import AdaLNZero, DiTBlock, Mlp, SinusoidalTimeEmbedding, modulate


class VelocityDiT(nn.Module):
    """Time- and class-conditioned DiT velocity field for rectified flow on a latent.

    Predicts ``v(z_t, t, y) ~= z1 - z0`` where ``z_t = (1 - t) z0 + t z1``. The flat
    latent ``(B, latent_dim)`` is reshaped into ``num_tokens`` tokens, processed by a
    stack of DiT blocks conditioned (adaLN-Zero) on the summed sinusoidal time and
    class-label embeddings, and projected back to the latent shape.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        num_classes: int = 10,
        hidden_dim: int = 256,
        time_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        num_tokens: int = 8,
    ):
        super().__init__()
        if latent_dim % num_tokens != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by num_tokens "
                f"({num_tokens})"
            )
        self.latent_dim = latent_dim
        self.num_tokens = num_tokens
        self.token_dim = latent_dim // num_tokens

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = Mlp(time_dim, hidden_dim, hidden_dim, act_layer=nn.SiLU)
        self.class_emb = nn.Embedding(num_classes, hidden_dim)
        self.in_proj = nn.Linear(self.token_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))

        self.blocks = nn.ModuleList(
            DiTBlock(hidden_dim, num_heads) for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaln = AdaLNZero(hidden_dim, 2)
        self.out_proj = nn.Linear(hidden_dim, self.token_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_emb, std=0.02)
        # adaLN-Zero: the adaLN modulators zero-init themselves (see AdaLNZero);
        # zero the final projection too so the model starts at zero velocity.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, z: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        cond = self.time_mlp(self.time_emb(t)) + self.class_emb(y)

        x = z.view(z.size(0), self.num_tokens, self.token_dim)
        x = self.in_proj(x) + self.pos_emb
        for block in self.blocks:
            x = block(x, cond)

        shift, scale = self.final_adaln(cond).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.out_proj(x)
        return x.reshape(z.size(0), self.latent_dim)


class TextVelocityDiT(nn.Module):
    """Time- and text-conditioned DiT velocity field for rectified flow on a *spatial* latent.

    Sibling of :class:`VelocityDiT` for an autoencoder whose latent stays a feature
    map ``(B, C, H, W)`` (e.g. :class:`~chimera.models.ImageAutoEncoder`). The latent
    is patchified into ``(H/p)*(W/p)`` tokens, processed by DiT blocks conditioned
    (adaLN-Zero) on the summed sinusoidal time embedding and a projected pooled text
    vector, and unpatchified back to the latent shape. Predicts
    ``v(z_t, t, text) ~= z1 - z0`` where ``z_t = (1 - t) z0 + t z1``.

    Conditioning is a single pooled vector (no token sequence), so it enters through
    the global adaLN path rather than cross-attention. A learned ``null_text`` vector
    stands in for the text when training with conditioning dropout and for the
    unconditional pass under classifier-free guidance.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        latent_size: int = 16,
        text_dim: int = 384,
        patch_size: int = 2,
        hidden_dim: int = 384,
        time_dim: int = 256,
        depth: int = 8,
        num_heads: int = 6,
    ):
        super().__init__()
        if latent_size % patch_size != 0:
            raise ValueError(
                f"latent_size ({latent_size}) must be divisible by patch_size "
                f"({patch_size})"
            )
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.text_dim = text_dim
        self.grid = latent_size // patch_size
        self.num_tokens = self.grid * self.grid
        self.patch_dim = latent_channels * patch_size * patch_size

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = Mlp(time_dim, hidden_dim, hidden_dim, act_layer=nn.SiLU)
        self.text_proj = Mlp(text_dim, hidden_dim, hidden_dim, act_layer=nn.SiLU)
        # Learned unconditional token: swapped in for dropped/uncond rows (CFG).
        self.null_text = nn.Parameter(torch.zeros(text_dim))

        # patch_size x patch_size patch embed via a strided conv over the latent map.
        self.patchify = nn.Conv2d(
            latent_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_dim))

        self.blocks = nn.ModuleList(
            DiTBlock(hidden_dim, num_heads) for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaln = AdaLNZero(hidden_dim, 2)
        self.out_proj = nn.Linear(hidden_dim, self.patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.null_text, std=0.02)
        # adaLN-Zero modulators zero-init themselves; zero the final projection too
        # so the model starts at zero velocity (identity flow).
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, num_tokens, patch_dim)`` -> ``(B, C, H, W)``."""
        b, c, p, g = x.size(0), self.latent_channels, self.patch_size, self.grid
        x = x.view(b, g, g, c, p, p)
        x = torch.einsum("bhwcpq->bchpwq", x)
        return x.reshape(b, c, g * p, g * p)

    def forward(
        self, z: torch.Tensor, t: torch.Tensor, text: torch.Tensor
    ) -> torch.Tensor:
        cond = self.time_mlp(self.time_emb(t)) + self.text_proj(text)

        x = self.patchify(z)  # (B, hidden, grid, grid)
        x = x.flatten(2).transpose(1, 2) + self.pos_emb  # (B, num_tokens, hidden)
        for block in self.blocks:
            x = block(x, cond)

        shift, scale = self.final_adaln(cond).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.out_proj(x)  # (B, num_tokens, patch_dim)
        return self._unpatchify(x)


class ClassVelocityDiT(nn.Module):
    """Time- and class-conditioned DiT velocity field for rectified flow on a *spatial* latent.

    The class-conditioned sibling of :class:`TextVelocityDiT`: same patchified spatial-latent
    body (for an autoencoder whose latent stays a feature map ``(B, C, H, W)``, e.g.
    :class:`~chimera.models.ConvAutoEncoder`), but conditioned on a discrete class label like
    :class:`VelocityDiT` rather than on a pooled text vector. The latent is patchified into
    ``(H/p)*(W/p)`` tokens, processed by DiT blocks conditioned (adaLN-Zero) on the summed
    sinusoidal time embedding and a learned class embedding, and unpatchified back to the
    latent shape. Predicts ``v(z_t, t, y) ~= z1 - z0`` where ``z_t = (1 - t) z0 + t z1``.

    Classifier-free guidance is handled by reserving the last class index as a learned null
    class: construct with ``num_classes = NUM_CLASSES + 1`` and pass that index for dropped /
    unconditional rows (so no separate null parameter is needed -- the embedding owns it).
    """

    def __init__(
        self,
        latent_channels: int = 8,
        latent_size: int = 16,
        num_classes: int = 5,
        patch_size: int = 2,
        hidden_dim: int = 384,
        time_dim: int = 256,
        depth: int = 8,
        num_heads: int = 6,
    ):
        super().__init__()
        if latent_size % patch_size != 0:
            raise ValueError(
                f"latent_size ({latent_size}) must be divisible by patch_size "
                f"({patch_size})"
            )
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.grid = latent_size // patch_size
        self.num_tokens = self.grid * self.grid
        self.patch_dim = latent_channels * patch_size * patch_size

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = Mlp(time_dim, hidden_dim, hidden_dim, act_layer=nn.SiLU)
        self.class_emb = nn.Embedding(num_classes, hidden_dim)

        # patch_size x patch_size patch embed via a strided conv over the latent map.
        self.patchify = nn.Conv2d(
            latent_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_dim))

        self.blocks = nn.ModuleList(
            DiTBlock(hidden_dim, num_heads) for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaln = AdaLNZero(hidden_dim, 2)
        self.out_proj = nn.Linear(hidden_dim, self.patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_emb, std=0.02)
        # adaLN-Zero modulators zero-init themselves; zero the final projection too
        # so the model starts at zero velocity (identity flow).
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, num_tokens, patch_dim)`` -> ``(B, C, H, W)``."""
        b, c, p, g = x.size(0), self.latent_channels, self.patch_size, self.grid
        x = x.view(b, g, g, c, p, p)
        x = torch.einsum("bhwcpq->bchpwq", x)
        return x.reshape(b, c, g * p, g * p)

    def forward(
        self, z: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        cond = self.time_mlp(self.time_emb(t)) + self.class_emb(y)

        x = self.patchify(z)  # (B, hidden, grid, grid)
        x = x.flatten(2).transpose(1, 2) + self.pos_emb  # (B, num_tokens, hidden)
        for block in self.blocks:
            x = block(x, cond)

        shift, scale = self.final_adaln(cond).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.out_proj(x)  # (B, num_tokens, patch_dim)
        return self._unpatchify(x)
