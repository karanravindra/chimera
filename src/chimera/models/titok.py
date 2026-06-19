import torch
import torch.nn.functional as F
from torch import nn

from chimera.nn.mlp import Mlp


# --------------------------------------------------------------------------------------
# ViT building blocks
# --------------------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(
        self, image_size: int, patch_size: int, in_channels: int, embed_dim: int
    ):
        super().__init__()
        assert image_size % patch_size == 0, (
            "image_size must be divisible by patch_size"
        )
        self.grid = image_size // patch_size
        self.num_patches = self.grid**2
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, E, grid, grid)
        return x.flatten(2).transpose(1, 2)  # (B, N, E), row-major (row, col)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, heads, N, head_dim)
        x = F.scaled_dot_product_attention(q, k, v)  # flash attn when available
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Pre-norm ViT block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=nn.GELU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------------------
# TiTok encoder / decoder
# --------------------------------------------------------------------------------------
class TiTokEncoder(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        embed_dim,
        depth,
        num_heads,
        num_latent_tokens,
        mlp_ratio,
    ):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        n = self.patch_embed.num_patches

        self.patch_pos_embed = nn.Parameter(torch.zeros(1, n, embed_dim))
        self.latent_tokens = nn.Parameter(torch.zeros(1, num_latent_tokens, embed_dim))
        self.latent_pos_embed = nn.Parameter(
            torch.zeros(1, num_latent_tokens, embed_dim)
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        patches = self.patch_embed(x) + self.patch_pos_embed
        latent = self.latent_tokens.expand(b, -1, -1) + self.latent_pos_embed
        x = torch.cat([patches, latent], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[
            :, -self.num_latent_tokens :
        ]  # keep only latent token outputs (B, K, E)


class TiTokDecoder(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        out_channels,
        embed_dim,
        depth,
        num_heads,
        num_latent_tokens,
        mlp_ratio,
    ):
        super().__init__()
        self.grid = image_size // patch_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_patches = self.grid**2

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.latent_pos_embed = nn.Parameter(
            torch.zeros(1, num_latent_tokens, embed_dim)
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b = latent.shape[0]
        mask = self.mask_token.expand(b, self.num_patches, -1) + self.patch_pos_embed
        latent = latent + self.latent_pos_embed
        x = torch.cat([mask, latent], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        patches = self.head(
            x[:, : self.num_patches]
        )  # keep mask outputs -> (B, N, p*p*C)
        return self.unpatchify(patches)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        b = patches.shape[0]
        g, p, c = self.grid, self.patch_size, self.out_channels
        x = patches.reshape(b, g, g, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, g, p, g, p)
        return x.reshape(b, c, g * p, g * p)


# --------------------------------------------------------------------------------------
# Autoencoder  (same encode / decode / forward interface as the conv version)
# --------------------------------------------------------------------------------------
class TiTokAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        image_size: int = 256,
        patch_size: int = 16,
        num_latent_tokens: int = 32,
        latent_dim: int = 16,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.latent_dim = latent_dim

        self.encoder = TiTokEncoder(
            image_size,
            patch_size,
            input_dim,
            embed_dim,
            depth,
            num_heads,
            num_latent_tokens,
            mlp_ratio,
        )
        # bottleneck on the token sequence: embed_dim <-> latent_dim (per token)
        self.to_latent = nn.Linear(embed_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, embed_dim)
        self.decoder = TiTokDecoder(
            image_size,
            patch_size,
            input_dim,
            embed_dim,
            depth,
            num_heads,
            num_latent_tokens,
            mlp_ratio,
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std=0.02)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)  # (B, K, embed_dim)
        return self.to_latent(z)  # (B, K, latent_dim)  <- VQ tokenizer goes here later

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.from_latent(z)  # (B, K, embed_dim)
        return self.decoder(z).sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


if __name__ == "__main__":
    from torchinfo import summary

    # 1x32x32
    model = TiTokAutoEncoder(
        input_dim=1,
        image_size=32,
        patch_size=8,
        num_latent_tokens=4,
        latent_dim=4,
        embed_dim=48,
        depth=4,
        num_heads=2,
    )
    summary(model, input_size=(1, 1, 32, 32))

    # # 3x128x128
    # model = TiTokAutoEncoder(
    #     input_dim=3,
    #     image_size=128,
    #     patch_size=16,
    #     num_latent_tokens=32,
    #     latent_dim=64,
    #     embed_dim=512,
    #     depth=8,
    #     num_heads=8,
    #     mlp_ratio=4.0,
    # )
    # summary(model, input_size=(1, 3, 128, 128))
