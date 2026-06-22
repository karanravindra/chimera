import torch
import torch.nn.functional as F
from torch import nn

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv: a depthwise conv followed by 2 pointwise convs with a nonlinearity in between."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int | None = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2  # keep spatial dims; for k=1 this is 0, not 1
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.pointwise1 = nn.Conv2d(in_channels, out_channels * 2, 1)
        self.pointwise2 = nn.Conv2d(out_channels * 2, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.norm(x)
        x = F.gelu(self.pointwise1(x))
        return self.pointwise2(x)


class PixelUnshuffleChannelAveragingDownSample(nn.Module):
    """space-to-channel + channel averaging. No learned params."""

    def __init__(self, in_channels: int, out_channels: int, factor: int = 2):
        super().__init__()
        assert (in_channels * factor**2) % out_channels == 0, (
            f"in_channels*factor^2 ({in_channels * factor**2}) must be divisible "
            f"by out_channels ({out_channels})"
        )
        self.out_channels = out_channels
        self.factor = factor
        self.group_size = in_channels * factor**2 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_unshuffle(x, self.factor)  # (B, Cin*r^2, H/r, W/r)
        b, _, h, w = x.shape
        x = x.view(b, self.out_channels, self.group_size, h, w)
        return x.mean(dim=2)  # (B, Cout, H/r, W/r)


class ChannelDuplicatingPixelShuffleUpSample(nn.Module):
    """channel duplicating + channel-to-space. No learned params."""

    def __init__(self, in_channels: int, out_channels: int, factor: int = 2):
        super().__init__()
        assert (out_channels * factor**2) % in_channels == 0, (
            f"out_channels*factor^2 ({out_channels * factor**2}) must be divisible "
            f"by in_channels ({in_channels})"
        )
        self.factor = factor
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)  # (B, Cout*r^2, H, W)
        return F.pixel_shuffle(x, self.factor)  # (B, Cout, H*r, W*r)


class ResBlock(nn.Module):
    """Standard pre-norm residual conv block run at a fixed resolution."""

    def __init__(self, channels: int, norm_groups: int = 32):
        super().__init__()
        g = min(norm_groups, channels)
        self.body = nn.Sequential(
            nn.GroupNorm(g, channels),
            nn.SiLU(),
            DepthwiseSeparableConv(channels, channels),
            nn.GroupNorm(g, channels),
            nn.SiLU(),
            DepthwiseSeparableConv(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class DCDownBlock(nn.Module):
    """
    Downsample by `factor`, channels in_channels -> out_channels.

    Main (learned) path: stride-1 conv -> pixel_unshuffle, so spatial info is reorganized
    into channels rather than discarded by a strided conv. Output is summed with the
    non-parametric averaging shortcut, so the conv only has to learn the residual.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 2,
        n_res: int = 1,
        shortcut: bool = True,
    ):
        super().__init__()
        assert out_channels % factor**2 == 0, (
            "out_channels must be divisible by factor^2"
        )
        self.res = nn.Sequential(*[ResBlock(in_channels) for _ in range(n_res)])
        # conv outputs out_channels // factor^2, pixel_unshuffle inflates back to out_channels
        self.conv = DepthwiseSeparableConv(in_channels, out_channels // factor**2)
        self.factor = factor
        self.shortcut = (
            PixelUnshuffleChannelAveragingDownSample(in_channels, out_channels, factor)
            if shortcut
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res(x)
        out = F.pixel_unshuffle(self.conv(x), self.factor)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return out


class DCUpBlock(nn.Module):
    """
    Upsample by `factor`, channels in_channels -> out_channels.

    Main (learned) path: conv -> pixel_shuffle. Summed with the non-parametric duplicating
    shortcut so the conv learns only the residual.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 2,
        n_res: int = 1,
        shortcut: bool = True,
    ):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels * factor**2)
        self.factor = factor
        self.shortcut = (
            ChannelDuplicatingPixelShuffleUpSample(in_channels, out_channels, factor)
            if shortcut
            else None
        )
        self.res = nn.Sequential(*[ResBlock(out_channels) for _ in range(n_res)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.pixel_shuffle(self.conv(x), self.factor)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return self.res(out)


class ConvAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        latent_dim: int = 16,
        base_channels: int = 64,
        dim_per_block: tuple[int, ...] = (64, 128),
        layers_per_block: tuple[int, ...] = (2, 2),
    ):
        super().__init__()
        assert len(dim_per_block) == len(layers_per_block), (
            "dim_per_block and layers_per_block must have the same length"
        )

        # stem: lift input_dim -> base_channels at full resolution.
        # No residual shortcut here: the channel jump (e.g. 1 -> 64) can't form a
        # space-to-channel averaging shortcut (needs in*factor^2 % out == 0).
        self.stem = DepthwiseSeparableConv(input_dim, base_channels)

        # encoder: residual downsample blocks operating on hidden channels
        enc_blocks = []
        in_channels = base_channels
        for out_channels, n_res in zip(dim_per_block, layers_per_block):
            enc_blocks.append(DCDownBlock(in_channels, out_channels, n_res=n_res))
            in_channels = out_channels
        self.encoder = nn.Sequential(*enc_blocks)

        # bottleneck: project hidden channels <-> latent_dim with 1x1 convs
        self.to_latent = DepthwiseSeparableConv(in_channels, latent_dim, 1)
        self.from_latent = DepthwiseSeparableConv(latent_dim, in_channels, 1)

        # decoder: mirror of the encoder. Each encoder block maps enc_in -> enc_out
        # (at a halved resolution); the matching decoder block must invert that,
        # mapping enc_out -> enc_in while doubling the resolution back.
        dec_blocks = []
        enc_in_channels = [base_channels, *dim_per_block[:-1]]
        for enc_in, n_res in reversed(list(zip(enc_in_channels, layers_per_block))):
            dec_blocks.append(DCUpBlock(in_channels, enc_in, n_res=n_res))
            in_channels = enc_in
        self.decoder = nn.Sequential(*dec_blocks)

        # head: project hidden channels back to input_dim
        self.head = nn.Conv2d(in_channels, input_dim, 3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.encoder(x)
        return self.to_latent(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z)
        x = self.decoder(x)
        return self.head(x).sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


if __name__ == "__main__":
    from torchinfo import summary

    # # 1x32x32
    # model = ConvAutoEncoder(
    #     input_dim=1,
    #     latent_dim=4,
    #     base_channels=4,
    #     dim_per_block=(8, 16, 16, 16),
    #     layers_per_block=(2, 2, 3, 3),
    # )
    # summary(model, input_size=(1, 1, 32, 32))

    # 3x128x128
    model = ConvAutoEncoder(
        input_dim=3,
        latent_dim=8,
        base_channels=64,
        dim_per_block=(128, 256, 256, 256),
        layers_per_block=(2, 2, 3, 3),
        
    )
    summary(model, input_size=(1, 3, 128, 128))
