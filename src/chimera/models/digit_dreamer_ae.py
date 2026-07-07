import torch
import torch.nn as nn
import torch.nn.functional as F


class FSQ(nn.Module):
    def __init__(self, levels: list[int]):
        super().__init__()
        self.register_buffer("_levels", torch.tensor(levels))
        self.register_buffer("_half_width", torch.tensor(levels) // 2)
        self.d = len(levels)
        self.codebook_size = int(torch.prod(torch.tensor(levels)).item())

    def _round_ste(self, z):
        return z + (torch.round(z) - z).detach()

    def _bound(self, z):
        half_l = (self._levels - 1) * (1 - 1e-3) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = torch.atan(offset / half_l)  # atan, not tan
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z):
        # z: (..., d) with channels last
        q = self._round_ste(self._bound(z))
        return q / self._half_width  # normalized to ~[-1, 1]

    def forward(self, z):
        return self.quantize(z)

    def codes_to_indices(self, zq):
        z = (zq * self._half_width) + self._half_width  # -> [0, l_i)
        z = z.round().long()
        strides = torch.cumprod(
            torch.cat(
                [torch.ones(1, dtype=torch.long, device=z.device), self._levels[:-1]]
            ),
            0,
        )
        return (z * strides).sum(-1)


class ResBlock(nn.Module):
    """DC-AE style residual block: two 3x3 convs with a projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU(approximate="tanh")
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + self.shortcut(x))


class DCDownBlock(nn.Module):
    """DC-AE downsample: space-to-channel (pixel-unshuffle) + conv, with an
    averaged pixel-unshuffle residual shortcut.

    Halves H,W. The shortcut pixel-unshuffles (4x channels) then averages down
    to ``out_channels`` groups, preserving information the strided path would
    otherwise discard.
    """

    def __init__(self, in_channels: int, out_channels: int, factor: int = 2):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv2d(
            in_channels * factor * factor, out_channels, kernel_size=3, padding=1
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def _shortcut(self, x):
        # (B, C, H, W) -> (B, C*f*f, H/f, W/f) -> average groups -> out_channels
        x = F.pixel_unshuffle(x, self.factor)  # C * f^2 channels
        c_expanded = self.in_channels * self.factor * self.factor
        rep = c_expanded // self.out_channels
        if rep * self.out_channels != c_expanded:
            # fall back to a simple mean over all channels broadcast out
            x = x.mean(1, keepdim=True).repeat(1, self.out_channels, 1, 1)
            return x
        B, _, H, W = x.shape
        return x.view(B, self.out_channels, rep, H, W).mean(2)

    def forward(self, x):
        s = self._shortcut(x)
        h = self.conv(F.pixel_unshuffle(x, self.factor))
        return h + s


class DCUpBlock(nn.Module):
    """DC-AE upsample: conv + channel-to-space (pixel-shuffle), with a
    duplicated channel-to-space residual shortcut.

    Doubles H,W. The shortcut repeats channels to feed pixel-shuffle, mirroring
    the averaging done on the way down.
    """

    def __init__(self, in_channels: int, out_channels: int, factor: int = 2):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv2d(
            in_channels, out_channels * factor * factor, kernel_size=3, padding=1
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def _shortcut(self, x):
        # repeat channels to out_channels * f^2, then pixel-shuffle
        c_target = self.out_channels * self.factor * self.factor
        rep = c_target // self.in_channels
        if rep * self.in_channels != c_target:
            x = x.repeat(1, (c_target // self.in_channels) + 1, 1, 1)[:, :c_target]
        else:
            x = x.repeat(1, rep, 1, 1)
        return F.pixel_shuffle(x, self.factor)

    def forward(self, x):
        s = self._shortcut(x)
        h = F.pixel_shuffle(self.conv(x), self.factor)
        return h + s


class DigitDreamerAE(nn.Module):
    """A small DC-AE style convolutional autoencoder for MNIST with a *spatial*
    latent.

    Downsampling uses space-to-channel (pixel-unshuffle) DC blocks and
    upsampling uses channel-to-space (pixel-shuffle) DC blocks, each with the
    residual shortcut from the DC-AE paper. The bottleneck is a ``1x1``
    convolution, so ``encode`` returns a ``(B, latent_dim, 4, 4)`` feature map
    that can be patchified into a token sequence for a latent diffusion /
    rectified-flow transformer (MM-DiT), while still working as a plain
    reconstruction autoencoder via ``forward``.

    ``latent_dim`` is the number of latent *channels* on the 4x4 grid.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 4,
        base_channels: int = 16,
        fsq_levels=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        c1, c2, c4 = base_channels, base_channels * 2, base_channels * 4

        self.in_to_channels = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        # 28 -> 14 -> 7 -> ~4 spatially (MNIST padded/handled by pixel ops)
        self.encoder = nn.Sequential(
            ResBlock(c1, c1),
            DCDownBlock(c1, c2),  # /2
            ResBlock(c2, c2),
            DCDownBlock(c2, c4),  # /2
            ResBlock(c4, c4),
            ResBlock(c4, c4),
            DCDownBlock(c4, c4),  # /2
            ResBlock(c4, c4),
        )
        self.to_latent = nn.Conv2d(c4, latent_dim, kernel_size=1)

        self.from_latent = nn.Conv2d(latent_dim, c4, kernel_size=1)
        self.decoder = nn.Sequential(
            ResBlock(c4, c4),
            DCUpBlock(c4, c4),  # x2
            ResBlock(c4, c4),
            ResBlock(c4, c2),
            DCUpBlock(c2, c2),  # x2
            ResBlock(c2, c1),
            DCUpBlock(c1, c1),  # x2
            ResBlock(c1, c1),
        )
        self.out_from_channels = nn.Conv2d(c1, in_channels, kernel_size=3, padding=1)

        self.fsq = FSQ(fsq_levels) if fsq_levels is not None else None
        if fsq_levels is not None:
            assert len(fsq_levels) == latent_dim, (
                "FSQ needs one level count per latent channel"
            )

    def encode(self, x):
        """Encode an image into a spatial latent feature map."""
        x = self.in_to_channels(x)
        x = self.encoder(x)
        x = self.to_latent(x)
        if self.fsq is not None:
            x = x.permute(0, 2, 3, 1)  # (B, H, W, d)
            x = self.fsq(x)
            x = x.permute(0, 3, 1, 2)  # back to (B, d, H, W)
        return x

    def decode(self, z):
        """Decode a spatial latent feature map back into an image."""
        z = self.from_latent(z)
        z = self.decoder(z)
        z = self.out_from_channels(z)
        return z.sigmoid()  # Ensure output is in [0, 1]

    def forward(self, x):
        """Reconstruct an image from itself."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def tokenize(self, x):
        """Return integer token indices: (B, H, W)."""
        assert self.fsq is not None
        x = self.in_to_channels(x)
        x = self.encoder(x)
        x = self.to_latent(x).permute(0, 2, 3, 1)
        zq = self.fsq(x)
        return self.fsq.codes_to_indices(zq)

    @staticmethod
    def from_variant(
        variant: str,
        in_channels: int = 1,
        latent_dim: int = 4,
        fsq_levels=None,
    ) -> "DigitDreamerAE":
        if variant == "tiny":
            base_channels = 16
        elif variant == "small":
            base_channels = 24
        elif variant == "medium":
            base_channels = 32
        elif variant == "large":
            base_channels = 48
        else:
            raise ValueError(f"Unknown model variant: {variant}")
        return DigitDreamerAE(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            fsq_levels=fsq_levels,
        )


if __name__ == "__main__":
    from torchinfo import summary

    for variant in ["tiny", "small", "medium", "large"]:
        model = DigitDreamerAE.from_variant(variant)
        print(f"Model variant: {variant}")
        summary(model, input_size=(1, 1, 32, 32))
