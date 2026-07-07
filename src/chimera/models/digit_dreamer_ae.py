import torch
import torch.nn as nn



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
            torch.cat([torch.ones(1, dtype=torch.long, device=z.device),
                       self._levels[:-1]]), 0
        )
        return (z * strides).sum(-1)


class Block(nn.Module):
    """A simple convolutional block with a conv, batch norm, and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DigitDreamerAE(nn.Module):
    """A small convolutional autoencoder for MNIST with a *spatial* latent.

    The bottleneck is a ``1x1`` convolution rather than a flatten + linear, so
    ``encode`` returns a ``(B, latent_dim, 4, 4)`` feature map instead of a 1D
    vector. Keeping the latent spatial lets it be patchified into a token sequence
    for a latent diffusion / rectified-flow transformer (MM-DiT), while still
    working as a plain reconstruction autoencoder via ``forward``.

    ``latent_dim`` is the number of latent *channels* on the 4x4 grid.
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 4, base_channels: int = 16, fsq_levels=None):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        self.in_to_channels = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.encoder = nn.Sequential(
            Block(base_channels, base_channels),
            nn.MaxPool2d(2),
            Block(base_channels, base_channels * 2),
            nn.MaxPool2d(2),
            Block(base_channels * 2, base_channels * 4),
            Block(base_channels * 4, base_channels * 4),
            Block(base_channels * 4, base_channels * 4),
            nn.MaxPool2d(2),
            Block(base_channels * 4, base_channels * 4)
        )
        self.to_latent = nn.Conv2d(base_channels * 4, latent_dim, kernel_size=1)

        self.from_latent = nn.Conv2d(latent_dim, base_channels * 4, kernel_size=1)
        self.decoder = nn.Sequential(
            Block(base_channels * 4, base_channels * 4),
            nn.Upsample(scale_factor=2, mode="nearest"),
            Block(base_channels * 4, base_channels * 4),
            Block(base_channels * 4, base_channels * 4),
            Block(base_channels * 4, base_channels * 2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            Block(base_channels * 2, base_channels),
            nn.Upsample(scale_factor=2, mode="nearest"),
            Block(base_channels, base_channels),
        )
        self.out_from_channels = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

        self.fsq = FSQ(fsq_levels) if fsq_levels is not None else None
        if fsq_levels is not None:
            assert len(fsq_levels) == latent_dim, \
                "FSQ needs one level count per latent channel"

    def encode(self, x):
        """Encode an image into a spatial latent feature map."""
        x = self.in_to_channels(x)
        x = self.encoder(x)
        x = self.to_latent(x)
        if self.fsq is not None:
            x = x.permute(0, 2, 3, 1)  # (B, 4, 4, d)
            x = self.fsq(x)
            x = x.permute(0, 3, 1, 2)  # back to (B, d, 4, 4)
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
        """Return integer token indices: (B, 4, 4)."""
        assert self.fsq is not None
        x = self.in_to_channels(x)
        x = self.encoder(x)
        x = self.to_latent(x).permute(0, 2, 3, 1)
        zq = self.fsq(x)
        return self.fsq.codes_to_indices(zq)
