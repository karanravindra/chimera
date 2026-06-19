import torch
import torch.nn as nn
import torch.nn.functional as F


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


if __name__ == "__main__":
    # Example usage
    downsample = PixelUnshuffleChannelAveragingDownSample(
        in_channels=3, out_channels=3, factor=2
    )
    upsample = ChannelDuplicatingPixelShuffleUpSample(
        in_channels=3, out_channels=3, factor=2
    )

    x = torch.randn(1, 3, 64, 64)  # (B, Cin, H, W)
    downsampled = downsample(x)  # (B, Cout, H/2, W/2)
    upsampled = upsample(downsampled)  # (B, Cout, H, W)

    print("Input shape:", x.shape)
    print("Downsampled shape:", downsampled.shape)
    print("Upsampled shape:", upsampled.shape)
