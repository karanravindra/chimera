import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.conv(self.act(self.norm(x)))


class ResidualBlock(nn.Module):
    def __init__(self, main_path: nn.Module, skip_path: nn.Module = None):
        super(ResidualBlock, self).__init__()
        self.main_path = main_path
        self.skip_path = skip_path

    def forward(self, x):
        skip = self.skip_path(x) if self.skip_path is not None else x
        return self.main_path(x) + skip


def residual(in_channels: int, out_channels: int) -> ResidualBlock:
    main = Block(in_channels, out_channels)
    skip = (
        nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if in_channels != out_channels
        else None
    )

    return ResidualBlock(main, skip)


class DigitNet(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super(DigitNet, self).__init__()
        self.backbone = nn.Sequential(
            residual(1, 16),
            nn.AvgPool2d(2),
            residual(16, 32),
            nn.AvgPool2d(2),
            residual(32, 64),
            nn.AvgPool2d(2),
            residual(64, 64),
            residual(64, 64),
            residual(64, 64),
            nn.AvgPool2d(2),
            residual(64, 64),
            nn.AdaptiveAvgPool2d(1),
        )
        self.norm = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        x = self.backbone(x)
        x = self.norm(x)
        x = self.dropout(x)
        return self.head(x.reshape(x.shape[0], -1))


class DigitNetLPIPS(nn.Module):
    """Perceptual loss using intermediate features from a pretrained DigitNet.

    Taps backbone blocks 0, 2, 4, 6, 7, 8, 10 (before pooling at each scale) to get features
    at 32x32, 8x8, and 2x2 spatial resolutions. Features are L2-normalized per
    channel before MSE, matching the LPIPS formulation.
    """

    # indices into backbone Sequential that are ResidualBlocks (before each pool)
    _TAP_INDICES = (0, 2, 4, 6, 7, 8, 10)

    def __init__(self, checkpoint_path: str):
        super().__init__()
        net = DigitNet()
        net.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        # Keep the full backbone and tap by index: running it as-is reproduces
        # DigitNet.forward exactly. (Splitting into block/pool pairs mis-handles
        # the consecutive ResidualBlocks at indices 6,7,8 -- there is no pool
        # between them -- and would apply some blocks twice.)
        self.backbone = net.backbone
        self._last_tap = max(self._TAP_INDICES)
        for p in self.parameters():
            p.requires_grad_(False)

    def _features(self, x):
        feats = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self._TAP_INDICES:
                feats.append(x)
            if i == self._last_tap:  # no taps past here; skip the rest
                break
        return feats

    @staticmethod
    def _normalize(f):
        return F.normalize(f, dim=1)

    def forward(self, pred, target):
        pred_feats = self._features(pred)
        with torch.no_grad():
            tgt_feats = self._features(target)
        return sum(
            F.mse_loss(self._normalize(p), self._normalize(t))
            for p, t in zip(pred_feats, tgt_feats)
        ) / len(pred_feats)
