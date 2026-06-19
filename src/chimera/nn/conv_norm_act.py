from collections.abc import Callable

from torch import nn


def _default_act() -> nn.Module:
    return nn.GELU(approximate="tanh")


class Conv2dNormAct(nn.Module):
    """``Conv2d -> BatchNorm2d -> activation``.

    The workhorse downsampling/feature block shared by the conv encoders
    (``ConvAutoEncoder``, ``ImageAutoEncoder``) and the ``Backbone``. Pass
    ``stride=2`` to halve the spatial resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        act_layer: Callable[[], nn.Module] = _default_act,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act_layer()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvTranspose2dNormAct(nn.Module):
    """``ConvTranspose2d -> BatchNorm2d -> activation``.

    The upsampling counterpart to :class:`Conv2dNormAct`, used in the autoencoder
    decoders. Defaults (``kernel_size=4, stride=2, padding=1``) double the
    spatial resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        act_layer: Callable[[], nn.Module] = _default_act,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act_layer()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
