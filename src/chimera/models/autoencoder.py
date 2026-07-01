import torch
import torch.nn.functional as F
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv: a depthwise conv followed by 2 pointwise convs with a nonlinearity in between."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2  # keep spatial dims; for k=1 this is 0, not 1
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding=padding, groups=in_channels
        )
        self.norm = nn.BatchNorm2d(in_channels)
        self.pointwise1 = nn.Conv2d(in_channels, out_channels * 2, 1)
        self.pointwise2 = nn.Conv2d(out_channels * 2, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.norm(x)
        x = F.gelu(self.pointwise1(x))
        return self.pointwise2(x)

class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2  # keep spatial dims; for k=1 this is 0, not 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

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
    """Standard pre-norm residual conv block run at a fixed resolution.

    Optional (default off, backward-compatible):
      * ``layer_scale``: a learnable per-channel gain on the residual branch (LayerScale,
        init 0.1). In pre-norm blocks the GroupNorm gamma barely trains (its scale is
        absorbed by the following conv), leaving the residual path ungated; an explicit
        gate restores a trainable gain and typically improves deep-AE fidelity.
      * ``zero_init``: zero the block's last conv so the block starts as identity
        (standard for residual AEs/diffusion — steadier early training)."""

    def __init__(self, channels: int, norm_groups: int = 32,
                 layer_scale: bool = False, zero_init: bool = False,
                 depthwise: bool = False):
        super().__init__()
        g = min(norm_groups, channels)

        def make_conv():
            # depthwise-separable (depthwise 3x3 + pointwise 1x1) is ~8-9x cheaper than a
            # dense 3x3 conv, so the same compute buys a much deeper stack (more capacity ->
            # lower rFID) at NO change to the latent/compression. Norm+act come from the
            # pre-norm ResBlock wrapper, so the DW/PW pair carries no internal norm here.
            if depthwise:
                return nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                    nn.Conv2d(channels, channels, 1),
                )
            return Conv2d(channels, channels)

        c1, c2 = make_conv(), make_conv()
        self.body = nn.Sequential(
            nn.GroupNorm(g, channels), nn.SiLU(), c1,
            nn.GroupNorm(g, channels), nn.SiLU(), c2,
        )
        if zero_init:
            last = c2[-1] if depthwise else c2.conv  # pointwise (dw) or Conv2d wrapper
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        self.gamma = (
            nn.Parameter(torch.full((1, channels, 1, 1), 0.1)) if layer_scale else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        if self.gamma is not None:
            out = self.gamma * out
        return x + out


# ======================================================================================
# DC-AE-faithful pieces (arXiv:2410.10733, mit-han-lab/efficientvit).
#
# These reproduce the official EfficientViT / linear-attention machinery that DC-AE places
# in its DEEP (low-resolution) stages, on top of plain conv ResBlocks in the high-res
# stages. They are OPT-IN: the classes below are only instantiated when a caller requests
# attention stages (see ``ConvAutoEncoder(attn_stages=...)`` and the ``DCAE`` class). The
# existing ConvAutoEncoder default path is untouched.
#
# Faithfulness notes vs the official code:
#   * ``RMSNorm2d`` == DC-AE's "trms2d" (channel-wise RMSNorm over C at each HxW location).
#   * ``LiteMLA`` reproduces the ReLU-linear (kernel) attention with the KV-then-Q linear
#     matmul, the +1 "denominator" pad, and the 1e-15 eps. ImageNet DC-AE configs use
#     ``EViT_GLU`` which is EfficientViTBlock(local_module="GLUMBConv", scales=()) -- i.e.
#     single-scale linear attention with NO multi-scale depthwise aggregation. The (5,)
#     multi-scale variant ("EViTS5_GLU") is available via ``scales``.
#   * ``EfficientViTBlock`` == context(LiteMLA)+residual, then local(GLUMBConv)+residual.
# ======================================================================================
class RMSNorm2d(nn.Module):
    """Channel-wise RMSNorm for (B, C, H, W) tensors (DC-AE's "trms2d").

    Normalizes over the channel dim at each spatial location (no mean subtraction),
    then applies a learnable per-channel gain. Cheaper/more stable than GroupNorm in the
    deep attention stages, which is what DC-AE uses there."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rms over channels (dim=1)
        dtype = x.dtype
        x = x.float()
        norm = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        x = (x * norm).to(dtype)
        return x * self.weight.view(1, -1, 1, 1)


class LiteMLA(nn.Module):
    """Lightweight multi-scale linear attention (DC-AE / EfficientViT).

    ReLU-kernel linear attention: with q,k >= 0 (via ReLU), attention is computed as
    ``out = (v @ kᵀ) @ q`` normalized by ``(1 @ kᵀ) @ q`` -- linear in the number of
    tokens instead of quadratic. Optional multi-scale depthwise aggregation of the QKV map
    (``scales``) adds local context; the ImageNet DC-AE configs use ``scales=()``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int | None = None,
        heads_ratio: float = 1.0,
        dim: int = 32,
        scales: tuple[int, ...] = (),
        norm: str | None = "rms",
        eps: float = 1e-15,
    ):
        super().__init__()
        self.eps = eps
        heads = int(in_channels // dim * heads_ratio) if heads is None else heads
        total_dim = heads * dim
        self.dim = dim
        # qkv projection (1x1), no norm/act (matches DC-AE norm=(None, ...))
        self.qkv = nn.Conv2d(in_channels, 3 * total_dim, 1, bias=False)
        # optional multi-scale aggregation: depthwise conv + grouped pointwise per scale
        self.aggreg = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(3 * total_dim, 3 * total_dim, s, padding=s // 2,
                          groups=3 * total_dim, bias=False),
                nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=False),
            )
            for s in scales
        )
        self.kernel_func = nn.ReLU(inplace=False)
        proj_in = total_dim * (1 + len(scales))
        self.proj = nn.Conv2d(proj_in, out_channels, 1, bias=False)
        self.proj_norm = RMSNorm2d(out_channels) if norm == "rms" else None

    def _linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        b, _, h, w = qkv.shape
        qkv = qkv.float().reshape(b, -1, 3 * self.dim, h * w)
        q = self.kernel_func(qkv[:, :, : self.dim])
        k = self.kernel_func(qkv[:, :, self.dim : 2 * self.dim])
        v = qkv[:, :, 2 * self.dim :]
        # pad v with a ones row so the same matmul yields the normalizer denominator
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1.0)
        vk = torch.matmul(v, k.transpose(-1, -2))  # (B, heads, dim+1, dim)
        out = torch.matmul(vk, q)                   # (B, heads, dim+1, HW)
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        return out.reshape(b, -1, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        multi = [qkv]
        for op in self.aggreg:
            multi.append(op(qkv))
        qkv = torch.cat(multi, dim=1)
        out = self._linear_att(qkv).to(x.dtype)
        out = self.proj(out)
        if self.proj_norm is not None:
            out = self.proj_norm(out)
        return out


class GLUMBConv(nn.Module):
    """Gated-Linear-Unit inverted bottleneck (DC-AE local module).

    1x1 expand -> depthwise conv -> GLU gate (split, SiLU one half, multiply) -> 1x1
    project. This is the "local" token-mixing half of an EfficientViT block."""

    def __init__(self, channels: int, expand_ratio: float = 4.0,
                 kernel_size: int = 3, norm: str | None = "rms"):
        super().__init__()
        mid = round(channels * expand_ratio)
        self.inverted = nn.Conv2d(channels, mid * 2, 1, bias=True)
        self.depth = nn.Conv2d(mid * 2, mid * 2, kernel_size,
                               padding=kernel_size // 2, groups=mid * 2, bias=True)
        self.act = nn.SiLU(inplace=False)
        self.point = nn.Conv2d(mid, channels, 1, bias=False)
        self.norm = RMSNorm2d(channels) if norm == "rms" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted(x)
        x = self.depth(x)
        x, gate = x.chunk(2, dim=1)
        x = x * self.act(gate)  # gated linear unit
        x = self.point(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class EfficientViTBlock(nn.Module):
    """DC-AE deep-stage block = LiteMLA (context) + GLUMBConv (local), each residual.

    Corresponds to the official ``EViT_GLU`` (``scales=()``) / ``EViTS5_GLU``
    (``scales=(5,)``) block used in the low-resolution encoder/decoder stages."""

    def __init__(self, channels: int, heads_ratio: float = 1.0, dim: int = 32,
                 expand_ratio: float = 4.0, scales: tuple[int, ...] = (),
                 norm: str = "rms"):
        super().__init__()
        self.context = LiteMLA(channels, channels, heads_ratio=heads_ratio, dim=dim,
                               scales=scales, norm=norm)
        self.local = GLUMBConv(channels, expand_ratio=expand_ratio, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.context(x)
        x = x + self.local(x)
        return x


def _make_stage_blocks(
    channels: int, n: int, block_type: str, *,
    layer_scale: bool, zero_init: bool, depthwise: bool = False,
    attn_dim: int, attn_expand: float, attn_scales: tuple[int, ...],
) -> list[nn.Module]:
    """Build ``n`` blocks of the given type at a fixed resolution/width.

    ``block_type`` is "res" (plain conv ResBlock, high-res stages) or "attn"
    (EfficientViTBlock linear-attention, deep low-res stages)."""
    if block_type == "attn":
        return [EfficientViTBlock(channels, dim=attn_dim, expand_ratio=attn_expand,
                                  scales=attn_scales) for _ in range(n)]
    return [ResBlock(channels, layer_scale=layer_scale, zero_init=zero_init,
                     depthwise=depthwise) for _ in range(n)]


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
        layer_scale: bool = False,
        zero_init: bool = False,
        depthwise: bool = False,
        block_type: str = "res",
        attn_dim: int = 32,
        attn_expand: float = 4.0,
        attn_scales: tuple[int, ...] = (),
    ):
        super().__init__()
        assert out_channels % factor**2 == 0, (
            "out_channels must be divisible by factor^2"
        )
        # in-stage blocks run at in_channels (pre-downsample), then the conv/shuffle
        # projects to out_channels. Blocks may be plain ResBlocks or attention blocks.
        self.res = nn.Sequential(*_make_stage_blocks(
            in_channels, n_res, block_type, layer_scale=layer_scale, zero_init=zero_init, depthwise=depthwise,
            attn_dim=attn_dim, attn_expand=attn_expand, attn_scales=attn_scales,
        ))
        # conv outputs out_channels // factor^2, pixel_unshuffle inflates back to out_channels
        self.conv = Conv2d(in_channels, out_channels // factor**2)
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
        layer_scale: bool = False,
        zero_init: bool = False,
        depthwise: bool = False,
        block_type: str = "res",
        attn_dim: int = 32,
        attn_expand: float = 4.0,
        attn_scales: tuple[int, ...] = (),
    ):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels * factor**2)
        self.factor = factor
        self.shortcut = (
            ChannelDuplicatingPixelShuffleUpSample(in_channels, out_channels, factor)
            if shortcut
            else None
        )
        # in-stage blocks run at out_channels (post-upsample). Plain or attention blocks.
        self.res = nn.Sequential(*_make_stage_blocks(
            out_channels, n_res, block_type, layer_scale=layer_scale, zero_init=zero_init, depthwise=depthwise,
            attn_dim=attn_dim, attn_expand=attn_expand, attn_scales=attn_scales,
        ))

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
        dec_layers_per_block: tuple[int, ...] | None = None,
        layer_scale: bool = False,
        zero_init_res: bool = False,
        depthwise: bool = False,
        refine_head: bool = False,
        attn_stages: tuple[int, ...] = (),
        attn_dim: int = 32,
        attn_expand: float = 4.0,
        attn_scales: tuple[int, ...] = (),
    ):
        super().__init__()
        assert len(dim_per_block) == len(layers_per_block), (
            "dim_per_block and layers_per_block must have the same length"
        )
        rk = dict(layer_scale=layer_scale, zero_init=zero_init_res, depthwise=depthwise)
        # OPT-IN DC-AE mimicry: ``attn_stages`` lists the stage indices (0 = highest-res /
        # first downsample) whose in-stage blocks become EfficientViT linear-attention
        # blocks instead of plain conv ResBlocks. DC-AE places attention only in the DEEP
        # (low-resolution, high-channel) stages, so a typical value is the last 1-2 stages
        # (e.g. (2,) for a 3-stage encoder). Empty () == original conv-only behavior.
        attn_stages = tuple(attn_stages)
        ak = dict(attn_dim=attn_dim, attn_expand=attn_expand, attn_scales=attn_scales)

        def _btype(stage_idx: int) -> str:
            return "attn" if stage_idx in attn_stages else "res"
        # Decoder can be made heavier than the encoder independently: at a fixed
        # bottleneck, reconstruction rFID is driven by decoder capacity while encoder
        # size is ~uncorrelated (ViTok, arXiv:2501.09755). Defaults to a symmetric
        # mirror of the encoder for backward compatibility.
        if dec_layers_per_block is None:
            dec_layers_per_block = layers_per_block
        assert len(dec_layers_per_block) == len(dim_per_block), (
            "dec_layers_per_block must have the same length as dim_per_block"
        )

        # stem: lift input_dim -> base_channels at full resolution.
        # No residual shortcut here: the channel jump (e.g. 1 -> 64) can't form a
        # space-to-channel averaging shortcut (needs in*factor^2 % out == 0).
        self.stem = Conv2d(input_dim, base_channels)

        # encoder: residual downsample blocks operating on hidden channels
        enc_blocks = []
        in_channels = base_channels
        for stage_idx, (out_channels, n_res) in enumerate(zip(dim_per_block, layers_per_block)):
            enc_blocks.append(DCDownBlock(in_channels, out_channels, n_res=n_res,
                                          block_type=_btype(stage_idx), **rk, **ak))
            in_channels = out_channels
        self.encoder = nn.Sequential(*enc_blocks)

        # bottleneck: project hidden channels <-> latent_dim with 1x1 convs
        self.to_latent = Conv2d(in_channels, latent_dim, 1)
        self.from_latent = Conv2d(latent_dim, in_channels, 1)

        # decoder: mirror of the encoder. Each encoder block maps enc_in -> enc_out
        # (at a halved resolution); the matching decoder block must invert that,
        # mapping enc_out -> enc_in while doubling the resolution back.
        # Decoder attention is placed at the SAME stage indices as the encoder, so the deep
        # low-res stages get attention on both sides (DC-AE is symmetric this way).
        dec_blocks = []
        enc_in_channels = [base_channels, *dim_per_block[:-1]]
        for stage_idx, (enc_in, n_res) in reversed(
            list(enumerate(zip(enc_in_channels, dec_layers_per_block)))
        ):
            dec_blocks.append(DCUpBlock(in_channels, enc_in, n_res=n_res,
                                        block_type=_btype(stage_idx), **rk, **ak))
            in_channels = enc_in
        self.decoder = nn.Sequential(*dec_blocks)

        # head: project hidden channels back to input_dim. The plain single 3-ch conv is
        # the hardest-working, most under-parameterized layer (it renders the whole RGB
        # output from a low-energy feature map); refine_head adds one conv+norm+act of
        # real capacity before the final projection.
        if refine_head:
            g = min(32, in_channels)
            self.head = nn.Sequential(
                Conv2d(in_channels, in_channels),
                nn.GroupNorm(g, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, input_dim, 3, padding=1),
            )
        else:
            self.head = nn.Conv2d(in_channels, input_dim, 3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.encoder(x)
        return self.to_latent(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z)
        x = self.decoder(x)
        return self.head(x).sigmoid()

    def forward(
        self, x: torch.Tensor, return_latent: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return (recon, z) if return_latent else recon


# --------------------------------------------------------------------------------------
# DCAE: a DC-AE-faithful convenience wrapper over ConvAutoEncoder.
# --------------------------------------------------------------------------------------
# The official DC-AE (arXiv:2410.10733) is exactly our residual-autoencoding conv AE with
# EfficientViT linear-attention blocks in its deep low-resolution stages. ``DCAE`` is a
# thin factory that (a) auto-places attention in the deepest ``n_attn_stages`` stages and
# (b) ships named presets scaled to a single 16GB card + AFHQ, rather than the original
# ImageNet widths (128..2048). It builds and returns a plain ``ConvAutoEncoder`` so all
# existing training/eval code (``.encode``/``.decode``/``forward``) works unchanged.
#
# Official ImageNet reference (for the report; NOT what we build here):
#   f32c32:  width [128,256,512,512,1024,1024], depth [0,4,8,2,2,2],
#            block  [Res,Res,Res,EViT,EViT,EViT], latent_channels=32  (E=32*8*8=2048, rFID 0.69)
#   f64c128: width [...,2048], 7 stages, latent_channels=128          (E=128*4*4=2048)
# DC-AE puts the 3 high-res stages as conv ResBlocks and all deeper stages as EViT_GLU
# (single-scale ReLU-linear attention + GLU-MBConv), norm=RMSNorm(trms2d)/BN, act silu/relu.


def DCAE(
    input_dim: int = 3,
    latent_dim: int = 32,
    base_channels: int = 128,
    dim_per_block: tuple[int, ...] = (128, 256, 512),
    layers_per_block: tuple[int, ...] = (2, 4, 8),
    dec_layers_per_block: tuple[int, ...] | None = None,
    n_attn_stages: int = 1,
    attn_dim: int = 32,
    attn_expand: float = 4.0,
    attn_scales: tuple[int, ...] = (),
    **kwargs,
) -> ConvAutoEncoder:
    """Build a DC-AE-style ``ConvAutoEncoder`` with attention in the deepest stages.

    ``n_attn_stages`` deepest stages (highest-channel, lowest-res) use EfficientViT
    linear-attention blocks; the rest stay conv ResBlocks -- matching DC-AE's
    conv-shallow / attention-deep split. All other args pass through to ConvAutoEncoder."""
    n = len(dim_per_block)
    attn_stages = tuple(range(max(0, n - n_attn_stages), n))
    return ConvAutoEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        base_channels=base_channels,
        dim_per_block=dim_per_block,
        layers_per_block=layers_per_block,
        dec_layers_per_block=dec_layers_per_block,
        attn_stages=attn_stages,
        attn_dim=attn_dim,
        attn_expand=attn_expand,
        attn_scales=attn_scales,
        **kwargs,
    )


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
    c = 64
    model = ConvAutoEncoder(
        input_dim=3,
        latent_dim=8,
        base_channels=c,
        dim_per_block=(c, 2 * c, 4 * c),
        layers_per_block=(2, 2, 4)
    )
    summary(model, input_size=(1, 3, 128, 128))
