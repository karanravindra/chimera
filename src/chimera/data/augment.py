"""Batched, on-GPU, train-only image augmentation for reconstruction objectives.

:class:`ReconstructionAugment` is a ``gpu_transform`` (see :class:`chimera.data.base.ImageDataModule`):
the loader ships a raw ``uint8`` NCHW batch and this runs *batched on the device* in
``on_after_batch_transfer``, returning the ``bf16 [0,1]`` batch the model trains on -- the same
dtype/range the plain bf16 collate produces, so it drops in transparently.

It applies, **independently per sample** (unlike a torchvision ``v2`` transform, which samples one
parameter set for the whole batch): a random-resized-crop (one ``affine_grid`` + ``grid_sample``), a
horizontal flip, and photometric jitter (brightness / contrast / saturation). All are label-free and
**reconstruction-safe**: the augmented image is what the autoencoder both ingests and reconstructs,
so the target moves with the input. Pair it with ``augment_eval=False`` so val/test see clean
(cast-only) images.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Rec.601 luma weights, used to derive the grayscale anchor for contrast/saturation jitter.
_LUMA = (0.299, 0.587, 0.114)


class ReconstructionAugment:
    """Per-sample random-resized-crop + flip + photometric jitter on a uint8 GPU batch -> bf16 [0,1].

    Crop ``area`` is sampled in ``[min_scale, max_scale]`` (fraction of the image area) with
    log-uniform aspect ``ratio`` -- matching :class:`torchvision.transforms.v2.RandomResizedCrop`
    semantics -- then resized back to the input size. ``jitter`` is the half-width of the uniform
    brightness/contrast/saturation factor ranges (``[1 - jitter, 1 + jitter]``), i.e. a single
    ColorJitter-style strength knob; ``0`` disables photometric augmentation. Hue rotation is
    deliberately omitted -- it is the costly HSV-roundtrip term and least useful for faces/animals.

    For an autoencoder the crop is kept milder than a classifier's (``min_scale`` default 0.4, not
    0.08): the model must still reconstruct high-frequency detail, so an extreme crop would turn
    every target into a blurry upsample.
    """

    def __init__(
        self,
        min_scale: float = 0.4,
        max_scale: float = 1.0,
        hflip_p: float = 0.5,
        jitter: float = 0.3,
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.hflip_p = hflip_p
        self.jitter = jitter
        self.log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    def _rand_factor(self, b: int, dev) -> torch.Tensor:
        # Per-sample factor in [1 - jitter, 1 + jitter], shaped to broadcast over C,H,W.
        lo = max(0.0, 1.0 - self.jitter)
        return torch.empty(b, 1, 1, 1, device=dev).uniform_(lo, 1.0 + self.jitter)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: uint8 NCHW on the GPU. Augment in fp32, then hand back bf16 [0,1].
        xf = x.to(torch.float32).div_(255.0)
        b, c, _, _ = xf.shape
        dev = xf.device

        # Per-sample crop box as a normalized affine: a crop of width-fraction w_frac centered at
        # cx maps the output's [-1,1] grid onto input coords [cx - w_frac, cx + w_frac]. Keeping
        # |c| + frac <= 1 (the (1 - frac) factor below) guarantees the crop stays in-bounds.
        area = torch.empty(b, device=dev).uniform_(self.min_scale, self.max_scale)
        ratio = torch.empty(b, device=dev).uniform_(*self.log_ratio).exp()
        w_frac = (area * ratio).sqrt().clamp(max=1.0)
        h_frac = (area / ratio).sqrt().clamp(max=1.0)
        cx = torch.empty(b, device=dev).uniform_(-1.0, 1.0) * (1.0 - w_frac)
        cy = torch.empty(b, device=dev).uniform_(-1.0, 1.0) * (1.0 - h_frac)

        theta = torch.zeros(b, 2, 3, device=dev)
        theta[:, 0, 0] = w_frac
        theta[:, 1, 1] = h_frac
        theta[:, 0, 2] = cx
        theta[:, 1, 2] = cy
        grid = F.affine_grid(theta, xf.shape, align_corners=False)
        xf = F.grid_sample(
            xf, grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )

        # Independent horizontal flip per sample.
        flip = torch.rand(b, device=dev) < self.hflip_p
        xf = torch.where(flip.view(b, 1, 1, 1), xf.flip(-1), xf)

        # Per-sample photometric jitter (brightness -> contrast -> saturation), each a lerp toward a
        # reference: 0 for brightness, the per-image mean luma for contrast, the grayscale image for
        # saturation -- the torchvision ColorJitter definitions, applied with per-sample factors.
        if self.jitter > 0:
            xf = xf * self._rand_factor(b, dev)  # brightness
            if c == 3:
                luma = (
                    _LUMA[0] * xf[:, 0:1] + _LUMA[1] * xf[:, 1:2] + _LUMA[2] * xf[:, 2:3]
                )
                gray_mean = luma.mean(dim=(1, 2, 3), keepdim=True)
                xf = torch.lerp(gray_mean, xf, self._rand_factor(b, dev))  # contrast
                xf = torch.lerp(luma, xf, self._rand_factor(b, dev))  # saturation
            else:
                mean = xf.mean(dim=(1, 2, 3), keepdim=True)
                xf = torch.lerp(mean, xf, self._rand_factor(b, dev))  # contrast
            xf = xf.clamp_(0.0, 1.0)

        return xf.to(torch.bfloat16)
