"""Frozen DINOv2 patch-feature extractor for Representation Alignment (REPA).

REPA (Yu et al. 2024) regularizes a generative model's internal representation toward the
patch features of a strong frozen self-supervised encoder. This wraps a HuggingFace DINOv2
checkpoint as that frozen target: it takes ``[0,1]`` images, resizes + ImageNet-normalizes
them, and returns the per-patch hidden states (CLS / register tokens dropped).

The module is meant to be used the way ``LitAutoEncoder`` uses LPIPS/FID: built lazily on the
training device, ``.eval()``-ed and frozen, and held *off* the training module's ``state_dict``
(so the ~90MB weights don't bloat every checkpoint and are trivially rebuilt).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# ImageNet statistics DINOv2 was trained with (its default image processor uses these).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Patch-feature width per published DINOv2 checkpoint, so callers can size a projection head
# without instantiating (and downloading) the model first.
DINOV2_HIDDEN_SIZE = {
    "facebook/dinov2-small": 384,
    "facebook/dinov2-base": 768,
    "facebook/dinov2-large": 1024,
    "facebook/dinov2-giant": 1536,
}


class Dinov2Features(nn.Module):
    """A frozen DINOv2 encoder that maps ``[0,1]`` images to patch tokens.

    Parameters
    ----------
    model_name:
        HuggingFace id, e.g. ``"facebook/dinov2-small"``.
    image_size:
        Side length the input is resized to before the encoder. Must be a multiple of the
        model's patch size (14 for DINOv2); the resulting patch grid is
        ``(image_size // patch_size)`` per side.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        image_size: int = 224,
    ):
        super().__init__()
        from transformers import AutoModel

        self.model_name = model_name
        self.image_size = image_size
        self.model = AutoModel.from_pretrained(model_name)
        self.model.requires_grad_(False)
        self.model.eval()

        cfg = self.model.config
        self.patch_size: int = cfg.patch_size
        self.hidden_size: int = cfg.hidden_size
        # dinov2-* has no register tokens; dinov2-with-registers checkpoints report 4.
        self.num_register_tokens: int = getattr(cfg, "num_register_tokens", 0)
        assert image_size % self.patch_size == 0, (
            f"image_size {image_size} must be a multiple of patch_size {self.patch_size}"
        )
        self.grid_size: int = image_size // self.patch_size

        # Normalization constants as buffers so .to(device)/dtype moves them with the module.
        self.register_buffer(
            "mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    @torch.no_grad()
    def forward(self, x_01: torch.Tensor) -> torch.Tensor:
        """``[0,1]`` images ``(B,3,H,W)`` -> patch tokens ``(B, grid^2, hidden_size)``."""
        x = F.interpolate(
            x_01, size=self.image_size, mode="bilinear", align_corners=False
        )
        x = (x - self.mean) / self.std
        out = self.model(pixel_values=x.to(self.mean.dtype))
        tokens = out.last_hidden_state  # (B, 1 + num_register_tokens + grid^2, hidden)
        return tokens[:, 1 + self.num_register_tokens :]

    def as_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        """Patch tokens ``(B, grid^2, C)`` -> spatial grid ``(B, C, grid, grid)``."""
        b, n, c = tokens.shape
        g = self.grid_size
        assert n == g * g, f"expected {g * g} patch tokens, got {n}"
        return tokens.transpose(1, 2).reshape(b, c, g, g)
