"""Minimal LoRA: wrap nn.Linear layers with a trainable low-rank delta.

``apply_lora(model)`` swaps every targeted ``nn.Linear`` for a
:class:`LoRALinear` (frozen base weight + ``B @ A`` delta scaled by
``alpha / r``) and freezes everything else; only the A/B matrices train.
``merge_lora(model)`` folds the deltas back into plain Linears so the merged
model is architecturally identical to the base (checkpoint-compatible).
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)  # B stays zero => delta starts at 0

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale

    def merged(self) -> nn.Linear:
        lin = nn.Linear(
            self.base.in_features, self.base.out_features,
            bias=self.base.bias is not None,
        )
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.scale
            lin.weight.copy_(self.base.weight + delta.to(self.base.weight.dtype))
            if self.base.bias is not None:
                lin.bias.copy_(self.base.bias)
        return lin.to(self.base.weight.device, self.base.weight.dtype)


def apply_lora(model: nn.Module, r: int = 16, alpha: float = 32.0) -> list[nn.Parameter]:
    """Freeze the model, wrap every nn.Linear in LoRA; returns trainable params."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha))
    params = [p for p in model.parameters() if p.requires_grad]
    assert params, "no nn.Linear found to LoRA-wrap"
    return params


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every LoRALinear back into a plain Linear (in place)."""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merged())
    return model
