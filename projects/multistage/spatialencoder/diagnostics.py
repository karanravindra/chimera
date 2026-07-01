"""Capture per-layer ACTIVATIONS and GRADIENTS of a trained stage-1 autoencoder on
REAL AFHQ images, and dump a thorough text report for analysis.

Usage: diagnostics.py <ckpt.pt> [--report out.md]
Loads the checkpoint's own config, runs a real 128px batch through encode->decode,
computes the training loss (MSE + LPIPS-VGG), backprops, and records:
  * activations (forward hooks on every leaf module): mean/std/absmax, %dead (post-act
    <1e-4), %saturated, shape  -- with special focus on the latent z.
  * gradients (from .grad after backward): per-param grad norm, weight norm,
    grad/weight ratio, %zero  -- to spot vanishing/exploding/dead layers.
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from chimera.data import AFHQDataModule
from chimera.models import ConvAutoEncoder
from chimera.utils.seed import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--report", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=128)
    args = ap.parse_args()
    seed_everything(0, deterministic=True)
    dev = "cuda"

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck.get("config", {})
    model = ConvAutoEncoder(
        input_dim=cfg.get("input_dim", 3),
        latent_dim=cfg.get("latent_dim", 16),
        base_channels=cfg.get("base_channels", 64),
        dim_per_block=tuple(cfg.get("dim_per_block", (64, 128, 256))),
        layers_per_block=tuple(cfg.get("layers_per_block", (1, 2, 4))),
        dec_layers_per_block=(tuple(cfg["dec_layers_per_block"])
                              if cfg.get("dec_layers_per_block") else None),
    )
    model.load_state_dict(ck["model"])
    model = model.to(dev).float()  # fp32 for clean grad/act stats
    model.train()

    dm = AFHQDataModule(batch_size=args.batch, image_size=args.image_size, num_workers=4)
    dm.prepare_data(); dm.setup("fit")
    x = next(iter(dm.test_dataloader()))[0][: args.batch].to(dev).float()

    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
    lp.requires_grad_(False); lp = lp.to(dev).eval()

    # ---- forward hooks: capture output activation of every leaf module ----
    acts = {}

    def make_hook(name):
        def hook(mod, inp, out):
            if isinstance(out, torch.Tensor):
                acts[name] = out.detach()
        return hook

    handles = []
    leaf_names = {}
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:  # leaf
            handles.append(mod.register_forward_hook(make_hook(name)))
            leaf_names[name] = type(mod).__name__

    # ---- forward + loss + backward ----
    z = model.encode(x)
    z.retain_grad()
    recon = model.decode(z)
    mse = F.mse_loss(recon, x)
    lpips_val = lp.net(recon, x, normalize=True).mean()
    loss = mse + 0.1 * lpips_val
    model.zero_grad(set_to_none=True)
    loss.backward()

    for h in handles:
        h.remove()

    lines = []
    def w(s=""): lines.append(s)

    w(f"# AE diagnostics on real AFHQ images")
    w(f"- ckpt: `{args.ckpt}`")
    w(f"- config: {cfg}")
    w(f"- batch={args.batch} image_size={args.image_size}")
    w(f"- **loss={loss.item():.5f}  mse={mse.item():.6f}  psnr={10*math.log10(1/mse.item()):.2f}dB  lpips={lpips_val.item():.4f}**")
    w()

    # ---- LATENT analysis (the 12x-compression bottleneck) ----
    w("## Latent z (bottleneck)")
    zc = z.detach()
    per_ch_std = zc.std(dim=(0, 2, 3))          # (C,)
    per_ch_mean = zc.mean(dim=(0, 2, 3))
    w(f"- shape {tuple(zc.shape)}  global mean {zc.mean():.4f}  std {zc.std():.4f}  "
      f"absmax {zc.abs().max():.3f}")
    dead_ch = int((per_ch_std < 1e-3).sum())
    w(f"- per-channel std: min {per_ch_std.min():.4f}  max {per_ch_std.max():.4f}  "
      f"**dead channels (std<1e-3): {dead_ch}/{zc.shape[1]}**")
    w(f"- per-channel std vector: {[round(float(s),3) for s in per_ch_std]}")
    w(f"- per-channel mean vector: {[round(float(s),3) for s in per_ch_mean]}")
    if z.grad is not None:
        zg = z.grad.detach()
        w(f"- latent grad: norm {zg.norm():.4e}  per-elem-rms {zg.pow(2).mean().sqrt():.4e}")
        gstd = zg.std(dim=(0, 2, 3))
        w(f"- per-channel grad-std: min {gstd.min():.2e} max {gstd.max():.2e}")
    w()

    # ---- ACTIVATIONS per leaf ----
    w("## Activations (forward, per leaf module)")
    w("| module | type | shape | mean | std | absmax | %~0 |")
    w("|---|---|---|---|---|---|---|")
    for name, a in acts.items():
        a = a.float()
        pct0 = 100.0 * (a.abs() < 1e-4).float().mean().item()
        shape = "x".join(str(s) for s in a.shape[1:])
        w(f"| {name} | {leaf_names.get(name,'')} | {shape} | {a.mean():.3f} | "
          f"{a.std():.3f} | {a.abs().max():.2f} | {pct0:.1f} |")
    w()

    # ---- GRADIENTS per parameter ----
    w("## Gradients (per parameter, after backward)")
    w("| param | shape | weight_norm | grad_norm | grad/weight | %zero |")
    w("|---|---|---|---|---|---|")
    total_gn = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            w(f"| {name} | {tuple(p.shape)} | {p.norm():.3e} | NONE | - | - |")
            continue
        g = p.grad.detach()
        gn = g.norm().item(); wn = p.norm().item()
        total_gn += gn ** 2
        ratio = gn / (wn + 1e-12)
        pz = 100.0 * (g == 0).float().mean().item()
        w(f"| {name} | {tuple(p.shape)} | {wn:.3e} | {gn:.3e} | {ratio:.2e} | {pz:.1f} |")
    w()
    w(f"**global grad norm: {math.sqrt(total_gn):.4e}**")

    report = args.report or str(Path(args.ckpt).parent / "diagnostics.md")
    Path(report).write_text("\n".join(lines))
    print(f"Wrote {report}  ({len(lines)} lines)")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
