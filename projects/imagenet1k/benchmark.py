"""Throughput benchmark for the ImageNet-1k streaming DataModule.

Measures steady-state images/sec for different pipeline variants over the prepared
val shards (representative: same decode/resize work as train). Warmup batches are
skipped so worker spin-up / first-batch JIT doesn't bias the number.

Variants
--------
  pil   : current pipeline -- PIL decode + resize + center-crop + to_tensor(float32)
  uint8 : PIL decode + resize + center-crop, emit uint8 CHW; cast/normalize on GPU
  gpu   : workers only untar + ship raw jpg bytes; nvjpeg decode + resize/crop on GPU

Usage::

    uv run python projects/imagenet1k/benchmark.py --variant pil --workers 4
    uv run python projects/imagenet1k/benchmark.py --sweep
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from chimera.data.imagenet import _stream_shard

SHARD_DIR = Path("/mnt/ai/data/imagenet/imagenet_256/val")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def shards() -> list[Path]:
    s = sorted(SHARD_DIR.glob("*.tar"))
    if not s:
        raise FileNotFoundError(f"no shards in {SHARD_DIR}")
    return s


# --------------------------------------------------------------------------- #
# Datasets: one per variant. All do worker-sharded streaming of the same tars.
# --------------------------------------------------------------------------- #


def _worker_shard(idx: int) -> tuple[int, int]:
    wi = get_worker_info()
    return (wi.id, wi.num_workers) if wi else (0, 1)


class PILDataset(IterableDataset):
    """Current pipeline: decode+resize+crop -> float32 (to_tensor) or uint8 CHW.

    ``draft=True`` lets libjpeg decode at a reduced DCT scale (powers of 1/2) when the
    target ``size`` is well below the stored resolution -- e.g. 256px JPEG -> 128 decodes
    at half scale, roughly halving decode cost. Only helps when size <= stored/2."""

    def __init__(self, shards, size, out="float", draft=False):
        self.shards, self.size, self.out, self.draft = shards, size, out, draft

    def __iter__(self):
        wid, nw = _worker_shard(0)
        i = 0
        for sh in self.shards:
            for _k, jpg, label in _stream_shard(sh):
                if (i := i + 1) % nw != wid:
                    continue
                try:
                    img = Image.open(io.BytesIO(jpg))
                    if self.draft:
                        img.draft("RGB", (self.size, self.size))  # libjpeg DCT downscale
                    img = img.convert("RGB")
                    img = TF.resize(img, self.size, antialias=True)
                    img = TF.center_crop(img, [self.size, self.size])
                except Exception:
                    continue
                if self.out == "float":
                    yield TF.to_tensor(img), label  # float32 CHW [0,1]
                else:
                    yield TF.pil_to_tensor(img), label  # uint8 CHW


class BytesDataset(IterableDataset):
    """Workers only untar + shard; ship raw jpg bytes as a uint8 1-D tensor (no decode)."""

    def __init__(self, shards):
        self.shards = shards

    def __iter__(self):
        wid, nw = _worker_shard(0)
        i = 0
        for sh in self.shards:
            for _k, jpg, label in _stream_shard(sh):
                if (i := i + 1) % nw != wid:
                    continue
                yield torch.frombuffer(bytearray(jpg), dtype=torch.uint8), label


def _collate_float(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys)


def _collate_uint8(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys)


def _collate_bytes(batch):
    xs, ys = zip(*batch)
    return list(xs), torch.tensor(ys)  # variable-length byte tensors -> list


# --------------------------------------------------------------------------- #
# GPU post-transfer step for the uint8 / gpu variants.
# --------------------------------------------------------------------------- #

_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def gpu_from_uint8(x):
    """uint8 NCHW batch -> normalized float on GPU (the cast the collate deferred)."""
    return ((x.to(DEV, non_blocking=True).float() / 255.0) - _MEAN) / _STD


def gpu_decode(byte_list, size):
    """nvjpeg-decode a list of jpg byte tensors on GPU, resize shortest-side+crop, stack."""
    from torchvision.io import ImageReadMode, decode_jpeg

    imgs = decode_jpeg(byte_list, mode=ImageReadMode.RGB, device=DEV)  # list of uint8 CHW
    out = torch.empty(len(imgs), 3, size, size, device=DEV)
    for i, im in enumerate(imgs):
        im = TF.resize(im, size, antialias=True)
        im = TF.center_crop(im, [size, size])
        out[i] = im
    return ((out / 255.0) - _MEAN) / _STD


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def build_loader(variant, workers, size, batch_size):
    sh = shards()
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=True,
                  persistent_workers=workers > 0,
                  prefetch_factor=4 if workers > 0 else None)
    if variant == "pil":
        return DataLoader(PILDataset(sh, size, "float"), collate_fn=_collate_float, **common)
    if variant == "uint8":
        return DataLoader(PILDataset(sh, size, "uint8"), collate_fn=_collate_uint8, **common)
    if variant == "gpu":
        return DataLoader(BytesDataset(sh), collate_fn=_collate_bytes, **common)
    raise ValueError(variant)


def run(variant, workers, size, batch_size, max_batches, warmup):
    loader = build_loader(variant, workers, size, batch_size)
    n_img = 0
    t0 = None
    for bi, (x, y) in enumerate(loader):
        if variant == "gpu":
            x = gpu_decode(x, size)
        elif variant == "uint8":
            x = gpu_from_uint8(x)
        else:  # pil: emulate the H2D the trainer would do
            x = x.to(DEV, non_blocking=True)
        if DEV == "cuda":
            torch.cuda.synchronize()
        if bi == warmup:  # start clock after warmup
            t0 = time.perf_counter()
            n_img = 0
        if t0 is not None:
            n_img += y.numel()
        if bi + 1 >= max_batches:
            break
    dt = time.perf_counter() - t0
    return n_img / dt, dt, n_img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("pil", "uint8", "gpu"), default="pil")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-batches", type=int, default=60)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--sweep", action="store_true", help="sweep variants x workers")
    args = p.parse_args()

    if not args.sweep:
        ips, dt, n = run(args.variant, args.workers, args.size, args.batch_size,
                         args.max_batches, args.warmup)
        print(f"{args.variant:5s} w={args.workers} size={args.size} bs={args.batch_size}: "
              f"{ips:8.1f} img/s   ({n} imgs in {dt:.2f}s)")
        return

    print(f"sweep: size={args.size} bs={args.batch_size} "
          f"(warmup {args.warmup}, measure {args.max_batches - args.warmup} batches)\n")
    print(f"{'variant':8s} {'workers':>7s} {'img/s':>10s}")
    for variant in ("pil", "uint8", "gpu"):
        for w in (2, 4, 6, 8):
            try:
                ips, _, _ = run(variant, w, args.size, args.batch_size,
                                args.max_batches, args.warmup)
                print(f"{variant:8s} {w:7d} {ips:10.1f}", flush=True)
            except Exception as e:
                print(f"{variant:8s} {w:7d}   ERROR {type(e).__name__}: {str(e)[:60]}", flush=True)


if __name__ == "__main__":
    main()
