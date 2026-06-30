"""ImageNet-1k DataModule (compact webdataset shards -> (image, label) batches).

ImageNet-1k is ~1.28M train / 50k val images -- far too large for the
materialize-everything-to-``.npy`` approach the torchvision DataModules use
(:mod:`chimera.data.base`): a single 256px uint8 NCHW store of the train split
would be ~250 GB. So this follows the streaming-shard pattern of
``projects/text2image`` instead: a one-time :func:`prepare` step downsizes the
source shards to a compact local base, and an :class:`IterableDataset` streams
samples straight off those tar shards at train time.

Source: ``timm/imagenet-1k-wds`` -- 1024 train + 64 val webdataset shards, one
sample = ``{key}.jpg`` + ``{key}.cls`` (integer class label, ascii) + ``{key}.json``
(label/width/height/filename). The dataset is gated; accept the terms on the Hub
and be logged in (``huggingface-cli login``) before running :func:`prepare`.

Prepare (run once; resumable, shard by shard, peak disk ~one raw shard). Store just above
the train size: default 144px longest side for 128px training, so the train-split
random-resized-crop has spatial headroom (and decode stays cheap -- see the throughput log)::

    # full set, downsized to 144px longest side (~14 GB train + ~0.5 GB val)
    uv run python -m chimera.data.imagenet --split train --shards 0-1023
    uv run python -m chimera.data.imagenet --split val   --shards 0-63

    # a quick subset to smoke-test the pipeline
    uv run python -m chimera.data.imagenet --split val --shards 0-3

Train (in a Lightning module)::

    from chimera.data import ImageNetDataModule
    dm = ImageNetDataModule(image_size=128, batch_size=256, num_workers=8)  # max_size=144

Loaders yield raw uint8 NCHW batches; the bf16/255 -> [0, 1] cast runs on the GPU in
``on_after_batch_transfer`` (workers stay on the JPEG decode bottleneck, ~4x less H2D --
see ``projects/imagenet1k/RESULTS.md``). Train at any ``image_size`` <= the prepared
``max_size``; a larger size needs a prepare re-run.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomResizedCrop
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

DEFAULT_DIR = Path("/mnt/ai/data/imagenet")
REPO = "timm/imagenet-1k-wds"
# Shard naming in the source repo (train: 4-digit 0..1023, val: 2-digit 0..63).
_RAW_NAME = {
    "train": lambda i: f"imagenet1k-train-{i:04d}.tar",
    "val": lambda i: f"imagenet1k-validation-{i:02d}.tar",
}
NUM_RAW_SHARDS = {"train": 1024, "val": 64}


def _center_square(img: Image.Image, size: int) -> torch.Tensor:
    """Resize shortest side to ``size`` then center-crop to ``size`` x ``size``; -> uint8 CHW.

    Deterministic eval/val crop. Returns **uint8** (not float): the bf16/255 cast is deferred
    to the GPU in :meth:`ImageNetDataModule.on_after_batch_transfer`, which keeps the CPU
    workers on the JPEG decode (the bottleneck) only and ships ~4x fewer bytes H2D.
    Benchmarked ~1.5-2x faster than emitting float32 here (see ``projects/imagenet1k/RESULTS.md``).
    """
    img = img.convert("RGB")
    img = TF.resize(img, size, antialias=True)  # shortest side -> size
    img = TF.center_crop(img, [size, size])
    return TF.pil_to_tensor(img)  # uint8 CHW


def _random_crop_square(img: Image.Image, size: int, scale, ratio) -> torch.Tensor:
    """Random-resized-crop to ``size`` x ``size`` -> uint8 CHW (the standard ImageNet train aug).

    Samples a random area (``scale`` fraction of the image) and aspect ``ratio``, crops it, and
    resizes to ``size``. The shards are stored a touch larger than ``size`` (``max_size`` 144 vs
    128) so this crop has spatial headroom rather than just upsampling a center patch. ``scale``
    is kept mild (default lower bound 0.65) because the 144px source is already small. Uses torch
    RNG (seeded per dataloader worker), as is standard for this transform."""
    img = img.convert("RGB")
    i, j, h, w = RandomResizedCrop.get_params(img, scale=list(scale), ratio=list(ratio))
    img = TF.resized_crop(img, i, j, h, w, [size, size], antialias=True)
    return TF.pil_to_tensor(img)  # uint8 CHW


def _stream_shard(path: Path):
    """Yield (key, jpg_bytes, label) per sample; files for a key are consecutive."""
    cur_key, jpg, label = None, None, -1
    with tarfile.open(path) as t:
        for m in t:
            if not m.isfile():
                continue
            key, _, ext = m.name.lstrip("./").partition(".")
            if key != cur_key and cur_key is not None:
                if jpg is not None:
                    yield cur_key, jpg, label
                jpg, label = None, -1
            cur_key = key
            if ext == "jpg":
                jpg = t.extractfile(m).read()
            elif ext == "cls":
                label = int(t.extractfile(m).read())
        if cur_key is not None and jpg is not None:
            yield cur_key, jpg, label


class ImageNetIterable(IterableDataset):
    """Streams (image, label) from the compact tar shards with rank/worker sharding.

    Sample-level sharding (global index % total_workers) is used so it is correct for any
    (#shards, #ranks, #workers) combination -- including more workers than shards -- and so
    every sample is seen exactly once per epoch across the whole DDP world. A reservoir
    shuffle buffer decorrelates the on-disk (class-contiguous) order without loading a shard
    into RAM; set ``shuffle_buffer <= 1`` (val) to stream in order.

    ``train=True`` applies a random-resized-crop (the standard ImageNet train aug, using the
    144-vs-128 storage headroom); ``train=False`` uses a deterministic center crop.
    """

    def __init__(self, shards, image_size, *, train=False, scale=(0.65, 1.0),
                 ratio=(3 / 4, 4 / 3), shuffle_buffer=4096, seed=0, epoch=0):
        self.shards = [Path(s) for s in shards]
        self.image_size = image_size
        self.train = train
        self.scale = scale
        self.ratio = ratio
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.epoch = epoch

    def _transform(self, img: Image.Image) -> torch.Tensor:
        if self.train:
            return _random_crop_square(img, self.image_size, self.scale, self.ratio)
        return _center_square(img, self.image_size)

    def _global_worker(self) -> tuple[int, int]:
        """(global_worker_id, total_workers) across DDP ranks x dataloader workers."""
        rank, world = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        return rank * nw + wid, world * nw

    def __iter__(self):
        import random

        gwid, total = self._global_worker()
        rng = random.Random(self.seed + self.epoch * 100003)
        buf: list[tuple[torch.Tensor, int]] = []
        idx = 0
        for shard in self.shards:
            for _key, jpg, label in _stream_shard(shard):
                owned = (idx % total) == gwid
                idx += 1
                if not owned:
                    continue
                try:
                    img = self._transform(Image.open(io.BytesIO(jpg)))
                except Exception:
                    continue  # skip a rare corrupt jpg rather than crash the epoch
                item = (img, label)
                if self.shuffle_buffer <= 1:
                    yield item
                    continue
                buf.append(item)
                if len(buf) >= self.shuffle_buffer:
                    yield buf.pop(rng.randrange(len(buf)))
        rng.shuffle(buf)
        yield from buf


def _collate(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.int64)  # uint8 NCHW, int64


def _to_bf16_scaled(x: torch.Tensor) -> torch.Tensor:
    """uint8 image batch -> bf16 scaled to [0, 1] (matches chimera.data.base)."""
    return x.to(torch.bfloat16).div_(255)


class ImageNetDataModule(LightningDataModule):
    """Lightning DataModule over the compact ImageNet-1k shards built by :func:`prepare`.

    Loaders yield raw **uint8** NCHW batches; :meth:`on_after_batch_transfer` casts them to
    **bf16 [0, 1]** on the GPU (same contract as the other datamodules, just with the cast
    moved off the CPU workers). Throughput note: the decode cost is set by the *stored* JPEG
    resolution, so prepare the shards just above the train size and construct with the matching
    ``max_size`` -- that is the single biggest throughput win (~4.5x over 256px+float32, see
    ``projects/imagenet1k/RESULTS.md``). Default stores at 144px and trains at 128, giving the
    train-split random-resized-crop spatial headroom over a plain center crop.
    """

    def __init__(self, data_dir=DEFAULT_DIR, max_size=144, image_size=128, batch_size=256,
                 num_workers=8, shuffle_buffer=4096, seed=0,
                 scale=(0.65, 1.0), ratio=(3 / 4, 4 / 3)):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.max_size = max_size
        if image_size > max_size:
            raise ValueError(f"image_size {image_size} > prepared max_size {max_size}; "
                             f"re-run prepare with a larger --max-size")
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        # Train-split RandomResizedCrop params (val always center-crops).
        self.scale = scale
        self.ratio = ratio
        self.train_shards: list[Path] = []
        self.val_shards: list[Path] = []

    def _split_shards(self, split: str) -> list[Path]:
        d = self.data_dir / f"imagenet_{self.max_size}" / split
        shards = sorted(d.glob("*.tar"))
        if not shards:
            raise FileNotFoundError(
                f"no {split} shards in {d}; run `python -m chimera.data.imagenet "
                f"--split {split} ...` first"
            )
        return shards

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_shards = self._split_shards("train")
        if stage in ("fit", "validate", "test", None):
            self.val_shards = self._split_shards("val")

    def train_dataloader(self):
        ds = ImageNetIterable(
            self.train_shards, self.image_size, train=True, scale=self.scale, ratio=self.ratio,
            shuffle_buffer=self.shuffle_buffer, seed=self.seed,
            epoch=self.trainer.current_epoch if self.trainer else 0,
        )
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=_collate, pin_memory=True, drop_last=True,
                          persistent_workers=self.num_workers > 0,
                          prefetch_factor=4 if self.num_workers > 0 else None)

    def val_dataloader(self):
        # No shuffle buffer: stream val in order so every sample is seen once per epoch.
        ds = ImageNetIterable(self.val_shards, self.image_size, shuffle_buffer=0, seed=self.seed)
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=_collate, pin_memory=True,
                          persistent_workers=self.num_workers > 0,
                          prefetch_factor=4 if self.num_workers > 0 else None)

    def test_dataloader(self):
        return self.val_dataloader()

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        # Cast the uint8 batch to bf16 [0, 1] on-device -- deferred here from the workers so
        # the CPU stays on JPEG decode (the bottleneck) and H2D ships uint8, not float.
        x, y = batch
        return _to_bf16_scaled(x), y


# ---------------------------------------------------------------------------
# Prepare: stream-download + downsize the source shards into a compact local base.
# Mirrors projects/text2image/prepare.py -- shard by shard, deleting each raw shard
# after processing so peak disk stays ~one raw shard.
# ---------------------------------------------------------------------------


def _resize_jpeg(jpg: bytes, max_size: int, quality: int) -> bytes:
    """Decode -> shrink longest side to <= max_size (aspect kept) -> re-encode JPEG."""
    img = Image.open(io.BytesIO(jpg)).convert("RGB")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _add(tar: tarfile.TarFile, name: str, data: bytes, mtime: float) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = mtime
    tar.addfile(info, io.BytesIO(data))


def _parse_shards(spec: str) -> list[int]:
    """'0-3' -> [0,1,2,3]; '0,5,9' -> [0,5,9]; '7' -> [7]."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def _process_shard(split: str, idx: int, args, mtime: float) -> None:
    import os
    import time

    from huggingface_hub import hf_hub_download

    out_dir = args.out / f"imagenet_{args.max_size}" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tar = out_dir / f"{split}_{idx:04d}.tar"
    if out_tar.exists() and not args.overwrite:
        print(f"[{split} {idx}] {out_tar.name} exists -> skip", flush=True)
        return

    print(f"[{split} {idx}] downloading raw shard ...", flush=True)
    raw = hf_hub_download(REPO, _RAW_NAME[split](idx), repo_type="dataset", local_dir=str(args.out))
    raw_size = os.path.getsize(raw)

    tmp = out_tar.with_suffix(".tar.tmp")
    n = 0
    t0 = time.monotonic()
    # Each sample's files are written consecutively (jpg then cls) so the streaming
    # reader can pair them. The label byte string is carried through verbatim.
    with tarfile.open(tmp, "w") as dst:
        for key, jpg, label in _stream_shard(Path(raw)):
            _add(dst, f"{key}.jpg", _resize_jpeg(jpg, args.max_size, args.quality), mtime)
            _add(dst, f"{key}.cls", str(label).encode(), mtime)
            n += 1

    tmp.rename(out_tar)  # atomic publish
    out_size = os.path.getsize(out_tar)
    print(
        f"[{split} {idx}] {n} imgs  {raw_size / 1e9:.2f} GB -> {out_size / 1e9:.2f} GB "
        f"({out_size / raw_size:.0%})  in {time.monotonic() - t0:.0f}s",
        flush=True,
    )
    if not args.keep_raw:
        Path(raw).unlink(missing_ok=True)  # tolerate an already-gone raw shard
        print(f"[{split} {idx}] removed raw shard to reclaim {raw_size / 1e9:.2f} GB", flush=True)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Stream-download + downsize ImageNet-1k (timm/imagenet-1k-wds) shards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--shards", default=None,
                   help="indices e.g. '0-1023' or '0,5,9'; default = the whole split")
    p.add_argument("--max-size", type=int, default=144, help="longest side after resize")
    p.add_argument("--quality", type=int, default=95, help="output JPEG quality")
    p.add_argument("--out", type=Path, default=DEFAULT_DIR)
    p.add_argument("--keep-raw", action="store_true", help="do not delete raw shards")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    shards = _parse_shards(args.shards) if args.shards else list(range(NUM_RAW_SHARDS[args.split]))
    mtime = 1_700_000_000.0  # fixed mtime -> reproducible shard contents
    print(f"preparing {args.split} shards {shards[0]}..{shards[-1]} -> "
          f"{args.out}/imagenet_{args.max_size}/{args.split}  (max_size={args.max_size})")
    for idx in shards:
        _process_shard(args.split, idx, args, mtime)
    print("done.")


if __name__ == "__main__":
    main()
