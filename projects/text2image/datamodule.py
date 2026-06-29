"""text-to-image-2M DataModule: compact webdataset shards -> (image, caption) batches.

Reads the ``data_<size>`` shards produced by ``prepare.py`` (jpg + ``{prompt}`` json per
sample). An :class:`IterableDataset` streams samples straight from the tar shards --
sharded across DDP ranks *and* dataloader workers so every sample is seen once per epoch
-- decodes the jpg, resizes/center-crops to the requested **square** train resolution (any
size <= the stored max), and pairs it with a training caption.

Caption handling realizes the "randomized length" finding (arXiv:2506.16679): text-to-image-2M
ships one *dense* caption per image, so training on it verbatim is single-register and hurts
short-prompt following. Each step we instead serve the prompt at a random length tier --
**long** (full), **short** (first sentence), or **tag** (short head) -- with **10% caption
dropout** to the empty string for classifier-free guidance. Richer tag/terse caption variants
can be supplied later via an offline augment file without changing this module.

Images are yielded as float32 CHW in [0, 1] (the repo's collate casts to bf16), matching the
convention of the CelebA-HQ/AFHQ datamodules.

Demo::

    uv run python projects/text2image/datamodule.py --image-size 128 --batch-size 8
"""

from __future__ import annotations

import io
import json
import random
import re
import tarfile
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

DEFAULT_DIR = Path("/mnt/ai/data/text-to-image-2M")

# Length-tier draw for randomized-length training (long favored; short+tag keep the model
# responsive to terse prompts) and the standard 10% classifier-free-guidance dropout.
DEFAULT_TIER_PROBS = {"long": 0.4, "short": 0.3, "tag": 0.3}
DEFAULT_DROPOUT = 0.10
_SENT = re.compile(r"(?<=[.!?])\s+")


def sample_caption(
    prompt: str,
    rng: random.Random,
    *,
    tier_probs: dict[str, float] = DEFAULT_TIER_PROBS,
    dropout: float = DEFAULT_DROPOUT,
    tag_words: int = 8,
    tiers: tuple[str, ...] | None = None,
    weights: tuple[float, ...] | None = None,
) -> str:
    """Serve a dense prompt at a randomly chosen length tier (or "" on dropout).

    ``tiers``/``weights`` may be supplied pre-zipped from ``tier_probs`` to skip the per-call
    ``zip`` in hot loops; when omitted they're derived from ``tier_probs`` so standalone calls
    keep working."""
    if rng.random() < dropout or not prompt:
        return ""
    if tiers is None or weights is None:
        tiers, weights = zip(*tier_probs.items())
    tier = rng.choices(tiers, weights=weights, k=1)[0]
    full = prompt.strip()
    if tier == "long":
        return full
    first = _SENT.split(full, 1)[0].strip()
    if tier == "short":
        result = first
    else:
        # tag: a short head of the first sentence -- a cheap keyword-ish prompt
        result = " ".join(first.split()[:tag_words]).rstrip(",;:")
    # Empty tier results (e.g. a prompt with leading punctuation) would inject unintended
    # unconditional samples outside the dropout path; fall back to the full prompt instead.
    return result or full


def _stream_shard(path: Path):
    """Yield (key, jpg_bytes, prompt) per sample; files for a key are consecutive."""
    cur_key, jpg, prompt = None, None, ""
    with tarfile.open(path) as t:
        for m in t:
            if not m.isfile():
                continue
            key, _, ext = m.name.lstrip("./").partition(".")
            if key != cur_key and cur_key is not None:
                if jpg is not None:
                    yield cur_key, jpg, prompt
                jpg, prompt = None, ""
            cur_key = key
            if ext == "jpg":
                jpg = t.extractfile(m).read()
            elif ext == "json":
                prompt = json.loads(t.extractfile(m).read()).get("prompt", "")
        if cur_key is not None and jpg is not None:
            yield cur_key, jpg, prompt


def _to_square(img: Image.Image, size: int) -> torch.Tensor:
    """Resize shortest side to ``size`` then center-crop to ``size`` x ``size``; -> CHW [0,1]."""
    img = img.convert("RGB")
    img = TF.resize(img, size, antialias=True)  # shortest side -> size
    img = TF.center_crop(img, [size, size])
    return TF.to_tensor(img)  # float32 CHW in [0,1]


class T2I2MIterable(IterableDataset):
    """Streams (image, caption) from the compact tar shards with rank/worker sharding.

    Sample-level sharding (global index % total_workers) is used so it is correct for any
    (#shards, #ranks, #workers) combination -- including more workers than shards. A reservoir
    shuffle buffer decorrelates the on-disk order without loading a shard into RAM."""

    def __init__(self, shards, image_size, *, shuffle_buffer=2048, seed=0, dropout=DEFAULT_DROPOUT,
                 tier_probs=DEFAULT_TIER_PROBS, epoch=0):
        self.shards = [Path(s) for s in shards]
        self.image_size = image_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.dropout = dropout
        self.tier_probs = tier_probs
        # Pre-zip once: sample_caption is called per sample in the dataloader hot loop, so
        # re-deriving (tiers, weights) from tier_probs every call is wasted work.
        self._tiers, self._weights = zip(*tier_probs.items())
        self.epoch = epoch

    def _global_worker(self) -> tuple[int, int]:
        """(global_worker_id, total_workers) across DDP ranks x dataloader workers."""
        rank, world = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        return rank * nw + wid, world * nw

    def __iter__(self):
        gwid, total = self._global_worker()
        rng = random.Random(self.seed + self.epoch * 100003)
        cap_rng = random.Random(self.seed + 7919 + gwid + self.epoch * 100003)
        buf: list[tuple[torch.Tensor, str]] = []
        idx = 0
        for shard in self.shards:
            for key, jpg, prompt in _stream_shard(shard):
                owned = (idx % total) == gwid
                idx += 1
                if not owned:
                    continue
                try:
                    img = _to_square(Image.open(io.BytesIO(jpg)), self.image_size)
                except Exception:
                    continue  # skip a rare corrupt jpg rather than crash the epoch
                item = (img, sample_caption(prompt, cap_rng, dropout=self.dropout,
                                            tiers=self._tiers, weights=self._weights))
                if self.shuffle_buffer <= 1:
                    yield item
                    continue
                buf.append(item)
                if len(buf) >= self.shuffle_buffer:
                    yield buf.pop(rng.randrange(len(buf)))
        rng.shuffle(buf)
        yield from buf


def _collate(batch):
    imgs, caps = zip(*batch)
    return torch.stack(imgs), list(caps)


class T2I2MDataModule(LightningDataModule):
    """Lightning DataModule over the compact text-to-image-2M shards."""

    def __init__(self, data_dir=DEFAULT_DIR, max_size=256, image_size=128, batch_size=64,
                 num_workers=8, shuffle_buffer=2048, seed=0, dropout=DEFAULT_DROPOUT,
                 tier_probs=DEFAULT_TIER_PROBS):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.max_size = max_size
        if image_size > max_size:
            raise ValueError(f"image_size {image_size} > stored max_size {max_size}; "
                             f"re-run prepare.py with a larger --max-size")
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.dropout = dropout
        self.tier_probs = tier_probs
        self.shards: list[Path] = []

    def setup(self, stage=None):
        self.shards = sorted((self.data_dir / f"data_{self.max_size}").glob("data_*.tar"))
        if not self.shards:
            raise FileNotFoundError(
                f"no shards in {self.data_dir}/data_{self.max_size}; run prepare.py first"
            )

    def train_dataloader(self):
        ds = T2I2MIterable(
            self.shards, self.image_size, shuffle_buffer=self.shuffle_buffer,
            seed=self.seed, dropout=self.dropout, tier_probs=self.tier_probs,
            epoch=self.trainer.current_epoch if self.trainer else 0,
        )
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=_collate, pin_memory=True,
                          persistent_workers=self.num_workers > 0)


def main() -> None:
    import argparse
    from collections import Counter

    p = argparse.ArgumentParser(description="Inspect the text-to-image-2M DataModule.")
    p.add_argument("--data-dir", default=str(DEFAULT_DIR))
    p.add_argument("--max-size", type=int, default=256)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    dm = T2I2MDataModule(data_dir=args.data_dir, max_size=args.max_size,
                         image_size=args.image_size, batch_size=args.batch_size,
                         num_workers=args.num_workers)
    dm.setup()
    print(f"shards: {len(dm.shards)}  image_size: {args.image_size}")
    imgs, caps = next(iter(dm.train_dataloader()))
    print(f"image batch: {tuple(imgs.shape)} {imgs.dtype} range[{imgs.min():.3f},{imgs.max():.3f}]")
    print("sampled captions:")
    for c in caps:
        print(f"    {c!r}" if c else "    <dropout: unconditional>")

    # Confirm the tier/dropout distribution over many single-caption draws.
    rng = random.Random(0)
    prompt = ("A rustic wooden bridge crosses over a calm river surrounded by lush "
              "greenery. The water reflects the autumn-colored leaves. Soft morning light.")
    tally = Counter()
    for _ in range(20000):
        c = sample_caption(prompt, rng)
        tally["dropout" if c == "" else ("long" if c == prompt.strip() else
              "short" if c.endswith("greenery.") else "tag")] += 1
    print("\n20k caption draws from one prompt:",
          {k: f"{v/20000:.0%}" for k, v in tally.items()})


if __name__ == "__main__":
    main()
