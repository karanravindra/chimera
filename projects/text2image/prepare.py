"""Stream-download + downsize text-to-image-2M shards into a compact local base.

The source (``jackyhate/text-to-image-2M``, subset ``data_512_2M``) is 47 webdataset
shards of 512px JPEGs -- ~7.5 GB / 50k images each, ~351 GB / 2.35M total. We train at
64/128/256, never 512, so storing the 512px originals wastes disk. This processes the
set **shard by shard**:

  download a raw 512px shard  ->  resize each image so its longest side <= --max-size
  (default 256, LANCZOS, only ever shrinks)  ->  re-encode JPEG (q=--quality)  ->  write
  a compact shard under ``data_<max-size>/`` preserving the ``{prompt}`` json  ->  delete
  the raw shard (unless --keep-raw) so peak disk stays ~one raw shard.

Resumable: a shard whose compact output already exists is skipped. Train at any size
<= --max-size by downsampling in the dataloader (see datamodule.py); training above
--max-size needs a re-run with a larger value.

All data lives under --out (default /mnt/ai/data/text-to-image-2M); set HF_HOME to a
/mnt/ai path so the HF download cache also stays off the root disk.

Examples
--------
    # the recommended 200k-image subset, downsized to 256px
    uv run python projects/text2image/prepare.py --shards 0-3

    # the full 2.35M set
    uv run python projects/text2image/prepare.py --shards 0-46

    # keep the raw 512px shards too (needs ~351 GB more)
    uv run python projects/text2image/prepare.py --shards 0-46 --keep-raw
"""

from __future__ import annotations

import argparse
import io
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from huggingface_hub import hf_hub_download

REPO = "jackyhate/text-to-image-2M"
SUBSET = "data_512_2M"
DEFAULT_OUT = Path("/mnt/ai/data/text-to-image-2M")


def parse_shards(spec: str) -> list[int]:
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


def _stream_samples(tar: tarfile.TarFile):
    """Yield (key, {ext: bytes}) per webdataset sample. All files for one key are
    consecutive in a webdataset tar, so we flush when the key changes -- only one
    sample is held in memory at a time."""
    cur_key, cur = None, {}
    for m in tar:
        if not m.isfile():
            continue
        name = m.name.lstrip("./")
        key, _, ext = name.partition(".")
        if key != cur_key and cur:
            yield cur_key, cur
            cur = {}
        cur_key = key
        cur[ext] = tar.extractfile(m).read()
    if cur:
        yield cur_key, cur


def _resize_jpeg(jpg: bytes, max_size: int, quality: int) -> bytes:
    """Decode -> shrink longest side to <= max_size (aspect kept) -> re-encode JPEG."""
    img = Image.open(io.BytesIO(jpg))
    img = img.convert("RGB")
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


def process_shard(idx: int, args, mtime: float) -> None:
    out_dir = args.out / f"data_{args.max_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tar = out_dir / f"data_{idx:06d}.tar"
    if out_tar.exists() and not args.overwrite:
        print(f"[shard {idx}] {out_tar.name} exists -> skip")
        return

    print(f"[shard {idx}] downloading raw 512px shard ...", flush=True)
    raw = hf_hub_download(
        REPO, f"{SUBSET}/data_{idx:06d}.tar", repo_type="dataset", local_dir=str(args.out)
    )
    raw_size = os.path.getsize(raw)

    tmp = out_tar.with_suffix(".tar.tmp")
    n = 0
    t0 = time.monotonic()
    with tarfile.open(raw) as src, tarfile.open(tmp, "w") as dst, ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        # Chunked parallel resize: read a batch of samples, resize concurrently, write
        # each sample's files consecutively (jpg then json) so webdataset can pair them.
        batch: list[tuple[str, dict]] = []

        def flush(batch):
            nonlocal n
            jpgs = list(pool.map(lambda s: _resize_jpeg(s[1]["jpg"], args.max_size, args.quality), batch))
            for (key, files), jpg in zip(batch, jpgs):
                _add(dst, f"{key}.jpg", jpg, mtime)
                if "json" in files:
                    _add(dst, f"{key}.json", files["json"], mtime)
                n += 1

        for sample in _stream_samples(src):
            if "jpg" not in sample[1]:
                continue
            batch.append(sample)
            if len(batch) >= args.chunk:
                flush(batch)
                batch = []
        if batch:
            flush(batch)

    tmp.rename(out_tar)  # atomic publish
    out_size = os.path.getsize(out_tar)
    dt = time.monotonic() - t0
    print(
        f"[shard {idx}] {n} imgs  {raw_size / 1e9:.2f} GB -> {out_size / 1e9:.2f} GB "
        f"({out_size / raw_size:.0%})  in {dt:.0f}s",
        flush=True,
    )
    if not args.keep_raw:
        os.remove(raw)
        print(f"[shard {idx}] removed raw shard to reclaim {raw_size / 1e9:.2f} GB")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--shards", default="0-3", help="shard indices, e.g. '0-3' or '0,5,9'")
    p.add_argument("--max-size", type=int, default=256, help="longest side after resize")
    p.add_argument("--quality", type=int, default=95, help="output JPEG quality")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--workers", type=int, default=8, help="parallel resize threads")
    p.add_argument("--chunk", type=int, default=512, help="samples resized per batch")
    p.add_argument("--keep-raw", action="store_true", help="do not delete raw 512px shards")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    shards = parse_shards(args.shards)
    # A fixed mtime keeps shard contents reproducible (and avoids wall-clock churn in tars).
    mtime = 1_700_000_000.0
    print(f"preparing shards {shards} -> {args.out}/data_{args.max_size}  (max_size={args.max_size})")
    for idx in shards:
        process_shard(idx, args, mtime)
    print("done.")


if __name__ == "__main__":
    main()
