"""Stage raw parquet shards for the tiny-LM mixture — NO tokenization.

For each source in ``sources.SOURCES`` this downloads the parquet shards that back
its slice (the first ``n_shards`` under ``file_prefix``, or the whole slice when
``n_shards is None``) into::

    /mnt/ai/data/tiny-llm/raw/<key>/

sized to cover the source's target unique tokens (weight x TARGET_TOKENS) with
headroom. Then it measures staged rows + bytes and writes a ``manifest.json`` per
source with a rough token estimate (bytes/≈4). Tokenization is deliberately left
to a later stage (this project trains its own tokenizer).

Usage:
    uv run python download.py            # stage every source
    uv run python download.py KEY [KEY]  # stage only these keys
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")

import sources as S  # noqa: E402


def _shard_files(repo: str, prefix: str, n: int | None) -> list[str]:
    from huggingface_hub import HfApi

    files = sorted(
        f
        for f in HfApi().list_repo_files(repo, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"no parquet under {prefix!r} in {repo}")
    return files if n is None else files[:n]


def stage(src: S.Source) -> dict:
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq

    out = Path(S.RAW_ROOT) / src.key
    out.mkdir(parents=True, exist_ok=True)

    patterns = _shard_files(src.hf_repo, src.file_prefix, src.n_shards)
    print(f"\n[{src.key}] {src.hf_repo} :: {len(patterns)} shard(s) -> {out}")
    snapshot_download(
        src.hf_repo,
        repo_type="dataset",
        local_dir=str(out),
        allow_patterns=patterns,
    )

    # measure what actually landed
    local = [out / p for p in patterns]
    n_bytes = sum(p.stat().st_size for p in local if p.exists())
    n_rows = 0
    cols = None
    for p in local:
        if not p.exists():
            continue
        pf = pq.ParquetFile(p)
        n_rows += pf.metadata.num_rows
        if cols is None:
            cols = pf.schema_arrow.names
    est_tok = int(n_bytes / 4)  # crude bytes/≈4 until our tokenizer exists

    meta = {
        "key": src.key,
        "hf_repo": src.hf_repo,
        "category": src.category,
        "weight": src.weight,
        "sft_weight": src.sft_weight,
        "license": src.license,
        "text_column": src.text_column,
        "columns": cols,
        "n_shards_staged": len(patterns),
        "n_rows": n_rows,
        "bytes": n_bytes,
        "est_tokens": est_tok,
        "target_tokens": int(src.weight * S.TARGET_TOKENS),
        "shards": patterns,
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    have_col = (
        "OK"
        if (cols and src.text_column in cols)
        else f"!! missing {src.text_column!r}"
    )
    print(
        f"   rows={n_rows:,}  size={n_bytes / 1e9:.2f}GB  ~{est_tok / 1e6:.0f}M tok"
        f"  cols={cols}  [{have_col}]"
    )
    return meta


def main(keys: list[str]):
    srcs = [S.get(k) for k in keys] if keys else S.SOURCES
    metas = [stage(s) for s in srcs]

    print("\n=== staged summary ===")
    pre_tok = sum(m["est_tokens"] for m in metas if m["weight"] > 0)
    for m in metas:
        tag = "SFT" if m["weight"] == 0 else f"{m['weight']:.2f}"
        print(
            f"  {m['key']:<24} {tag:>5}  {m['n_rows']:>10,} rows  "
            f"{m['bytes'] / 1e9:>6.2f}GB  ~{m['est_tokens'] / 1e6:>6.0f}M tok  "
            f"(target {m['target_tokens'] / 1e6:.0f}M)"
        )
    print(
        f"  pretrain staged (est): ~{pre_tok / 1e9:.2f}B unique tokens "
        f"for a {S.TARGET_TOKENS / 1e9:.1f}B budget"
    )
    print(f"  raw root: {S.RAW_ROOT}")


if __name__ == "__main__":
    main(sys.argv[1:])
