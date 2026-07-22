"""Command-line inspection and compilation for Chimera data."""

from __future__ import annotations

import argparse
from fnmatch import fnmatch
import json
from pathlib import Path

from .cache import atomic_json_save, content_hash


def _text_lock(keys: list[str]) -> None:
    from huggingface_hub import HfApi

    from .text.catalog import LOCK_PATH, SOURCES

    payload = json.loads(LOCK_PATH.read_text())
    selected = keys or sorted(SOURCES)
    api = HfApi()
    for key in selected:
        source = SOURCES[key]
        info = api.dataset_info(source.source.repo)
        repo_files = {sibling.rfilename for sibling in info.siblings}
        configured = source.source.data_files
        if isinstance(configured, str):
            selected_files = {path for path in repo_files if fnmatch(path, configured)}
        else:
            selected_files = set(configured or ())
        missing = selected_files - repo_files
        if missing:
            raise RuntimeError(
                f"{key} catalog files are absent upstream: {sorted(missing)}"
            )
        if isinstance(configured, str) and not selected_files:
            raise RuntimeError(
                f"{key} catalog pattern {configured!r} matched no upstream files"
            )
        previous_files = set(payload["sources"].get(key, {}).get("files", ()))
        files = sorted((selected_files | previous_files) & repo_files)
        payload["sources"][key] = {
            "repo": source.source.repo,
            "revision": info.sha,
            "gated": bool(info.gated),
            "license": source.license,
            "files": files,
            "schema": content_hash(files),
        }
        print(f"locked {key}: {info.sha}")
    atomic_json_save(payload, LOCK_PATH)


def _text_inspect(key: str) -> None:
    from .manifest import CatalogLock
    from .text.catalog import LOCK_PATH, SOURCES, VIEWS, get_source, get_view

    if key in VIEWS:
        view = get_view(key)
        source = get_source(view.source)
        result = {"view": view, "source": source}
    elif key in SOURCES:
        source = get_source(key)
        result = {"source": source}
    else:
        raise SystemExit(f"unknown text source or view: {key}")
    lock = CatalogLock.load(LOCK_PATH).require(source.key, source.source.repo)
    print(result)
    print(lock)


def _text_list() -> None:
    from .manifest import CatalogLock
    from .text.catalog import LOCK_PATH, SOURCES, VIEWS

    locks = CatalogLock.load(LOCK_PATH)
    for key, source in sorted(SOURCES.items()):
        lock = locks.require(key, source.source.repo)
        views = ", ".join(
            sorted(view.key for view in VIEWS.values() if view.source == key)
        )
        print(f"{key}\t{source.source.repo}@{lock.revision[:12]}\t{views or '-'}")


def _text_sample(key: str, count: int, data_dir: Path) -> None:
    from .text.catalog import get_view, load_rows

    view = get_view(key)
    rows = load_rows(view, "train", data_dir=data_dir, streaming=True)
    for index, example in enumerate(view.adapter.iter_examples(rows)):
        print(f"--- {index}\n{example.text}")
        if index + 1 >= count:
            break


def _text_build(args) -> None:
    from .text import MixtureSource, TextDataModule, TextMixtureSpec, TokenizerSpec

    dm = TextDataModule(
        TextMixtureSpec(
            sources=(
                MixtureSource(
                    args.view,
                    max_train_tokens=args.max_train_tokens,
                    max_val_tokens=args.max_val_tokens,
                ),
            ),
            tokenizer=TokenizerSpec.pinned(args.tokenizer),
            add_bos=args.add_bos,
        ),
        data_dir=str(args.data_dir),
        seq_len=args.seq_len,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup("fit")
    print(dm.source_train_tokens)


def _text_validate(directory: Path) -> None:
    from .text.artifacts import ShardedTokenStore

    store = ShardedTokenStore(directory, verify=True)
    print(
        f"valid v{store.manifest.version} artifact: "
        f"{store.manifest.tokens:,} tokens, {store.manifest.documents:,} documents"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chimera-data")
    modalities = parser.add_subparsers(dest="modality", required=True)
    text = modalities.add_parser("text")
    commands = text.add_subparsers(dest="command", required=True)

    lock = commands.add_parser("lock")
    lock.add_argument("sources", nargs="*")
    commands.add_parser("list")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("key")
    sample = commands.add_parser("sample")
    sample.add_argument("view")
    sample.add_argument("--count", type=int, default=3)
    sample.add_argument("--data-dir", type=Path, default=Path("./data"))
    build = commands.add_parser("build")
    build.add_argument("view")
    build.add_argument("--tokenizer", type=Path, required=True)
    build.add_argument("--data-dir", type=Path, default=Path("./data"))
    build.add_argument("--seq-len", type=int, default=512)
    build.add_argument("--max-train-tokens", type=int, default=10_000_000)
    build.add_argument("--max-val-tokens", type=int, default=1_000_000)
    build.add_argument("--add-bos", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("directory", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "lock":
        _text_lock(args.sources)
    elif args.command == "list":
        _text_list()
    elif args.command == "inspect":
        _text_inspect(args.key)
    elif args.command == "sample":
        _text_sample(args.view, args.count, args.data_dir)
    elif args.command == "build":
        _text_build(args)
    elif args.command == "validate":
        _text_validate(args.directory)
