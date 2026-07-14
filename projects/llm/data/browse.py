"""Browse the planned LLM pretraining mixture.

Preview real, streamed samples from each dataset that will feed the blend, so the
data can be inspected before any tokenization/packing pipeline is written.

Examples
--------
    # list the mixture and per-category weights
    uv run python projects/llm/data/browse.py --list

    # peek at 3 Python files from The Stack v2
    uv run python projects/llm/data/browse.py --source stackv2-python --n 3

    # sample every (non-deferred) source, 1 row each, truncated
    uv run python projects/llm/data/browse.py --all --n 1 --chars 800

    # print a sample in full (no truncation)
    uv run python projects/llm/data/browse.py --source finemath --n 1 --full
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sources import (  # noqa: E402
    SOURCES,
    TARGET_TOKENS,
    category_weights,
    get,
    plan,
)

BOLD, DIM, CYAN, GREEN, YELLOW, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[36m",
    "\033[32m",
    "\033[33m",
    "\033[0m",
)


def print_list() -> None:
    print(f"\n{BOLD}Planned LLM pretraining mixture{RESET}\n")
    cats = category_weights()
    total = sum(cats.values())
    row = "  {:<16} {:<40} {:>7}  {}"
    print(row.format("KEY", "SOURCE", "WEIGHT", "CATEGORY"))
    print("  " + "-" * 78)
    for s in SOURCES:
        tag = f"{YELLOW}(deferred){RESET}" if s.deferred else ""
        key = f"{s.key} {tag}".strip()
        print(
            row.format(
                key, s.title[:40], f"{s.weight:.0%}", f"{CYAN}{s.category}{RESET}"
            )
        )
    print("  " + "-" * 78)
    print(f"\n{BOLD}By category:{RESET}")
    for cat, w in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<8} {w:>6.0%}")
    print(f"  {'total':<8} {total:>6.0%}\n")


def _fmt_tokens(t: float) -> str:
    if t >= 1e9:
        return f"{t / 1e9:.2f}B"
    if t >= 1e6:
        return f"{t / 1e6:.1f}M"
    return f"{t:.0f}"


def print_plan(target: int) -> None:
    print(f"\n{BOLD}Training plan for {_fmt_tokens(target)} total tokens{RESET}\n")
    row = "  {:<16} {:<7} {:>6} {:>9} {:>9} {:>8}"
    print(row.format("KEY", "CAT", "WEIGHT", "TARGET", "AVAIL", "REPEAT"))
    print("  " + "-" * 66)
    for r in plan(target):
        rep = r["repeat"]
        if rep == float("inf"):
            rep_s = "—"
        elif rep > 3:
            rep_s = f"{YELLOW}{rep:.1f}x{RESET}"
        elif rep > 1:
            rep_s = f"{rep:.1f}x"
        else:
            rep_s = f"{DIM}{rep:.2f}x{RESET}"
        print(
            row.format(
                r["key"],
                r["category"],
                f"{r['weight']:.0%}",
                _fmt_tokens(r["target_tokens"]),
                _fmt_tokens(r["avail_tokens"]),
                rep_s,
            )
        )
    print("  " + "-" * 66)
    # category rollup
    cats: dict[str, list] = {}
    for r in plan(target):
        cats.setdefault(r["category"], []).append(r)
    print(f"\n{BOLD}By category:{RESET}")
    for cat, rs in cats.items():
        tgt = sum(r["target_tokens"] for r in rs)
        avail = sum(r["avail_tokens"] for r in rs)
        rep = tgt / avail if avail else float("inf")
        flag = f"  {YELLOW}<- heavy repetition{RESET}" if rep > 3 else ""
        print(
            f"  {cat:<7} {_fmt_tokens(tgt):>8}  (avail {_fmt_tokens(avail):>7}, "
            f"{rep:.1f}x){flag}"
        )
    print()


def show_source(key: str, n: int, chars: int, full: bool) -> None:
    src = get(key)
    header = f"{BOLD}{CYAN}{src.key}{RESET}  {src.title}  {DIM}({src.weight:.0%}, {src.category}){RESET}"
    print("\n" + header)
    print(f"{DIM}repo={src.hf_repo or '—'} config={src.config} split={src.split}{RESET}")
    print(f"{DIM}{src.notes}{RESET}")
    if src.deferred:
        print(f"{YELLOW}  deferred — no data to show yet.{RESET}")
        return
    print("=" * 80)
    try:
        for i, sample in enumerate(src.sample(n)):
            meta = " ".join(f"{k}={v}" for k, v in sample.meta.items() if v is not None)
            print(f"\n{GREEN}[{key} #{i}]{RESET} {DIM}{meta}{RESET}")
            text = sample.text
            if not full and len(text) > chars:
                text = text[:chars] + f"\n{DIM}... [+{len(sample.text) - chars} chars]{RESET}"
            print(text)
        sh = src.last_shard
        if sh is not None:
            state = "cached" if sh.cached else "downloaded once, now cached"
            print(
                f"\n{DIM}(read from shard {sh.path.split('/')[-1]}, "
                f"{sh.size / 1e6:.0f} MB — {state}; not the full dataset){RESET}"
            )
    except Exception as e:
        print(f"{YELLOW}  error reading {key}: {e}{RESET}")
    print("\n" + "=" * 80)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list the mixture and weights")
    p.add_argument("--source", help="source key to preview (see --list)")
    p.add_argument("--all", action="store_true", help="preview every non-deferred source")
    p.add_argument("--n", type=int, default=3, help="rows per source (default 3)")
    p.add_argument("--chars", type=int, default=1500, help="truncate each sample to N chars")
    p.add_argument("--full", action="store_true", help="print samples in full (no truncation)")
    p.add_argument(
        "--plan",
        nargs="?",
        const=TARGET_TOKENS,
        type=float,
        metavar="TOTAL_TOKENS",
        help="show the training plan (target tokens + repeat factors) for a budget",
    )
    args = p.parse_args()

    if args.plan is not None:
        print_plan(int(args.plan))
        if not (args.source or args.all):
            return

    if args.list or not (args.source or args.all or args.plan is not None):
        print_list()
        if not (args.source or args.all):
            return

    if args.all:
        for s in SOURCES:
            if not s.deferred:
                show_source(s.key, args.n, args.chars, args.full)
    elif args.source:
        show_source(args.source, args.n, args.chars, args.full)


if __name__ == "__main__":
    main()
