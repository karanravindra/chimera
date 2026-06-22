"""Sweep the LPIPS perceptual term (net x weight) for the celeba_afhq autoencoder.

Runs ``train.py`` once per ``(lpips_net, lpips_weight)`` combo as an isolated subprocess and
ranks the results. Subprocess isolation is deliberate: each run gets a fresh CUDA-graph /
``torch.compile`` state, fresh LPIPS/FID metric networks, a fresh in-RAM dataset, and a fresh
EMA shadow, so combos can't contaminate each other. The on-disk uint8 dataset cache is built
once by the first run and reused by the rest.

The runs are grouped in wandb via env vars (no ``train.py`` change): ``WANDB_RUN_GROUP`` puts
all combos in one group and ``WANDB_TAGS`` tags each with its net/weight.

Ranking uses the *net-independent* metrics only -- ``test/rfid`` (primary), ``test/psnr``,
``test/ssim``. The logged ``*/lpips`` and ``*/loss`` use the network in the loss, so their
scale differs per net and weight and they are NOT comparable across runs; rFID/PSNR/SSIM are
computed independently of the LPIPS net, so they are the fair yardstick.

Examples
--------
    # full 3-net x 4-weight sweep (12 runs, 20 epochs each), then print the ranking
    uv run python projects/celeba_afhq/autoencoder/sweep.py

    # smoke test: one combo, one epoch
    uv run python projects/celeba_afhq/autoencoder/sweep.py --nets alex --weights 0.1 --epochs 1

    # just (re)print the ranking table for an existing group
    uv run python projects/celeba_afhq/autoencoder/sweep.py --report-only

    # pass extra flags straight through to train.py (after a --)
    uv run python projects/celeba_afhq/autoencoder/sweep.py -- --compile-mode off
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path

import wandb

TRAIN = Path(__file__).parent / "train.py"
PROJECT = "celeba-afhq-autoencoder"

# Net-independent quality metrics we rank on (logged on the test split by train.py).
RANK_METRICS = ("test/rfid", "test/psnr", "test/ssim")


def run_sweep(nets, weights, epochs, group, extra) -> None:
    """Run ``train.py`` once per (net, weight) combo as an isolated subprocess, grouped in
    wandb under ``group``. One crashing combo is reported and the sweep continues."""
    combos = list(itertools.product(nets, weights))
    results: list[tuple[str, float, int]] = []  # (net, weight, returncode)
    for i, (net, weight) in enumerate(combos, 1):
        env = {
            **os.environ,
            "WANDB_RUN_GROUP": group,
            "WANDB_TAGS": f"lpips-sweep,net-{net},w-{weight}",
        }
        cmd = [
            sys.executable,
            str(TRAIN),
            "--epochs",
            str(epochs),
            "--lpips-net",
            net,
            "--lpips-weight",
            str(weight),
            *extra,
        ]
        print(f"\n[sweep {i}/{len(combos)}] net={net} weight={weight} epochs={epochs}")
        print(f"[sweep] $ {' '.join(cmd)}")
        proc = subprocess.run(cmd, env=env, check=False)
        if proc.returncode != 0:
            print(f"[sweep] !! net={net} weight={weight} exited with {proc.returncode}")
        results.append((net, weight, proc.returncode))

    ok = sum(1 for *_, rc in results if rc == 0)
    print(f"\n[sweep] finished: {ok}/{len(results)} runs ok")
    for net, weight, rc in results:
        status = "ok" if rc == 0 else f"FAILED (rc={rc})"
        print(f"[sweep]   net={net:<8} weight={weight:<5} {status}")


def report(group: str) -> None:
    """Fetch the group's runs from wandb and print a table sorted by ``test/rfid`` ascending
    (lower is better), cross-checked against PSNR/SSIM."""
    api = wandb.Api()
    runs = list(
        api.runs(
            f"{api.default_entity}/{PROJECT}",
            filters={"group": group},
        )
    )
    if not runs:
        print(f"[report] no runs found in group {group!r} for project {PROJECT}")
        return

    rows = []
    for run in runs:
        summary = run.summary
        rows.append(
            {
                "net": run.config.get("training", {}).get("lpips_net", "?"),
                "weight": run.config.get("training", {}).get("lpips_weight", "?"),
                "rfid": summary.get("test/rfid"),
                "psnr": summary.get("test/psnr"),
                "ssim": summary.get("test/ssim"),
                "name": run.name,
                "state": run.state,
            }
        )

    # Sort by rFID ascending; runs missing the metric (crashed/unfinished) sink to the bottom.
    rows.sort(key=lambda r: (r["rfid"] is None, r["rfid"] if r["rfid"] is not None else 0.0))

    def fmt(value, spec):
        return format(value, spec) if isinstance(value, (int, float)) else "    -"

    print(f"\n[report] group={group!r}  ({len(rows)} runs, ranked by test/rfid asc)")
    print(f"{'rank':>4}  {'net':<8} {'weight':>7}  {'rFID':>8} {'PSNR':>7} {'SSIM':>7}  {'state':<8} run")
    for rank, r in enumerate(rows, 1):
        print(
            f"{rank:>4}  {r['net']:<8} {str(r['weight']):>7}  "
            f"{fmt(r['rfid'], '8.3f')} {fmt(r['psnr'], '7.3f')} {fmt(r['ssim'], '7.4f')}  "
            f"{r['state']:<8} {r['name']}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nets", nargs="+", default=["alex", "vgg", "squeeze"])
    p.add_argument("--weights", nargs="+", type=float, default=[0.05, 0.1, 0.5, 1.0])
    p.add_argument("--epochs", type=int, default=20, help="epochs per sweep run")
    p.add_argument("--group", default="lpips-sweep", help="wandb run group for the sweep")
    p.add_argument(
        "--report-only",
        action="store_true",
        help="skip training; just print the ranking table for --group",
    )
    p.add_argument(
        "extra",
        nargs="*",
        help="extra flags passed straight through to train.py (put after a --)",
    )
    args = p.parse_args()

    if not args.report_only:
        run_sweep(args.nets, args.weights, args.epochs, args.group, args.extra)
    report(args.group)


if __name__ == "__main__":
    main()
