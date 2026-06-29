"""W&B Bayesian sweep over the two Muon learning rates for the text2image TiTok autoencoder.

``train.py`` defaults to the Muon optimizer (Muon on the ViT's 2D matmul weights, AdamW aux on
the embeddings/norms/biases -- see :func:`chimera.optim.muon_adam_param_groups`), so the LR sweep
searches **both** group LRs: ``--muon-lr`` and ``--adam-lr``. They live on very different scales
(the Muon update is orthogonalized, the AdamW one is entry-wise adaptive), so each is sampled
log-uniformly over its own range and Bayesian optimization proposes the next pair from the runs so
far.

This is a real Weights & Biases sweep (``wandb.sweep`` + ``wandb.agent``), not a hand-rolled loop.
Each trial is launched by the agent as a *fresh* ``python train.py ...`` subprocess via the sweep's
``program``/``command`` -- W&B's native mechanism. That subprocess isolation is deliberate: every
trial gets a fresh CUDA-graph / ``torch.compile`` state, fresh LPIPS/FID metric networks, a fresh
in-RAM dataset and a fresh optimizer, so trials can't contaminate each other. The on-disk uint8
dataset cache is built once by the first trial and reused by the rest.

``train.py`` needs no sweep-specific code: the agent passes each sampled LR as the matching CLI flag
(``--muon-lr``, ``--adam-lr``) and the ``WandbLogger`` inside ``train.py`` auto-joins the sweep from
the agent's env vars, so all trials land in one sweep on the W&B dashboard.

The sweep optimizes ``test/rfid`` (reconstruction FID, lower is better). rFID/PSNR/SSIM are computed
independently of the LPIPS net used in the loss, so they are the fair cross-trial yardstick; the
logged ``*/lpips`` and ``*/loss`` are on the loss's own scale and are NOT what we rank by.

Examples
--------
    # default: bayes over (muon-lr, adam-lr), 10 trials, 10 epochs each
    uv run python projects/text2image/titok/sweep.py

    # widen the Muon LR range and run more epochs per trial
    uv run python projects/text2image/titok/sweep.py --muon-lr-range 1e-3 1e-1 --epochs 20

    # smoke test: one trial, one epoch
    uv run python projects/text2image/titok/sweep.py --count 1 --epochs 1

    # attach another agent (e.g. a second GPU/machine) to an existing sweep
    uv run python projects/text2image/titok/sweep.py --sweep-id entity/text2image-titok/abc123

    # just (re)print the ranking table for an existing sweep
    uv run python projects/text2image/titok/sweep.py --report-only --sweep-id abc123

    # pass extra flags straight through to train.py (after a --)
    uv run python projects/text2image/titok/sweep.py -- --compile-mode off
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import wandb

TRAIN = Path(__file__).parent / "train.py"
PROJECT = "text2image-titok"

# Net-independent quality metric the sweep optimizes (logged on the test split by train.py).
METRIC = "test/rfid"


def build_sweep_config(muon_range, adam_range, epochs, method, extra) -> dict:
    """Build the W&B sweep config. Parameter keys are the exact ``train.py`` flag names (minus the
    ``--``) so the ``${args}`` macro expands them straight onto the command line. The two LRs are
    sampled log-uniformly over ``[min, max]`` (``log_uniform_values`` takes real bounds, not logs);
    the optimizer is pinned to ``muon`` so a future change to train.py's default can't silently
    turn the sweep into a no-op. Each trial runs as a fresh subprocess; ``extra`` is appended."""
    muon_lo, muon_hi = muon_range
    adam_lo, adam_hi = adam_range
    return {
        "program": str(TRAIN),
        "method": method,
        "metric": {"name": METRIC, "goal": "minimize"},
        "parameters": {
            # Swept dimensions: bayes proposes one (muon-lr, adam-lr) pair per trial.
            "muon-lr": {
                "distribution": "log_uniform_values",
                "min": float(muon_lo),
                "max": float(muon_hi),
            },
            "adam-lr": {
                "distribution": "log_uniform_values",
                "min": float(adam_lo),
                "max": float(adam_hi),
            },
            # Fixed for every trial.
            "optimizer": {"value": "muon"},
            "epochs": {"value": int(epochs)},
        },
        # Launch each trial with the SAME interpreter running this sweep (uv venv), not a bare
        # `python` off PATH (the ${interpreter} default), then expand the sampled params as
        # `--muon-lr=... --adam-lr=...` and tack on any passthrough flags.
        "command": ["${env}", sys.executable, "${program}", "${args}", *extra],
    }


def run_sweep(sweep_id: str, count: int | None) -> None:
    """Run an agent against ``sweep_id``. ``count`` caps trials (required for bayes/random). Trials
    run sequentially in this process's agent, each as its own ``train.py`` subprocess."""
    print(f"[sweep] starting agent on {sweep_id} (count={count if count is not None else 'all'})")
    wandb.agent(sweep_id, count=count)


def report(sweep_id: str) -> None:
    """Fetch the sweep's runs from wandb and print a table sorted by ``test/rfid`` ascending (lower
    is better), cross-checked against PSNR/SSIM."""
    api = wandb.Api()
    # Accept either a bare sweep id or a fully-qualified entity/project/id path.
    path = sweep_id if sweep_id.count("/") == 2 else f"{api.default_entity}/{PROJECT}/{sweep_id}"
    sweep = api.sweep(path)
    runs = list(sweep.runs)
    if not runs:
        print(f"[report] no runs found for sweep {path!r}")
        return

    rows = []
    for run in runs:
        training = run.config.get("training", {})
        summary = run.summary
        rows.append(
            {
                "muon_lr": training.get("muon_lr", "?"),
                "adam_lr": training.get("adam_lr", "?"),
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
        return format(value, spec) if isinstance(value, (int, float)) else "       -"

    print(f"\n[report] sweep={path!r}  ({len(rows)} runs, ranked by {METRIC} asc)")
    print(
        f"{'rank':>4}  {'muon_lr':>9} {'adam_lr':>9}  "
        f"{'rFID':>8} {'PSNR':>7} {'SSIM':>7}  {'state':<8} run"
    )
    for rank, r in enumerate(rows, 1):
        print(
            f"{rank:>4}  {fmt(r['muon_lr'], '9.2e')} {fmt(r['adam_lr'], '9.2e')}  "
            f"{fmt(r['rfid'], '8.3f')} {fmt(r['psnr'], '7.3f')} {fmt(r['ssim'], '7.4f')}  "
            f"{r['state']:<8} {r['name']}"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Log-uniform ranges (min max) for each LR group. Defaults bracket train.py's defaults
    # (muon-lr 0.02, adam-lr 8e-4) and reach down toward the ViT LRs reported by Southworth et
    # al. (arXiv:2605.24770: Muon ~1e-3, AdamW ~3e-4) so bayes can explore both regimes.
    p.add_argument(
        "--muon-lr-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=[2e-3, 5e-2],
        help="log-uniform range for the Muon (2D weight) group LR",
    )
    p.add_argument(
        "--adam-lr-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=[1e-4, 2e-3],
        help="log-uniform range for the AdamW aux (embeddings/norms/biases) group LR",
    )
    p.add_argument("--epochs", type=int, default=20, help="epochs per trial")
    p.add_argument(
        "--method",
        choices=["grid", "random", "bayes"],
        default="bayes",
        help="W&B search strategy; bayes/random need --count to bound them",
    )
    p.add_argument(
        "--count",
        type=int,
        default=10,
        help="max trials for the agent (default: 10)",
    )
    p.add_argument(
        "--sweep-id",
        default=None,
        help="attach an agent to (and report on) an existing sweep instead of creating a new one; "
        "bare id or entity/project/id",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="skip training; just print the ranking table for --sweep-id",
    )
    p.add_argument(
        "extra",
        nargs="*",
        help="extra flags passed straight through to train.py (put after a --)",
    )
    args = p.parse_args()

    if args.report_only:
        if not args.sweep_id:
            p.error("--report-only requires --sweep-id")
        report(args.sweep_id)
        return

    if args.method != "grid" and args.count is None:
        p.error(f"--method {args.method} is open-ended; pass --count to bound the number of trials")

    sweep_id = args.sweep_id
    if sweep_id is None:
        config = build_sweep_config(
            args.muon_lr_range, args.adam_lr_range, args.epochs, args.method, args.extra
        )
        sweep_id = wandb.sweep(config, project=PROJECT)
        print(f"[sweep] created sweep {sweep_id} (project={PROJECT})")
        print(
            f"[sweep] {args.method} over muon-lr in {args.muon_lr_range} x "
            f"adam-lr in {args.adam_lr_range}, {args.count} trials"
        )
    else:
        print(f"[sweep] attaching agent to existing sweep {sweep_id}")

    run_sweep(sweep_id, args.count)
    report(sweep_id)


if __name__ == "__main__":
    main()
