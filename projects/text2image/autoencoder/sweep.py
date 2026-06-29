"""W&B sweep over the LPIPS perceptual term (net x weight) and the REPA weight for the
text2image autoencoder.

This is a real Weights & Biases sweep (``wandb.sweep`` + ``wandb.agent``), not a hand-rolled
loop. Each trial is launched by the agent as a *fresh* ``python train.py ...`` subprocess via the
sweep's ``program``/``command`` -- W&B's native mechanism. That subprocess isolation is
deliberate: every trial gets a fresh CUDA-graph / ``torch.compile`` state, fresh LPIPS/FID metric
networks, a fresh in-RAM dataset and a fresh EMA shadow, so trials can't contaminate each other.
The on-disk uint8 dataset cache is built once by the first trial and reused by the rest.

``train.py`` needs no sweep-specific code: the agent passes each sampled hyperparameter as the
matching CLI flag (``--lpips-net``, ``--lpips-weight``, ``--repa-weight``, ``--epochs``) and the
``WandbLogger`` inside ``train.py`` auto-joins the sweep from the agent's env vars, so all trials
land in one sweep on the W&B dashboard.

The sweep optimizes the *net-independent* metric ``test/rfid`` (lower is better). The logged
``*/lpips`` and ``*/loss`` use the network in the loss, so their scale differs per net/weight and
they are NOT comparable across trials; rFID/PSNR/SSIM are computed independently of the LPIPS net,
so they are the fair yardstick.

Examples
--------
    # default: bayes search over net x lpips-weight x repa-weight, 25 trials, 10 epochs each
    uv run python projects/text2image/autoencoder/sweep.py

    # exhaustive grid instead of bayes (3 nets x 3 lpips x 3 repa = 27 trials)
    uv run python projects/text2image/autoencoder/sweep.py --method grid

    # pin the LPIPS net and just search the two weight axes
    uv run python projects/text2image/autoencoder/sweep.py --nets squeeze

    # smoke test: one combo, one epoch
    uv run python projects/text2image/autoencoder/sweep.py --nets alex --weights 0.1 --repa-weights 0.5 --epochs 1

    # attach another agent (e.g. a second GPU/machine) to an existing sweep
    uv run python projects/text2image/autoencoder/sweep.py --sweep-id entity/text2image-autoencoder/abc123

    # just (re)print the ranking table for an existing sweep
    uv run python projects/text2image/autoencoder/sweep.py --report-only --sweep-id abc123

    # pass extra flags straight through to train.py (after a --)
    uv run python projects/text2image/autoencoder/sweep.py -- --compile-mode off
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import wandb

TRAIN = Path(__file__).parent / "train.py"
PROJECT = "text2image-autoencoder"

# Net-independent quality metric the sweep optimizes (logged on the test split by train.py).
METRIC = "test/rfid"


def build_sweep_config(nets, weights, repa_weights, epochs, method, extra) -> dict:
    """Build the W&B sweep config. Parameter keys are the exact ``train.py`` flag names (minus
    the ``--``) so the ``${args}`` macro expands them straight onto the command line. Each trial
    runs as a fresh subprocess (``program``/``command``); ``extra`` flags are appended verbatim."""
    return {
        "program": str(TRAIN),
        "method": method,
        "metric": {"name": METRIC, "goal": "minimize"},
        "parameters": {
            # Swept dimensions: the agent picks one value per trial.
            "lpips-net": {"values": list(nets)},
            "lpips-weight": {"values": [float(w) for w in weights]},
            "repa-weight": {"values": [float(w) for w in repa_weights]},
            # Fixed for every trial.
            "epochs": {"value": int(epochs)},
        },
        # Launch each trial with the SAME interpreter running this sweep (uv venv), not a bare
        # `python` off PATH (the ${interpreter} default), then expand the sampled params as
        # `--lpips-net=... --lpips-weight=...` and tack on any passthrough flags.
        "command": ["${env}", sys.executable, "${program}", "${args}", *extra],
    }


def run_sweep(sweep_id: str, count: int | None) -> None:
    """Run an agent against ``sweep_id``. For ``grid`` the agent stops once the grid is exhausted;
    ``count`` caps trials (required to bound ``random``/``bayes``). Trials run sequentially in this
    process's agent, each as its own ``train.py`` subprocess."""
    print(f"[sweep] starting agent on {sweep_id} (count={count if count is not None else 'all'})")
    wandb.agent(sweep_id, count=count)


def report(sweep_id: str) -> None:
    """Fetch the sweep's runs from wandb and print a table sorted by ``test/rfid`` ascending
    (lower is better), cross-checked against PSNR/SSIM."""
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
                "net": training.get("lpips_net", "?"),
                "weight": training.get("lpips_weight", "?"),
                "repa": training.get("repa_weight", "?"),
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

    print(f"\n[report] sweep={path!r}  ({len(rows)} runs, ranked by {METRIC} asc)")
    print(
        f"{'rank':>4}  {'net':<8} {'weight':>7} {'repa':>6}  "
        f"{'rFID':>8} {'PSNR':>7} {'SSIM':>7}  {'state':<8} run"
    )
    for rank, r in enumerate(rows, 1):
        print(
            f"{rank:>4}  {r['net']:<8} {str(r['weight']):>7} {str(r['repa']):>6}  "
            f"{fmt(r['rfid'], '8.3f')} {fmt(r['psnr'], '7.3f')} {fmt(r['ssim'], '7.4f')}  "
            f"{r['state']:<8} {r['name']}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nets", nargs="+", default=["alex", "vgg", "squeeze"])
    # LPIPS weight: geometric spread spanning ViT-VQGAN's ~0.1 to the LDM/SD-autoencoder ~1.0.
    p.add_argument("--weights", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    p.add_argument(
        "--repa-weights",
        nargs="+",
        type=float,
        # 0 = no-REPA baseline control; 0.5 = the REPA-paper anchor; 0.1 = the lower optimum
        # found by REPA variants. The literature says alignment weight > ~0.5 only hurts.
        default=[0.0, 0.1, 0.5],
        help="REPA latent-alignment weights to sweep; 0 disables REPA (matches train.py default)",
    )
    p.add_argument("--epochs", type=int, default=10, help="epochs per trial")
    p.add_argument(
        "--method",
        choices=["grid", "random", "bayes"],
        default="bayes",
        help="W&B search strategy; grid is exhaustive, random/bayes need --count to bound them",
    )
    p.add_argument(
        "--count",
        type=int,
        default=25,
        help="max trials for the agent (default: 25; for grid, the grid still bounds it)",
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
            args.nets, args.weights, args.repa_weights, args.epochs, args.method, args.extra
        )
        sweep_id = wandb.sweep(config, project=PROJECT)
        n_grid = len(args.nets) * len(args.weights) * len(args.repa_weights)
        print(f"[sweep] created sweep {sweep_id} (project={PROJECT})")
        if args.method == "grid":
            print(f"[sweep] grid has {n_grid} trials")
    else:
        print(f"[sweep] attaching agent to existing sweep {sweep_id}")

    run_sweep(sweep_id, args.count)
    report(sweep_id)


if __name__ == "__main__":
    main()
