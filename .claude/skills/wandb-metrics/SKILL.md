---
name: wandb-metrics
description: Pull metrics from a local wandb run in this repo — the most recent run by default, or a named run. Use when asked to get/show/pull wandb metrics, check the latest run's loss/psnr/ssim/lpips/fid, list recent runs, get a metric's time-series (history), or inspect per-layer gradients/weights (vanishing/exploding/dead flags) — all without opening the wandb UI.
---

# wandb-metrics

Reads metrics straight from the local `wandb/` run dirs — no network, no W&B
login. The driver is `.claude/skills/wandb-metrics/pull_metrics.py`. It finds
the most recent run via the `wandb/latest-run` symlink and prints the final
scalar metrics from that run's `wandb-summary.json`.

**Why a script:** `wandb-summary.json` is ~500 KB because it embeds a
per-layer gradient/parameter histogram for every layer. `cat`ing it floods the
context. The driver filters those out and prints only scalars.

All paths below are relative to the repo root (`/root/Code/chimera`). Run from
there.

## Run (agent path)

Default — most recent run, final scalar metrics. Stdlib only, so plain
`python3` works:

```bash
python3 .claude/skills/wandb-metrics/pull_metrics.py
```

List the N most recent runs (newest first; default 10):

```bash
python3 .claude/skills/wandb-metrics/pull_metrics.py --list 5
```

A specific run — bare run id (the suffix) or a run dir path:

```bash
python3 .claude/skills/wandb-metrics/pull_metrics.py ik1cbw35
```

Machine-readable JSON (meta + scalar metrics):

```bash
python3 .claude/skills/wandb-metrics/pull_metrics.py --json
```

### Gradients / weights

`--gradients` reconstructs per-tensor gradient-magnitude stats (abs-mean, std,
min, max) from the histograms wandb logs via `watch(...)`, sorts tensors by
magnitude, and flags **dead** (all-zero), **vanishing** (<1e-6), and
**exploding** (>1 or |v|>10) tensors. Needs the venv (reads histograms):

```bash
.venv/bin/python .claude/skills/wandb-metrics/pull_metrics.py --gradients
.venv/bin/python .claude/skills/wandb-metrics/pull_metrics.py --gradients 20   # top/bottom 20
.venv/bin/python .claude/skills/wandb-metrics/pull_metrics.py lrs7fuic --gradients
```

`--params` does the same for parameter (weight) distributions, **if** the run
logged them (`wandb.watch(log='all'|'parameters')`):

```bash
.venv/bin/python .claude/skills/wandb-metrics/pull_metrics.py --params
```

The report reads histograms from the run **summary** when present (finished
runs) and falls back to the **live history datastore** for an in-progress run
that hasn't flushed its summary yet — the printed `source:` line tells you
which. Add `--json` for the raw per-tensor stats.

### Time-series (history)

`--history` reads the run's local `run-*.wandb` datastore. This needs the
`wandb` package, so use the project venv (`.venv/bin/python`) — still no
network. With no key it prints n / first / min / max / last per metric (great
for spotting regressions); with a key it dumps that metric's full series:

```bash
.venv/bin/python .claude/skills/wandb-metrics/pull_metrics.py --history
.venv/bin/python .claude/skills/wandb-metrics/pull_metrics.py --history val/psnr
```

## Gotchas

- **Bare `python3` lacks `wandb` and `yaml`.** The default/summary path is
  deliberately stdlib-only so it works regardless. Only `--history` needs the
  venv (`.venv/bin/python`, wandb 0.27.2). Don't `uv run` the summary path —
  it's unnecessary.
- **`latest-run` is a symlink** wandb maintains, pointing at the newest run.
  The driver resolves it; `--list` and explicit-id selection sort run dirs by
  mtime instead (so they also catch `offline-run-*` dirs).
- **The run id is the dir suffix**, e.g. `run-20260626_181415-lrs7fuic` → id
  `lrs7fuic`. Pass just that suffix as the positional arg.
- **A still-initializing run** may have no `wandb-summary.json` yet — the
  driver errors clearly instead of printing a half-run. Pick a finished run
  from `--list`.
- **Summary `_step` ≠ `trainer/global_step`.** `_step` is wandb's log counter;
  `trainer/global_step` is the Lightning optimizer step. The driver prints
  both.
- **`val/*` metrics are sparse** (one point per epoch) vs `train/*` (per log
  interval) — expected in `--history` (e.g. n=14 vs n=188), not a bug.
- **Gradient histograms live in two shapes.** The summary stores each as a
  clean nested dict (`{bins, values}`); the live history datastore stores them
  *flattened* into `<tensor>.bins` / `.values` / `._type` items. The driver
  handles both — don't be surprised the same histogram looks different raw.
- **`--params` can be empty** even when `--gradients` works: a run calling
  `wandb.watch(log='gradients')` logs only gradients. The error says so.
- **Stats are reconstructed from histograms**, so abs-mean/std are bin-center
  approximations, not exact tensor stats — fine for spotting vanishing/
  exploding trends, not for precise values.

## Troubleshooting

- `no wandb/ directory found` → you're not under the repo. `cd /root/Code/chimera`.
- `--history` ⇒ `needs the wandb package` → you used bare `python3`; switch to
  `.venv/bin/python`.
- `key 'X' not in history` → the error lists available keys; copy one exactly
  (they contain `/`, e.g. `train/loss`).
