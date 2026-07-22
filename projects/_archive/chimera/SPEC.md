# Chimera reboot — cleanup, standardization, and project redo spec

Status: **draft — pending approval** (2026-07-18)

> **Superseded cleanup decision (2026-07-19):** archive-only data, model, Lightning
> module, and scheduler implementations have been removed from `src/chimera` rather
> than retained for future ports. The exact deletion and dependency manifest, archived
> consumers, and recovery procedure live in [`projects/_archive/README.md`](projects/_archive/README.md).
> This supersedes the relevant library-retention non-goal and vision redo steps below.

## 1. Goals

- Archive every existing project at a recoverable point; the working tree starts clean.
- Extract the training boilerplate every project has been copy-pasting into a shared
  `chimera.train` layer, so a redone project's `train.py` is ~30 lines of config + wiring.
- One written project template, so every redone project has the same shape, tooling,
  and hygiene (no generated artifacts in git, results tables in READMEs, sweeps as
  real wandb sweeps).
- Redo projects incrementally against the template, cheapest first, reusing existing
  checkpoints/baselines instead of retraining where old runs remain valid comparisons.

### Non-goals

- No rewrite of `src/chimera` model/data code for its own sake. The library modules
  (`data/`, `models/`, `modules/`, `optim/`, `tokenizers/`, `utils/`) stay; they only
  change where the new train layer or a redone project needs them to.
- No re-running of settled experiments (seq-len sweeps, EMA verdicts, mixture
  rebalances, aux-objective A/Bs). Their conclusions carry over as recorded.
- No new infra (no docker, no CI beyond what exists, no experiment tracker migration).

## 2. Current state (audit, 2026-07-18)

**Library** (`src/chimera/`): `data/` (17 modules), `models/` (13), `modules/` (7 Lightning
modules), `optim/` (Muon, warmup-cosine), `tokenizers/` (BPE), `utils/` (EMA, loggers,
device). No `train/` layer — every project owns its own argparse + Trainer + callback
boilerplate, which is the root cause of the drift below.

**Projects** (three generations of conventions):

| Project | Shape | Status |
|---|---|---|
| mnist (3 tasks), cifar10 (2), afhq, clevr | `train.py` + `sweep.yaml` + `main.ipynb` | gen-1 vision; uniform but argparse-heavy |
| fineweb-edu (gpt, sft) | + bench/coord-check scripts | superseded twice over |
| llm (data, gpt, sft) | + data pipeline, bpb.py, mfu bench | superseded by tiny-llm for active work |
| tiny-llm (data, gpt) | + `module.py`, `sweeps/`, README | gen-3; the shape the template is based on |

**Duplication**: `sources.py` / `tokenize_source.py` / `train_tokenizer.py` /
`build_mixture.py` exist in near-parallel in `projects/llm/data` and
`projects/tiny-llm/data`. `bench.py` + `bpb.py` exist in both llm and tiny-llm gpt.

**Cruft**: tracked generated artifacts (`projects/llm/gpt/bpb_cache.json`, `eval.html`,
`projects/fineweb-edu/gpt/coord_check.png`); ~1.1 GB untracked `lightning_logs/` under
`projects/afhq/autoencoder`; `session_tables.md` at repo root; `pyproject.toml` has a
placeholder description and no lint/format config.

## 3. Phase 0 — Archive (one commit, fully reversible)

1. `git tag archive/2026-07-pre-redo` — everything below is recoverable at this tag.
2. `git rm -r projects/fineweb-edu` — superseded by llm and tiny-llm; the tag keeps it.
3. `git mv` the remaining projects to `projects/_archive/<name>` — notebooks and
   READMEs stay browsable in-tree, but `_archive/` is read-only by convention: no
   fixes, no runs, no imports from it.
4. Untrack generated artifacts: `projects/llm/gpt/bpb_cache.json`, `projects/llm/gpt/eval.html`,
   `session_tables.md` (they stay on disk, untracked; no new `.gitignore` patterns —
   generated outputs are kept out of git by review, not blanket ignores).
5. Disk (not git): salvage any wanted checkpoints from
   `projects/afhq/autoencoder/lightning_logs/` (~1.1 GB) into `/mnt/ai/runs/afhq/`,
   then delete the directory. **Requires explicit per-checkpoint confirmation first.**

## 4. Phase 1 — `chimera.train`: the shared training layer

New package `src/chimera/train/` plus `tyro` as a dependency (replaces argparse;
gives typed dataclass configs, subcommands, and `--help` for free).

### 4.1 API sketch

```python
# src/chimera/train/config.py
@dataclass
class TrainConfig:
    # paths — every project inherits these defaults
    data_dir: Path = Path("/mnt/ai/data")
    run_dir: Path                      # set per-project, e.g. /mnt/ai/runs/mnist/classifier
    # schedule
    epochs: int = 1
    max_steps: int = -1                # -1 = no cap
    batch_size: int = 128
    lr: float = 1e-3
    warmup_steps: int = 100
    seed: int = 42
    # precision / compile
    precision: str = "bf16-mixed"
    compile: bool = False
    # logging
    wandb_project: str                 # set per-project
    wandb_offline: bool = False
    run_name: str | None = None
    tags: tuple[str, ...] = ()
    # opt-in extras
    ema_decay: float | None = None
```

Projects subclass it (`class Config(TrainConfig): arch: str = "small"; muon_lr: float = 0.013`)
and get the full CLI via `tyro.cli(Config)`.

```python
# src/chimera/train/run.py
def run(cfg: TrainConfig, module: LightningModule, dm: LightningDataModule,
        *, monitor: str = "val/loss", mode: str = "min",
        callbacks: list[Callback] = (), test: bool = True) -> RunResult:
    """Owns: seed_everything, ModelCheckpoint (to cfg.run_dir/checkpoints),
    build_run_loggers (wandb + csv), EMACallback when cfg.ema_decay is set,
    ProgressPrinter/ETA, Trainer construction, fit, optional test from best
    checkpoint, and printing/returning the best checkpoint path."""
```

`RunResult` carries `best_ckpt: Path`, `metrics: dict`, and the wandb run id (so
`bench.py --wandb-id` backfills evals without retraining).

### 4.2 Design rules

- `run()` owns everything a run *always* needs; anything task-specific stays in the
  project (custom callbacks pass through `callbacks=`). If a third project needs the
  same passthrough, it moves into the layer.
- Optimizer/scheduler construction stays in the module or project — Muon vs AdamW
  splits, muP grouping, and LR schedules are model decisions, not harness decisions.
  The layer provides helpers, not policy.
- Defaults encode the standing agreements: checkpoints under `/mnt/ai/runs/…`,
  bf16-mixed, wandb + CSV loggers, evals/benchmarks/doc-masking default-ON for LLM
  projects (never `--no-*` in launch commands).

### 4.3 Shared LLM tooling

The duplicated llm/tiny-llm data + eval scripts consolidate into the library:

- `src/chimera/data/pipeline/` ← `sources.py` registry pattern, `tokenize_source.py`,
  `build_mixture.py`, `train_tokenizer.py` (parameterized by a per-project sources
  table, which is the only thing that stays in the project).
- `src/chimera/evals/` ← `bpb.py` (bits-per-byte, per-source, cache under
  `/mnt/ai/data/<project>/bpb_cache.json` — never in-repo), `bench.py` core
  (BLiMP/LAMBADA/etc runners, wandb backfill, table printing with GPT-2-small
  reference row and best-per-row bolding).

## 5. Phase 1b — Repo standards

- **pyproject**: real description; add `tyro`; add `[tool.ruff]` (line-length 100,
  `ruff format` + `ruff check --select I` for import sorting) and run it once over
  `src/` + `tests/`. Notebooks exempt.
- **Tests**: `tests/` currently minimal — each redone project must leave behind at
  least a smoke test (`config parses`, `one train step runs on CPU/tiny shapes`).
  Library changes (train layer, pipeline, evals) get unit tests as they land.
- **Docs**: repo `README.md` gets the project index table (name, task, status, wandb
  project link). `session_tables.md` is retired (content folded into project READMEs
  where still relevant).

## 6. The project template

```
projects/<dataset-or-corpus>/<task>/
  README.md        # what/why, how to run, results table (best-per-row bolded,
                   # GPT-2-small reference row for LLM benches), key run links
  train.py         # Config(TrainConfig) + model/module wiring + run(cfg, ...)
  module.py        # only if the Lightning module is project-specific
  sweeps/          # wandb sweep YAMLs (values ordered most-promising-first)
  main.ipynb       # analysis ONLY — loads checkpoints, never trains
  figures/         # committed images referenced by README (only exemption to the
                   # no-generated-artifacts rule)
```

Rules (enforced by review, recorded in CLAUDE.md):

1. Nothing generated is committed except `figures/` and README tables.
2. Sweeps are real wandb sweeps (`wandb sweep` + `wandb agent`), never sequential
   loops — per the existing CLAUDE.md agreement.
3. Every run costing >10 min or any run beyond what was asked requires explicit
   approval first; reuse checkpoints/baselines wherever a valid comparison exists.
4. Data and checkpoints live under `/mnt/ai` (`HF_HOME=/mnt/ai/data/hf`); image
   datasets come from HF Hub (`karanravindra/*`), built from raw sources.
5. `main.ipynb` moves models to CUDA explicitly before generation/inference
   (Lightning leaves them on CPU after fit/test).

## 7. Phase 2 — Redo roadmap

Order: cheapest first to shake out the template, LLM last because it's largest and
its data layer lands in Phase 1.

| # | Project | Scope of redo | Reuse |
|---|---|---|---|
| 1 | `mnist/classifier` | Template pilot: port to `chimera.train`, smallest possible diff | existing baselines; retrain only if the port changes metrics |
| 2 | `mnist/autoencoder`, `mnist/rectified_flow` | Port; validates EMA + generation paths in the layer | existing checkpoints for analysis notebooks |
| 3 | `cifar10/{classify,autoencoder}` | Port; validates multi-task-per-dataset layout | existing runs |
| 4 | `clevr/vqa` | Port; validates multi-modal module wiring | existing runs |
| 5 | `afhq/autoencoder` | Port; FSQ/LPIPS knobs (lr=1e-4 codebook guard, lpips-weight 1.0 default) | salvaged checkpoints from Phase 0 |
| 6 | `llm/` (unified) | **One** project replacing llm + tiny-llm: shared data pipeline from `chimera.data.pipeline`, tiny-llm's train/module shape, benches from `chimera.evals`. Tiny (5–20M) and full-size are `--arch` presets of the same code, not separate projects | tokenizers (4k/8k/16k), packed mixes on `/mnt/ai`, phase-2 baseline run 0dj5aixt, all sweep verdicts |

Each redo is one PR-sized commit: port → smoke test → (only if needed) one
validation run compared against the archived baseline → README results table.

Settled conclusions that carry over without re-running: muP LR transfer
(0.013/0.006), depth-6 wall-optimality, seq512 > 1024 > 2048 at 6M, EMA no-op under
cosine, CCE > fp8 lm_head, mixture rebalance a wash, MTP/NextLat lose to NTP at 6M
(capacity-bound → next lever is scale).

## 8. Execution order & commit plan

| Step | Deliverable | Approval gate |
|---|---|---|
| 0 | Archive commit (tag, `_archive/`, delete fineweb-edu, cruft, .gitignore) | plan approval covers it, **except** afhq checkpoint deletion (per-item confirm) |
| 1 | `chimera.train` (config + run + tests) | API review before implementation |
| 1b | pyproject/ruff/README standards commit | — |
| 1c | `chimera.data.pipeline` + `chimera.evals` extraction (tests) | — |
| 2.1–2.5 | Vision project ports, one commit each | any training run >10 min |
| 2.6 | Unified `llm/` project | scope review before starting |

## 9. Open questions

1. **Naming**: does the unified LLM project live at `projects/llm/` (reclaiming the
   name once the old one is archived), or something new (`projects/lm/`)?
2. **`main.ipynb` requirement**: keep a notebook in every project, or only where
   there's real analysis to show (recommend: only where needed)?
3. **muP dependency**: keep the `mup` package dependency or is muP fully hand-rolled
   in `models/gpt.py` now (affects whether coord-check tooling moves to `chimera.evals`)?
4. **tests scope**: smoke tests only, or also golden-metric regression tests pinned
   to the bit-reproducible tiny-llm runs?
