# GRPO — RL fine-tuning of a small LLM on verifiable tasks

From-scratch **Group Relative Policy Optimization** (DeepSeekMath, [arXiv:2402.03300](https://arxiv.org/abs/2402.03300))
for **LiquidAI/LFM2.5-230M**, implemented as a `LightningModule` on the repo's training
rails (`chimera.utils.experiment` + wandb), with HuggingFace `transformers` generation and a
LoRA policy. The first task is **GSM8K** grade-school math, where correctness is a cheap
exact-match on the final number — an easily verifiable reward, no reward model needed.

## Why GRPO

GRPO drops PPO's learned value network. For each prompt it samples a *group* of `G`
completions, scores them with a verifiable reward, and uses the **group mean** as the
baseline: completions above the group average get pushed up, those below get pushed down.

```
advantage_i = (reward_i − mean(group)) / (std(group) + eps)
loss = − (1 / Σ|completions|) · Σ_i Σ_t  advantage_i · log π(token_{i,t})   [+ β·KL]
```

We do one optimizer step per generation (`μ = 1`), so the PPO importance ratio is exactly 1
and there is no `θ_old` cache. KL to a reference policy is **off by default** (`β = 0`, the
modern GRPO default); when enabled, the reference is the **LoRA-disabled base model**, so no
separate reference model sits in memory.

## Layout

Runnable **scripts** live here in `projects/grpo/`; the reusable **library** lives in the
`chimera.grpo` package (`src/chimera/grpo/`), matching the repo's split between `projects/*/`
entry points and the importable `chimera` package.

**Scripts** — `projects/grpo/`

| file | role |
|------|------|
| `train.py`      | `LitGRPO` + `main()` — the GRPO step, periodic greedy pass@1, optimizer/schedule |
| `eval.py`       | standalone pass@1 on the held-out test set (base vs. trained) |

**Library** — `chimera.grpo` (`import chimera.grpo`)

| module | role |
|--------|------|
| `core.py`       | pure algorithm: group advantages, per-token log-probs, the loss (unit-testable) |
| `rewards.py`    | verifiable reward primitives: answer extraction, `correctness_reward`, `format_reward` |
| `tasks.py`      | `Task` dataclass + `TASKS` registry (the extensibility seam); `gsm8k` implemented |
| `data.py`       | `PromptDataModule` — serves chat-templated prompt + gold batches |
| `generation.py` | shared rollout helper (`generate_completions`, completion masking) used by train + eval |

## Setup

The only extra dependency over the repo baseline is **`peft`** (already added via `uv add peft`).
Models/datasets cache under `/mnt/ai/data/hf` (`HF_HOME`).

## Usage

```bash
# baseline: untrained base model on the full GSM8K test set
uv run python projects/grpo/eval.py --task gsm8k

# train (fresh run)
uv run python projects/grpo/train.py --task gsm8k --epochs 2

# resume a run for more epochs (same wandb run, rebuilds model + adapter from the checkpoint)
uv run python projects/grpo/train.py --resume <run_id> --epochs 4

# after training: score the trained adapter on the same test set
uv run python projects/grpo/eval.py --task gsm8k --resume <run_id>

# fast smoke run (few prompts, small group, tiny val/test)
uv run python projects/grpo/train.py --train-size 32 --val-size 16 \
    --batch-size 2 --num-generations 4 --max-completion-len 128 --epochs 1
```

### Key flags (`train.py`)

- `--num-generations G` — completions per prompt (group size). Effective rollouts/step = `batch_size · G`.
- `--batch-size` — **prompts** per step (default 4).
- `--beta` — KL coefficient (default 0; >0 adds the reference-model KL term).
- `--no-scale-rewards` — disable the group-std division in the advantage ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783)).
- `--lora-r / --lora-alpha / --lora-dropout` — LoRA config (`target_modules="all-linear"`).
- `--temperature / --top-p` — sampling for rollouts (greedy is always used for val/test).
- `--grad-checkpoint` — trade generation speed for lower activation memory (for larger batches).
- Inherited from `add_common_args`: `--lr`, `--grad-clip`, `--epochs`, `--seed`, `--data-dir`,
  `--resume`. `--compile-mode` defaults to `off` and `--ema-decay` to `0` (both ill-suited to a
  generate-in-the-loop LoRA RL run).

## Reward design (GSM8K)

- **`correctness_reward`** (weight 1.0): `1.0` if the completion's final number equals the gold
  number, else `0.0`. The number is parsed from the last `#### <n>` marker (fallback: the last
  number in the text), normalizing `$`, commas, and decimals.
- **`format_reward`** (weight 1.0 × 0.1): a small bonus for emitting a parseable `#### <number>`
  line, which keeps grading reliable and nudges a consistent output format.

Watch `train/reward`, `val/accuracy` (↑), `train/frac_reward_zero_std` (should be < 1 — groups
need a mix of right/wrong to give signal), and `completions/mean_length` in wandb.

## Adding a new verifiable task

1. Add reward primitives to `rewards.py` if the existing ones don't fit (e.g. symbolic
   equivalence via `math-verify` for competition MATH, or a sandboxed runner for code).
2. Add a `Task(...)` entry to `TASKS` in `tasks.py`: its `load_splits` / `load_test`,
   `build_prompt` (system + user chat messages), `gold_of`, `reward_funcs` + `reward_weights`,
   and the `correctness` func used for pass@1.
3. Train it: `--task <name>`. No trainer changes needed.
