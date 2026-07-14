# LLM training data — composition by phase

Detailed composition of the data the LLM is trained on, per phase. Weights and
token counts are derived from the source registry (`data/sources.py`,
`TARGET_TOKENS = 10B`) and the mixture builder (`data/build_mixture.py`). Browse
any source with `data/browse.py`; see the running plan with `browse.py --plan`.

**Shared setup (all phases)**
- **Tokenizer:** default `LiquidAI/LFM2.5-230M` byte-level BPE, vocab **64,402** →
  token ids stored as **`uint16`** memmaps. Carries ChatML specials
  (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`). A **custom** tokenizer trained
  on this exact blend can be swapped in — see *Custom tokenizer* below.
- **Packing:** documents concatenated with `<|endoftext|>`, sliced into
  non-overlapping `seq_len` windows, **block-shuffled** (1M-token blocks) across
  sources so no single source sits in one contiguous run.
- **Caches:** per-source token streams in `/mnt/ai/data/llm-mix/tok/<key>/`;
  packed mixes in `/mnt/ai/data/llm-mix/mix[_sft]/<name>/`.
- **x-axis:** every metric is logged against `trainer/trained_tokens`.

Two budgets are built for each phase: a **1B** and a **10B** token subset (the 1B
is the same composition, one-tenth the tokens).

---

## Chat template & special tokens

Single source of truth: `chimera.data.chat_template` (used by the tokenizer
trainer, the pretrain/SFT tokenizers, and the DataModule so they never drift).
Format is **ChatML**, extended for reasoning + tool use:

```
<|im_start|>system
{system}

# Tools
<tools>
[{"name": "...", "description": "...", "parameters": {...}}]
</tools><|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
<think>
{reasoning}
</think>
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call><|im_end|>
<|im_start|>tool
<tool_response>
{result}
</tool_response><|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
```

- **Reasoning** → `<think>…</think>` at the start of an assistant turn.
- **Tool calls** → Hermes-style JSON inside `<tool_call>…</tool_call>` (canonical).
  All source formats (pythonic, OpenAI `tool_calls`, `function_call` role, inline
  tags) are normalized into this by `normalize_messages()`.
- **Tool results** → a `role="tool"` turn wrapped in `<tool_response>…</tool_response>`.

**Special tokens** (reserved at stable low ids; the custom tokenizer bakes these in
— structural first, then semantic):

| id | token | role |
|---:|---|---|
| 0 | `<\|endoftext\|>` | EOS / document separator |
| 1 | `<\|startoftext\|>` | BOS |
| 2 | `<\|pad\|>` | padding |
| 3 | `<\|im_start\|>` | turn start |
| 4 | `<\|im_end\|>` | turn end |
| 5–6 | `<think>` `</think>` | reasoning block |
| 7–8 | `<tool_call>` `</tool_call>` | assistant tool call |
| 9–10 | `<tool_response>` `</tool_response>` | tool result |

(Ids look up **by name** downstream, so LFM2.5 — which lacks the semantic markers —
still works; those tokens just render as ordinary subwords until the custom
tokenizer is used.)

---

## Custom tokenizer (optional, blend-tuned)

The default is the off-the-shelf LFM2.5 BPE. A tokenizer trained on *this* mixture
can compress code identifiers, math notation, and ChatML/tool-call markup better
than one tuned for generic web text. `data/train_tokenizer.py` builds one:

- Samples a **weighted** text corpus straight from the sources (same rows / same
  per-`kind` rendering as `tokenize_source.py`, including Stack v2 from S3); each
  source contributes chars ∝ its mixture weight, so merges reflect the real blend.
- Trains a byte-level BPE with the ChatML specials baked in at stable low ids
  (`<|endoftext|>`, `<|startoftext|>`, `<|pad|>`, `<|im_start|>`, `<|im_end|>`).
- Writes `tokenizer.json` + `meta.json` (vocab, special ids, realized per-source
  char mix) to `/mnt/ai/data/llm-mix/tokenizer/<name>/`.
- `--eval` reports chars/token (compression) per source vs the incumbent.
- Vocab must fit **uint16** (≤ 65535) since ids are stored as `uint16`.

A **suite** of vocab sizes trains from ONE corpus sample: the (S3-expensive) blend
is sampled once, cached to `<name>-corpus.jsonl`, then every vocab size trains from
that cache — outputs land in `<name>-<tag>/` (e.g. `llm-bpe-32k/`).

```bash
# 4k/8k/16k/32k tokenizers over a single ~2GB weighted sample of the full blend
uv run python projects/llm/data/train_tokenizer.py --name llm-bpe \
    --vocab-sizes 4096 8192 16384 32768 --total-chars 2e9 --eval
```

To use one, thread the same path everywhere (id lookups are by name, so IDs need
not match LFM2.5 — but **caches are tokenizer-specific**, so re-tokenize):
```bash
DIR=/mnt/ai/data/llm-mix/tokenizer/llm-bpe-32k
uv run python projects/llm/data/tokenize_source.py --all --budget 17e9 --tokenizer $DIR
uv run python projects/llm/data/build_mixture.py --name mix_10B --total 10e9
uv run python projects/llm/gpt/train.py --mix mix_10B --tokenizer $DIR
```

`BPETokenizer.from_pretrained` accepts either a hub id or a local
`tokenizer.json`/dir, so `--tokenizer` on every stage takes the same value.

---

## Phase 1 — Pretraining (next-token, unmasked)

Raw next-token prediction over the full blend. Everything is fully supervised
(chat/tool rows are rendered to ChatML text and learned end-to-end; masking is a
Phase-2 concern).

### 1a. Target composition (with code) — the eventual mix

This is the intended blend once the Stack v2 code caches finish.

| Category | Source | Weight | Tokens @1B | Tokens @10B | License |
|---|---|---:|---:|---:|---|
| **code 40%** | `stackv2-python` (Stack v2 dedup, Python) | 28% | 280M | 2.80B | other (gated) |
| | `stackv2-json` (Stack v2 dedup, JSON) | 6% | 60M | 600M | other (gated) |
| | `stackv2-bash` (Stack v2 dedup, Shell) | 6% | 60M | 600M | other (gated) |
| **web 25%** | `fineweb-edu` (sample-10BT) | 25% | 250M | 2.50B | ODC-By |
| **math 20%** | `finemath` (FineMath-4+) | 10% | 100M | 1.00B | ODC-By |
| | `openmath-cot` (OpenMathReasoning, CoT) | 10% | 100M | 1.00B | CC-BY-4.0 |
| **tools 15%** | `tool-toucan` (Toucan-1.5M) | 10% | 100M | 1.00B | Apache-2.0 |
| | `tool-xlam` (xLAM-60k, Hermes mirror) | 1% | 10M | 100M | CC-BY-4.0 |
| | `tool-pythonic` (Dria pythonic FC) | 1% | 10M | 100M | see repo |
| | `tool-toolace` (ToolACE) | 1% | 10M | 100M | Apache-2.0 |
| | `tool-apigen-mt` (APIGen-MT-5k) | 1% | 10M | 100M | **CC-BY-NC-4.0** |
| | `tool-hermes` (Hermes-FC v1) | 1% | 10M | 100M | Apache-2.0 |

**Epochs / repetition @10B** (tokens ÷ unique available): code, web, math all draw
**<1 epoch** (supply-rich). Tools: `tool-toucan` 0.17×; but the five small tool
sets *want* 100M each while only ~10–51M unique exists, so at 10B they are **capped
/ repeated** (`tool-toolace`/`tool-hermes` ~10× if forced; the builder caps at
available, so they simply contribute their full unique set). At **1B** every source
is under one epoch except `tool-toolace`/`tool-hermes` (~1× / slightly capped).

> ⚠️ `tool-apigen-mt` is **CC-BY-NC** (non-commercial). Drop it if that matters —
> it's only 1% and Toucan covers deep multi-turn trajectories.

### 1b. Interim composition (code deferred) — **what is being built now**

Stack v2 code content is fetched per-file from Software Heritage S3 and is the
slow pole, so code is **dropped from the immediate subsets** and the remaining
weights are **renormalized to sum to 1.0**. The builder renormalizes over whichever
sources have a complete cache, so **re-running `build_mixture.py` once the code
caches finish automatically restores the 1a target** (code back to 40%).

| Category | Source | Weight (renorm) | Tokens @1B | Tokens @10B |
|---|---|---:|---:|---:|
| **web 41.7%** | `fineweb-edu` | 41.67% | 417M | 4.17B |
| **math 33.3%** | `finemath` | 16.67% | 167M | 1.67B |
| | `openmath-cot` | 16.67% | 167M | 1.67B |
| **tools 25%** | `tool-toucan` | 16.67% | 167M | 1.67B |
| | `tool-xlam` | 1.67% | 17M | 167M |
| | `tool-pythonic` | 1.67% | 17M | 167M |
| | `tool-toolace` | 1.67% | 17M | 167M* |
| | `tool-apigen-mt` | 1.67% | 17M | 167M |
| | `tool-hermes` | 1.67% | 17M | 167M* |

\* capped at available unique tokens (`tool-toolace`/`tool-hermes` ≈ 10M each), so
the realized 10B mix lands slightly under 10B until code (or more tool data) fills
the gap.

Built as `mix_1B` and `mix_10B` under `mix/`. Rebuild to fold code in later:
```bash
uv run python projects/llm/data/build_mixture.py --name mix_10B --total 10e9
```

---

## Phase 2 — SFT (masked ChatML)

Supervised fine-tuning of the Phase-1 base checkpoint. Only conversational sources
are used, rendered to ChatML with **loss masking**: the model's own turns are
supervised, everything else is `-100` (ignored).

**Masking policy** (in `chimera.data.chat_template.iter_segments`) — supervised =
the assistant's own output: its `<think>` block, its content, its `<tool_call>`
blocks, **and** the closing `<|im_end|>` (so it learns to stop). Masked = system
prompts, `<tools>` schema blocks, user turns, and **tool responses** (the model
must never learn to fabricate tool outputs).

Sources = the six chat/tool sets + OpenMathReasoning (rendered user=problem /
assistant=solution), weights renormalized over just these:

| Source | Weight (renorm) | Supervised frac | Tokens @1B | License |
|---|---:|---:|---:|---|
| `openmath-cot` (problem → CoT solution) | 40% | ~99% | 400M | CC-BY-4.0 |
| `tool-toucan` (tool-agent trajectories) | 40% | ~9% | 400M | Apache-2.0 |
| `tool-xlam` (single-turn FC) | 4% | ~mid | 40M | CC-BY-4.0 |
| `tool-pythonic` (pythonic FC) | 4% | ~mid | 40M | see repo |
| `tool-toolace` (multi-turn FC) | 4% | ~mid | 40M | Apache-2.0 |
| `tool-apigen-mt` (deep agentic) | 4% | ~low | 40M | **CC-BY-NC-4.0** |
| `tool-hermes` (multi-turn FC) | 4% | ~mid | 40M | Apache-2.0 |

**Supervised fraction** (measured) varies sharply by source: `openmath-cot` ~99%
(the solution is nearly the whole example), `tool-toucan` ~9% (most tokens are tool
schemas, user turns, and tool results — all masked). So the *effective* supervised
signal is reasoning-heavy; raise the tool weights if you want more tool-call
supervision relative to reasoning.

Built as `sft_1B` (and optionally `sft_10B`) under `mix_sft/`:
```bash
uv run python projects/llm/data/tokenize_source.py --sft --all --budget 1e9
uv run python projects/llm/data/build_mixture.py --sft --name sft_1B --total 1e9
uv run python projects/llm/sft/train.py --mix sft_1B --init-ckpt <base>/gpt.ckpt
```

---

## Provenance / reproduction

| Step | Command |
|---|---|
| Inspect a source | `data/browse.py --source <key> --n 3` |
| See the plan + repeat factors | `data/browse.py --plan [TOTAL]` |
| Tokenize (pretrain) | `data/tokenize_source.py --all --budget 17e9` |
| Tokenize (SFT, masked) | `data/tokenize_source.py --sft --all --budget 1e9` |
| Pack a mix | `data/build_mixture.py --name mix_10B --total 10e9 [--sft]` |

Each packed mix writes a `manifest.json` recording the exact realized token counts,
per-source weights, repeat factors, and seed — the ground truth for a given run.
