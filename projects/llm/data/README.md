# LLM pretraining data

Browsing layer for the planned pretraining mixture. Lets you eyeball real samples
from each dataset before any tokenization/packing pipeline is built.

## Mixture

| Slice | Source | Weight | Unique tokens | Category |
|---|---|---:|---:|---|
| `stackv2-python` | The Stack v2 (dedup) — Python | 28% | ~60B* | code |
| `stackv2-bash` | The Stack v2 (dedup) — Shell/Bash | 6% | ~4B* | code |
| `stackv2-json` | The Stack v2 (dedup) — JSON | 6% | ~40B* | code |
| `fineweb-edu` | FineWeb-Edu (`sample-10BT`) | 25% | 10B | web |
| `finemath` | FineMath 4+ | 10% | 9.5B | math |
| `openmath-cot` | OpenMathReasoning — CoT | 10% | 24.9B | math |
| `tool-toucan` | Toucan-1.5M (synthetic tool-agent, Apache-2.0) | 10% | ~6.0B | tools |
| `tool-xlam` | xLAM-60k (Hermes reformat, **ungated**) | 1% | 51M | tools |
| `tool-pythonic` | Dria pythonic function-calling | 1% | 49M | tools |
| `tool-toolace` | ToolACE (multi-turn) | 1% | 10.4M | tools |
| `tool-apigen-mt` | APIGen-MT-5k (deep agentic, **NC license**) | 1% | 30M | tools |
| `tool-hermes` | Hermes function-calling v1 | 1% | 10.4M | tools |

By category: **code 40% · web 25% · math 20% · tools 15%** (= 100%).
Token counts are measured with the LFM2.5 tokenizer on cached shards; *Stack v2 is
a published order-of-magnitude (its content is gated + S3-only, not scanned here).

**The tools slice is no longer supply-constrained.** `tool-toucan` alone is ~6B
tokens (Apache-2.0), so 15% of a 10B run is ~0.2 epochs of the tools category —
see the plan below. The five smaller sets carry only ~1% each, for format diversity
(single-turn FC, pythonic calls, deep human-in-the-loop trajectories); their tiny
weight means their per-source repeat factor is high but low-impact.

## Training plan (weights → tokens → repeat factors)

Open `main.ipynb` and run the **Training plan** section; set `BUDGET` to any token
count (default 10B).

At a **10B-token** budget this shows, per source, target tokens (`weight × budget`)
and the **repeat factor** (`target ÷ unique available`). Every category now sits
well under 1× (code 0.0×, web 0.2×, math 0.1×, tools 0.2×) — no repetition trap.
`TARGET_TOKENS` in `sources.py` sets the default budget.

### Phasing (optional)
Tool-calling is a format/behavior skill, so you can still learn it in a short
high-weight anneal/SFT phase rather than spread across pretraining — but with
Toucan's ~6B tokens it's equally fine to keep 15% in a single run without
repetition. The choice is now about curriculum, not supply.

## Usage

Everything runs from **`main.ipynb`** (the old CLIs — `browse.py`,
`tokenize_source.py`, `train_tokenizer.py`, `build_mixture.py` — were folded into
it; the last three stay as importable library modules the notebook drives).
Edit the config variables at the top of a section and run the cell:

- **The mixture / Training plan** — list slices, per-category weights, target
  tokens + repeat factors for any `BUDGET`.
- **Peek at real samples** — set `SOURCE` (or `None` to sweep all), `N`, `CHARS`,
  `FULL`; streams real rows from the smallest cached shard (Stack v2 from S3).
- **Document-length distributions** — token-length stats + histograms per source.
- **Tokenize / Train tokenizer / Build mixture** — the packing pipeline.

## How sampling works (and why)

These datasets are **Xet-backed**. In this environment two lighter peek paths do
*not* work: the HF dataset-viewer `/rows` API is unreachable, and ranged HTTP
reads are rejected by the Xet CDN. The one dependable content path is
`huggingface_hub`'s native Xet downloader, which fetches whole files.

So `browse.py` downloads the **smallest shard** of a slice (e.g. a partial final
FineWeb-Edu shard is ~0.5 GB vs ~2 GB), caches it under `HF_HOME`
(`/mnt/ai/data/hf`), and reads only the first rows + needed columns locally. One
shard is a *chunk* of the corpus, not the whole dataset, and it's cached — so
repeated browsing is free. Each run prints which shard it read and its size.

The Stack v2 is different: its rows are **pointers**, not code. Each row carries a
`blob_id` whose bytes live in the public Software Heritage S3 bucket, so those
slices read a tiny metadata shard and then fetch each file's content on demand
from `s3://softwareheritage/content/` (anonymous access via `boto3`).

## Where to get more tool-call data

The registered tool sources cover the clean, permissively-licensed core. The big
one is **`Agent-Ark/Toucan-1.5M`** (~6B tokens, Apache-2.0) — the largest open
tool-agent dataset, which by itself makes the 15% slice comfortable.

**Large, high-value (add these first if you want even more):**
- `Agent-Ark/Toucan-1.5M` — **registered** as `tool-toucan`. 1.5M multi-turn
  trajectories over real MCP tools, ~6B tokens, Apache-2.0.
- `nvidia/Nemotron-SFT-Agentic-v2` — ~2B tokens across `interactive_agent` (6.3GB),
  `search`, and `tool_calling` splits; single + multi-turn, permissive. NVIDIA quality.
- `nebius/SWE-agent-trajectories` — 80k SWE-agent trajectories (~0.3B tokens),
  CC-BY-4.0. Agentic *coding* (edit/run/observe), a different flavor from FC.
- `microsoft/orca-agentinstruct-1M-v1` — 1M agentic instruction pairs, CDLA-permissive.
  Broad (not pure FC), but big.

**Smaller / niche:**
- `glaiveai/glaive-function-calling-v2` — ~70M tokens, multi-turn, Apache-2.0. Older/noisier.
- `AymanTarig/function-calling-v0.2-with-r1-cot` — xLAM + DeepSeek-R1 reasoning chains
  (tool-use *with* CoT).
- `nvidia/Llama-Nemotron-Post-Training-Dataset` → `train/when2call_train_sft.jsonl`
  — "When2Call": deciding *whether* to call a tool (and abstaining). CC-BY-4.0.
- ToolLLM's **ToolBench** (large RapidAPI multi-tool trajectories) — HF mirrors like
  `Yhyu13/ToolBench_toolllama_G123_dfs`; older, messy schema.
- `xingyaoww/code-act` — CodeAct agentic trajectories (code-as-action).

**Generate your own** (APIGen/ToolACE-style pipelines) if you need control over the
specific tool schemas your model will see in deployment.

## Access requirements

- **The Stack v2 is gated.** Accept the terms at
  <https://huggingface.co/datasets/bigcode/the-stack-v2-dedup> while logged in as
  the account behind `HF_TOKEN`, or the `stackv2-*` slices raise a 403 with the
  request-access URL. FineWeb-Edu, FineMath, OpenMathReasoning, and all registered
  tool sources are open (xLAM is reached via the ungated `minpeter` mirror).
- **`tool-apigen-mt` is CC-BY-NC** (non-commercial) — drop it if that matters.
- `HF_TOKEN` must be set (it is in this environment) and `HF_HOME=/mnt/ai/data/hf`
  so shards land on the big volume.
- Software Heritage content is fetched anonymously; if that ever returns
  AccessDenied, S3 credentials / requester-pays may be needed.

## Files

- `main.ipynb` — the interactive workbench (browse, plan, peek, length stats,
  tokenize, train tokenizer, build mixture). This is the entry point.
- `sources.py` — the mixture registry (`SOURCES`), one `Source` per slice with its
  weight, backing repo, and a streaming `sample()`; plus the shard-peek and SWH
  content helpers.
- `tokenize_source.py` / `train_tokenizer.py` / `build_mixture.py` — library
  modules (row rendering, tokenization, corpus sampling, weighted packing) that
  the notebook imports and drives.

## Next steps (not built yet)

- Decide phasing (single-run vs pretrain + tool-heavy anneal/SFT) and set final
  weights + `TARGET_TOKENS` accordingly.
- Optionally add `nvidia/Nemotron-SFT-Agentic-v2` / `nebius/SWE-agent-trajectories`
  for more agentic-coding flavor (see "Where to get more tool-call data").
- Build the tokenization + weighted-packing DataModule that turns this mixture
  (weights from `SOURCES`) into training batches — mirrors `chimera.data.fineweb_edu`
  / `ultrachat`, adding weighted interleaving + per-source repeat.
