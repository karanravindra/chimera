# GRPO on GSM8K — Engineering Journal

**Append-only.** Newest entries go at the bottom; earlier entries are never edited — if
something turns out wrong, a later entry corrects it. Each entry records **what I did**, the
**results**, and the **implications** for what comes next. For the current best result, read
the most recent `Run` entry.

**Goal:** improve LFM2.5-230M GSM8K pass@1 via GRPO (LoRA). Budget: up to 5 runs; stop once a
run clearly beats baseline.

**Standing setup** (unless an entry overrides it): LFM2.5-230M, bf16 base + fp32 LoRA
(`all-linear`, r=16, ~3.9M trainable); reward = **correctness only**; eval = 64 held-out GSM8K
problems, greedy, pass@1 — deterministic and comparable across runs (noise ≈ ±0.016 per
problem); GRPO μ=1, std-scaled group advantages, token-level (DAPO) loss, KL off; single
RTX 5070 Ti, 16 GB; generation via HF `transformers` (no vLLM yet).

---

## J01 — Scaffolding + baseline · 2026-06-28
**Did:** Built `projects/grpo` (from-scratch GRPO as a Lightning module, `transformers`
generation, LoRA policy) and measured the untrained base model.
**Results:** Baseline pass@1 = **0.141 (9/64)** at max_completion=160; **0.156 (10/64)** at
max_completion=200 (fewer truncations). Format-line rate 1.00 from the system prompt alone.
**Implications:** Baseline depends on the completion cap (truncation), so max_completion must be
held fixed when comparing runs. ~0.15 is the number to beat.

## J02 — Run 0: lr=2e-4 + additive format reward → COLLAPSE · 2026-06-28
**Did:** A quick run with a +0.1 format bonus alongside correctness, lr=2e-4 to force fast movement.
**Results:** Collapsed within ~3 steps to bare 4-token `#### N` outputs. correctness 0.25→0;
pass@1 0.156 → **0.031**; completion length 119 → 5.
**Implications:** Two faults — (1) LR ~20× too high; (2) the format reward is exploitable (a bare
answer line always earns it), so GRPO maximized the bonus by abandoning reasoning. This is
reward-hacking, not a training failure. → Drop the format reward; lower the LR.

## J03 — Fixes + RAM envelope · 2026-06-28
**Did:** Dropped the format reward (correctness-only in `tasks.py`). Fixed a real bug:
`PromptDataModule` never forwarded `val_size` to `task.load_splits`, so the eval set was always
200 regardless of config. Measured memory at two scales.
**Results:** 32 rollouts/step (batch 4 × G 8) at ~360-token sequences → peak **11.3 GB**, stable
across steps (no leak). 64 rollouts/step → **OOM** (~22 GB; dies on the training-forward
`lm_head` over all sequences). At max_completion=200, peak rises to ~14.6 GB on long-completion steps.
**Implications:** Practical ceiling ≈ **32 rollouts at these lengths on 16 GB**. To go wider:
shorter completions, gradient checkpointing, or micro-batching the log-prob forward.

## J04 — Run 1: correctness-only, lr=1e-5 → BIG WIN · 2026-06-28
**Did:** correctness-only reward, lr=1e-5, G=8, batch=4, max_completion=200, 48 steps (1 epoch
over 192 prompts). `scratchpad/run.py --tag run1`.
**Results:** pass@1 **0.156 → 0.375 (24/64)**, **+0.219** (+14 problems, well beyond noise).
Truncation 0.30 → 0.17; completion length healthy (~140 tok), no collapse. Per-step reward trend
0.17 → 0.28 (noisy at the batch level; `zero_std` swings 0–1 as batches are all-right/all-wrong).
**Implications:** **Goal met on the first real run.** The earlier collapse was the format reward,
not the algorithm — correctness-only + low LR trains cleanly and improves fast. Caveat: 64-problem
eval is small; should confirm on a larger / official test set before over-claiming. Banking the
win; 4 runs of budget unused.

## J05 — Throughput observation: GPU ~10% util, RAM high · 2026-06-28
**Did:** Noted GPU compute utilization ~10% while VRAM sat at ~12–15 GB during training.
**Diagnosis:** RL wall-clock is dominated by autoregressive **decoding** inside
`transformers.generate()` — one token at a time, tiny per-token matmuls on a 230M model → the GPU
is kernel-launch / memory-bandwidth bound, not FLOP-bound. Low util is expected for this path. The
high VRAM is unrelated to util: it comes from the wide 32-rollout batch plus the full-sequence
training forward/backward and logits. So we are simultaneously **time-bound on decode** and
**memory-bound on the backward** — two different constraints.
**Implications / levers (ranked):**
1. **Faster generation engine** — biggest win. vLLM (paged attention + continuous batching,
   ~10–20× decode; the isolated `/root/vllm-env` already exists), or HF continuous batching, or
   static KV cache + compiled `generate` (in-env, lower effort).
2. **More rollouts per decode pass** to fill idle compute — but first cut training-forward memory
   (micro-batch the log-prob forward, or gradient checkpointing) since we're at the RAM ceiling.
3. **Shorter max_completion** trades correctness for speed.
Next (pending direction): wire in a faster generation backend; vLLM is the standard GRPO choice.

## J06 — Throughput fix: lfm2 static cache fails; concurrency is the lever · 2026-06-28
**Did:** Benchmarked generation paths on lfm2+LoRA at training shape (32 seqs, 160 new tok).
**Results:** Baseline 844 tok/s. **Static KV cache unsupported on lfm2** (`KeyError: 'conv'` /
`LinearAttentionLayer has no max_batch_size` — the hybrid conv/linear-attention layers don't
implement the StaticCache interface), so the cudagraph route is out. `torch.compile(default)`
gave ~0% (848 vs 844). But decode throughput scales ~linearly with **concurrency** at nearly
constant wall time, and generation memory is tiny:

| concurrent seqs | wall | tok/s | peak VRAM |
|---|---|---|---|
| 32  | 6.07s | 844  | 0.68 G |
| 64  | 6.11s | 1676 | 0.98 G |
| 128 | 6.16s | 3323 | 1.44 G |
| 256 | 6.69s | 6121 (7.2×) | 2.37 G |

**Implications:** The ~10% util was from running far too few concurrent sequences — the decode
is latency/bandwidth-bound and the tiny 230M model leaves the GPU idle at 32 seqs. Custom
kernels won't help (cuBLAS already fine; the gap is launch/occupancy). The win is to **generate
a wide rollout batch** (e.g. 32 prompts × G8 = 256 rollouts in ~the same wall time → 7× samples)
and **micro-batch the backward** to fit 16 GB. Next: implement big-gen + chunked backward with
global token-normalization, then a larger run.

## J07 — Run 2: wide gen (256 rollouts) + micro-batched backward · 2026-06-28
**Did:** Implemented wide generation + chunked backward with global token-normalization
(`scratchpad/run2.py`): generate 32 prompts × G8 = 256 rollouts in one decode pass, then run the
policy-gradient backward in chunks of 32 rollouts. lr=1e-5, 40 steps.
**Results:** pass@1 **0.188 → 0.391 (25/64)** (within-run baseline; note eval baseline shifted
0.156→0.188 vs Run 1 purely from eval **batch size** 16→32 — greedy decoding is padding-sensitive,
so compare only within a run). Truncation 0.31 → 0.12. Throughput: **256 rollouts in ~11s/step**
vs Run 1's 32 in ~8s → ~5–7× more samples/sec; GPU util rose from ~10% to 33–100%. Peak 14.8 GB.
**Implications:** Throughput goal achieved. But accuracy matched Run 1 (~0.39) despite 6.7× more
rollouts — and `gnorm` stayed ~0.11 (vs Run 1's ~0.4). The big batch gives a cleaner, smaller
gradient and is **under-stepping at lr=1e-5**. Standard large-batch fix: raise the LR. → Run 3 at
lr=3e-5. Also: greedy eval is batch-size sensitive; future evals should fix batch size (or use a
larger eval set) for cross-run comparability.

**Note on library integration:** the speedup lives in the harness. Folding it into `LitGRPO`
cleanly needs manual optimization (`automatic_optimization=False`) to do the chunked backward —
deferred (documented here) rather than rushed.

## J08 — Run 3: same wide pipeline at lr=3e-5 · 2026-06-28
**Did:** Repeated Run 2's 256-rollout/step pipeline with lr=3e-5 (large batches usually want a
higher LR; Run 2's gnorm≈0.11 showed headroom). 40 steps, seed 42 (same data order as Run 2).
**Results:** pass@1 **0.188 → 0.359 (23/64)**. Training reward ran higher (0.260 → 0.336 vs Run 2's
0.214 → 0.296) and completions tightened further (truncation 0.31 → 0.06, len ~121). But eval
pass@1 (0.359) is within 64-problem noise of Run 2 (0.391) — 23 vs 25 correct.
**Implications:** 3e-5 raised the *reward* the model optimizes but not the held-out pass@1 — a hint
of mild reward/eval divergence (and/or just eval noise at n=64). At this step budget the result
plateaus ~0.36–0.39 regardless of 1e-5 vs 3e-5. To push further would need a larger eval to
resolve differences, more steps/data, or curriculum — not just more LR.

## J09 — Session summary (1-hour budget) · 2026-06-28
**Throughput (the main win):** diagnosed ~10% GPU util as decode under-concurrency, not a kernel
problem (lfm2 can't use static-cache/cudagraphs; compile gave 0%). Switched to **wide generation
(256 rollouts/step) + micro-batched backward with global token-normalization**. Result: ~5–7×
more samples/sec at the same wall-time, GPU util 10% → 33–100%, peak 14.8 GB (fits 16 GB).
**Performance:** GRPO lifts base pass@1 from ~0.16–0.19 to **~0.36–0.39 (best 0.391, Run 2)**,
≈2.4×, stable across Runs 1/2/3 and free of the Run 0 collapse once the format reward was dropped.
**Method/eval lessons:** correctness-only reward avoids reward-hacking; greedy eval is sensitive
to batch-size/padding (compare within-run); n=64 eval is too noisy to separate ~0.36 vs ~0.39.
**Carried-forward / not done:** (a) fold the wide-gen+micro-backward path into `LitGRPO` (needs
`automatic_optimization=False`); (b) **other math data** — `microsoft/orca-math-word-problems-200k`
is the clean add (same `question`/`answer` columns, numeric answers → works with the existing
reward; MetaMathQA mixes in non-numeric MATH answers our reward can't grade). Plan: mix it into the
train pool, keep eval on GSM8K; (c) confirm best run on the full 1319-problem GSM8K test set.

---

## J10 — Session 2 kickoff: VibeThinker recipe + 12h plan + open-domain direction · 2026-06-28
**Goal (revised):** 12 hours to improve the model. Beyond pushing GSM8K pass@1, the user asked to
(1) adopt a **VibeThinker**-style process and stack optimizations on top, (2) make the trainer
**open-domain** (not just math) — which datasets/rewards, (3) scope an **agentic research**
direction (the model as a tool-using search/RAG agent with verifiable rewards).

**VibeThinker brief (refs: arXiv:2511.06221 the 1.5B report; `WeiboAI/VibeThinker-1.5B`, base
`Qwen2.5-Math-1.5B`, MIT).** Its recipe is the **Spectrum-to-Signal Principle (SSP)**: deliberately
*decouple* SFT and RL. SFT is tuned for **Pass@K, not Pass@1** (broaden the diversity of correct
trajectories — "spectrum"); RL then narrows to correctness ("signal") *without* killing diversity.
RL stage = **MGPO (MaxEnt-Guided Policy Optimization)**, a GRPO variant that reweights the
group-relative advantage by problem difficulty: `A' = w(p_c)·A`, with
`w(p_c)=exp(−λ·D_ME(p_c‖0.5))` and `D_ME` the binary-entropy deviation of the group pass-rate
`p_c` from 0.5. Net effect: problems the model solves ~50% of the time get the **most** weight;
trivially-easy (p→1) and currently-impossible (p→0) groups are exponentially down-weighted. They
sample math rollouts at **T=1.0** (we already do). Headline: 1.5B matches DeepSeek-R1-671B on
AIME/HMMT for ~$8K — but on a *strong* math base; for our weak 230M base the brief flags the
**SFT/rejection-sampling warm-start as more important than it was for them.**

**Transferable techniques, ranked (impact × ease on 230M/16GB):** ① T=1.0 rollouts (done);
② **MGPO continuous difficulty weighting** (few lines); ③ **dynamic sampling / difficulty
filtering** (the discrete version of ②: drop all-right & all-wrong groups, which carry zero GRPO
advantage anyway — saves the dead forward/backward); ④ **rejection-sampling SFT (STaR/ReST) warm
start**, Pass@K-early-stopped not Pass@1; ⑤ light entropy/KL to preserve exploration.

**Plan for the 12h (each run journaled):**
- **Measurement fix first:** eval n=64 → **n=200** (deterministic, batched) so we can actually
  resolve ~0.36 vs ~0.39; full 1319-test confirmation at the end. New harness `scratchpad/run4.py`.
- **Run A** (in flight): gsm8k + **dynamic sampling** (③) at n=200 — establishes a reliable
  baseline and tests whether focusing on informative groups beats Run 2/3's ~0.39.
- **Run B:** `mathmix` (gsm8k + orca-math 200k) + dynamic — does more/varied math data help?
- **Run C:** add **MGPO weighting** (②, sweep λ).
- **Run D (if time):** rejection-sampling SFT warm start (④) → GRPO.
- **Infra:** added `orcamath` + `mathmix` tasks to the registry (orca gold parses 99.7%, ~85%
  integer); `--task` selects. Open-domain task seam + agentic-research design documented below as
  they land.
**Status:** harness smoke-tested (3 steps lifted a 32-problem eval 0.156→0.281, peak 13.3 GB, fits).

## J11 — Open-domain expansion: task registry now spans 5 verifiable tasks · 2026-06-28
**Did:** Used the open-domain RLVR brief to extend the `tasks.py` registry beyond GSM8K,
proving the "one Task entry per domain" seam. Added:
- **`orcamath`** / **`mathmix`** — more/varied numeric math (orca-math 200k; mix = gsm8k+orca).
  Reuses the numeric reward; eval stays GSM8K. (orca gold parses 99.7%.)
- **`dapomath`** — `open-r1/DAPO-Math-17k-Processed`, harder competition math with bare-answer
  `solution` gold; filtered to numeric-parseable rows; reuses the numeric reward; eval on GSM8K.
- **`countdown`** — `Jiayi-Pan/Countdown-Tasks-3to4`, a *non-math* verifiable domain
  (generate-and-check arithmetic: reach the target using each number once). New reward
  `countdown_reward` safe-evals a regex-gated expression (only digits/`+-*/()`, no names or
  builtins), checks the number multiset is used exactly and the value hits the target. Gold is
  encoded `"<target>|<n1>,<n2>,..."`. Unit-tested 8 cases incl. division `6/(1-3/4)=24` — all pass.
**Why these three:** the brief ranked them as the lowest-friction open-domain adds — DAPO reuses
our exact reward; **Countdown is the cleanest unhackable objective for a tiny model** (no stored
answer, ~0 luck floor), and it proves the pipeline generalizes past numeric-answer math.
**Rationale on reward hackability:** kept correctness as the sole signal everywhere (our own J02
collapse showed additive format bonuses wreck this model). Countdown's parser is deliberately
strict — lenient prose-scraping would invite false-positive reward hacking.
**Implications:** open-domain is now a config flag (`--task`). Larger curated multi-domain
backbones (`LLM360/guru-RL-92k`, MIT, ships verifiers; code via `AceCode-87K` + a sandbox) are
the documented next tier but need a code-exec sandbox. Next: run GRPO on `countdown` to confirm a
non-math verifiable task trains, and the agentic-search direction (below).

## J12 — MGPO difficulty weighting added to the core · 2026-06-28
**Did:** Implemented `mgpo_difficulty_weights(rewards, G, lam, p0=0.5)` in `core.py` (VibeThinker's
RL technique): `w(p_c)=exp(-lam·KL(Bernoulli(p_c)‖Bernoulli(0.5)))`, broadcast per group, applied
as `A' = w·A`. Soft, continuous version of dynamic sampling — keeps all groups but concentrates
gradient on ~50%-pass prompts. Wired as `--mgpo-lambda` in the harness (0 = plain GRPO). To be
ablated in an upcoming run (sweep λ∈{1,2,3}).

## J13 — Run A: gsm8k + dynamic sampling, reliable n=200 eval · 2026-06-28
**Did:** First long run on the new harness — gsm8k, **dynamic sampling** (filter+refill to 24
informative groups/step, up to 3 passes), lr=1e-5, 80 steps, **n=200 held-out eval**.
**Results:** pass@1 **0.160 → 0.370 (74/200)**, **+0.210** (≈2.3×). Truncation **0.34 → 0.07**
(much cleaner than Run 2's 0.12) and completion length steady ~125. Step time ~18–27s (2–3 gen
passes to refill); **peak only 7.7 GB** (micro=16 + `expandable_segments` — huge headroom on 16 GB).
**Implications:** This is the *reliable* number — at n=200 the model sits at ~0.37, consistent with
Runs 2/3's noisy n=64 ~0.37–0.39. So **dynamic sampling matches the plateau on accuracy but
produces far cleaner outputs** (trunc 0.07); gnorm stayed ~0.18–0.23 (still head-room). The ~0.37
plateau is real, not eval noise. Breaking it needs **data** (mathmix), **method** (MGPO), or a
**warm start** (STaR) — exactly the B/C/D + STaR queue.
**Tooling:** runs now log **live to wandb** (`grpo-vibe` project) — reward, held-out pass@1 curve,
gnorm, step time — so runs compare side-by-side in the browser. Iteration runs shortened to
**40 steps (~13 min)** with start/mid/end n=200 evals; long runs reserved for final confirmation.
Added `eval.py --adapter <dir>` to score saved adapters on the full 1319-problem GSM8K test set.

## J14 — Agentic direction: tool layer + architecture (make it a tool user, not a pure reasoner) · 2026-06-29
**Did:** Started the "agentic rather than pure reasoning" track. Two complementary uses of one
shared machinery (multi-turn loop + tool-call parse + observation injection + **observation-token
masking** + outcome reward — masking is the only change vs single-turn GRPO; the existing core
applies once obs tokens are masked out of the loss):
- **Tool-Integrated Reasoning (TIR) for math** — model calls a calculator (`<calc>..</calc>` →
  `<result>..</result>`) instead of computing in its head. For a 230M model this targets the
  *dominant* error source (arithmetic), so it should beat pure-reasoning RL on the very objective
  we're optimizing. cf. ToRA (arXiv:2309.17452).
- **Search/RAG agent for knowledge work** — Search-R1 loop (BM25 over HotpotQA's own context, EM
  reward); env already validated (BM25 recall@2 0.647 vs 0.20 random; genuinely multi-hop).
**Built (backend-agnostic, library):** `chimera/grpo/tools.py` — `Tool` protocol, `CalculatorTool`
(sandboxed regex-gated eval, reuses the countdown guard), `SearchTool` (injected retriever),
`find_tool_call` / `extract_answer` / `format_observation`. Unit-tested (calc incl. `2/3+1/6`,
div-by-zero, injection attempt `import os`; mixed-tool parsing). The single-seq HotpotQA rollout
prototype is `scratchpad/agent_search.py`.
**Sequencing decision (ties to throughput):** the multi-turn *generation engine* (batched HF loop
vs vLLM) is deferred until the throughput audit returns the **vLLM-lfm2 verdict** — if vLLM serves
lfm2+LoRA it becomes the engine for both agentic rollouts and ordinary GRPO (potential 10–20×
decode). Building the engine before that verdict would mean building it twice.
**Throughput:** dispatched a background benchmarking agent (per the standing directive to keep
hunting throughput wins) to (a) settle vLLM-lfm2 support with measured tok/s, (b) quantify the
dynamic-refill multi-pass waste, (c) check how high `micro` can go now that peak is only 7.7/16 GB.
Memory-capped to not disturb the running queue. Results to be journaled.
