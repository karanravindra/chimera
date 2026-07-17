# Tiny-LM Datasets — Reference

A catalog of the datasets the tiny / small-LM community uses to **pretrain and
fine-tune models in the ~1M–100M param range** (focus: **5–20M**), assembled to
decide the data composition for a general-language + chat tiny model.

> **Provenance.** Compiled from a fan-out web-research pass (5 angles → 24 sources
> fetched → adversarial 3-vote verification). Confidence markers below:
> - ✅ **verified** — survived 3-0 adversarial verification against a primary source (dataset card / paper / official blog).
> - 🔷 **sourced** — from a primary source (HF card, arXiv, GitHub) but not put through the final verification budget; figures are reliable but treat as "read the card before relying on exact numbers."
>
> **Size caveat that bites budgeting:** HF *repo file size* ≠ *parquet export size* ≠ *token count*. TinyStories shows 7.62 GB repo but ~1 GB parquet. Live HF row/token counts also drift from the numbers in origin papers. Always confirm the split you'll actually load.

---

## TL;DR recommendation for a 5–20M general + chat model

At 5–20M params the binding constraint is **distribution width, not data volume** —
every bit of capacity spent modeling rare/complex tokens is capacity not spent on
fluency. This is why TinyStories works at this scale and "train a 10M model on raw
FineWeb" does not. Budget ~1–3B tokens (overtrained regime).

**Pretrain (fluency backbone + light knowledge):**
| Source | Share | Why |
|---|---|---|
| **TinyStoriesV2** | 55–70% | Narrow, clean, coherent English — the thing that makes a tiny model *readable*. |
| **tiny-strange-textbooks** (or Cosmopedia slice) | 15–25% | World knowledge / expository style. Downweight — higher complexity. |
| **FineWeb-Edu sample** or **finephrase `tutorial` slice** | 10–20% | A little natural/explanatory register. Keep small; broad web hurts coherence here. |

**Chat (separate short SFT phase — don't blend into pretrain):**
- **smol-smoltalk** (`HuggingFaceTB/smol-smoltalk`) — the SmolTalk subset *purpose-trimmed for <1B models*; drops advanced math + long function-calling that overwhelm tiny models. Best off-the-shelf chat SFT for this scale.
- **TinyStories-Instruct** — a simpler, more distributionally-matched SFT analogue if smol-smoltalk proves too hard at 5–20M.

**Explicitly exclude at this scale:** LongPage (book-length, far too complex), full SmolTalk advanced-math, raw C4/SlimPajama/The Pile (too wide).

**If you want to go further:** look at **TinyHelen** (Leaner-Pretrain/Instruct) — a
"simpler language environment" designed precisely for this regime, and **BabyLM**
for a natural (non-synthetic) sample-efficient alternative to TinyStories.

---

## Summary table

| Dataset | HF repo ID | Type | Size | License | Role |
|---|---|---|---|---|---|
| TinyStories | `roneneldan/TinyStories` | synthetic kids' stories | 2.14M rows / 7.62 GB repo (~1 GB parquet) / ~0.5B tok | cdla-sharing-1.0 | fluency pretrain |
| **TinyStoriesV2** | `roneneldan/TinyStories` (V2-GPT4 files) · mirror `noanabeshima/TinyStoriesV2` | synthetic kids' stories, GPT-4 only | 2.75M rows / 2.26 GB repo (~1.1 GB parquet) | cdla-sharing-1.0 | fluency pretrain (**preferred**) |
| TinyStories-Instruct | `roneneldan/TinyStoriesInstruct` | instruction-conditioned stories | ~21.8M train / 218K val (line rows) | cdla-sharing-1.0 | SFT analogue |
| Multilingual TinyStories | `deeponh/multilingual-tinystories` | synthetic stories, 17 Indic langs | ~133K stories / ~94M tok | CC-BY-4.0 | fluency (multilingual) |
| Cosmopedia | `HuggingFaceTB/cosmopedia` | synthetic textbooks/blogs/stories | 31.1M rows / 25B tok / 92.2 GB | apache-2.0 | knowledge injection + fluency |
| SmolLM-Corpus | `HuggingFaceTB/smollm-corpus` | packaged mix (synth + web + code) | Cosmopedia v2 28B + FineWeb-Edu 220B + Python-Edu 4B | apache-2.0 (mixed) | reference general-purpose recipe |
| tiny-strange-textbooks | `nampdn-ai/tiny-strange-textbooks` 🔒 gated | 100% synthetic textbooks | 2.7M docs / 16 GB raw (4.44 GB gz) | apache-2.0 | knowledge injection |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | natural educational web | 1.3T tok default; samples 10BT/100BT/350BT | ODC-BY | natural web pretrain |
| finephrase | `HuggingFaceFW/finephrase` | synthetic web-rephrasings | ~486B tok; configs math/table/tutorial (~338M rows / >1 TB each) | ODC-BY | large-scale synthetic pretrain |
| C4 | `allenai/c4` | cleaned Common Crawl (T5) | en ~305 GB / ~175B tok | ODC-BY | natural web baseline |
| SlimPajama | `cerebras/SlimPajama-627B` | cleaned/deduped RedPajama | 627B tok (82% web, rest wiki/books/code/arxiv/SE) | mixed | balanced natural mix |
| MiniPile | `JeanKaddour/minipile` | diverse Pile subset | 1,010,500 docs / ~6 GB / ~1.5B tok (est.) | MIT | data-efficient research corpus |
| **SmolTalk** | `HuggingFaceTB/smoltalk` | synthetic + curated SFT mix | ~1.1M samples / 4.15 GB / 14 subsets | apache-2.0 (new subsets) | chat/instruction SFT |
| **smol-smoltalk** | `HuggingFaceTB/smol-smoltalk` | trimmed SmolTalk for <1B models | 484,570 rows / 971 MB | apache-2.0 | chat SFT (**tiny-scale**) |
| UltraFeedback (binarized) | `HuggingFaceH4/ultrafeedback_binarized` | preference pairs | — | mixed | DPO alignment |
| LongPage | `Pageshift-Entertainment/LongPage` | Gutenberg books + reasoning traces | 6,067 books, 40K–600K+ tok each / 1.5 GB | PD text + CC-BY-4.0 traces | long-context SFT |
| BabyLM | `babylm/babylm_100M` (+ 10M track) | natural developmental corpus | 100M / 10M words | mixed (source-dependent) | sample-efficient fluency |
| TinyHelen | GitHub `EmpathYang/TinyHelen` | simplified "leaner" corpora | Pretrain 71M tok / Instruct 7M tok | see repo | efficient tiny-LM train+eval |

---

## By role

### 1. Fluency backbone — synthetic children's stories

**TinyStories** — `roneneldan/TinyStories` ✅
Synthetic short stories generated by GPT-3.5 **and** GPT-4, constrained to the
vocabulary of a 3–4-year-old. 2,141,709 rows / 7.62 GB repo / cdla-sharing-1.0.
Backs *TinyStories: How Small Can Language Models Be and Still Speak Coherent
English?* (arXiv:2305.07759) — models **below 10M params**, even a single
transformer block, produce fluent multi-paragraph English. The canonical proof
that a narrow distribution beats scale at this size.

**TinyStoriesV2** — `roneneldan/TinyStories` (TinyStoriesV2-GPT4 files); mirror `noanabeshima/TinyStoriesV2` ✅
Cleaned, **GPT-4-only**, significantly larger successor that drops the
lower-quality GPT-3.5 generations; a strict superset of V1's GPT-4 portion.
2,745,323 rows / 2.26 GB repo / 27.6K val / cdla-sharing-1.0. **The preferred
modern fluency-pretrain corpus** for compact (10M–200M) models.

**TinyStories-Instruct** — `roneneldan/TinyStoriesInstruct` ✅
Instruction-conditioned variant: each story is preceded by a set of instructions
(word lists, required sentences, feature flags — Dialogue/BadEnding/MoralValue/
Foreshadowing/PlotTwist — and summaries). ~21.8M train / 218K val line-rows. The
SFT / instruction-conditioning analogue that stays inside the TinyStories
distribution — good fallback if full chat SFT is too hard at 5–20M.

**Multilingual TinyStories** — `deeponh/multilingual-tinystories` ✅
Extends the paradigm to 17 Indic languages. ~132,942 stories / ~93.9M tokens
(paper) / CC-BY-4.0. Purpose-built for 10M–200M decoder-only LMs. Evidence the
synthetic-story recipe is still a live 2025–26 direction.

### 2. Synthetic textbooks / knowledge injection (the phi lineage)

**phi-1.5** — methodology only, datasets **not public** ✅
*Textbooks Are All You Need II* (arXiv:2309.05463). 1.3B-param model trained on
~30B tokens of synthetic "textbook-quality" text, **deliberately excluding generic
web-crawl**. Establishes the "data quality substitutes for scale" thesis that
motivates every synthetic-textbook corpus below. Note: phi's own data was never
released — only the *idea*, plus the open re-implementations.

**Cosmopedia** — `HuggingFaceTB/cosmopedia` ✅
Largest open phi-style synthetic corpus: textbooks, blogposts, stories, posts,
WikiHow articles from Mixtral-8x7B-Instruct-v0.1. 31,064,744 rows / 25B tok /
92.2 GB / apache-2.0. Prompts ~80% web-seeded (RefinedWeb → 145 clusters / 112
topics / 23M prompts) + ~20% curated (Stanford, Khan Academy, OpenStax, WikiHow).
8 splits by seed source. Used to train Cosmo-1B (~1.8B).

**SmolLM-Corpus** — `HuggingFaceTB/smollm-corpus` ✅
The canonical *packaged* small-model recipe combining all four ingredients:
**Cosmopedia v2** (synthetic, ~28B tok / 39M docs, Mixtral-8x7B, seeded via BISAC
34K+ topics + FineWeb web seeds) + **FineWeb-Edu deduplicated** (natural
educational web, 220B tok) + **Python-Edu** (code filtered from The Stack v2,
4B tok from 40B). This is the reference mix SmolLM (135M/360M/1.7B) was trained on
— the nearest published template to scale *down* from.

**tiny-strange-textbooks** — `nampdn-ai/tiny-strange-textbooks` (🔒 **gated**) 🔷
100% AI-generated synthetic textbooks explicitly aligned with *Textbooks Are All
You Need*. 2.7M docs / 16 GB raw (4.44 GB compressed) / apache-2.0. Part of a
family worth knowing: `tiny-textbooks`, `tiny-code-textbooks`,
`tiny-math-textbooks`, `tiny-webtext`, `tiny-lessons`, `tiny-bridgedict`
(domain-split variants for code/math). **Gated — accept terms on the HF page
before downloading.**

### 3. Natural web text

**FineWeb-Edu** — `HuggingFaceFW/fineweb-edu` ✅
The standard natural (non-synthetic) educational web corpus: Common Crawl
(2013–2025) filtered by an educational-quality classifier (trained on Llama3-70B
annotations). 1.3T tok default / 1.53B docs / ODC-BY. Ships tiny-budget samples:
**10BT** (9.67M rows), **100BT** (97.3M rows), **350BT** (339M rows). Beats
C4/Dolma/SlimPajama/The Pile at equal token budgets.
> ⚠️ Verification note: the claim that FineWeb-Edu is *specifically positioned for
> small models* was **refuted (0-3)** — it's a general pretraining corpus, not
> small-model-specific. Use the sample subsets for tiny budgets, but don't expect a
> tiny model to absorb its full width.

**C4** — `allenai/c4` 🔷
Colossal Clean Crawled Corpus (T5's data). `en` ~305 GB / ~175B tok; also
en.noclean (2.3 TB), realnewslike (15 GB), multilingual mC4. ODC-BY. Classic
nanoGPT-scale baseline and a SlimPajama component.

**SlimPajama** — `cerebras/SlimPajama-627B` 🔷 *(source quality flagged; verify on card)*
627B-token cleaned/deduped RedPajama: ~82% web (67% CommonCrawl + 15% C4), 4.5%
Wikipedia, 4.5% books, 4.5% GitHub, 2.5% ArXiv, 2.0% StackExchange. The
SlimPajama-DC paper (arXiv:2309.10818) studies domain-mixture ratios — a useful
reference even if the full corpus is oversized for 5–20M.

**MiniPile** — `JeanKaddour/minipile` ✅
Diversity-preserving 1,010,500-doc / ~6 GB subset of the deduplicated Pile,
curated via embedding + k-means cluster filtering (English-only, MIT). **Built
precisely for data-efficient / small-budget research** — a compact, domain-diverse
*natural*-text alternative when TinyStories is too narrow but full web corpora are
too big.

### 4. Instruction / chat SFT

**SmolTalk** — `HuggingFaceTB/smoltalk` ✅
The primary modern SFT mixture for small chat models (built SmolLM2-Instruct).
~1.1M samples (~2.2M rows w/ splits) / 4.15 GB / 14 subsets / apache-2.0 (new
subsets; public sets keep their licenses). Core: **Smol-Magpie-Ultra** (431K, via
Magpie from Llama-3.1-405B-Instruct), Smol-Summarize (101K), Smol-Rewrite (56.2K),
Smol-Constraints (36.2K), plus OpenHermes2.5 (100K), MetaMathQA (50K),
NuminaMath-CoT (112K), Self-OSS-Instruct (50.7K), SystemChats (30K), LongAlign
(3.73K), APIGen (87.5K). Spans chat / math / code / long-context / persona.

**smol-smoltalk** — `HuggingFaceTB/smol-smoltalk` ✅ ← **use this for tiny chat**
The SmolTalk subset *purpose-trimmed for <1B models* (trained SmolLM2-135M/360M-
Instruct). 484,570 rows (460K train / 24.2K test) / 971 MB / apache-2.0. Shorter
Magpie-Ultra conversations, **no advanced-math sets** (they overwhelm tiny models),
less rewriting/summarization, no function calling. Sources: smol-magpie-ultra-short,
smol-constraints, self-oss-instruct, openhermes-50k, smollm-rewrite-30k. This is
the closest published answer to "which SFT subset for the smallest models."

**UltraFeedback (binarized)** — `HuggingFaceH4/ultrafeedback_binarized` 🔷
Preference pairs used for the **DPO** alignment stage after SFT in the SmolLM2
recipe. Optional at 5–20M; alignment gains may be marginal at this capacity.

*Reference recipe:* **SmolLM2** paper (arXiv:2502.02737) — the best single source
for tiny-model data composition end-to-end: pretrain (Cosmopedia v0.2 / FineWeb-Edu
/ Stack-Edu / FineMath / DCLM) → SFT (SmolTalk) → DPO (UltraFeedback).

### 5. Long-context

**LongPage** — `Pageshift-Entertainment/LongPage` 🔷
6,067 full Project Gutenberg books (40K–600K+ tokens each) paired with **synthetic
hierarchical reasoning/planning traces** (character archetypes, story arcs, world
rules, scene breakdowns) generated by Qwen3-32B — a three-part
`prompt / thinking / book-text` structure. 1.5 GB. Dual license (public-domain text
+ CC-BY-4.0 scaffolds). Role: **long-context / creative-writing cold-start SFT**,
not fluency pretrain. **Book-length samples are far too heavy for a 5–20M model** —
exclude at this scale.

### 6. Developmental / cognitive / niche (sample-efficient)

**BabyLM corpus** — `babylm/babylm_100M` (+ 10M "strict-small" track) 🔷
Developmentally-inspired *natural* corpus replicating a child's linguistic
exposure: CHILDES child-directed speech, BNC, OpenSubtitles, children's fiction
(Gutenberg, **Children's Book Test** ~14% of original mix), Simple/Standard
Wikipedia. >50% spoken tokens. Two tracks: **10M words** (strict-small) / **100M
words** (strict). Deliberately **excludes synthetic/web-scrape/technical** text;
minimal preprocessing (keeps disfluencies, no lowercasing). THE canonical
sample-efficient pretraining benchmark — a natural-text alternative to TinyStories
if you want cognitive plausibility over synthetic cleanliness.
- *Components also usable standalone:* Children's Book Test (CBT), Simple Wikipedia.

**TinyHelen** — GitHub `EmpathYang/TinyHelen` (arXiv:2501.00522) 🔷
2025 "simpler language environment" for tiny LMs. A refinement pipeline eliminates
noise and minimizes vocabulary while preserving genre patterns (books,
conversation, code), yielding: **Leaner-Pretrain** (71M tok), **Leaner-Instruct**
(7M tok), **Leaner-Glue** (linguistic-proficiency eval), **Leaner-Eval**
(instruction-following eval). Tiny LMs trained on it beat those on original data at
instruction-following with less model size + data. Directly on-target for both the
pretrain and chat halves at 5–20M.

**finephrase** — `HuggingFaceFW/finephrase` 🔷
HuggingFace's ~486B-token synthetic corpus produced by **rephrasing FineWeb-Edu
into structured formats** (tables, math problems, FAQs, tutorials) via
SmolLM2-1.7B-Instruct — winner of a 90-experiment / 1T-token synthetic-data-recipe
study; reportedly outperforms Cosmopedia while cutting generation cost up to 30×,
and found generator models beyond ~1B params give no extra benefit. Configs:
**`math` / `table` / `tutorial`** (~338M rows each, **>1 TB parquet per config on
the Hub**) plus `all`/`faq`. ODC-BY. **Massive — must slice** (e.g. `tutorial`,
first few GB) for tiny-model use; the `tutorial` register is the closest to a
general/chat-adjacent style.

---

## The four datasets originally requested — verdict at 5–20M

| Requested | Verdict | Notes |
|---|---|---|
| `noanabeshima/TinyStoriesV2` | ✅ **Core backbone** | Exactly right width for this scale. |
| `nampdn-ai/tiny-strange-textbooks` | 🔷 **Minority add** | Knowledge/expository; gated; downweight hard. |
| `HuggingFaceFW/finephrase` | ⚠️ **Small slice only** | >1 TB/config — take a bounded `tutorial` slice. |
| `Pageshift-Entertainment/LongPage` | ❌ **Exclude** | Book-length long-context; wrong tool below ~100M. |

---

## Caveats

- **phi-1/phi-1.5 datasets are NOT public** — only the methodology and open
  re-implementations (Cosmopedia, tiny-strange-textbooks) are usable.
- **Repo size ≠ token budget.** TinyStories 7.62 GB repo → ~1 GB parquet; V2 2.26 GB
  → ~1.1 GB parquet. Confirm the split you load.
- **Live counts drift** from paper figures (Multilingual TinyStories, FineWeb-Edu
  extended with 2025 snapshots).
- **Published mixture ratios were validated at 100M–1.8B (SmolLM/Cosmo scale)**, not
  at 5–20M. The exact fluency/knowledge/web/code ratios at *this* scale are an open
  empirical question — expect to sweep them.
- **Licenses vary and matter:** TinyStories/V2 cdla-sharing-1.0; Cosmopedia /
  tiny-strange-textbooks / SmolTalk apache-2.0; Multilingual TinyStories CC-BY-4.0;
  FineWeb-Edu / C4 / finephrase ODC-BY; MiniPile MIT; LongPage dual (PD + CC-BY-4.0).

## Open questions

1. Is SmolTalk-class multi-turn chat data actually *learnable* at 5–20M, or do
   tiny models need simplified/templated chat (TinyStories-Instruct / TinyHelen
   Leaner-Instruct)?
2. What fluency vs. synthetic-textbook vs. natural-web vs. code ratios work best at
   5–20M specifically (vs. the SmolLM-scale recipes)?
3. Is long-context training tractable/beneficial below ~100M at all?

## Key sources

- TinyStories paper — arXiv:2305.07759
- phi-1.5 — arXiv:2309.05463 (Microsoft Research)
- Cosmopedia — HF blog `huggingface.co/blog/cosmopedia` + card
- SmolLM / SmolLM-Corpus — HF blog `huggingface.co/blog/smollm`
- SmolLM2 (data-centric recipe) — arXiv:2502.02737
- FineWeb / FineWeb-Edu — NeurIPS 2024 D&B ("Decanting the Web")
- MiniPile — arXiv:2304.08442
- TinyHelen — arXiv:2501.00522
- SlimPajama-DC — arXiv:2309.10818
- Multilingual TinyStories — arXiv:2603.14563
- Dataset cards: `roneneldan/TinyStories`, `noanabeshima/TinyStoriesV2`,
  `HuggingFaceTB/{cosmopedia,smollm-corpus,smoltalk,smol-smoltalk}`,
  `nampdn-ai/tiny-strange-textbooks`, `HuggingFaceFW/{fineweb-edu,finephrase}`,
  `allenai/c4`, `cerebras/SlimPajama-627B`, `JeanKaddour/minipile`,
  `Pageshift-Entertainment/LongPage`, `babylm/*`
