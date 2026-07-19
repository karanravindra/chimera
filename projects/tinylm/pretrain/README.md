# tinylm / pretrain

Pretrains a ~6M-param GPT (dim 384, 12 heads, 6 layers, ReLU² MLP, RoPE + QK-norm,
tied embeddings, 16k BPE vocab) on TinyStories v2 + Tiny Textbooks, packed at
seq-len 512 with FlexAttention causal+document masking and per-document RoPE
positions, Cut Cross Entropy, and Muon+AdamW.

## Layout

- `model.py` — the GPT (project-local on purpose: per-doc RoPE position reset, no muP —
  diverges from `chimera.models.gpt`; candidate for the library once the unified LLM
  redo picks the canonical GPT)
- `train.py` — raw PyTorch training loop; saves the checkpoint to
  `/mnt/ai/runs/tinylm/pretrain/chimera_gpt6m.pt`
- `main.ipynb` — analysis only: loads the checkpoint, mask visualization, samples,
  zero-shot benchmarks (`chimera.evals`), train-step profile

## Run

```sh
cd projects/tinylm/pretrain
uv run python train.py
```

## Results — data-mix ablation (1k iters, ~65M tokens)

Zero-shot `lm_eval` scores (%), headline metric per task (`acc_norm > acc`).
Best of the three mixes bolded per row; GPT-2 small is a reference, not a rival —
it has 20x the params and ~50x the training tokens.

| task           | metric   | TinyStories v2 | Tiny Textbooks | 50/50 mix | chance | GPT-2 small (124M) |
|----------------|----------|----------------|----------------|-----------|--------|--------------------|
| blimp          | acc      | 62.93          | 63.72          | **65.03** | 50.0   | 82.29              |
| lambada_openai | acc      | 10.87          | 6.95           | **12.59** | 0.0    | 32.16              |
| piqa           | acc_norm | 52.34          | **56.96**      | 56.53     | 50.0   | 62.62              |
| sciq           | acc_norm | 27.40          | **55.10**      | 54.70     | 25.0   | 64.40              |
| arc_easy       | acc_norm | 26.94          | **33.63**      | 31.99     | 25.0   | 39.52              |

Tiny Textbooks dominates everything except LAMBADA (long-range narrative coherence,
where TinyStories' long stories help); the 50/50 mix lands between the two rather
than combining their strengths.
