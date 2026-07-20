"""Benchmark sampled-softmax CE against CCE and full CE on the real pretrain step.

Runs the actual tinylm 6M GPT (same config as train.py: dim 384, 6 layers,
vocab 16k, seq 512, microbatch 64) through fwd+bwd with each loss and reports
step time, tokens/sec, peak memory, and the loss value at init (to show the
sampled estimator's bias). Model compiled max-autotune-no-cudagraphs like the
real run; losses stay outside the compiled region, matching train.py.

    uv run python bench_sampled_ce.py
"""

import os
import time

os.environ.setdefault("CCE_AUTOTUNE", "1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/mnt/ai/data/torchinductor_cache")

import torch
import torch.nn.functional as F
from cut_cross_entropy import linear_cross_entropy
from model import GPT
from sampled_ce import _sampled_ce_from_negatives, sampled_cross_entropy

from chimera.models.attention import build_block_mask_and_pos

DEVICE = "cuda"
DTYPE = torch.bfloat16

VOCAB_SIZE = 16_384
SEQ_LEN = 512
DIM = 384
N_HEADS = 12
MLP_MULT = 3
N_LAYERS = 6
LOGIT_SOFTCAP = 30.0
EOS_ID = 0

MICRO_BATCH = 64  # train.py: global 128 / grad-accum 2
WARMUP_STEPS = 10
TIMED_STEPS = 30
SAMPLE_COUNTS = [1024, 4096, 8192]


def make_batch(generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(
        1, VOCAB_SIZE, (MICRO_BATCH, SEQ_LEN), device=DEVICE, generator=generator
    )
    # Sprinkle EOS at the training data's typical doc granularity so the
    # block mask has realistic document structure.
    doc_break = torch.rand(x.shape, device=DEVICE, generator=generator) < (1 / 192)
    x[doc_break] = EOS_ID
    y = torch.roll(x, -1, dims=1)
    return x, y


def full_ce_loss(model, x, y):
    block_mask, pos_ids = build_block_mask_and_pos(x, EOS_ID)
    logits = model(x, block_mask=block_mask, pos_ids=pos_ids)
    return F.cross_entropy(logits.view(-1, VOCAB_SIZE).float(), y.view(-1))


def cce_loss(model, x, y):
    block_mask, pos_ids = build_block_mask_and_pos(x, EOS_ID)
    hidden = model(x, return_hidden=True, block_mask=block_mask, pos_ids=pos_ids)
    weight = getattr(model, "_orig_mod", model).token_emb.weight
    return linear_cross_entropy(hidden, weight, y, softcap=LOGIT_SOFTCAP)


def make_sampled_loss(num_samples: int, compile_loss: bool = False):
    core = _sampled_ce_from_negatives
    if compile_loss:
        core = torch.compile(
            _sampled_ce_from_negatives, mode="max-autotune-no-cudagraphs"
        )

    def loss_fn(model, x, y):
        block_mask, pos_ids = build_block_mask_and_pos(x, EOS_ID)
        hidden = model(x, return_hidden=True, block_mask=block_mask, pos_ids=pos_ids)
        weight = getattr(model, "_orig_mod", model).token_emb.weight
        return sampled_cross_entropy(
            hidden,
            weight,
            y,
            num_samples=num_samples,
            softcap=LOGIT_SOFTCAP,
            core=core,
        )

    return loss_fn


def bench(name: str, loss_fn, model, batches) -> dict:
    for x, y in batches[:WARMUP_STEPS]:
        loss = loss_fn(model, x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    last_loss = None
    start = time.perf_counter()
    for x, y in batches[WARMUP_STEPS : WARMUP_STEPS + TIMED_STEPS]:
        loss = loss_fn(model, x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
        last_loss = loss.detach()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    step_ms = elapsed / TIMED_STEPS * 1000
    tok_s = MICRO_BATCH * SEQ_LEN * TIMED_STEPS / elapsed
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    result = {
        "name": name,
        "step_ms": step_ms,
        "tok_s": tok_s,
        "peak_gb": peak_gb,
        "loss": last_loss.item(),
    }
    print(
        f"{name:>16}: {step_ms:8.2f} ms/step  {tok_s / 1e3:8.1f}k tok/s  "
        f"peak {peak_gb:5.2f} GB  loss {result['loss']:.4f}"
    )
    return result


def main():
    torch.manual_seed(0)
    model = GPT(
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        dim=DIM,
        n_heads=N_HEADS,
        mlp_mult=MLP_MULT,
        n_layers=N_LAYERS,
        eos_id=EOS_ID,
        logit_softcap=LOGIT_SOFTCAP,
    ).to(DEVICE, DTYPE)
    model = torch.compile(model, mode="max-autotune-no-cudagraphs")

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(0)
    batches = [make_batch(generator) for _ in range(WARMUP_STEPS + TIMED_STEPS)]

    print(
        f"tinylm 6M step benchmark: microbatch {MICRO_BATCH}x{SEQ_LEN}, "
        f"vocab {VOCAB_SIZE}, {WARMUP_STEPS} warmup / {TIMED_STEPS} timed steps\n"
    )

    results = [
        bench("full CE", full_ce_loss, model, batches),
        bench("CCE (baseline)", cce_loss, model, batches),
    ]
    for k in SAMPLE_COUNTS:
        results.append(bench(f"sampled k={k}", make_sampled_loss(k), model, batches))
    for k in SAMPLE_COUNTS:
        results.append(
            bench(
                f"sampled+tc k={k}",
                make_sampled_loss(k, compile_loss=True),
                model,
                batches,
            )
        )

    baseline = next(r for r in results if r["name"] == "CCE (baseline)")
    print("\nrelative to CCE:")
    for r in results:
        speedup = baseline["step_ms"] / r["step_ms"]
        print(f"{r['name']:>16}: {speedup:5.2f}x speed, {r['peak_gb']:.2f} GB")


if __name__ == "__main__":
    main()
