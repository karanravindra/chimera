"""Throughput benchmark: dense GQA/MLP vs MLA vs MoE vs MLA+MoE.

Measures steady-state training-step throughput (forward + backward +
``Muon.step()``, under the same ``bf16`` autocast ``train.py`` uses) for each
architecture variant at a fixed arch/batch/seq-len, then profiles the
MLA+MoE variant with ``torch.profiler`` to show exactly which ops dominate
wall-clock time -- useful for answering "why is this slower" rather than just
"how much slower".

    uv run python projects/fineweb-edu/gpt/throughput_bench.py

No ``torch.compile`` here (matches the sweep's ``--no-compile``, since MoE's
data-dependent routing makes compile unreliable — see train.py's own warning).
"""

import argparse
import time

import torch

from chimera.models import GPT
from chimera.optim import Muon, muon_param_groups


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-embd", type=int, default=48)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--n-kv-head", type=int, default=1)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--timed-steps", type=int, default=20)
    # MLA dims, scaled to roughly match this arch's GQA head_dim (n_embd // n_head).
    p.add_argument("--kv-lora-rank", type=int, default=32)
    p.add_argument("--qk-nope-head-dim", type=int, default=16)
    p.add_argument("--qk-rope-head-dim", type=int, default=8)
    p.add_argument("--v-head-dim", type=int, default=24)
    # MoE dims.
    p.add_argument("--n-routed-experts", type=int, default=8)
    p.add_argument("--n-activated-experts", type=int, default=2)
    return p.parse_args()


def build_model(args, use_mla: bool, use_moe: bool) -> GPT:
    return GPT(
        vocab_size=65536,
        block_size=args.seq_len,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_layer=args.n_layer,
        use_mla=use_mla,
        kv_lora_rank=args.kv_lora_rank,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        v_head_dim=args.v_head_dim,
        use_moe=use_moe,
        n_routed_experts=args.n_routed_experts,
        n_activated_experts=args.n_activated_experts,
    ).cuda()


def time_variant(model: GPT, args, name: str) -> dict:
    optimizer = Muon(muon_param_groups(model, adamw_name_keywords=("emb", "head", "gate")))
    x = torch.randint(0, 65536, (args.batch_size, args.seq_len), device="cuda")

    def step():
        # return_hidden=True: never materializes full (B,T,vocab) logits, same
        # as train.py's Cut Cross Entropy path -- a naive model(x) full-vocab
        # projection OOMs here (32*2048*65536*4 bytes = ~17GB for fp32 logits).
        # We're benchmarking the transformer body, not the vocab head, so a
        # surrogate loss straight off the hidden state is representative.
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = model(x, return_hidden=True)
            loss = hidden.float().pow(2).mean()
        loss.backward()
        optimizer.step()

    for _ in range(args.warmup_steps):
        step()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(args.timed_steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    it_s = args.timed_steps / elapsed
    tok_s = it_s * args.batch_size * args.seq_len
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"{name:16s}  {n_params / 1e6:6.2f}M params  {it_s:6.2f} it/s  "
        f"{tok_s / 1e3:8.1f}k tok/s  {1000 / it_s:7.1f} ms/step  {peak_mb:7.0f} MB peak"
    )
    return {"it_s": it_s, "tok_s": tok_s, "ms_step": 1000 / it_s, "peak_mb": peak_mb}


def profile_variant(model: GPT, args, name: str, steps: int = 3):
    """torch.profiler pass to see exactly which ops dominate wall-clock time."""
    optimizer = Muon(muon_param_groups(model, adamw_name_keywords=("emb", "head", "gate")))
    x = torch.randint(0, 65536, (args.batch_size, args.seq_len), device="cuda")

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = model(x, return_hidden=True)
            loss = hidden.float().pow(2).mean()
        loss.backward()
        optimizer.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(steps):
            step()
            prof.step()
    torch.cuda.synchronize()

    print(f"\n--- torch.profiler top ops for {name} (sorted by CUDA time) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "this benchmark needs a CUDA device"
    torch.manual_seed(0)

    print(
        f"arch: n_embd={args.n_embd} n_head={args.n_head} n_kv_head={args.n_kv_head} "
        f"n_layer={args.n_layer}  seq_len={args.seq_len} batch_size={args.batch_size}\n"
    )

    variants = [
        ("dense", False, False),
        ("mla", True, False),
        ("moe", False, True),
        ("mla+moe", True, True),
    ]
    results = {}
    for name, use_mla, use_moe in variants:
        torch.manual_seed(0)
        model = build_model(args, use_mla, use_moe)
        results[name] = time_variant(model, args, name)
        del model
        torch.cuda.empty_cache()

    dense_ms = results["dense"]["ms_step"]
    print("\nslowdown vs dense (ms/step ratio):")
    for name, r in results.items():
        print(f"  {name:16s} {r['ms_step'] / dense_ms:5.2f}x")

    # Profile the two variants that matter for the "why slower" question: MoE
    # alone (isolates MoE's own overhead) and MLA+MoE (the actual sweep config).
    torch.manual_seed(0)
    moe_model = build_model(args, use_mla=False, use_moe=True)
    profile_variant(moe_model, args, "moe")
    del moe_model
    torch.cuda.empty_cache()

    torch.manual_seed(0)
    combo_model = build_model(args, use_mla=True, use_moe=True)
    profile_variant(combo_model, args, "mla+moe")


if __name__ == "__main__":
    main()
