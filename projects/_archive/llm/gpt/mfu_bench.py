"""MFU benchmark for the GPT training step, mirroring train.py exactly.

Measures steady-state training-step wall time for the *real* optimizer step —
bf16 weights (Lightning ``bf16-true``), Cut Cross Entropy loss (with the muP
output multiplier), grad-clip 1.0, and ``Muon.step()`` — at the train.py
default arch/batch/seq, then converts to Model FLOPs Utilization:

    MFU = (analytic model FLOPs per step) / (step time * hardware peak FLOPS)

FLOPs use the standard PaLM convention: ``6 * N_matmul`` per token (fwd+bwd
for every weight matmul, embedding lookup excluded, tied lm_head counted
once) plus ``12 * n_layer * n_head * head_dim * T`` for attention (no causal
discount -- comparable with nanoGPT/PaLM numbers).

Two MFU denominators are reported: the device's *measured* large-GEMM bf16
ceiling (what a perfect implementation could actually reach here) and the
spec-sheet peak passed via ``--peak-tflops``.

Compares eager vs torch.compile (default) vs torch.compile(reduce-overhead)
(the train.py default), plus a fwd/loss/bwd/clip/opt breakdown in eager mode
and an optional ``--profile`` top-ops table.

    uv run python projects/llm/gpt/mfu_bench.py

Synthetic data: this isolates the model+optimizer step. Real-run MFU is
additionally reduced by dataloading, validation passes, and Lightning
logging overhead, which are not measured here.
"""

import argparse
import os
import time

# CCE ships hardcoded Triton block configs tuned on Apple's dev hardware; on
# this repo's shapes/GPU letting Triton autotune is ~15% faster on both CCE
# kernels (~8% whole-step). Must be set before cut_cross_entropy is imported.
os.environ.setdefault("CCE_AUTOTUNE", "1")

import torch
from cut_cross_entropy import linear_cross_entropy

from chimera.models import GPT
from chimera.optim import Muon, muon_param_groups


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # train.py defaults
    p.add_argument("--vocab-size", type=int, default=65536)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--n-kv-head", type=int, default=3)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--global-token-count", type=int, default=65536)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--timed-steps", type=int, default=30)
    p.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="spec-sheet bf16 dense TFLOPS; defaults to the measured GEMM ceiling",
    )
    p.add_argument(
        "--modes",
        default="flex-compile,compile,fp8-compile",
        help="comma-separated; each mode is dash-joined tokens: "
        "eager|compile|compile-ro (execution), flex (flex_attention "
        "instead of the default SDPA/flash causal path), fp8 "
        "(torchao float8 on the body Linears), sparse (sliding-"
        "window + global-prefix attention via flex block masks; "
        "see --attn-window), mla, moe, attnres (architecture "
        "variants at train.py-default dims). "
        "e.g. 'eager', 'sparse-fp8-compile', 'mla-moe-compile'",
    )
    p.add_argument("--attn-window", type=int, default=256)
    p.add_argument("--attn-global-tokens", type=int, default=16)
    # MLA dims (train.py defaults)
    p.add_argument("--kv-lora-rank", type=int, default=128)
    p.add_argument("--qk-nope-head-dim", type=int, default=64)
    p.add_argument("--qk-rope-head-dim", type=int, default=32)
    p.add_argument("--v-head-dim", type=int, default=64)
    # MoE dims (train.py defaults)
    p.add_argument("--n-routed-experts", type=int, default=8)
    p.add_argument("--n-shared-experts", type=int, default=1)
    p.add_argument("--n-activated-experts", type=int, default=2)
    p.add_argument("--moe-inter-dim", type=int, default=None)
    # AttnRes: full attention-over-depth (n_blocks == n_layer) by default,
    # since train.py's default 8 doesn't divide the default n_layer 6.
    p.add_argument("--attn-res-n-blocks", type=int, default=None)
    p.add_argument(
        "--profile",
        action="store_true",
        help="after timing each mode, profile 3 steps of it and print "
        "the top kernels by self CUDA time",
    )
    p.add_argument(
        "--trace-dir",
        default=None,
        help="with --profile: also export a chrome trace per mode "
        "(<trace-dir>/<mode>.json, viewable in ui.perfetto.dev)",
    )
    return p.parse_args()


def model_flops_per_token(args, mla: bool = False, moe: bool = False) -> float:
    """PaLM-convention train FLOPs per token: 6*N_matmul + attention.

    Attention (no causal discount, comparable with nanoGPT/PaLM) generalized
    to asymmetric QK/V head dims: 6*L*H*T*(qk_dim + v_dim), which reduces to
    the standard 12*L*H*hd*T for GQA. MoE counts only *activated* expert
    FLOPs (top-k routed + shared), matching what actually executes per token.
    AttnRes adds only vector einsums (no weight matmuls) -- same formula.
    """
    d, H, L = args.n_embd, args.n_head, args.n_layer
    hd = d // H

    if mla:
        qk_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        v_dim = args.v_head_dim
        attn_proj = (
            d * H * qk_dim  # q_proj (q_lora_rank=0)
            + d * (args.kv_lora_rank + args.qk_rope_head_dim)  # w_dkv
            + args.kv_lora_rank * H * args.qk_nope_head_dim  # w_uk
            + args.kv_lora_rank * H * v_dim  # w_uv
            + H * v_dim * d  # out proj
        )
    else:
        qk_dim = v_dim = hd
        attn_proj = (
            d * H * hd  # q_proj
            + d * args.n_kv_head * hd * 2  # kv_proj
            + H * hd * d  # out proj
        )

    if moe:
        inter = args.moe_inter_dim if args.moe_inter_dim else d // 2
        mlp = (
            3 * d * inter * args.n_shared_experts  # shared expert (SwiGLU)
            + 3 * d * inter * args.n_activated_experts  # top-k routed experts
            + d * args.n_routed_experts  # gate
        )
    else:
        mlp = d * 4 * d * 2  # dense fc1 + fc2

    n_matmul = L * (attn_proj + mlp) + d * args.vocab_size  # + tied lm_head
    attn = 6 * L * H * args.seq_len * (qk_dim + v_dim)
    return 6 * n_matmul + attn


def measure_gemm_ceiling(n: int = 8192, iters: int = 20) -> float:
    """Achievable bf16 dense-GEMM TFLOPS on this device (practical peak)."""
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return 2 * n**3 * iters / (time.perf_counter() - t0) / 1e12


def build(
    args,
    fp8: bool = False,
    flash: bool = True,
    sparse: bool = False,
    mla: bool = False,
    moe: bool = False,
    attnres: bool = False,
):
    model = (
        GPT(
            vocab_size=args.vocab_size,
            block_size=args.seq_len,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_kv_head=args.n_kv_head,
            n_layer=args.n_layer,
            use_flash_attn=flash,
            attn_window=args.attn_window if sparse else None,
            attn_global_tokens=args.attn_global_tokens,
            use_mla=mla,
            kv_lora_rank=args.kv_lora_rank,
            qk_nope_head_dim=args.qk_nope_head_dim,
            qk_rope_head_dim=args.qk_rope_head_dim,
            v_head_dim=args.v_head_dim,
            use_moe=moe,
            n_routed_experts=args.n_routed_experts,
            n_shared_experts=args.n_shared_experts,
            n_activated_experts=args.n_activated_experts,
            moe_inter_dim=args.moe_inter_dim,
            use_attn_res=attnres,
            attn_res_n_blocks=args.attn_res_n_blocks or args.n_layer,
        )
        .cuda()
        .to(torch.bfloat16)
    )  # bf16-true: bf16 weights, no autocast
    if fp8:
        # torchao float8 training (dynamic tensorwise scaling): swaps the
        # transformer-body Linears for Float8Linear, which casts weight and
        # activation to float8_e4m3fn around each matmul (fwd + both bwd
        # GEMMs). The tied lm_head is deliberately excluded -- CCE needs the
        # bf16 weight, and a prior fp8-lm_head experiment here was 7-12x
        # slower than CCE. Master weights stay bf16, so Muon is unaffected.
        from torchao.float8 import convert_to_float8_training

        convert_to_float8_training(
            model, module_filter_fn=lambda mod, fqn: fqn.startswith("blocks.")
        )
    optimizer = Muon(
        muon_param_groups(model, adamw_name_keywords=("emb", "head", "gate"))
    )
    return model, optimizer


def make_step(model, optimizer, args):
    raw = getattr(model, "_orig_mod", model)
    B = args.global_token_count // args.seq_len
    x = torch.randint(0, args.vocab_size, (B, args.seq_len), device="cuda")
    y = torch.randint(0, args.vocab_size, (B, args.seq_len), device="cuda")

    def step():
        optimizer.zero_grad(set_to_none=True)
        hidden = model(x, return_hidden=True)
        hidden = hidden * raw.output_mult
        loss = linear_cross_entropy(
            hidden.reshape(-1, hidden.size(-1)), raw.lm_head_weight, y.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        optimizer.step()
        model.update_moe_bias()

    return step


def time_mode(args, mode: str, gemm_tflops: float, spec_tflops: float):
    tokens = set(mode.split("-"))
    mla, moe = "mla" in tokens, "moe" in tokens
    flops_per_step = model_flops_per_token(args, mla, moe) * args.global_token_count
    torch.manual_seed(0)
    model, optimizer = build(
        args,
        fp8="fp8" in tokens,
        flash="flex" not in tokens,
        sparse="sparse" in tokens,
        mla=mla,
        moe=moe,
        attnres="attnres" in tokens,
    )
    n_params = sum(p.numel() for p in model.parameters())
    if {"compile", "ro"} <= tokens:
        model = torch.compile(model, mode="reduce-overhead")
    elif "compile" in tokens:
        model = torch.compile(model)

    step = make_step(model, optimizer, args)
    for _ in range(args.warmup_steps):
        step()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(args.timed_steps):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / args.timed_steps * 1000

    tok_s = args.global_token_count / (ms / 1000)
    tflops = flops_per_step / (ms / 1000) / 1e12
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"{mode:20s} {n_params / 1e6:6.1f}M {ms:8.1f} ms/step  {tok_s / 1e3:7.1f}k tok/s  "
        f"{tflops:6.2f} TFLOPS  MFU {tflops / gemm_tflops * 100:5.1f}% (gemm) "
        f"{tflops / spec_tflops * 100:5.1f}% (spec)  {peak_gb:5.2f} GB peak"
    )
    if args.profile:
        profile_kernels(args, mode, step, ms)
    del model, optimizer
    torch._dynamo.reset()
    torch.cuda.empty_cache()


def component_breakdown(args):
    """CUDA-event timing of each step phase (eager, so phases are separable)."""
    torch.manual_seed(0)
    model, optimizer = build(args)
    B = args.global_token_count // args.seq_len
    x = torch.randint(0, args.vocab_size, (B, args.seq_len), device="cuda")
    y = torch.randint(0, args.vocab_size, (B, args.seq_len), device="cuda")

    names = ["forward", "loss", "backward", "clip", "opt"]
    acc = dict.fromkeys(names, 0.0)
    iters = 10

    def run(record=False):
        evs = [torch.cuda.Event(enable_timing=True) for _ in range(len(names) + 1)]
        optimizer.zero_grad(set_to_none=True)
        evs[0].record()
        hidden = model(x, return_hidden=True) * model.output_mult
        evs[1].record()
        loss = linear_cross_entropy(
            hidden.reshape(-1, hidden.size(-1)), model.lm_head_weight, y.reshape(-1)
        )
        evs[2].record()
        loss.backward()
        evs[3].record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        evs[4].record()
        optimizer.step()
        evs[5].record()
        if record:
            torch.cuda.synchronize()
            for i, n in enumerate(names):
                acc[n] += evs[i].elapsed_time(evs[i + 1])

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    for _ in range(iters):
        run(record=True)

    total = sum(acc.values())
    print(f"\nphase breakdown (eager, CUDA events, {iters}-step avg):")
    for n in names:
        print(f"  {n:9s} {acc[n] / iters:8.1f} ms  {acc[n] / total * 100:5.1f}%")
    print(f"  {'total':9s} {total / iters:8.1f} ms")
    del model, optimizer
    torch.cuda.empty_cache()


def profile_kernels(args, mode: str, step, ms_per_step: float, steps: int = 3):
    """Profile ``steps`` steps of an already-warm mode and print the kernels
    that dominate GPU time (self CUDA time -- actual device execution, not
    op-tree attribution). Optionally exports a chrome trace for perfetto."""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        for _ in range(steps):
            step()
        torch.cuda.synchronize()

    kernels = [
        e
        for e in prof.key_averages()
        if e.device_type == torch.autograd.DeviceType.CUDA
        and e.self_device_time_total > 0
    ]
    kernels.sort(key=lambda e: -e.self_device_time_total)
    total_us = sum(e.self_device_time_total for e in kernels)
    print(
        f"\n--- {mode}: top kernels by self CUDA time "
        f"({total_us / steps / 1000:.1f} ms/step GPU-busy of {ms_per_step:.1f} ms wall) ---"
    )
    print(f"{'ms/step':>8}  {'%':>5}  {'x/step':>6}  kernel")
    for e in kernels[:15]:
        print(
            f"{e.self_device_time_total / steps / 1000:8.2f}  "
            f"{e.self_device_time_total / total_us * 100:5.1f}  "
            f"{e.count // steps:6d}  {e.key[:90]}"
        )

    if args.trace_dir:
        import os

        os.makedirs(args.trace_dir, exist_ok=True)
        path = os.path.join(args.trace_dir, f"{mode}.json")
        prof.export_chrome_trace(path)
        print(f"trace: {path}")


def main():
    args = parse_args()
    assert torch.cuda.is_available()

    B = args.global_token_count // args.seq_len
    gemm_tflops = measure_gemm_ceiling()
    spec_tflops = args.peak_tflops or gemm_tflops

    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(
        f"measured bf16 GEMM ceiling: {gemm_tflops:.1f} TFLOPS"
        + (f"  (spec: {spec_tflops:.1f})" if args.peak_tflops else "")
    )
    print(
        f"arch: {args.n_embd}-{args.n_head}-{args.n_kv_head}-{args.n_layer} "
        f"vocab={args.vocab_size}  "
        f"seq={args.seq_len} batch={B} ({args.global_token_count} tok/step)"
    )
    print(
        f"dense model FLOPs: {model_flops_per_token(args) / 1e6:.1f} MFLOPs/token "
        f"(variant modes use variant-aware FLOPs)\n"
    )

    for mode in args.modes.split(","):
        time_mode(args, mode.strip(), gemm_tflops, spec_tflops)

    component_breakdown(args)


if __name__ == "__main__":
    main()
