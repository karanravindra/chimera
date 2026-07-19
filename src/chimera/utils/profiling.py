"""Wall-time + kernel profiling for a single training step."""

import copy
import time

import torch
from torch.profiler import ProfilerActivity, profile


def profile_train_step(
    model,
    batch,
    loss_fn,
    make_optimizer,
    *,
    compile: bool = True,
    n_warmup: int = 8,
    n_iters: int = 20,
    loss_label: str = "",
):
    """Time forward / backward / optimizer for one train step.

    Runs on an isolated deepcopy + fresh optimizer + its own compile, so profiling never
    mutates the trained model. Every phase is CUDA-synced (wall-time) — the profiler's
    Self-CPU is inflated by async dispatch, so trust the per-phase wall times; the op
    table only shows which kernels dominate. n_warmup is high to absorb compilation.

    Args:
        model: the trained model (a torch.compile wrapper is unwrapped before copying).
        batch: one (x, y) batch; moved to the model's device.
        loss_fn: ``loss_fn(pmodel, x, y) -> scalar loss`` on the isolated copy. Tied
            parameters (e.g. an lm_head weight) should be read via
            ``getattr(pmodel, "_orig_mod", pmodel)`` since the copy may be compiled.
        make_optimizer: ``make_optimizer(pmodel_uncompiled) -> optimizer`` for the copy.
        compile: torch.compile the copy (CUDA only), matching the optimized train path.
        loss_label: shown in the header (e.g. "CutCrossEntropy").
    """
    base = getattr(model, "_orig_mod", model)  # unwrap torch.compile if already applied
    p = next(base.parameters())
    device, dtype = p.device, p.dtype
    is_cuda = device.type == "cuda"

    x, y = batch
    x, y = x.to(device), y.to(device)

    pmodel = copy.deepcopy(base).to(device, dtype=dtype).train()
    popt = make_optimizer(pmodel)
    if is_cuda and compile:
        pmodel = torch.compile(pmodel)

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    for _ in range(n_warmup):  # absorb torch.compile tracing + autotune + allocator warmup
        popt.zero_grad(set_to_none=True)
        loss_fn(pmodel, x, y).backward()
        popt.step()
    sync()

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    fwd = bwd = opt = 0.0
    for _ in range(n_iters):
        popt.zero_grad(set_to_none=True)
        sync()
        t0 = time.perf_counter()
        loss = loss_fn(pmodel, x, y)
        sync()
        t1 = time.perf_counter()
        loss.backward()
        sync()
        t2 = time.perf_counter()
        popt.step()
        sync()
        t3 = time.perf_counter()
        fwd += t1 - t0
        bwd += t2 - t1
        opt += t3 - t2

    fwd, bwd, opt = (v / n_iters * 1e3 for v in (fwd, bwd, opt))  # ms/step
    step = fwd + bwd + opt
    tokens = x.numel()

    header = f"batch {tuple(x.shape)}  ({tokens:,} tokens)"
    if loss_label:
        header += f"  |  loss: {loss_label}"
    header += f"  |  {n_iters} iters"
    print(header)
    print("-" * 60)
    print(f"forward    {fwd:8.3f} ms  ({fwd / step:5.1%})")
    print(f"backward   {bwd:8.3f} ms  ({bwd / step:5.1%})")
    print(f"optimizer  {opt:8.3f} ms  ({opt / step:5.1%})")
    print("-" * 60)
    print(f"step total {step:8.3f} ms   ->  {tokens / (step / 1e3):,.0f} tokens/s")
    if is_cuda:
        print(f"peak memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Op-level breakdown: which kernels dominate. Sort by CUDA time (Self-CPU unreliable).
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if is_cuda else [])
    with profile(activities=acts) as prof:
        for _ in range(5):
            popt.zero_grad(set_to_none=True)
            loss_fn(pmodel, x, y).backward()
            popt.step()
        sync()
    sort_key = "self_cuda_time_total" if is_cuda else "self_cpu_time_total"
    print("\n" + prof.key_averages().table(sort_by=sort_key, row_limit=15))
