"""Coordinate check: verify the GPT muP implementation is correct.

The muP promise is that per-layer activation magnitudes stay (roughly) invariant
to model width. This script trains the GPT at several widths for a handful of
optimizer steps (Muon on hidden matrices + AdamW on embedding/head/norms, exactly
as in ``train.py``) and records the mean-absolute activation ("coordinate size")
at each layer type after every step.

    uv run python projects/fineweb-edu/gpt/coord_check.py

Under correct muP the curves for different widths cluster together (magnitude does
NOT grow with width). To see the standard-parameterization (SP) failure mode for
contrast, force ``m_d == 1`` at every width by matching the base width to it:

    uv run python projects/fineweb-edu/gpt/coord_check.py --sp

which sets ``mup_base_width = n_embd`` per model — then the coordinates fan out
with width, the classic SP divergence.

Widths use a FIXED head_dim (``--head-dim``, default 64) so the attention-logit
scale is itself width-invariant; ``n_head = n_embd // head_dim`` grows instead.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from chimera.models import GPT
from chimera.optim import Muon, muon_param_groups


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="n_embd values to compare (head_dim fixed, so n_head scales).",
    )
    p.add_argument("--base-width", type=int, default=256, help="muP base width m_d=1.")
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--kv-ratio", type=int, default=4, help="n_head // n_kv_head.")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--adamw-lr", type=float, default=8e-4)
    p.add_argument("--mup-input-mult", type=float, default=1.0)
    p.add_argument("--mup-output-mult", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sp",
        action="store_true",
        help="Standard parameterization: force m_d=1 at every width (SP failure).",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out",
        default=str(Path(__file__).parent / "coord_check.png"),
        help="Output PNG path.",
    )
    return p.parse_args()


# Layer types whose coordinate size we track; per-block ones are averaged over depth.
LAYER_TYPES = ["emb", "attn_out", "mlp_out", "block_out", "logits"]


def register_hooks(model, record):
    """Hook the model so each forward writes mean-abs activations into ``record``.

    ``record`` is a dict {layer_type: [values across blocks]} reset per forward.
    """
    handles = []

    def mean_abs(t):
        return t.detach().float().abs().mean().item()

    # Embedding coordinate as it enters the residual stream (includes input mult).
    def emb_hook(_m, _inp, out):
        record["emb"].append(mean_abs(out) * model.mup_input_mult)

    handles.append(model.tok_emb.register_forward_hook(emb_hook))

    for block in model.blocks:
        # GroupedQueryAttention returns (out, present); MLP returns a tensor.
        handles.append(
            block.attn.register_forward_hook(
                lambda _m, _i, out: record["attn_out"].append(mean_abs(out[0]))
            )
        )
        handles.append(
            block.mlp.register_forward_hook(
                lambda _m, _i, out: record["mlp_out"].append(mean_abs(out))
            )
        )
        handles.append(
            block.register_forward_hook(
                lambda _m, _i, out: record["block_out"].append(mean_abs(out[0]))
            )
        )
    return handles


def run_width(args, n_embd):
    torch.manual_seed(args.seed)
    n_head = n_embd // args.head_dim
    assert n_head >= 1, f"n_embd={n_embd} < head_dim={args.head_dim}"
    n_kv_head = max(1, n_head // args.kv_ratio)
    base_width = n_embd if args.sp else args.base_width

    model = GPT(
        vocab_size=args.vocab,
        block_size=args.seq_len,
        n_embd=n_embd,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_layer=args.n_layer,
        tie_embedding=True,
        mup_base_width=base_width,
        mup_input_mult=args.mup_input_mult,
        mup_output_mult=args.mup_output_mult,
    ).to(args.device)

    optimizer = Muon(
        muon_param_groups(model, muon_lr=args.muon_lr, adamw_lr=args.adamw_lr)
    )

    record = defaultdict(list)
    handles = register_hooks(model, record)

    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    # Per-step coordinate size for each layer type (averaged over blocks).
    history = {lt: [] for lt in LAYER_TYPES}
    model.train()
    for _ in range(args.steps):
        for lt in LAYER_TYPES:
            record[lt].clear()
        x = torch.randint(
            0, args.vocab, (args.batch, args.seq_len), device=args.device, generator=gen
        )
        y = torch.randint(
            0, args.vocab, (args.batch, args.seq_len), device=args.device, generator=gen
        )
        logits = model(x)
        record["logits"].append(logits.detach().float().abs().mean().item())
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        for lt in LAYER_TYPES:
            vals = record[lt]
            history[lt].append(sum(vals) / len(vals) if vals else float("nan"))

    for h in handles:
        h.remove()
    return history


def main():
    args = parse_args()
    mode = "SP (m_d=1 forced)" if args.sp else f"muP (base_width={args.base_width})"
    print(f"coordinate check — {mode}, head_dim={args.head_dim}, device={args.device}")

    results = {}  # width -> {layer_type -> [per-step magnitude]}
    for w in args.widths:
        results[w] = run_width(args, w)
        final = {lt: results[w][lt][-1] for lt in LAYER_TYPES}
        print(f"  n_embd={w:5d}  " + "  ".join(f"{lt}={final[lt]:.4f}" for lt in LAYER_TYPES))

    # Final-step table: rows=layer type, cols=width. Flat across widths => muP ok.
    print("\nfinal-step coordinate size (row=layer, col=width):")
    header = "layer".ljust(12) + "".join(f"{w:>12d}" for w in args.widths)
    print(header)
    for lt in LAYER_TYPES:
        row = lt.ljust(12) + "".join(f"{results[w][lt][-1]:12.4f}" for w in args.widths)
        print(row)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available — skipping plot)")
        return

    fig, axes = plt.subplots(1, len(LAYER_TYPES), figsize=(4 * len(LAYER_TYPES), 4))
    steps = range(1, args.steps + 1)
    for ax, lt in zip(axes, LAYER_TYPES):
        for w in args.widths:
            ax.plot(steps, results[w][lt], marker="o", ms=3, label=f"n_embd={w}")
        ax.set_title(lt)
        ax.set_xlabel("step")
        ax.set_ylabel("mean |activation|")
        ax.set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"GPT coordinate check — {mode}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nsaved plot -> {args.out}")


if __name__ == "__main__":
    main()
