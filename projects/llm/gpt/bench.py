"""Zero-shot benchmark evaluation of a GPT model via lm-evaluation-harness.

Wraps a raw ``chimera.models.GPT`` in a minimal ``lm_eval.api.model.TemplateLM``
adapter and runs the standard loglikelihood-ranking benchmarks (HellaSwag, PIQA,
LAMBADA, ARC-Easy/Challenge, SciQ, WinoGrande by default). ``run_benchmarks`` is
imported directly by ``train.py`` to run the same suite as the test phase and log
it to wandb; this file also works standalone against a saved checkpoint:

    uv run python projects/fineweb-edu/gpt/bench.py

Named ``bench.py`` (not ``evaluate.py``) to avoid shadowing the ``evaluate`` PyPI
package pulled in transitively by ``lm-eval``.

Results are saved as JSON under ``--out-dir`` and printed as a markdown table.
"""

import argparse
import json
import os
from pathlib import Path

# lm_eval.caching.cache reads this env var at import time (not lazily), so it
# must be set before the first `import lm_eval*` anywhere in the process.
os.environ.setdefault("LM_HARNESS_CACHE_PATH", "/mnt/ai/data/lm_eval_cache")

import torch
import torch.nn.functional as F
from lm_eval.api.model import TemplateLM

from chimera.models import GPT
from chimera.tokenizers import BPETokenizer

DEFAULT_TASKS = ["hellaswag", "piqa", "lambada_openai", "arc_easy", "arc_challenge", "sciq", "winogrande"]

# Random-chance baselines (percent) for the accuracy-style metrics of each task,
# so a raw number can be read against "better than guessing" at a glance.
CHANCE = {
    "hellaswag": 25.0,
    "piqa": 50.0,
    "lambada_openai": 0.0,
    "arc_easy": 25.0,
    "arc_challenge": 25.0,
    "sciq": 25.0,
    "winogrande": 50.0,
}
PPL_METRICS = {"perplexity", "word_perplexity", "byte_perplexity", "bits_per_byte"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="/mnt/ai/runs/fineweb-edu/gpt/checkpoints/gpt.ckpt")
    p.add_argument("--out-dir", default="/mnt/ai/runs/fineweb-edu/gpt/eval")
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    p.add_argument("--tokenizer-id", default="LiquidAI/LFM2.5-230M")
    # Model dims match train.py's defaults; the checkpoint has no saved
    # hyper_parameters, so they must be supplied explicitly here.
    p.add_argument("--block-size", type=int, default=2048)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--n-kv-head", type=int, default=3)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--use-mla", action="store_true")
    p.add_argument("--kv-lora-rank", type=int, default=128)
    p.add_argument("--q-lora-rank", type=int, default=0)
    p.add_argument("--qk-nope-head-dim", type=int, default=64)
    p.add_argument("--qk-rope-head-dim", type=int, default=32)
    p.add_argument("--v-head-dim", type=int, default=64)
    p.add_argument("--use-moe", action="store_true")
    p.add_argument("--n-routed-experts", type=int, default=8)
    p.add_argument("--n-shared-experts", type=int, default=1)
    p.add_argument("--n-activated-experts", type=int, default=2)
    p.add_argument("--moe-inter-dim", type=int, default=None)
    p.add_argument("--route-scale", type=float, default=1.0)
    p.add_argument("--bias-update-speed", type=float, default=0.001)
    p.add_argument("--batch-tokens", type=int, default=32768)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(args, vocab_size: int) -> GPT:
    model = GPT(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_layer=args.n_layer,
        tie_embedding=True,
        use_mla=args.use_mla,
        kv_lora_rank=args.kv_lora_rank,
        q_lora_rank=args.q_lora_rank,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        v_head_dim=args.v_head_dim,
        use_moe=args.use_moe,
        n_routed_experts=args.n_routed_experts,
        n_shared_experts=args.n_shared_experts,
        n_activated_experts=args.n_activated_experts,
        moe_inter_dim=args.moe_inter_dim,
        route_scale=args.route_scale,
        bias_update_speed=args.bias_update_speed,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # Lightning + torch.compile prefixes every key with "model._orig_mod.".
    prefix = "model._orig_mod."
    state_dict = {
        k[len(prefix) :]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)
    }
    model.load_state_dict(state_dict)
    model.to(args.device).eval()
    return model


class ChimeraLM(TemplateLM):
    """Minimal lm-eval adapter around ``chimera.models.GPT`` for loglikelihood tasks."""

    def __init__(self, model: GPT, tokenizer: BPETokenizer, block_size: int, batch_tokens: int, device: str):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.backend = "causal"
        self.block_size = block_size
        self.batch_tokens = batch_tokens
        self._device = device
        eot_id = tokenizer._tok.token_to_id("<|endoftext|>")
        assert eot_id is not None, "tokenizer has no <|endoftext|> token"
        self._eot_id = eot_id

    @property
    def eot_token_id(self):
        return self._eot_id

    def tok_encode(self, string, add_special_tokens: bool = False, **kwargs):
        # add_special_tokens=False matches training-time tokenization
        # (mixture.py / tokenize_source.py): the LFM2.5 post-processor otherwise
        # auto-prepends <|startoftext|>, a token the model never saw in pretrain.
        return self.tokenizer._tok.encode(
            string, add_special_tokens=add_special_tokens
        ).ids

    def loglikelihood(self, requests, disable_tqdm=False):
        # Override the base loop (which tok_encodes each pair one at a time, ~18s
        # for the full suite) with a batched + disk-cached encode. The tokenized
        # (context_enc, continuation_enc) depend only on the tokenizer + the
        # request text, NOT the model weights, so the cache is reused across every
        # run with the same tokenizer/tasks (e.g. the smoke run and the baseline).
        new_reqs = self._encode_pairs_cached([req.args for req in requests])
        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def _encode_pairs_cached(self, pairs):
        import hashlib
        import pickle
        from pathlib import Path

        version = "v1"  # bump if the encoding logic below changes
        # fingerprint the tokenizer so a different tokenizer can't hit a stale key.
        fp = tuple(self.tokenizer._tok.encode(
            "The quick brown fox", add_special_tokens=False).ids)
        h = hashlib.md5(f"{version}|{fp}|{len(pairs)}".encode())
        for ctx, cont in pairs:
            h.update(ctx.encode("utf-8")); h.update(b"\x00")
            h.update(cont.encode("utf-8")); h.update(b"\x01")
        cache_dir = Path(os.environ.get("LM_HARNESS_CACHE_PATH", "/mnt/ai/data/lm_eval_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"chimera-tokcache-{h.hexdigest()}.pkl"
        if cache_file.exists():
            print(f"[bench] loaded pretokenized eval inputs from {cache_file.name}")
            return pickle.loads(cache_file.read_bytes())
        new_reqs = self._encode_pairs_batched(pairs)
        cache_file.write_bytes(pickle.dumps(new_reqs))
        print(f"[bench] pretokenized + cached {len(pairs)} eval inputs -> {cache_file.name}")
        return new_reqs

    def _encode_pairs_batched(self, pairs):
        """Batched equivalent of the base loglikelihood tokenize loop.

        Replicates lm-eval's ``_encode_pair`` trailing-space shift and empty-context
        handling, but encodes all (context) and (context+continuation) strings in
        two parallel ``encode_batch`` calls instead of 2 sequential ``encode`` calls
        per request -- ~10x faster on the ~70k-request suite. Returns the same
        ``((context, continuation), context_enc, continuation_enc)`` tuples the base
        class feeds to ``_loglikelihood_tokens``.
        """
        enc = self.tokenizer._tok
        ctxs, wholes, shifted = [], [], []
        for context, continuation in pairs:
            if context == "":
                shifted.append(("", continuation, True))
                continue
            n_spaces = len(context) - len(context.rstrip())
            if n_spaces > 0:
                continuation = context[-n_spaces:] + continuation
                context = context[:-n_spaces]
            shifted.append((context, continuation, False))
            ctxs.append(context)
            wholes.append(context + continuation)
        whole_encs = [e.ids for e in enc.encode_batch(wholes, add_special_tokens=False)] if wholes else []
        ctx_encs = [e.ids for e in enc.encode_batch(ctxs, add_special_tokens=False)] if ctxs else []

        new_reqs = []
        j = 0  # index into the non-empty-context batches
        for context, continuation, is_empty in shifted:
            if is_empty:
                # empty context: condition on prefix token (base loglikelihood).
                cont_enc = enc.encode(continuation, add_special_tokens=False).ids
                if self.prefix_token_id != cont_enc[0]:
                    context_enc, continuation_enc = [self.prefix_token_id], cont_enc
                else:
                    context_enc, continuation_enc = cont_enc[:1], cont_enc[1:]
                new_reqs.append((("", continuation), context_enc, continuation_enc))
                continue
            context_enc = ctx_encs[j]
            continuation_enc = whole_encs[j][len(context_enc):]
            new_reqs.append(((context, continuation), context_enc, continuation_enc))
            j += 1
        return new_reqs

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError("no configured task needs rolling loglikelihood")

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("no configured task needs generation")

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        prepared = []
        for idx, (_, context_enc, continuation_enc) in enumerate(requests):
            whole = list(context_enc) + list(continuation_enc)
            if len(whole) > self.block_size + 1:
                whole = whole[-(self.block_size + 1) :]  # left-truncate context only
            inp = whole[:-1]
            prepared.append((idx, inp, continuation_enc))

        # Largest sequences first: an OOM at this batch size surfaces immediately.
        prepared.sort(key=lambda x: len(x[1]), reverse=True)

        results = [None] * len(requests)
        from tqdm import tqdm

        batch, batch_max_len = [], 0
        pbar = tqdm(total=len(prepared), disable=disable_tqdm, desc="loglikelihood")
        for idx, inp, continuation_enc in prepared:
            candidate_max_len = max(batch_max_len, len(inp))
            if batch and candidate_max_len * (len(batch) + 1) > self.batch_tokens:
                self._score_batch(batch, results)
                pbar.update(len(batch))
                batch, batch_max_len = [], 0
            batch.append((idx, inp, continuation_enc))
            batch_max_len = max(batch_max_len, len(inp))
        if batch:
            self._score_batch(batch, results)
            pbar.update(len(batch))
        pbar.close()

        return results

    @torch.inference_mode()
    def _score_batch(self, batch, results):
        max_len = max(len(inp) for _, inp, _ in batch)
        input_ids = torch.full((len(batch), max_len), self.eot_token_id, dtype=torch.long)
        for i, (_, inp, _) in enumerate(batch):
            input_ids[i, : len(inp)] = torch.tensor(inp, dtype=torch.long)
        input_ids = input_ids.to(self._device)

        autocast_enabled = self._device.startswith("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            hidden = self.model(input_ids, return_hidden=True)  # (B, L, C)
            slices, meta = [], []
            for i, (idx, inp, continuation_enc) in enumerate(batch):
                cont_len = len(continuation_enc)
                cont_start = len(inp) - cont_len
                slices.append(hidden[i, cont_start : cont_start + cont_len, :])
                meta.append((idx, continuation_enc))
            flat_hidden = torch.cat(slices, dim=0)  # (total_cont, C) — never materializes full (B,L,V) logits
            logits = self.model.project(flat_hidden)  # (total_cont, V)

        log_probs = F.log_softmax(logits.float(), dim=-1)

        offset = 0
        for idx, continuation_enc in meta:
            cont_len = len(continuation_enc)
            lp = log_probs[offset : offset + cont_len]
            offset += cont_len
            cont_ids = torch.tensor(continuation_enc, device=lp.device)
            token_logprobs = lp.gather(-1, cont_ids.unsqueeze(-1)).squeeze(-1)
            is_greedy = bool((lp.argmax(-1) == cont_ids).all().item())
            results[idx] = (token_logprobs.sum().item(), is_greedy)


def run_benchmarks(
    model: GPT,
    tokenizer: BPETokenizer,
    tasks: list[str],
    block_size: int = 2048,
    batch_tokens: int = 32768,
    device: str = "cuda",
    limit: int | None = None,
) -> dict:
    """Run the lm-eval loglikelihood benchmark suite against an in-memory model.

    Shared by the standalone CLI below and ``train.py``'s test phase — takes an
    already-loaded model/tokenizer so the caller controls checkpoint loading and
    device placement.
    """
    import lm_eval

    was_training = model.training
    model.eval()
    lm = ChimeraLM(model, tokenizer, block_size, batch_tokens, device)
    try:
        # cache_requests: reuse the built (dataset-loaded, doc-formatted, tokenized)
        # request objects across runs, keyed by task/num_fewshot/limit — skips the
        # dataset download+map+tokenize overhead on every repeat benchmark call.
        output = lm_eval.simple_evaluate(
            model=lm, tasks=tasks, num_fewshot=0, limit=limit, cache_requests=True
        )
    finally:
        model.train(was_training)
    return output["results"]


def iter_metrics(results_by_task: dict):
    """Yield ``(task, metric_name, value, stderr)`` for every real numeric metric.

    Filters out lm-eval bookkeeping fields (``alias``, ``sample_len``) and stderr
    entries (paired with their metric instead of yielded standalone).
    """
    skip_metrics = {"alias", "sample_len"}
    for task, metrics in sorted(results_by_task.items()):
        for key, val in metrics.items():
            metric_name = key.split(",")[0]
            if metric_name in skip_metrics or not isinstance(val, (int, float)) or "_stderr" in key:
                continue
            filt = key.split(",", 1)[1] if "," in key else "none"
            stderr = metrics.get(f"{metric_name}_stderr,{filt}")
            yield task, metric_name, val, stderr


def print_table(results_by_task: dict):
    header = f"| {'task':<16} | {'metric':<12} | {'value':>7} | {'stderr':>7} | {'chance':>7} |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for task, metric_name, val, stderr in iter_metrics(results_by_task):
        is_ppl = metric_name in PPL_METRICS
        chance = None if is_ppl else CHANCE.get(task)
        val_s = f"{val:.2f}" if is_ppl else f"{val * 100:.2f}"
        stderr_s = "-" if stderr is None else (f"{stderr:.2f}" if is_ppl else f"{stderr * 100:.2f}")
        chance_s = "-" if chance is None else f"{chance:.1f}"
        print(f"| {task:<16} | {metric_name:<12} | {val_s:>7} | {stderr_s:>7} | {chance_s:>7} |")


def flatten_for_wandb(results_by_task: dict, prefix: str = "test") -> dict[str, float]:
    """Flatten lm-eval results into ``{"test/<task>/<metric>": value}`` for wandb logging.

    Prefixed ``test`` (not ``eval``) to sit alongside the ``test/loss`` and
    ``test/bpt`` that ``LanguageModelModule.test_step`` already logs. Percent-style
    metrics are kept as raw fractions (0-1), matching wandb/lm-eval convention, so
    they plot on the same 0-1 axis as other logged accuracies.
    """
    return {
        f"{prefix}/{task}/{metric_name}": val
        for task, metric_name, val, _ in iter_metrics(results_by_task)
    }


def main():
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    tokenizer = BPETokenizer.from_pretrained(args.tokenizer_id)
    model = load_model(args, vocab_size=tokenizer.vocab_size)
    results = run_benchmarks(
        model, tokenizer, tasks, args.block_size, args.batch_tokens, args.device, args.limit
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"saved results to {out_path}")

    print_table(results)


if __name__ == "__main__":
    main()
