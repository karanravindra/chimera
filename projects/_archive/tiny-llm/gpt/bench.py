"""Zero-shot benchmark eval of the tiny GPT via lm-evaluation-harness.

Wraps a raw ``chimera.models.GPT`` in a minimal ``lm_eval`` adapter and runs
loglikelihood-ranking benchmarks. ``run_benchmarks`` is imported by ``train.py``
to run the suite in the test phase and log it to wandb under ``test/<task>/<metric>``;
this file also runs standalone against a saved checkpoint::

    uv run python projects/tiny-llm/gpt/bench.py --ckpt .../smoke-small-8k/checkpoints/gpt.ckpt

Task set is tuned for the 5-20M scale, where most knowledge/reasoning benchmarks
sit at chance:
  * blimp          — grammatical minimal pairs; LINGUISTIC COMPETENCE, not
                     knowledge, so a tiny model moves above chance (the BabyLM
                     yardstick). The headline capability metric here.
  * lambada_openai — last-word prediction; long-range coherence.
  * piqa / sciq    — physical / science QA; expect near-chance below ~30M
                     (logged so the trend is visible, not because they'll move early).
  * arc_easy       — grade-school science MC; near-chance at this scale.

Named ``bench.py`` (not ``evaluate.py``) to avoid shadowing the ``evaluate``
PyPI package pulled in by lm-eval.
"""

import argparse
import json
import os
from pathlib import Path

# lm_eval.caching.cache reads this at import time — set before importing lm_eval.
os.environ.setdefault("LM_HARNESS_CACHE_PATH", "/mnt/ai/data/lm_eval_cache")

import torch
import torch.nn.functional as F
from lm_eval.api.model import TemplateLM

from chimera.models import GPT
from chimera.tokenizers import BPETokenizer

DEFAULT_TASKS = ["blimp", "lambada_openai", "piqa", "sciq", "arc_easy"]

# Kept identical to train.py's SAMPLE_PROMPTS so backfilled test/generations match
# what a fresh run would log (story / expository / FAQ / procedural registers).
SAMPLE_PROMPTS = [
    "Once upon a time, there was a little",
    "The sun is a star that",
    "Question: Why do birds fly south in the winter?\nAnswer:",
    "Here is how to plant a seed. First,",
]

# Random-chance baselines (percent) for accuracy metrics, so a number reads
# against "better than guessing" at a glance.
CHANCE = {
    "blimp": 50.0,  # 2-way minimal pairs
    "lambada_openai": 0.0,  # exact last-word match
    "piqa": 50.0,  # 2-way
    "sciq": 25.0,  # 4-way
    "arc_easy": 25.0,  # ~4-way
}
PPL_METRICS = {"perplexity", "word_perplexity", "byte_perplexity", "bits_per_byte"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt",
        default="/mnt/ai/runs/tiny-llm/gpt/smoke-small-8k/checkpoints/gpt.ckpt",
    )
    p.add_argument("--out-dir", default="/mnt/ai/runs/tiny-llm/gpt/eval")
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    p.add_argument("--tokenizer-id", default="/mnt/ai/data/tiny-llm/tokenizer/8k")
    # Checkpoint saves no hyper_parameters, so model dims are supplied here
    # (match train.py's defaults: the "small" preset).
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--n-embd", type=int, default=320)
    p.add_argument("--n-head", type=int, default=10)
    p.add_argument("--n-kv-head", type=int, default=1)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--mup-base-width", type=int, default=256)
    p.add_argument("--batch-tokens", type=int, default=32768)
    p.add_argument(
        "--limit", type=int, default=None, help="cap examples per task (smoke)"
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Backfill: log to an EXISTING wandb run (e.g. a training run that skipped eval).
    p.add_argument(
        "--wandb-id", default=None, help="existing wandb run id to backfill into"
    )
    p.add_argument("--wandb-project", default="tiny-llm-pretrain")
    p.add_argument(
        "--gen", action="store_true", help="also sample + log test/generations"
    )
    p.add_argument(
        "--no-bench",
        dest="bench",
        action="store_false",
        help="skip the benchmark suite (e.g. generations-only backfill)",
    )
    return p.parse_args()


def load_model(args, vocab_size: int) -> GPT:
    # Dense GQA (tiny-llm doesn't use MLA/MoE). mup_base_width must match training
    # so the muP output_mult in forward is reconstructed correctly.
    model = GPT(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_layer=args.n_layer,
        tie_embedding=True,
        mup_base_width=args.mup_base_width,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    prefix = "model._orig_mod."  # Lightning + torch.compile prefix
    state_dict = {
        k[len(prefix) :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith(prefix)
    }
    model.load_state_dict(state_dict)
    model.to(args.device).eval()
    return model


class ChimeraLM(TemplateLM):
    """Minimal lm-eval adapter around ``chimera.models.GPT`` for loglikelihood tasks."""

    def __init__(
        self,
        model: GPT,
        tokenizer: BPETokenizer,
        block_size: int,
        batch_tokens: int,
        device: str,
    ):
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
        # add_special_tokens=False matches training-time tokenization.
        return self.tokenizer._tok.encode(
            string, add_special_tokens=add_special_tokens
        ).ids

    def loglikelihood(self, requests, disable_tqdm=False):
        new_reqs = self._encode_pairs_cached([req.args for req in requests])
        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def _encode_pairs_cached(self, pairs):
        import hashlib
        import pickle

        version = "v2"
        # Fingerprint the FULL tokenizer (vocab size + serialized vocab/merges), NOT
        # a few common tokens: 4k/8k/16k share their low-id common tokens, so a short
        # fingerprint collides across vocab sizes -> one vocab's cached token ids get
        # fed to another model's embedding -> out-of-range index / device-side assert.
        fp = (
            self.tokenizer._tok.get_vocab_size(),
            hashlib.md5(self.tokenizer._tok.to_str().encode()).hexdigest(),
        )
        h = hashlib.md5(f"{version}|{fp}|{len(pairs)}".encode())
        for ctx, cont in pairs:
            h.update(ctx.encode("utf-8"))
            h.update(b"\x00")
            h.update(cont.encode("utf-8"))
            h.update(b"\x01")
        cache_dir = Path(
            os.environ.get("LM_HARNESS_CACHE_PATH", "/mnt/ai/data/lm_eval_cache")
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"chimera-tokcache-{h.hexdigest()}.pkl"
        if cache_file.exists():
            print(f"[bench] loaded pretokenized eval inputs from {cache_file.name}")
            return pickle.loads(cache_file.read_bytes())
        new_reqs = self._encode_pairs_batched(pairs)
        cache_file.write_bytes(pickle.dumps(new_reqs))
        print(
            f"[bench] pretokenized + cached {len(pairs)} eval inputs -> {cache_file.name}"
        )
        return new_reqs

    def _encode_pairs_batched(self, pairs):
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
        whole_encs = (
            [e.ids for e in enc.encode_batch(wholes, add_special_tokens=False)]
            if wholes
            else []
        )
        ctx_encs = (
            [e.ids for e in enc.encode_batch(ctxs, add_special_tokens=False)]
            if ctxs
            else []
        )

        new_reqs = []
        j = 0
        for context, continuation, is_empty in shifted:
            if is_empty:
                cont_enc = enc.encode(continuation, add_special_tokens=False).ids
                if self.prefix_token_id != cont_enc[0]:
                    context_enc, continuation_enc = [self.prefix_token_id], cont_enc
                else:
                    context_enc, continuation_enc = cont_enc[:1], cont_enc[1:]
                new_reqs.append((("", continuation), context_enc, continuation_enc))
                continue
            context_enc = ctx_encs[j]
            continuation_enc = whole_encs[j][len(context_enc) :]
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

        prepared.sort(
            key=lambda x: len(x[1]), reverse=True
        )  # largest first -> OOM early

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
        input_ids = torch.full(
            (len(batch), max_len), self.eot_token_id, dtype=torch.long
        )
        for i, (_, inp, _) in enumerate(batch):
            input_ids[i, : len(inp)] = torch.tensor(inp, dtype=torch.long)
        input_ids = input_ids.to(self._device)

        autocast_enabled = self._device.startswith("cuda")
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            hidden = self.model(input_ids, return_hidden=True)  # (B, L, C)
            slices, meta = [], []
            for i, (idx, inp, continuation_enc) in enumerate(batch):
                cont_len = len(continuation_enc)
                cont_start = len(inp) - cont_len
                slices.append(hidden[i, cont_start : cont_start + cont_len, :])
                meta.append((idx, continuation_enc))
            flat_hidden = torch.cat(
                slices, dim=0
            )  # never materializes full (B,L,V) logits
            logits = self.model.project(flat_hidden)

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


@torch.inference_mode()
def generate_rows(
    model,
    tokenizer,
    device,
    prompts=SAMPLE_PROMPTS,
    max_new_tokens=80,
    temperature=0.8,
    repetition_penalty=1.3,
    min_p=0.05,
):
    """Sample continuations from fixed prompts -> [[prompt, generation], ...].

    Defaults match train.py's log_generations (repetition_penalty + min_p) so
    backfilled generations don't degenerate into repetition loops."""
    model.eval()
    rows = []
    for p in prompts:
        ids = tokenizer._tok.encode(p, add_special_tokens=False).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
        )
        rows.append([p, tokenizer.decode(out[0].tolist()[len(ids) :])])
    return rows


def run_benchmarks(
    model: GPT,
    tokenizer: BPETokenizer,
    tasks: list[str],
    block_size: int = 1024,
    batch_tokens: int = 32768,
    device: str = "cuda",
    limit: int | None = None,
) -> dict:
    """Run the lm-eval loglikelihood suite against an in-memory model."""
    import lm_eval

    was_training = model.training
    model.eval()
    lm = ChimeraLM(model, tokenizer, block_size, batch_tokens, device)
    try:
        # cache_requests=False: lm-eval's request cache keys on a tokenizer hash
        # that DOESN'T distinguish our 4k/8k/16k BPE tokenizers (same class/config),
        # so it would feed one vocab's cached token ids to another model's embedding
        # -> out-of-range index / device-side assert. Our own tokenizer-aware
        # tokcache (_encode_pairs_cached) already caches the encoding safely.
        output = lm_eval.simple_evaluate(
            model=lm, tasks=tasks, num_fewshot=0, limit=limit, cache_requests=False
        )
    finally:
        model.train(was_training)
    return output["results"]


def iter_metrics(results_by_task: dict):
    skip_metrics = {"alias", "sample_len"}
    for task, metrics in sorted(results_by_task.items()):
        for key, val in metrics.items():
            metric_name = key.split(",")[0]
            if (
                metric_name in skip_metrics
                or not isinstance(val, (int, float))
                or "_stderr" in key
            ):
                continue
            filt = key.split(",", 1)[1] if "," in key else "none"
            stderr = metrics.get(f"{metric_name}_stderr,{filt}")
            yield task, metric_name, val, stderr


# One headline metric per task (length-normalized acc where it exists, else raw
# acc, else perplexity). Keeps the wandb test/ namespace to one key per task.
_METRIC_PREF = ["acc_norm", "acc", "perplexity"]


def headline_metrics(results_by_task: dict, tasks: list[str] | None):
    """Collapse lm-eval results to ONE metric per REQUESTED task.

    Drops auto-expanded group subtasks (e.g. BLiMP's 67 ``blimp_*``) by keeping
    only ``tasks`` (the top-level names actually requested), and drops the
    redundant secondary metrics (acc vs acc_norm) per task. Full per-subtask
    detail still lands in ``results.json``; this is only what gets tabled/logged.
    """
    keep = set(tasks) if tasks else None
    best: dict[str, tuple] = {}
    for task, metric, val, stderr in iter_metrics(results_by_task):
        if keep is not None and task not in keep:
            continue
        rank = (
            _METRIC_PREF.index(metric) if metric in _METRIC_PREF else len(_METRIC_PREF)
        )
        cur = best.get(task)
        if cur is None or rank < cur[0]:
            best[task] = (rank, metric, val, stderr)
    return [(t, m, v, s) for t, (_, m, v, s) in sorted(best.items())]


def print_table(results_by_task: dict, tasks: list[str] | None = None):
    """Print the headline metric per task. ``tasks=None`` prints every metric."""
    rows = (
        headline_metrics(results_by_task, tasks)
        if tasks is not None
        else list(iter_metrics(results_by_task))
    )
    header = f"| {'task':<16} | {'metric':<10} | {'value':>7} | {'stderr':>7} | {'chance':>7} |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for task, metric_name, val, stderr in rows:
        is_ppl = metric_name in PPL_METRICS
        chance = None if is_ppl else CHANCE.get(task)
        val_s = f"{val:.2f}" if is_ppl else f"{val * 100:.2f}"
        stderr_s = (
            "-"
            if stderr is None
            else (f"{stderr:.2f}" if is_ppl else f"{stderr * 100:.2f}")
        )
        chance_s = "-" if chance is None else f"{chance:.1f}"
        print(
            f"| {task:<16} | {metric_name:<10} | {val_s:>7} | {stderr_s:>7} | {chance_s:>7} |"
        )


def flatten_for_wandb(
    results_by_task: dict, tasks: list[str] | None = None, prefix: str = "test"
) -> dict[str, float]:
    """Flatten to ``{"test/<task>/<metric>": value}`` — ONE headline metric per
    requested task (pass ``tasks`` to collapse BLiMP's subtasks + drop secondary
    metrics; ``None`` keeps every metric, the old behavior)."""
    rows = (
        headline_metrics(results_by_task, tasks)
        if tasks is not None
        else list(iter_metrics(results_by_task))
    )
    return {f"{prefix}/{task}/{metric_name}": val for task, metric_name, val, _ in rows}


def main():
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    tokenizer = BPETokenizer.from_pretrained(args.tokenizer_id)
    model = load_model(args, vocab_size=tokenizer.vocab_size)

    results = None
    if args.bench:
        results = run_benchmarks(
            model,
            tokenizer,
            tasks,
            args.block_size,
            args.batch_tokens,
            args.device,
            args.limit,
        )
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(
            json.dumps(results, indent=2, default=str)
        )
        print(f"saved results to {out_dir / 'results.json'}")
        print_table(results, tasks=tasks)  # headline per task (full detail in json)

    gen_rows = None
    if args.gen:
        gen_rows = generate_rows(model, tokenizer, args.device)
        print("\n=== sample generations ===")
        for p, c in gen_rows:
            print(f"> {p!r}\n  {c!r}\n")

    if args.wandb_id:
        import wandb

        run = wandb.init(project=args.wandb_project, id=args.wandb_id, resume="must")
        if results is not None:
            metrics = flatten_for_wandb(results, tasks=tasks)
            run.log(metrics)
            print(
                f"backfilled {len(metrics)} benchmark metrics into wandb run {args.wandb_id}"
            )
        if gen_rows is not None:
            run.log(
                {
                    "test/generations": wandb.Table(
                        columns=["prompt", "generation"], data=gen_rows
                    )
                }
            )
            print(
                f"backfilled test/generations ({len(gen_rows)} prompts) into {args.wandb_id}"
            )
        run.finish()


if __name__ == "__main__":
    main()
