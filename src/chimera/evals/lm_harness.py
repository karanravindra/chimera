"""lm-eval adapter for chimera language models (loglikelihood-ranking tasks only)."""

import hashlib
import os
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")  # datasets cache
os.environ.setdefault("LM_HARNESS_CACHE_PATH", "/mnt/ai/data/lm_eval_cache")

from lm_eval.api.model import TemplateLM  # noqa: E402  (env must be set before import)


class ChimeraLM(TemplateLM):
    """Minimal lm-eval adapter around a chimera GPT for loglikelihood tasks.

    Works with any model whose ``model(x)`` returns full logits (B, L, V) directly,
    so scoring is a plain gather; ``tokenizer`` is a ``chimera.tokenizers.BPETokenizer``
    (only its HF-tokenizers backend ``._tok`` is used). We implement tok_encode,
    eot_token_id, and _loglikelihood_tokens. We also override loglikelihood() to
    cache + batch the (context, continuation) -> token-id encoding to disk. That
    encoding is weight-independent, so it runs once per (tokenizer, task set) and
    reloads on every later run — including after a retrain, where only the scoring
    must repeat. (Base TemplateLM.loglikelihood re-encodes all requests in a Python
    loop every run — the "Tokenizing inputs" pass.)
    """

    def __init__(
        self,
        model,
        tokenizer,
        eot_id,
        block_size,
        device=None,
        batch_tokens=16384,
        bos_id=None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.backend = "causal"
        self._eot_id = eot_id
        self._bos_id = bos_id
        self.block_size = block_size
        self.batch_tokens = batch_tokens
        # Derive the device from the model's own weights so inputs can never land on a
        # different device than the parameters (`device` is only a fallback).
        params = list(model.parameters())
        self._device = params[0].device if params else torch.device(device or "cpu")

    @property
    def eot_token_id(self):
        return self._eot_id

    @property
    def prefix_token_id(self):
        # Training docs start with BOS, so empty-context requests are prefixed with it
        # (falls back to the lm-eval default, eot, when the stream has no BOS).
        return self._bos_id if self._bos_id is not None else self._eot_id

    @property
    def tokenizer_name(self) -> str:
        # Fingerprints the request cache (cache_requests=True) so it invalidates if the
        # tokenizer changes. Full serialization, not a few tokens: different vocab sizes
        # share their low-id common tokens, so a short fingerprint would collide and feed
        # one vocab's cached token ids to another model's embedding.
        tok = self.tokenizer._tok
        return f"chimera-v{tok.get_vocab_size()}-{hashlib.md5(tok.to_str().encode()).hexdigest()[:12]}"

    def tok_encode(self, string, add_special_tokens=False, **kwargs):
        # add_special_tokens=False matches training-time tokenization.
        return self.tokenizer._tok.encode(
            string, add_special_tokens=add_special_tokens
        ).ids

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError("no configured task needs rolling loglikelihood")

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("no configured task needs generation")

    def loglikelihood(self, requests, disable_tqdm=False):
        # Replace the base "Tokenizing inputs" pass (re-encodes every request in Python
        # on every run) with a batched, disk-cached encode.
        new_reqs = self._encode_pairs_cached([req.args for req in requests])
        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def _encode_pairs_cached(self, pairs):
        # Key on the tokenizer + the exact (ctx, cont) pairs. Independent of model
        # weights, so it survives retraining: re-score, but never re-tokenize.
        h = hashlib.md5(
            f"v2|bos={self._bos_id}|{self.tokenizer_name}|{len(pairs)}".encode()
        )
        for ctx, cont in pairs:
            h.update(ctx.encode("utf-8"))
            h.update(b"\x00")
            h.update(cont.encode("utf-8"))
            h.update(b"\x01")
        cache_dir = Path(os.environ["LM_HARNESS_CACHE_PATH"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"chimera-tokcache-{h.hexdigest()}.pkl"
        if cache_file.exists():
            print(f"[bench] loaded pretokenized eval inputs from {cache_file.name}")
            return pickle.loads(cache_file.read_bytes())
        new_reqs = self._encode_pairs_batched(pairs)
        cache_file.write_bytes(pickle.dumps(new_reqs))
        print(f"[bench] pretokenized + cached {len(pairs)} inputs -> {cache_file.name}")
        return new_reqs

    def _encode_pairs_batched(self, pairs):
        # Reimplements lm-eval's _encode_pair (trailing-space / empty-context shift) but
        # via encode_batch, so the whole task set tokenizes in one parallel Rust call.
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
            if self._bos_id is not None:
                context_enc = [self._bos_id] + context_enc
            new_reqs.append(((context, continuation), context_enc, continuation_enc))
            j += 1
        return new_reqs

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
        # Right-pad with eot: attention is causal, so pad tokens after the real content
        # never affect the continuation-position logits we read.
        input_ids = torch.full(
            (len(batch), max_len), self.eot_token_id, dtype=torch.long
        )
        for i, (_, inp, _) in enumerate(batch):
            input_ids[i, : len(inp)] = torch.tensor(inp, dtype=torch.long)
        input_ids = input_ids.to(self._device)

        logits = self.model(input_ids)  # (B, L, V)
        for i, (idx, inp, continuation_enc) in enumerate(batch):
            cont_len = len(continuation_enc)
            cont_start = len(inp) - cont_len
            cont_logits = logits[i, cont_start : cont_start + cont_len, :].float()
            log_probs = F.log_softmax(cont_logits, dim=-1)
            cont_ids = torch.tensor(continuation_enc, device=log_probs.device)
            token_logprobs = log_probs.gather(-1, cont_ids.unsqueeze(-1)).squeeze(-1)
            is_greedy = bool((log_probs.argmax(-1) == cont_ids).all().item())
            results[idx] = (token_logprobs.sum().item(), is_greedy)
