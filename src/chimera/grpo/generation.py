"""Shared rollout generation used by both training and eval.

Given a list of (already chat-templated) prompt strings, tokenize them with left padding,
sample/greedy-decode completions, and return everything the GRPO step and the eval script
need: the full prompt+completion ids, the prompt length, attention masks, a completion mask
that is 1 through the first EOS (and 0 over the trailing pad), and the decoded completion
texts for reward scoring.
"""

from __future__ import annotations

import torch


def build_completion_mask(completion_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """1 for completion tokens up to and including the first EOS, 0 afterward.

    A completion that never emits EOS (hit ``max_new_tokens``) is all 1s. This is what masks
    the post-EOS pad tokens out of the loss and the attention mask.
    """
    batch, length = completion_ids.shape
    is_eos = completion_ids == eos_id
    first_eos = torch.full(
        (batch,), length, dtype=torch.long, device=completion_ids.device
    )
    has_eos = is_eos.any(dim=1)
    first_eos[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    positions = torch.arange(length, device=completion_ids.device)
    return (positions.unsqueeze(0) <= first_eos.unsqueeze(1)).long()


@torch.no_grad()
def generate_completions(
    model,
    tokenizer,
    prompts: list[str],
    *,
    num_return_sequences: int,
    max_prompt_len: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device,
) -> dict:
    """Sample ``num_return_sequences`` completions per prompt.

    Returns a dict with ``full_ids`` ``(B*G, L)``, ``prompt_len`` (int), ``prompt_mask``
    ``(B*G, prompt_len)`` (the input attention mask expanded per generation), ``completion_ids``
    / ``completion_mask`` ``(B*G, max_new_tokens')``, and ``texts`` (decoded completions).
    Rows are ordered so each prompt's ``G`` completions are contiguous (prompt index = row //
    G), matching :func:`grpo.compute_group_advantages`'s grouping.
    """
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
        add_special_tokens=False,  # the chat template already added them
    ).to(device)
    prompt_len = enc.input_ids.shape[1]

    gen_kwargs = dict(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)
    full_ids = model.generate(**gen_kwargs)

    completion_ids = full_ids[:, prompt_len:]
    completion_mask = build_completion_mask(completion_ids, tokenizer.eos_token_id)
    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    prompt_mask = enc.attention_mask.repeat_interleave(num_return_sequences, dim=0)
    return {
        "full_ids": full_ids,
        "prompt_len": prompt_len,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "texts": texts,
    }
