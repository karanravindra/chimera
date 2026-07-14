"""Canonical chat template + special tokens for the LLM.

Single source of truth for how a conversation becomes text, shared by the
tokenizer trainer (which reserves the special tokens), the pretrain/SFT
tokenizers (which render rows to text), and the DataModule (which builds
generation prompts). Defined once here so all four stay in lockstep.

Format — ChatML, extended for reasoning + tool use:

    <|im_start|>system
    {system}

    # Tools
    <tools>
    [{"name": "...", "description": "...", "parameters": {...}}]
    </tools><|im_end|>
    <|im_start|>user
    {user}<|im_end|>
    <|im_start|>assistant
    <think>
    {reasoning}
    </think>
    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Paris"}}
    </tool_call><|im_end|>
    <|im_start|>tool
    <tool_response>
    {result}
    </tool_response><|im_end|>
    <|im_start|>assistant
    {answer}<|im_end|>

- Reasoning goes in a ``<think>…</think>`` block at the start of an assistant turn.
- Tool calls are Hermes-style JSON inside ``<tool_call>…</tool_call>`` (one block
  per call); all source formats (pythonic, OpenAI ``tool_calls``, ``function_call``
  role) are normalized into this canonical form by :func:`normalize_messages`.
- Tool results come back in a ``role="tool"`` turn wrapped in
  ``<tool_response>…</tool_response>``.

Rendering is expressed once as a list of ``(text, supervised)`` segments
(:func:`iter_segments`); the plain renderer joins the text, and the SFT renderer
encodes each segment and masks by the ``supervised`` flag — so both can never
drift apart.
"""

from __future__ import annotations

import ast
import json
from typing import Callable, Iterable, Optional

# --------------------------------------------------------------------------- #
# Special tokens (order fixed: structural first -> low, stable ids)
# --------------------------------------------------------------------------- #
BOS = "<|startoftext|>"
EOS = "<|endoftext|>"
PAD = "<|pad|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_RESPONSE_START = "<tool_response>"
TOOL_RESPONSE_END = "</tool_response>"

# Passed to the tokenizer trainer; ids assigned in this order (see
# BPETokenizer._train_hf). Structural markers first so their ids are the lowest
# and most stable; semantic markers next.
SPECIAL_TOKENS: list[str] = [
    EOS, BOS, PAD, IM_START, IM_END,
    THINK_START, THINK_END,
    TOOL_CALL_START, TOOL_CALL_END,
    TOOL_RESPONSE_START, TOOL_RESPONSE_END,
]

# roles the model itself produces -> supervised at SFT time
ASSISTANT_ROLES = {"assistant", "gpt", "function_call", "model", "chatbot"}
# normalize source-specific role names to canonical ChatML roles
ROLE_MAP = {
    "human": "user", "user": "user", "prompter": "user",
    "gpt": "assistant", "assistant": "assistant", "model": "assistant",
    "chatbot": "assistant", "function_call": "assistant",
    "observation": "tool", "tool": "tool", "ipython": "tool",
    "tool_response": "tool", "function_response": "tool",
    "system": "system",
}


# --------------------------------------------------------------------------- #
# Normalization: heterogeneous source rows -> canonical turns
# --------------------------------------------------------------------------- #
# A canonical turn is:
#   {"role": system|user|assistant|tool,
#    "content": str,
#    "thinking": Optional[str],                      # assistant only
#    "tool_calls": Optional[list[{"name","arguments"}]]}  # assistant only
def _as_obj(v):
    """Best-effort parse of a value that may be a dict, JSON string, or py-repr."""
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        for parse in (json.loads, ast.literal_eval):
            try:
                return parse(v)
            except Exception:
                pass
    return v


def _coerce_tool_calls(msg: dict) -> list[dict]:
    """Pull a list of {"name","arguments"} from the many tool-call encodings."""
    calls = []
    raw = msg.get("tool_calls")
    if raw:
        for c in _as_obj(raw) or []:
            fn = c.get("function", c) if isinstance(c, dict) else {}
            name = fn.get("name")
            args = _as_obj(fn.get("arguments", fn.get("parameters", {})))
            if name:
                calls.append({"name": name, "arguments": args})
    # OpenAI-legacy single function_call field, or a function_call-role message
    fc = msg.get("function_call")
    if fc:
        fc = _as_obj(fc)
        if isinstance(fc, dict) and fc.get("name"):
            calls.append({"name": fc["name"],
                          "arguments": _as_obj(fc.get("arguments", {}))})
    return calls


def _split_thinking(content: str) -> tuple[Optional[str], str]:
    """Extract a leading <think>…</think> block already present in the content."""
    if content and content.lstrip().startswith(THINK_START) and THINK_END in content:
        head, _, rest = content.partition(THINK_END)
        thinking = head.split(THINK_START, 1)[1].strip()
        return thinking, rest.strip()
    return None, content


def normalize_messages(raw: Iterable[dict]) -> list[dict]:
    """Map a source message list into canonical turns (see schema above)."""
    turns: list[dict] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        role = ROLE_MAP.get(m.get("role") or m.get("from") or "user", "user")
        content = m.get("content")
        if content is None:
            content = m.get("value") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        turn: dict = {"role": role, "content": content}
        if role == "assistant":
            thinking = m.get("thinking") or m.get("reasoning")
            if not thinking:
                thinking, content = _split_thinking(content)
                turn["content"] = content
            if thinking:
                turn["thinking"] = thinking.strip() if isinstance(thinking, str) else thinking
            calls = _coerce_tool_calls(m)
            if calls:
                turn["tool_calls"] = calls
        turns.append(turn)
    return turns


# --------------------------------------------------------------------------- #
# Segmented rendering (one code path for plain text + SFT masking)
# --------------------------------------------------------------------------- #
def _tool_call_block(calls: list[dict]) -> str:
    out = []
    for c in calls:
        payload = {"name": c["name"], "arguments": c.get("arguments", {})}
        out.append(f"{TOOL_CALL_START}\n{json.dumps(payload, ensure_ascii=False)}\n{TOOL_CALL_END}")
    return "\n".join(out)


def _system_text(system: Optional[str], tools) -> Optional[str]:
    """Compose the system turn's body from an optional prompt + tools schema."""
    parts = []
    if system:
        parts.append(system.strip())
    if tools:
        tools_json = tools if isinstance(tools, str) else json.dumps(tools, ensure_ascii=False)
        parts.append(f"# Tools\n<tools>\n{tools_json}\n</tools>")
    return "\n\n".join(parts) if parts else None


def iter_segments(messages: list[dict], tools=None, system: Optional[str] = None,
                  add_generation_prompt: bool = False) -> list[tuple[str, bool]]:
    """Yield (text, supervised) segments for a normalized conversation.

    ``supervised`` marks the assistant's own output (its think block, content,
    tool calls, and the closing ``<|im_end|>``) — everything the model should
    learn to produce. Headers, system/tool text, user turns, and tool responses
    are not supervised.
    """
    segs: list[tuple[str, bool]] = []
    # a leading system turn, folding in the tools schema if present
    sys_body = _system_text(system, tools)
    if sys_body is not None:
        segs.append((f"{IM_START}system\n{sys_body}{IM_END}\n", False))

    for m in messages:
        role = m["role"]
        if role == "system" and sys_body is not None:
            continue  # already emitted (schema folded in above)
        segs.append((f"{IM_START}{role}\n", False))  # header, never supervised
        sup = role == "assistant"
        if role == "tool":
            segs.append((f"{TOOL_RESPONSE_START}\n{m['content']}\n{TOOL_RESPONSE_END}", False))
        elif role == "assistant":
            if m.get("thinking"):
                segs.append((f"{THINK_START}\n{m['thinking']}\n{THINK_END}\n", True))
            if m.get("content"):
                segs.append((m["content"], True))
            if m.get("tool_calls"):
                sep = "\n" if m.get("content") else ""
                segs.append((sep + _tool_call_block(m["tool_calls"]), True))
        else:
            segs.append((m["content"], False))
        segs.append((IM_END, sup))   # supervise the stop token on assistant turns
        segs.append(("\n", False))

    if add_generation_prompt:
        segs.append((f"{IM_START}assistant\n", False))
    return segs


def render(messages, tools=None, system=None, add_generation_prompt=False) -> str:
    """Render a (possibly raw) conversation to canonical ChatML text."""
    norm = normalize_messages(messages)
    return "".join(t for t, _ in iter_segments(
        norm, tools=tools, system=system, add_generation_prompt=add_generation_prompt))


def render_masked(messages, encode: Callable[[str], list[int]], tools=None,
                  system=None, eos_id: Optional[int] = None) -> tuple[list[int], list[int]]:
    """Encode a conversation to (ids, mask); mask=1 on supervised (assistant) tokens.

    ``encode`` maps text -> token ids (special markers become their atomic ids if
    the tokenizer reserves them, else ordinary subwords — masking is correct
    either way, since it aligns per segment).
    """
    norm = normalize_messages(messages)
    ids: list[int] = []
    mask: list[int] = []
    for text, sup in iter_segments(norm, tools=tools, system=system):
        piece = encode(text)
        ids.extend(piece)
        mask.extend([1 if sup else 0] * len(piece))
    if eos_id is not None:
        ids.append(eos_id)
        mask.append(0)
    return ids, mask
