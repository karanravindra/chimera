"""Tools for *agentic* GRPO -- turning the policy from a pure reasoner into a tool user.

An agentic rollout interleaves the model's own tokens with tool calls: the model emits a call
like ``<calc>12*47</calc>``, the harness executes it and injects the result as an observation
``<result>564</result>``, and the model continues. Training is unchanged GRPO *provided the
injected observation tokens are masked out of the policy-gradient loss* (you never train the
policy to predict text it did not generate). This module is the backend-agnostic core: the
:class:`Tool` protocol, concrete tools, and the parse/format helpers a generation loop needs.
The generation engine (batched HF loop or vLLM) lives in the trainer and calls these.

Two tools ship here:

* :class:`CalculatorTool` -- safe arithmetic. The highest-leverage agentic change for a tiny
  model on math: a 230M model reasons acceptably but mis-*computes*; offloading arithmetic to
  an exact evaluator (Tool-Integrated Reasoning, cf. ToRA arXiv:2309.17452) removes the dominant
  error source. Reuses the regex-gated safe ``eval`` from :mod:`rewards` (no names/builtins).
* :class:`SearchTool` -- retrieval over a corpus (knowledge work / RAG, the Search-R1 setup).
  A retriever callable is injected so the corpus (per-example HotpotQA paragraphs, a BM25 index,
  or later a real wiki index / vLLM-served engine) is a plug-in, not baked in.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol, runtime_checkable

# An arithmetic expression of integer/decimal literals and + - * / ( ) only -- nothing else is
# ever passed to eval(). Mirrors the gate used by countdown_reward.
_CALC_OK = re.compile(r"^[\d+\-*/().\s]+$")


@runtime_checkable
class Tool(Protocol):
    """A callable tool. ``open_tag``/``close_tag`` delimit a call in the model's output; the
    harness extracts the inner text, runs :meth:`__call__`, and injects the returned string."""

    name: str
    open_tag: str
    close_tag: str

    def __call__(self, query: str) -> str: ...


class CalculatorTool:
    """Exact arithmetic via a sandboxed eval. ``<calc>2/3 + 1/6</calc>`` -> ``0.8333...``."""

    name = "calculator"
    open_tag = "<calc>"
    close_tag = "</calc>"

    def __call__(self, query: str) -> str:
        expr = query.strip()
        if not _CALC_OK.match(expr):
            return "error: only numbers and + - * / ( ) are allowed"
        try:
            value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - regex-gated, no names
        except (SyntaxError, ZeroDivisionError, TypeError, NameError, ValueError):
            return "error: could not evaluate"
        # present integers without a trailing .0; round long floats for a compact observation
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, float):
            value = round(value, 6)
        return str(value)


class SearchTool:
    """Retrieval tool. ``retriever(query, k) -> list[str]`` is injected (BM25, dense, web...)."""

    name = "search"
    open_tag = "<search>"
    close_tag = "</search>"

    def __init__(self, retriever: Callable[[str, int], list[str]], *, top_k: int = 2):
        self.retriever = retriever
        self.top_k = top_k

    def __call__(self, query: str) -> str:
        hits = self.retriever(query.strip(), self.top_k)
        return " ".join(hits)


def find_tool_call(text: str, tools: list[Tool]) -> tuple[Tool, str] | None:
    """Return the (tool, inner_query) for the FIRST closed tool call in ``text``, else ``None``.

    "First" by close-tag position so a generation stopped at a stop-string yields exactly the
    call the model just completed. Robust to multiple tool types in one rollout.
    """
    best: tuple[int, Tool, str] | None = None
    for tool in tools:
        m = re.search(re.escape(tool.open_tag) + r"(.*?)" + re.escape(tool.close_tag), text, re.DOTALL)
        if m and (best is None or m.end() < best[0]):
            best = (m.end(), tool, m.group(1))
    if best is None:
        return None
    return best[1], best[2]


def format_observation(result: str, *, tag: str = "information") -> str:
    """Wrap a tool result as an observation block to inject back into the context."""
    return f"<{tag}>{result}</{tag}>"


# The answer the rollout extracts as the model's final response (graded by the task reward).
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(text: str) -> str | None:
    """Pull the final ``<answer>...</answer>`` content from a completed rollout, if present."""
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else None
