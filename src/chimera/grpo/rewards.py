"""Verifiable-reward primitives for math tasks.

GRPO only needs a *cheap, automatic* score per completion -- no reward model. For
grade-school math the ground truth is a single number, so we parse the model's final answer
and compare it numerically to the gold answer. Two reward signals are provided:

* :func:`correctness_reward` -- the learning signal: 1.0 for the right number, else 0.0.
* :func:`format_reward` -- a small shaping signal: a bonus for ending with a parseable
  ``#### <number>`` line, which makes :func:`extract_final_answer` reliable and nudges the
  model toward a consistent, gradeable output.

These are plain ``(completion: str, gold: str) -> float`` functions so they are trivial to
unit-test and to compose (the trainer sums them with per-function weights).
"""

from __future__ import annotations

import re

# A signed number with optional thousands separators / decimals, e.g. "-1,234.5".
_NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?"
# The GSM8K answer convention: the final answer follows a "####" marker.
_HASH_ANSWER = re.compile(r"####\s*(" + _NUMBER + r")")
_ANY_NUMBER = re.compile(_NUMBER)


def _normalize_number(text: str) -> float | None:
    """Parse a human-written number into a float, tolerating ``$``, commas, and whitespace.

    Returns ``None`` if ``text`` does not parse as a number.
    """
    cleaned = text.strip().replace(",", "").replace("$", "").rstrip(".")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_final_answer(text: str) -> float | None:
    """Extract the model's final numeric answer from a completion (or a gold answer string).

    Prefers the number after the last ``####`` marker (the GSM8K convention, and what
    :func:`format_reward` encourages). Falls back to the **last** number anywhere in the
    text, since a correct chain-of-thought usually ends on the answer. Returns ``None`` when
    no number is present.
    """
    hash_matches = _HASH_ANSWER.findall(text)
    if hash_matches:
        return _normalize_number(hash_matches[-1])
    any_matches = _ANY_NUMBER.findall(text)
    if any_matches:
        return _normalize_number(any_matches[-1])
    return None


def correctness_reward(completion: str, gold: str) -> float:
    """1.0 if the completion's final number equals the gold number, else 0.0.

    ``gold`` may be a bare number or a full GSM8K answer string (``... #### 72``); both parse
    via :func:`extract_final_answer`. Comparison is numeric with a tiny tolerance so
    ``72``, ``72.0`` and ``72`` all match.
    """
    predicted = extract_final_answer(completion)
    target = extract_final_answer(gold)
    if predicted is None or target is None:
        return 0.0
    return 1.0 if abs(predicted - target) < 1e-6 else 0.0


# --- countdown (generate-and-check arithmetic) --------------------------------------------
# The Countdown / Game-of-24 reward is the gold standard for a *verifiable* objective: there
# is no stored answer to memorize and the random-guess floor is ~0, so a tiny model cannot
# reward-hack it. Gold is encoded as "<target>|<n1>,<n2>,...": the model must write an
# arithmetic expression that uses each given number exactly once and evaluates to the target.

# An expression of integer literals, + - * / ( ), spaces -- nothing else may be eval'd.
_EXPR_OK = re.compile(r"^[\d+\-*/().\s]+$")
_INT_LITERAL = re.compile(r"\d+")


def safe_eval(expr: str) -> float | int | None:
    """Evaluate a pure-arithmetic expression in a sandbox, or return ``None``.

    The single safe-eval gate shared by :func:`countdown_reward` and the calculator tool:
    ``expr`` is accepted only if it matches :data:`_EXPR_OK` (integer/decimal literals and
    ``+ - * / ( )`` -- no names, calls, or builtins), then evaluated with an empty
    ``__builtins__``. Returns the numeric result, or ``None`` if the gate rejects ``expr`` or
    evaluation fails (syntax, division by zero, type/name/value errors).
    """
    if not _EXPR_OK.match(expr):
        return None
    try:
        return eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - sandboxed: regex-gated, no names
    except (SyntaxError, ZeroDivisionError, TypeError, NameError, ValueError):
        return None


def _extract_expression(completion: str) -> str | None:
    """Pull the candidate arithmetic expression from a completion.

    Prefers the text after the last ``Answer:`` marker (the requested format); otherwise the
    last line that looks like a bare arithmetic expression. An ``expr = value`` form is
    truncated to the left-hand side so we evaluate the model's own arithmetic.
    """
    text = completion
    if "Answer:" in text:
        text = text.rsplit("Answer:", 1)[1]
    # take the first line of the candidate region, drop a trailing "= 24"
    line = text.strip().splitlines()[0] if text.strip() else ""
    if "=" in line:
        line = line.split("=", 1)[0]
    line = line.strip()
    if line and _EXPR_OK.match(line):
        return line
    # fallback: scan lines bottom-up for the longest arithmetic substring (tolerating prose
    # like "The expression 50 * 2 - 2 = 98" by dropping a trailing "= value" and any words).
    for cand in reversed(completion.strip().splitlines()):
        c = cand.split("=", 1)[0]
        runs = re.findall(r"[\d+\-*/().\s]+", c)
        runs = [r.strip() for r in runs if any(op in r for op in "+-*/") and any(ch.isdigit() for ch in r)]
        if runs:
            return max(runs, key=len)
    return None


def countdown_reward(completion: str, gold: str) -> float:
    """1.0 iff the completion's expression uses each given number once and hits the target.

    ``gold`` is ``"<target>|<n1>,<n2>,..."``. The expression is safe-evaluated (only integer
    literals and ``+ - * / ( )`` are permitted -- no names, calls, or builtins), then checked
    for (a) numeric equality to the target and (b) using the multiset of input numbers exactly.
    """
    try:
        target_str, nums_str = gold.split("|")
        target = float(target_str)
        nums = sorted(int(n) for n in nums_str.split(",") if n.strip())
    except (ValueError, AttributeError):
        return 0.0
    expr = _extract_expression(completion)
    if expr is None or not _EXPR_OK.match(expr):
        return 0.0
    used = sorted(int(n) for n in _INT_LITERAL.findall(expr))
    if used != nums:  # must use each number exactly once, no extras
        return 0.0
    value = safe_eval(expr)
    if value is None:
        return 0.0
    return 1.0 if abs(float(value) - target) < 1e-6 else 0.0


def format_reward(completion: str, gold: str) -> float:  # noqa: ARG001 (gold unused by design)
    """0.1 if the completion contains a parseable ``#### <number>`` line, else 0.0.

    A small shaping bonus, independent of correctness, that rewards emitting the gradeable
    answer format. ``gold`` is accepted (and ignored) so every reward function shares one
    ``(completion, gold)`` signature.
    """
    return 0.1 if _HASH_ANSWER.search(completion) else 0.0
