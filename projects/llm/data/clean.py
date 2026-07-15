"""Text normalization for the pretraining corpus (esp. the prose/LAMBADA mix).

Applied at tokenization time via ``tokenize_source.render``. The guiding rule is
*standardize toward the eval distribution, not toward markdown*: LAMBADA is raw
narrative prose (BookCorpus-derived), so we clean noise that wastes model
capacity — inconsistent line endings, blank-line runs, publishing boilerplate —
without imposing any structural (markdown) formatting the eval never sees.

Two layers:
  1. ``normalize`` — universal, safe, applied to every text source: CR/CRLF ->
     LF, strip zero-width / non-breaking oddities, collapse 3+ blank lines to a
     paragraph break, trim leading/trailing whitespace.
  2. ``strip_boilerplate`` — line-level, source-scoped: drops publishing /
     front-matter cruft (copyright, Smashwords, cover-art credits, bare URLs,
     ISBNs, standalone date codes). Conservative — only whole lines that match a
     boilerplate pattern are dropped, never mid-sentence text.

Verify impact with ``python clean.py`` (before/after on the cached samples) and
quantify with a 1B raw-vs-clean A/B before committing to the 10B retokenize.
"""

import re
import unicodedata

# --- universal normalization ------------------------------------------------ #

# zero-width + BOM + other invisible junk that survives copy/paste from ebooks
_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")
# non-breaking / thin / figure spaces -> regular space
_NBSP = re.compile(r"[    ]")
_CRLF = re.compile(r"\r\n?")           # \r\n or lone \r -> \n
_TRAILING_WS = re.compile(r"[ \t]+(?=\n)")  # trailing spaces at line ends
_BLANKRUN = re.compile(r"\n{3,}")      # 3+ newlines -> paragraph break (\n\n)


def normalize(text: str) -> str:
    """Safe, source-agnostic cleanup applied to every text document."""
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _NBSP.sub(" ", text)
    text = _CRLF.sub("\n", text)
    text = _TRAILING_WS.sub("", text)
    text = _BLANKRUN.sub("\n\n", text)
    return text.strip()


# --- source-scoped boilerplate stripping ------------------------------------ #

# Whole-line patterns for publishing / front-matter cruft in ebook prose. Kept
# tight (anchored on distinctive tokens) so narrative lines are never dropped.
_BOILERPLATE = re.compile(
    r"""^\s*(
        .*copyright\s*(©|\(c\)).*                       # Text Copyright © 2018
      | .*all\s+rights\s+reserved.*
      | .*smashwords.*                                   # Smashwords Ebook Edition
      | .*cover\s+art\s+(created|by|design).*
      | this\s+ebook\s+is\s+licensed.*                   # standard Smashwords blurb
      | .*may\s+not\s+be\s+re-?sold.*
      | .*no\s+part\s+of\s+this\s+(book|publication)\s+may\s+be\s+reproduced.*
      | (isbn[- ]?(1[03])?[:\s].*)                       # ISBN lines
      | (https?://\S+|www\.\S+)                          # bare URL line
      | \d{6,10}                                         # standalone date/id code (e.g. 02212018)
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

# Sources that carry ebook publishing front-matter worth stripping.
_BOILERPLATE_SOURCES = {"bookcorpus", "gutenberg", "pg19"}


def strip_boilerplate(text: str) -> str:
    """Drop whole lines matching publishing/front-matter boilerplate patterns."""
    lines = [ln for ln in text.split("\n") if not _BOILERPLATE.match(ln)]
    return "\n".join(lines)


def clean_text(text: str, source_key: str | None = None) -> str:
    """Full cleaning pass: normalize always; strip boilerplate for prose sources.

    ``source_key`` scopes the line-level boilerplate filter (harmless but
    pointless on web/code); pass ``None`` for normalize-only.
    """
    if not text:
        return text
    text = normalize(text)
    if source_key in _BOILERPLATE_SOURCES:
        text = strip_boilerplate(text)
        text = _BLANKRUN.sub("\n\n", text).strip()  # re-collapse gaps left by drops
    return text


# --- dry-run reporter ------------------------------------------------------- #
if __name__ == "__main__":
    import os

    os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")
    import numpy as np

    from chimera.tokenizers import BPETokenizer

    tok = BPETokenizer.from_pretrained("LiquidAI/LFM2.5-230M")
    N = 6000
    for key in ["gutenberg", "bookcorpus", "pg19", "fineweb"]:
        p = f"/mnt/ai/data/llm-mix/tok/{key}/ids.bin"
        a = np.memmap(p, dtype=np.uint16, mode="r")
        raw = tok.decode(a[:N].tolist())
        clean = clean_text(raw, key)
        rt = len(tok._tok.encode(raw, add_special_tokens=False).ids)
        ct = len(tok._tok.encode(clean, add_special_tokens=False).ids)
        print("=" * 25, key, "=" * 25)
        print(f"  chars {len(raw):>6} -> {len(clean):>6}   "
              f"tokens {rt:>5} -> {ct:>5}  ({100*(rt-ct)/max(rt,1):+.1f}% tokens)")
        print(f"  CRLF {raw.count(chr(13)):>4} -> {clean.count(chr(13)):>3}   "
              f"blankruns {len(_BLANKRUN.findall(raw)):>3} -> {len(_BLANKRUN.findall(clean)):>3}")
        print("  --- after (first 400 chars) ---")
        print("  " + repr(clean[:400]))
        print()
