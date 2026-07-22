# Chimera

Chimera is the reusable Python layer behind the active TinyLM project. It keeps
the training surface intentionally small: text/chat data modules, byte-level BPE
tokenization, attention/RoPE/LoRA components, Muon, and optional language-model
evaluation.

## Install

```bash
uv sync
```

The default environment contains everything needed by the TinyLM training
scripts. Add only the tools needed for a particular workflow:

```bash
uv sync --extra eval       # lm-eval, pandas, transformers
uv sync --extra notebook   # Jupyter, plotting, Ollama client
```

## Public API

```python
from chimera.data.text import TextDataModule, TextMixtureSpec
from chimera.optim import Muon, muon_param_groups
from chimera.tokenizers import BPETokenizer
```

Package exports are lazy, so importing `chimera.data`, `chimera.models`, or
`chimera.evals` does not initialize unrelated frameworks. Token caches are
versioned, configuration-keyed, and written atomically. Build them in
`prepare_data()`; `setup()` only reads completed caches, which is safe under DDP.
The complete dataset inventory, upstream source-of-truth policy, and migration
guide live in [`src/chimera/data/README.md`](src/chimera/data/README.md).

The runnable model work lives in [`projects/tinylm`](projects/tinylm). Reusable
legacy modules that may return later are documented in
[`projects/_archive/chimera`](projects/_archive/chimera/README.md).
