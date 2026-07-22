# Archived chimera modules

These files were removed from the installed `chimera` package during the TinyLM
cleanup. They are kept here as source material because they may be useful again;
they are not maintained, tested, or importable from the current package.
The superseded root `SPEC.md` is archived alongside them for historical context.

| Archived area | Former import paths | Dependencies to restore |
| --- | --- | --- |
| Lightning run harness | `chimera.train`, `chimera.utils.ema`, `chimera.utils.loggers`, `chimera.utils.device` | `lightning`, `tyro`, `wandb` |
| Toy models | `chimera.models.rnn`, `chimera.models.lstm`, `chimera.models.vgg` | `torch` |
| Vision data | `chimera.data.cifar100`, `chimera.data.celebahq`, `chimera.data.imagenet1k`, `chimera.data._image_cache` | `datasets`, `lightning`, `torchvision`, Pillow |
| Standalone text demos | `chimera.data.text8`, `chimera.data.tinyshakespeare` | `lightning` |

To restore an area, move its files back under `src/chimera`, re-add deliberate
lazy exports, restore only its required dependency extra, and move its archived
tests back into `tests/`. Treat the code as a starting point: active cache and
tokenizer APIs may have changed since archival.

Before restoring the Lightning harness, fix its EMA checkpoint contract: the
archived callback replaces checkpoint model weights with EMA weights while the
optimizer moments still describe the raw weights. A resumable checkpoint should
keep the raw model/optimizer pair aligned and store EMA as separate state; export
an EMA-only inference artifact separately.

The older `fineweb_edu.py` and `ultrachat.py` implementations are intentionally
not archived. `fineweb_edu_text.py` and `chat_sft.py` are their supported
replacements, and Git history remains the recovery path for the superseded code.
