# Chimera data

`chimera.data` is organized by modality. Shared source, split, locking, cache,
and CLI contracts live at this level; text-specific rendering, objectives,
artifacts, and runtime sampling live in `chimera.data.text`. The
`chimera.data.vision` package is intentionally dependency-free scaffolding for
the next modality.

## Source-of-truth hierarchy

1. [`text/catalog.py`](text/catalog.py) declares stable source keys, upstream
   repositories, logical split policy, license metadata, and reusable views.
1. [`text/catalog.lock.json`](text/catalog.lock.json) pins each Hugging Face
   repository to an immutable 40-character commit SHA. Training and sampling
   always pass that revision to `datasets.load_dataset`; its optional file list
   is the allowlist for larger alternate shard selections.
1. [`text/adapters.py`](text/adapters.py) and
   [`text/chat_template.py`](text/chat_template.py) are the canonical mapping
   from an upstream row to model text and supervision segments.
1. Every compiled artifact contains the source revision, file selection, view,
   adapter configuration, objective, tokenizer hash, token caps, special-token
   IDs, and shard size in its manifest. That descriptor determines its cache
   directory.

The Hugging Face repository is the operational online source. The `provenance`
URL in the catalog points to the original project or canonical upstream when a
Hugging Face dataset is a mirror. Updating an upstream source is explicit:

```bash
chimera-data text lock SOURCE_KEY
```

Review and commit the lockfile diff before rebuilding artifacts.

## Dataset inventory

One raw source can expose multiple views. For example, SQuAD is both passage
pretraining text and assistant-only ChatML SFT; the bytes and pinned revision
are shared, while adapters and objectives differ.

| Source key               | Locked online dataset                                                                                       | Views                                         | Split policy                                                        | License note                                           |
| ------------------------ | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------ |
| `tinystories-v2`         | [noanabeshima/TinyStoriesV2](https://huggingface.co/datasets/noanabeshima/TinyStoriesV2)                    | `tinystories-v2.pretrain`                     | native train/validation                                             | CDLA Sharing 1.0                                       |
| `tiny-textbooks`         | [nampdn-ai/tiny-textbooks](https://huggingface.co/datasets/nampdn-ai/tiny-textbooks)                        | `tiny-textbooks.pretrain`                     | train/test                                                          | Apache-2.0; catalog marks access as gated              |
| `tiny-strange-textbooks` | [nampdn-ai/tiny-strange-textbooks](https://huggingface.co/datasets/nampdn-ai/tiny-strange-textbooks)        | `tiny-strange-textbooks.pretrain`             | ordered 1% validation carve                                         | Apache-2.0; four selected shards; gated                |
| `tiny-webtext`           | [nampdn-ai/tiny-webtext](https://huggingface.co/datasets/nampdn-ai/tiny-webtext)                            | `tiny-webtext.pretrain`                       | ordered 1% validation carve                                         | MIT; English parquet selection; gated                  |
| `fineweb-edu`            | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)                      | `fineweb-edu.pretrain`                        | ordered 1% validation carve                                         | ODC-BY; bounded 10BT shard selection                   |
| `cosmopedia-v2`          | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)                  | `cosmopedia-v2.pretrain`                      | ordered 1% validation carve                                         | ODC-BY; bounded Cosmopedia v2 shards                   |
| `gooaq`                  | [sentence-transformers/gooaq](https://huggingface.co/datasets/sentence-transformers/gooaq)                  | `gooaq.pretrain`, `gooaq.sft`                 | ordered 1% validation carve                                         | Apache-2.0; selected pair shard                        |
| `squad`                  | [rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad)                                          | `squad.pretrain`, `squad.sft`                 | native train/validation                                             | CC-BY-SA-4.0                                           |
| `coqa`                   | [stanfordnlp/coqa](https://huggingface.co/datasets/stanfordnlp/coqa)                                        | `coqa.pretrain`, `coqa.sft`                   | pretrain uses native validation; SFT preserves the ordered 1% carve | Upstream-specific/other                                |
| `stackexchange`          | [donfu/oa-stackexchange](https://huggingface.co/datasets/donfu/oa-stackexchange)                            | `stackexchange.pretrain`                      | ordered 1% validation carve                                         | CC-BY-SA-4.0                                           |
| `wikipedia`              | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)                                  | `wikipedia.pretrain`                          | ordered 1% validation carve                                         | CC-BY-SA-3.0 and GFDL; selected English snapshot shard |
| `smoltalk`               | [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) (`everyday-conversations`) | `smoltalk.everyday.sft`                       | train/test                                                          | Verify upstream dataset card                           |
| `no-robots`              | [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)                          | `no-robots.sft`                               | train/test                                                          | CC-BY-NC-4.0; review before non-research use           |
| `quac`                   | [yairfeldman/quac](https://huggingface.co/datasets/yairfeldman/quac)                                        | `quac.sft`                                    | ordered 1% validation carve                                         | Verify upstream dataset card; mirror of QuAC           |
| `soda`                   | [allenai/soda](https://huggingface.co/datasets/allenai/soda)                                                | `soda.sft`                                    | ordered 1% validation carve                                         | CC-BY-4.0; synthetic dialog                            |
| `oasst1`                 | [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)                                | `oasst1.sft` (best-ranked English tree paths) | native train/validation                                             | Apache-2.0                                             |
| `ultrachat-200k`         | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)                | `ultrachat-200k.sft`                          | `train_sft`/`test_sft`                                              | MIT                                                    |

`LocalTextView` adds deterministic local files without pretending they are an
online catalog source. File names and contents are hashed into the artifact
descriptor; repetition applies only to training. Local views are excluded from
tokenizer training by default so adding project notes does not silently change
the shared vocabulary.

## How the pieces relate

```text
HF source + locked revision        local files + content hashes
             |                                  |
             +--------------+-------------------+
                            |
                    source-specific adapter
                            |
              TextExample[TextSegment(...)]
                            |
                  loss objective + tokenizer
                            |
              v3 manifest + document-aligned
                 mmap-able PyTorch shards
                            |
             packed or single-document windows
                            |
               TextDataModule mixture sampler
```

The canonical intermediate schema is `TextExample`: adapters own source schema
differences; objectives own label masking; sampling owns runtime window policy.
This keeps datasets out of the DataModule inheritance tree and makes adding a
new dataset a catalog-plus-adapter change.

## Runtime API

```python
from chimera.data.text import (
    MixtureSource,
    TextDataModule,
    TextMixtureSpec,
    TokenizerSpec,
)

dm = TextDataModule(
    TextMixtureSpec(
        sources=(
            MixtureSource("fineweb-edu.pretrain", weight=0.7),
            MixtureSource("tinystories-v2.pretrain", weight=0.3),
        ),
        tokenizer=TokenizerSpec.pinned("tokenizer.json"),
        add_bos=True,
    ),
    data_dir="/mnt/ai/data",
    batch_size=64,
    seq_len=512,
)
dm.prepare_data()
dm.setup("fit")
```

Explicit `weight` values control sampled item share. Without weights, source
share follows the number of compiled windows, preserving the old token-cap
mixture behavior. Validation is available as one combined loader and through
`val_dataloaders_by_source()`.

## Disk layout

For `data_dir=/mnt/ai/data`, new text state is isolated under:

```text
/mnt/ai/data/
  hf_cache/                         # datasets library downloads
  text/
    tokenizers/tokenizer-HASH.json
    artifacts/v3/VIEW/BUILD_HASH/
      manifest.json
      shard-00000.pt
      ...
```

Shards are document-aligned, checksummed, atomically published, and opened with
`torch.load(..., mmap=True)`. The manifest is written last. Version-2 caches are
not read or deleted; they can be removed separately after successful migration.

Useful commands:

```bash
chimera-data text list
chimera-data text inspect squad.sft
chimera-data text sample squad.sft --count 2
chimera-data text build squad.sft --tokenizer tokenizer.json
chimera-data text validate /mnt/ai/data/text/artifacts/v3/VIEW/HASH
```

## Imports

The supported module paths live under `chimera.data.text`; dataset-specific
compatibility modules at the `chimera.data` package root have been removed.
Public classes remain available as lazy attributes of `chimera.data`, while new
code should use catalog views and `TextDataModule`.
