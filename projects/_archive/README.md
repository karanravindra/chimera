# Archived projects and removed library support

The projects in this directory are historical, read-only snapshots. They are kept for
results and implementation reference, but active code must not import or run them.

## Library cleanup manifest

The active project moved to `projects/tinylm`, so the following library code was
removed because its only consumers were archived projects. This manifest is the source
of truth for restoring an archived experiment later.

### Data modules

| removed file | public API | archived consumer |
|---|---|---|
| `src/chimera/data/afhq.py` | `AFHQDataModule` | `afhq/autoencoder` |
| `src/chimera/data/cifar10.py` | `CIFAR10DataModule` | `cifar10/classify`, `cifar10/autoencoder` |
| `src/chimera/data/clevr.py` | `CLEVRVQADataModule` | `clevr/vqa` |
| `src/chimera/data/mixture.py` | `MixtureDataModule` | `llm/gpt`, `llm/sft`, `tiny-llm/gpt` |
| `src/chimera/data/mnist.py` | `MNISTDataModule` | `mnist/classifier`, `mnist/autoencoder` |
| `src/chimera/data/mnist_latents.py` | `MNISTLatentDataModule` | `mnist/rectified_flow` |

Their imports and `__all__` entries were also removed from
`src/chimera/data/__init__.py`.

### Models

| removed file | public API | archived consumer |
|---|---|---|
| `src/chimera/models/cifar_autoencoder.py` | `CIFARAutoencoder` | `cifar10/autoencoder` |
| `src/chimera/models/clevr_vqa.py` | `CLEVRVQAModel` | `clevr/vqa` |
| `src/chimera/models/digit_dreamer.py` | `DigitDreamer` | `mnist/rectified_flow` |
| `src/chimera/models/digit_dreamer_ae.py` | `DigitDreamerAE` | `mnist/autoencoder`, `mnist/rectified_flow` |
| `src/chimera/models/digit_net.py` | `DigitNet` | `mnist/classifier` |
| `src/chimera/models/gpt.py` | `GPT` | `llm/gpt`, `llm/sft`, `tiny-llm/gpt` |
| `src/chimera/models/patchgan.py` | `PatchGANDiscriminator` | `afhq/autoencoder` |
| `src/chimera/models/pet_palette_ae.py` | `PetPaletteAE` | `afhq/autoencoder` |
| `src/chimera/models/resnet.py` | `ResNet` | `cifar10/classify` |

Their imports and `__all__` entries were also removed from
`src/chimera/models/__init__.py`. The active tinylm architecture remains project-local
at `projects/tinylm/pretrain/model.py`.

### Lightning modules and scheduler

The entire legacy `src/chimera/modules` package was archive-only and was removed:

- `__init__.py`
- `adversarial_autoencoder.py`
- `autoencoder.py`
- `classifier.py`
- `language_model.py`
- `rectified_flow.py`
- `vqa.py`

`src/chimera/optim/linear_warmup_cosine_annealing_lr.py` and its
`LinearWarmupCosineAnnealingLR` export were also removed. Active tinylm training owns
its raw PyTorch loop and learning-rate schedule.

### Direct dependencies

The following direct requirements were removed from `pyproject.toml` because their
only source consumers disappeared:

- `lpips`
- `mup`
- `torch-fidelity`
- `torchmetrics[image]`

`torchmetrics` may remain in the lockfile transitively through Lightning.
Regenerating `uv.lock` also removed `seaborn`, which was only a transitive dependency
of the removed packages.

## Recovery

Find the last revision containing a removed path:

```sh
git log --all -- src/chimera/models/gpt.py
```

Restore only the files required by the archived project, using the commit immediately
before this cleanup:

```sh
git restore --source=<commit-before-cleanup> -- \
  src/chimera/models/gpt.py \
  src/chimera/modules/language_model.py \
  src/chimera/optim/linear_warmup_cosine_annealing_lr.py \
  src/chimera/data/mixture.py
```

Then restore the corresponding imports in the package `__init__.py` files and add only
the dependencies that project needs. For example:

```sh
uv add mup
uv lock
```

Prefer restoring into a dedicated branch or worktree. Do not restore an old
`pyproject.toml` or package `__init__.py` wholesale after newer active work has landed;
merge the needed entries instead.
