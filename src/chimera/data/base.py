"""Shared base for the image DataModules.

The baseline bottleneck (benchmarked in ``benchmarks/dataloading.py``) is *not*
disk IO -- torchvision already loads MNIST/CIFAR fully into RAM -- it is the
per-sample PIL round-trip its ``__getitem__`` runs for every image. Killing
that is the whole win: ~8k -> ~100k+ img/s on the benchmark machine, well past
the point where data loading bounds training.

Design (chosen for speed *and* drop-in compatibility):

  * **Materialize once, to disk.** Each split is pre-stacked into one contiguous
    uint8 NCHW ``.npy`` artifact under ``{data_dir}/.chimera_cache`` (built once
    in ``prepare_data``). A sample is then a plain index into it -- no PIL, no
    per-sample IO. The artifact doubles as a cache: later runs skip the
    (potentially slow) materialization entirely.
  * **RAM vs mmap, by flag.** ``in_memory=True`` (default) loads the artifact
    fully into RAM -- one sequential read, then pure-RAM random access every
    epoch (best for slow storage / shuffled training, and unchanged from the
    original behavior). ``in_memory=False`` ``np.load(mmap_mode="r")``s it --
    ~0 resident RAM, random access served by the OS page cache (best on fast
    local SSD, or when a split is larger than RAM). The flag is the RAM<->IO
    knob, chosen for where the data lives and how big it is.
  * **Batched cast.** The uint8 -> bf16/255 scale runs once per batch in a
    ``collate_fn`` (parallelized across workers when ``num_workers > 0``), so the
    loader still yields ready-to-use bf16 [0, 1] tensors -- the same output
    contract as the original PIL pipeline. Code that iterates a loader directly
    (e.g. ``extract_latents``) or reads ``.dataset.data`` / ``.dataset.targets``
    keeps working unchanged.

For augmentation-heavy pipelines (DINO), per-sample CPU work is the real cost,
so callers pass a ``gpu_transform``: the loader then ships raw uint8 batches and
the transform runs *batched on the GPU* in ``on_after_batch_transfer`` (see
:class:`chimera.trainers.dino.GPUDINOAugmentation`).
"""

import os
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningDataModule
from PIL import Image
from torch import bfloat16
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive


def to_bf16_scaled(x: torch.Tensor) -> torch.Tensor:
    """uint8 image batch -> bf16 scaled to [0, 1]."""
    return x.to(bfloat16).div_(255)


def _to_nchw_uint8(tv_dataset) -> torch.Tensor:
    """Stack a torchvision MNIST/CIFAR split into a contiguous uint8 NCHW tensor."""
    data = tv_dataset.data
    if isinstance(data, np.ndarray):  # CIFAR: HWC uint8 ndarray
        return torch.from_numpy(data).permute(0, 3, 1, 2).contiguous()
    return data.unsqueeze(1).contiguous()  # MNIST: [N,H,W] uint8 tensor


def _atomic_save(path: str, arr: np.ndarray) -> None:
    """``np.save`` to a temp file in the same dir, then atomically rename in.

    Guards against a half-written artifact if the (possibly slow) write is
    interrupted -- a partial file would otherwise be picked up as a valid cache.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npy"  # ends in .npy so np.save won't re-append
    np.save(tmp, arr)
    os.replace(tmp, path)


class FastImageDataset(Dataset):
    """Indexes a materialized uint8 NCHW store (no PIL, no per-sample IO).

    ``images`` is the contiguous uint8 ``[N, C, H, W]`` store -- either a torch
    tensor held in RAM or a ``np.memmap`` over the on-disk artifact. ``labels``
    are the int64 class indices. ``data`` / ``targets`` are optional pass-through
    proxies so existing code that reaches into ``.data`` / ``.targets`` (e.g.
    notebook visualizations on MNIST/CIFAR) keeps seeing the original torchvision
    layout; they default to ``None`` / the labels for datasets with no such
    consumer (avoiding a second full-resolution copy).
    """

    def __init__(self, images, labels, *, data=None, targets=None):
        self._images = images  # uint8 [N,C,H,W]: torch tensor (RAM) or np.memmap
        self._labels = torch.as_tensor(labels, dtype=torch.int64)
        self.data = data  # original torchvision layout, for callers (or None)
        self.targets = targets if targets is not None else self._labels

    @classmethod
    def from_torchvision(cls, tv_dataset) -> "FastImageDataset":
        """Build directly from a torchvision dataset, in RAM (no disk artifact)."""
        return cls(
            _to_nchw_uint8(tv_dataset),
            tv_dataset.targets,
            data=tv_dataset.data,
            targets=tv_dataset.targets,
        )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, i: int):
        img = self._images[i]
        if isinstance(img, np.ndarray):  # memmap row -> own, writable tensor
            img = torch.from_numpy(img.copy())
        return img, self._labels[i]


def _collate_uint8(samples):
    """Stack uint8 samples into a raw uint8 NCHW batch (for GPU-side transforms)."""
    xs = torch.stack([s[0] for s in samples])
    ys = torch.stack([s[1] for s in samples])
    return xs, ys


def _collate_bf16(samples):
    """Stack and cast the batch to bf16/255 -- ready-to-use loader output."""
    xs, ys = _collate_uint8(samples)
    return to_bf16_scaled(xs), ys


class ImageDataModule(LightningDataModule):
    """Materialized-uint8 DataModule for torchvision image datasets.

    Subclasses set :attr:`dataset_cls` (e.g. ``datasets.MNIST``).

    Each split is precomputed once into an on-disk uint8 NCHW ``.npy`` artifact
    (see the module docstring) and then loaded either fully into RAM
    (``in_memory=True``, default) or memory-mapped (``in_memory=False``).

    Without ``gpu_transform`` the loader yields ready-to-use bf16 [0, 1] NCHW
    batches (batched cast in the collate). With a ``gpu_transform`` the loader
    ships raw uint8 batches and the transform runs batched on-device in
    ``on_after_batch_transfer`` -- used by DINO to do its multi-view
    augmentation on the GPU.
    """

    dataset_cls: type = None

    def __init__(
        self,
        data_dir: str = "/mnt/ai/data",
        batch_size: int = 64,
        num_workers: int = 0,
        gpu_transform: Callable[[torch.Tensor], object] | None = None,
        pin_memory: bool | None = None,
        in_memory: bool = True,
        augment_eval: bool = True,
        image_size: int | None = None,
        drop_last: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gpu_transform = gpu_transform
        # When set, each split is resized to image_size x image_size once at
        # materialization time (and keyed into the cache), so loaders yield that
        # resolution directly -- no per-batch resize in the training loop.
        self.image_size = image_size
        # Ship uint8 (cast/augment on the GPU) only when a gpu_transform is set;
        # otherwise cast in the collate so the loader yields bf16 directly.
        self._collate = _collate_uint8 if gpu_transform else _collate_bf16
        # Pinning only helps async H2D when workers prefetch; off for the nw=0
        # path where it is pure overhead.
        self.pin_memory = pin_memory if pin_memory is not None else num_workers > 0
        # Load the precomputed artifact fully into RAM, or mmap it -- see the
        # module docstring for the RAM<->IO tradeoff.
        self.in_memory = in_memory
        # When False, the gpu_transform (augmentation) is applied to *training*
        # batches only; val/test/predict batches get just the uint8->bf16 cast the
        # collate deferred to the GPU path. Lets reconstruction metrics see clean,
        # unaugmented images. Default True preserves the augment-everywhere behavior
        # DINO relies on for its self-supervised views.
        self.augment_eval = augment_eval
        # Drop the final uneven batch of every split (train/val/test). Off by default;
        # turned on for CUDA-graph torch.compile modes, where a variable last-batch shape
        # would force an extra graph capture -- so all loaders must yield one static shape.
        self.drop_last = drop_last

    # ---- cache layout + per-dataset hooks (overridden by StarGAN subclass) ----

    @property
    def _cache_key(self) -> str:
        """Identifier for the on-disk artifact dir (one per dataset[+size])."""
        name = self.dataset_cls.__name__
        # image_size changes the materialized tensor, so key on it (when set) to
        # regenerate the artifact rather than reuse a stale one; None keeps the
        # original key so existing caches stay valid.
        return name if self.image_size is None else f"{name}-{self.image_size}"

    def _cache_paths(self, split: str) -> tuple[str, str]:
        d = os.path.join(self.data_dir, ".chimera_cache", self._cache_key)
        stem = os.path.join(d, split)
        return f"{stem}.images.npy", f"{stem}.labels.npy"

    def _ensure_downloaded(self) -> None:
        # Download once on a single process; no state assigned here.
        self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)

    def _build_split(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Materialize one split into (uint8 NCHW images, int64 labels) arrays."""
        tv = self.dataset_cls(self.data_dir, train=(split == "train"))
        images = _to_nchw_uint8(tv)
        if self.image_size is not None:
            images = (
                F.interpolate(
                    images.float(),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
            )
        return images.numpy(), np.asarray(tv.targets, dtype=np.int64)

    def _proxy(self, split: str):
        """``(data, targets)`` proxy in the original torchvision layout.

        Cheap re-read of the already-on-disk originals; what MNIST/CIFAR notebook
        visualizations (and ``benchmarks/correctness.py``) reach into.
        """
        tv = self.dataset_cls(self.data_dir, train=(split == "train"))
        return tv.data, tv.targets

    # ---- lifecycle ----

    def prepare_data(self):
        # Single process: download, then precompute each split's artifact once.
        self._ensure_downloaded()
        for split in ("train", "test"):
            img_path, lbl_path = self._cache_paths(split)
            if not (os.path.exists(img_path) and os.path.exists(lbl_path)):
                images, labels = self._build_split(split)
                _atomic_save(img_path, images)
                _atomic_save(lbl_path, labels)

    def _load_split(self, split: str) -> FastImageDataset:
        img_path, lbl_path = self._cache_paths(split)
        images = np.load(img_path, mmap_mode=None if self.in_memory else "r")
        if self.in_memory:
            images = torch.from_numpy(images)  # one-time; rows then index as views
        data, targets = self._proxy(split)
        return FastImageDataset(images, np.load(lbl_path), data=data, targets=targets)

    def setup(self, stage: str):
        # The val loader is the test split (val == test by design here), so the
        # test set must also be loaded during fit/validate. Guarded so a manual
        # setup("fit") + setup("test") doesn't reload.
        if stage == "fit" and not hasattr(self, "train_set"):
            self.train_set = self._load_split("train")
        if stage in ("fit", "validate", "test") and not hasattr(self, "test_set"):
            self.test_set = self._load_split("test")

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            collate_fn=self._collate,
            drop_last=self.drop_last,  # applies to every split (train/val/test) when enabled
        )

    def use_single_process_loaders(self):
        """Make subsequently-created loaders worker-free (``num_workers=0``).

        Persistent dataloader workers spawned during training are torn down when a
        notebook cell is interrupted; reusing the datamodule afterwards (to extract
        features, compute rFID, etc.) would then hit "DataLoader worker exited
        unexpectedly". Training scripts call this before returning so post-training
        analysis loaders run in-process and are interrupt-safe. These datasets are
        RAM-resident, so num_workers=0 is plenty fast for a single eval pass.
        """
        self.num_workers = 0
        self.pin_memory = False
        return self

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        # No separate validation split exists; monitor on the test set.
        return self._loader(self.test_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_set, shuffle=False)

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        # Only does work in the gpu_transform path; otherwise the collate already
        # produced bf16 and this is a no-op.
        if self.gpu_transform is None:
            return batch
        x, y = batch
        if not self.augment_eval and not self.trainer.training:
            # Train-only augmentation: skip the transform on eval/predict batches,
            # but still apply the uint8->bf16 cast the collate left to this hook.
            return to_bf16_scaled(x), y
        return self.gpu_transform(x), y


class ConcatImageDataModule(ImageDataModule):
    """Train/eval over the *union* of several ImageDataModules.

    Each child owns its own download + uint8 cache (and resize, when it sets
    ``image_size``); this module just concatenates their per-split datasets at load time
    with ``ConcatDataset``. The bf16 collate, ``in_memory`` handling, ``pin_memory``, and
    the loaders all carry over unchanged from :class:`ImageDataModule`.

    Children must share spatial size and channel count so their samples stack into a batch
    -- construct them with the same ``image_size``. Labels are passed through unchanged;
    overlapping class ids across children are fine for label-free objectives (e.g. an
    autoencoder), which is the intended use.
    """

    def __init__(
        self,
        datamodules,
        *,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool | None = None,
        in_memory: bool = True,
    ):
        # No gpu_transform: the loader yields ready-to-use bf16 [0,1] batches via the collate.
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            in_memory=in_memory,
        )
        self.datamodules = list(datamodules)

    def prepare_data(self):
        # Single process: each child downloads + materializes its own cache.
        for dm in self.datamodules:
            dm.prepare_data()

    def setup(self, stage: str):
        for dm in self.datamodules:
            dm.setup(stage)
        # Mirror ImageDataModule's split policy (val == test) over the concatenated children.
        if stage == "fit" and not hasattr(self, "train_set"):
            self.train_set = ConcatDataset([dm.train_set for dm in self.datamodules])
        if stage in ("fit", "validate", "test") and not hasattr(self, "test_set"):
            self.test_set = ConcatDataset([dm.test_set for dm in self.datamodules])


def _load_imagefolder_nchw_uint8(root: str, image_size: int):
    """Load + resize an ImageFolder split into (uint8 NCHW images, int64 labels).

    Fills a preallocated NCHW array directly (per-image HWC->CHW transpose), so
    there is no full-array permute copy and no second NHWC array kept alive --
    important at ``image_size=256`` where each split is multiple GB.
    """
    folder = ImageFolder(root)
    n = len(folder.samples)
    images = np.empty((n, 3, image_size, image_size), dtype=np.uint8)
    labels = np.empty(n, dtype=np.int64)
    for i, (path, label) in enumerate(folder.samples):
        with Image.open(path) as img:
            img = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            images[i] = np.asarray(img).transpose(2, 0, 1)  # HWC -> CHW
        labels[i] = label
    return images, labels


class StarGANImageDataModule(ImageDataModule):
    """DataModule for the StarGAN v2 ImageFolder datasets (AFHQ, CelebA-HQ).

    These aren't in torchvision, so we replicate StarGAN v2's ``download.sh``:
    fetch the Dropbox zip into ``data_dir`` and unpack it into a
    ``{dirname}/{train,val}/<class>/*.jpg`` tree. Each split is then resized to
    ``image_size`` and materialized into the same uint8 NCHW ``.npy`` artifact as
    the other DataModules, so the bf16 collate, ``gpu_transform``, ``in_memory``
    flag, and loaders all carry over unchanged. No ``.data`` proxy is kept (no
    consumer reads the originals), which is what avoids a second full-res copy.

    Subclasses set :attr:`url`, :attr:`archive_name`, and :attr:`dirname`.
    """

    url: str = None
    archive_name: str = None
    dirname: str = None

    def __init__(
        self,
        data_dir: str = "/mnt/ai/data",
        batch_size: int = 64,
        num_workers: int = 0,
        gpu_transform: Callable[[torch.Tensor], object] | None = None,
        pin_memory: bool | None = None,
        in_memory: bool = True,
        augment_eval: bool = True,
        *,
        image_size: int = 256,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            gpu_transform=gpu_transform,
            pin_memory=pin_memory,
            in_memory=in_memory,
            augment_eval=augment_eval,
        )
        self.image_size = image_size

    @property
    def _cache_key(self) -> str:
        # image_size changes the materialized tensor, so key on it: changing the
        # size regenerates the artifact rather than reusing a stale one.
        return f"{self.dirname}-{self.image_size}"

    def _ensure_downloaded(self) -> None:
        # Download + unzip once, mirroring download.sh (wget + unzip + rm). The
        # `dl=1` suffix turns the Dropbox share link into a direct download; use
        # `&` when the URL already carries a query (the newer `/scl/fi/` links
        # ship an `?rlkey=...`), `?` otherwise (legacy `/s/` links).
        # Guard on the split dirs we actually read (not just the top dir) so a
        # partial/interrupted extraction re-downloads instead of looking done.
        target = os.path.join(self.data_dir, self.dirname)
        if not all(os.path.isdir(os.path.join(target, s)) for s in ("train", "val")):
            sep = "&" if "?" in self.url else "?"
            download_and_extract_archive(
                f"{self.url}{sep}dl=1",
                download_root=self.data_dir,
                filename=self.archive_name,
                remove_finished=True,
            )

    def _split_root(self, split: str) -> str:
        # StarGAN v2 ships {train,val}; our "test" split reads the val dir.
        sub = "train" if split == "train" else "val"
        return os.path.join(self.data_dir, self.dirname, sub)

    def _build_split(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        return _load_imagefolder_nchw_uint8(self._split_root(split), self.image_size)

    def image_paths(self, split: str) -> list[str]:
        """Ordered per-image relative paths for ``split`` (stable image ids).

        Mirrors ``ImageFolder``'s deterministic ``samples`` order -- the same order
        ``_build_split`` materializes into the uint8 cache and that loaders iterate
        with ``shuffle=False`` -- so ``image_paths(split)[i]`` names the i-th image
        a caption/feature extractor sees. Scans the dir tree only (no image decode).
        """
        root = self._split_root(split)
        folder = ImageFolder(root)
        return [os.path.relpath(path, root) for path, _ in folder.samples]

    def _proxy(self, split: str):
        # No .data consumer for these datasets, so skip the proxy (and its second
        # full-res copy); FastImageDataset falls back to the labels for .targets.
        return None, None
