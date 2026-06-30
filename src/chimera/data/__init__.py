"""Data loading utilities.

The image DataModules are eagerly exported below. The text and vision helpers are
imported explicitly to avoid pulling heavy optional dependencies (``datasets``/
``tiktoken`` for text, ``torchvision`` extras for vision) unless used:

    from chimera.data.text import StreamingTextDataModule
    from chimera.data.vision import get_qmnist_blob
"""

from chimera.data.mnist import MNISTDataModule
from chimera.data.cifar10 import CIFAR10DataModule
from chimera.data.cifar100 import CIFAR100DataModule
from chimera.data.afhq import AFHQDataModule
from chimera.data.celeba_hq import CelebAHQDataModule
from chimera.data.imagenet import ImageNetDataModule
from chimera.data.base import ConcatImageDataModule
from chimera.data.augment import ReconstructionAugment

__all__ = [
    "MNISTDataModule",
    "CIFAR10DataModule",
    "CIFAR100DataModule",
    "AFHQDataModule",
    "CelebAHQDataModule",
    "ImageNetDataModule",
    "ConcatImageDataModule",
    "ReconstructionAugment",
]
