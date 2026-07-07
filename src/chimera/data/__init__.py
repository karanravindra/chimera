from .afhq import AFHQDataModule
from .celebahq import CelebAHQDataModule
from .cifar10 import CIFAR10DataModule
from .cifar100 import CIFAR100DataModule
from .fineweb_edu import FineWebEduDataModule
from .imagenet1k import ImageNet1kDataModule
from .mnist import MNISTDataModule
from .text8 import Text8DataModule
from .tinyshakespeare import TinyShakespeareDataModule

__all__ = [
    "AFHQDataModule",
    "CIFAR10DataModule",
    "CIFAR100DataModule",
    "CelebAHQDataModule",
    "FineWebEduDataModule",
    "ImageNet1kDataModule",
    "MNISTDataModule",
    "Text8DataModule",
    "TinyShakespeareDataModule",
]
