from .afhq import AFHQDataModule
from .celebahq import CelebAHQDataModule
from .cifar10 import CIFAR10DataModule
from .cifar100 import CIFAR100DataModule
from .clevr import CLEVRVQADataModule
from .fineweb_edu import FineWebEduDataModule
from .imagenet1k import ImageNet1kDataModule
from .mnist import MNISTDataModule
from .mnist_latents import MNISTLatentDataModule
from .text8 import Text8DataModule
from .tinyshakespeare import TinyShakespeareDataModule

__all__ = [
    "AFHQDataModule",
    "CIFAR10DataModule",
    "CIFAR100DataModule",
    "CLEVRVQADataModule",
    "CelebAHQDataModule",
    "FineWebEduDataModule",
    "ImageNet1kDataModule",
    "MNISTDataModule",
    "MNISTLatentDataModule",
    "Text8DataModule",
    "TinyShakespeareDataModule",
]
