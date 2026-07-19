from .afhq import AFHQDataModule
from .celebahq import CelebAHQDataModule
from .cifar10 import CIFAR10DataModule
from .chat_sft import (
    ChatSFTDataModule,
    EverydayConversationsDataModule,
    GooAQChatDataModule,
    SQuADChatDataModule,
)
from .cifar100 import CIFAR100DataModule
from .clevr import CLEVRVQADataModule
from .concat_text import ConcatTextDataModule
from .cosmopedia_v2 import CosmopediaV2DataModule
from .fineweb_edu import FineWebEduDataModule
from .fineweb_edu_text import FineWebEduTextDataModule
from .gooaq import GooAQDataModule
from .imagenet1k import ImageNet1kDataModule
from .local_documents import LocalDocumentsDataModule
from .mixture import MixtureDataModule
from .mnist import MNISTDataModule
from .mnist_latents import MNISTLatentDataModule
from .squad_text import SQuADTextDataModule
from .text8 import Text8DataModule
from .tinyshakespeare import TinyShakespeareDataModule
from .tiny_strange_textbooks import TinyStrangeTextbooksDataModule
from .tiny_textbooks import TinyTextbooksDataModule
from .tiny_webtext import TinyWebTextDataModule
from .tinystories_v2 import TinyStoriesV2DataModule
from .ultrachat import UltraChatDataModule

__all__ = [
    "AFHQDataModule",
    "CIFAR10DataModule",
    "CIFAR100DataModule",
    "CLEVRVQADataModule",
    "CelebAHQDataModule",
    "ChatSFTDataModule",
    "EverydayConversationsDataModule",
    "GooAQChatDataModule",
    "SQuADChatDataModule",
    "ConcatTextDataModule",
    "CosmopediaV2DataModule",
    "FineWebEduDataModule",
    "FineWebEduTextDataModule",
    "GooAQDataModule",
    "ImageNet1kDataModule",
    "LocalDocumentsDataModule",
    "MixtureDataModule",
    "MNISTDataModule",
    "MNISTLatentDataModule",
    "SQuADTextDataModule",
    "Text8DataModule",
    "TinyShakespeareDataModule",
    "TinyStoriesV2DataModule",
    "TinyStrangeTextbooksDataModule",
    "TinyTextbooksDataModule",
    "TinyWebTextDataModule",
    "UltraChatDataModule",
]
