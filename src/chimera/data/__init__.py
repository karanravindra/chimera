from .celebahq import CelebAHQDataModule
from .chat_sft import (
    ChatSFTDataModule,
    CoQAChatDataModule,
    EverydayConversationsDataModule,
    GooAQChatDataModule,
    NoRobotsChatDataModule,
    QuACChatDataModule,
    SODAChatDataModule,
    SQuADChatDataModule,
)
from .cifar100 import CIFAR100DataModule
from .concat_text import ConcatTextDataModule
from .context_mix import ContextMixDataModule
from .coqa import CoQADataModule
from .cosmopedia_v2 import CosmopediaV2DataModule
from .fineweb_edu import FineWebEduDataModule
from .fineweb_edu_text import FineWebEduTextDataModule
from .gooaq import GooAQDataModule
from .imagenet1k import ImageNet1kDataModule
from .local_documents import LocalDocumentsDataModule
from .squad_text import SQuADTextDataModule
from .stackexchange import StackExchangeDataModule
from .text8 import Text8DataModule
from .tinyshakespeare import TinyShakespeareDataModule
from .tiny_strange_textbooks import TinyStrangeTextbooksDataModule
from .tiny_textbooks import TinyTextbooksDataModule
from .tiny_webtext import TinyWebTextDataModule
from .tinystories_v2 import TinyStoriesV2DataModule
from .ultrachat import UltraChatDataModule
from .wikipedia import WikipediaDataModule

__all__ = [
    "CIFAR100DataModule",
    "CelebAHQDataModule",
    "ChatSFTDataModule",
    "CoQAChatDataModule",
    "EverydayConversationsDataModule",
    "GooAQChatDataModule",
    "NoRobotsChatDataModule",
    "QuACChatDataModule",
    "SODAChatDataModule",
    "SQuADChatDataModule",
    "ConcatTextDataModule",
    "ContextMixDataModule",
    "CoQADataModule",
    "CosmopediaV2DataModule",
    "FineWebEduDataModule",
    "FineWebEduTextDataModule",
    "GooAQDataModule",
    "ImageNet1kDataModule",
    "LocalDocumentsDataModule",
    "SQuADTextDataModule",
    "StackExchangeDataModule",
    "Text8DataModule",
    "TinyShakespeareDataModule",
    "TinyStoriesV2DataModule",
    "TinyStrangeTextbooksDataModule",
    "TinyTextbooksDataModule",
    "TinyWebTextDataModule",
    "UltraChatDataModule",
    "WikipediaDataModule",
]
