"""Text data sources, compilation, artifacts, and runtime APIs."""

from importlib import import_module

_EXPORTS = {
    "TextDataModule": ("datamodule", "TextDataModule"),
    "TextMixtureSpec": ("datamodule", "TextMixtureSpec"),
    "MixtureSource": ("datamodule", "MixtureSource"),
    "TokenizerSpec": ("datamodule", "TokenizerSpec"),
    "Packed": ("datamodule", "Packed"),
    "DocumentWindow": ("datamodule", "DocumentWindow"),
    "ChatSFTDataModule": ("chat_sft", "ChatSFTDataModule"),
    "CoQAChatDataModule": ("coqa_chat", "CoQAChatDataModule"),
    "EverydayConversationsDataModule": (
        "everyday_conversations",
        "EverydayConversationsDataModule",
    ),
    "GooAQChatDataModule": ("gooaq_chat", "GooAQChatDataModule"),
    "NoRobotsChatDataModule": ("no_robots", "NoRobotsChatDataModule"),
    "QuACChatDataModule": ("quac_chat", "QuACChatDataModule"),
    "SODAChatDataModule": ("soda_chat", "SODAChatDataModule"),
    "SQuADChatDataModule": ("squad_chat", "SQuADChatDataModule"),
    "ConcatTextDataModule": ("concat_text", "ConcatTextDataModule"),
    "ContextMixDataModule": ("context_mix", "ContextMixDataModule"),
    "CoQADataModule": ("coqa", "CoQADataModule"),
    "CosmopediaV2DataModule": ("cosmopedia_v2", "CosmopediaV2DataModule"),
    "FineWebEduTextDataModule": ("fineweb_edu", "FineWebEduTextDataModule"),
    "GooAQDataModule": ("gooaq", "GooAQDataModule"),
    "HFTextDataModule": ("hf_text", "HFTextDataModule"),
    "LocalDocumentsDataModule": ("local_documents", "LocalDocumentsDataModule"),
    "SQuADTextDataModule": ("squad", "SQuADTextDataModule"),
    "StackExchangeDataModule": ("stackexchange", "StackExchangeDataModule"),
    "TinyStoriesV2DataModule": ("tinystories_v2", "TinyStoriesV2DataModule"),
    "TinyStrangeTextbooksDataModule": (
        "tiny_strange_textbooks",
        "TinyStrangeTextbooksDataModule",
    ),
    "TinyTextbooksDataModule": ("tiny_textbooks", "TinyTextbooksDataModule"),
    "TinyWebTextDataModule": ("tiny_webtext", "TinyWebTextDataModule"),
    "WikipediaDataModule": ("wikipedia", "WikipediaDataModule"),
    "MaskedTokenDataset": ("datasets", "MaskedTokenDataset"),
    "TokenDataset": ("datasets", "TokenDataset"),
    "WindowSampledDataset": ("datasets", "WindowSampledDataset"),
    "window_worker_init_fn": ("datasets", "window_worker_init_fn"),
    "PackedArtifactDataset": ("artifacts", "PackedArtifactDataset"),
    "DocumentWindowArtifactDataset": (
        "artifacts",
        "DocumentWindowArtifactDataset",
    ),
    "ShardedTokenStore": ("artifacts", "ShardedTokenStore"),
    "TextExample": ("schema", "TextExample"),
    "TextSegment": ("schema", "TextSegment"),
    "get_source": ("catalog", "get_source"),
    "get_view": ("catalog", "get_view"),
    "LocalTextView": ("catalog", "LocalTextView"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted([*globals(), *__all__])
