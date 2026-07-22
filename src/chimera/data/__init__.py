"""TinyLM data API with lazy imports for optional runtime dependencies."""

from importlib import import_module

_EXPORTS = {
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
    "MaskedTokenDataset": ("_text", "MaskedTokenDataset"),
    "TokenDataset": ("_text", "TokenDataset"),
    "WindowSampledDataset": ("_text", "WindowSampledDataset"),
    "window_worker_init_fn": ("_text", "window_worker_init_fn"),
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
