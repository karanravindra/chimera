"""TinyLM data API with lazy imports for optional runtime dependencies."""

from importlib import import_module

_EXPORTS = {
    name: ("text", name)
    for name in (
        "TextDataModule",
        "TextMixtureSpec",
        "MixtureSource",
        "TokenizerSpec",
        "Packed",
        "DocumentWindow",
        "LocalTextView",
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
        "FineWebEduTextDataModule",
        "GooAQDataModule",
        "LocalDocumentsDataModule",
        "SQuADTextDataModule",
        "StackExchangeDataModule",
        "TinyStoriesV2DataModule",
        "TinyStrangeTextbooksDataModule",
        "TinyTextbooksDataModule",
        "TinyWebTextDataModule",
        "WikipediaDataModule",
        "MaskedTokenDataset",
        "TokenDataset",
        "WindowSampledDataset",
        "window_worker_init_fn",
    )
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
