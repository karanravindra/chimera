"""Reusable TinyLM model components, imported lazily."""

from importlib import import_module

_EXPORTS = {
    "LoRALinear": ("lora", "LoRALinear"),
    "RotaryEmbedding": ("rope", "RotaryEmbedding"),
    "apply_lora": ("lora", "apply_lora"),
    "apply_rotary": ("rope", "apply_rotary"),
    "build_block_mask_and_pos": ("attention", "build_block_mask_and_pos"),
    "doc_ids_and_pos": ("attention", "doc_ids_and_pos"),
    "flex_attn": ("attention", "flex_attn"),
    "make_causal_mask": ("attention", "make_causal_mask"),
    "make_document_mask": ("attention", "make_document_mask"),
    "merge_lora": ("lora", "merge_lora"),
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
