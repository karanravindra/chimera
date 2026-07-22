"""Optional lm-eval integration, imported only when requested."""

from importlib import import_module

_EXPORTS = {
    "CHANCE": ("bench", "CHANCE"),
    "GPT2_SMALL": ("bench", "GPT2_SMALL"),
    "TASKS": ("bench", "TASKS"),
    "ChimeraLM": ("lm_harness", "ChimeraLM"),
    "headline": ("bench", "headline"),
    "results_table": ("bench", "results_table"),
    "run_eval": ("bench", "run_eval"),
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
