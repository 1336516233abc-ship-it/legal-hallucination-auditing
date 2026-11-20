"""Shared utilities; importing logging helpers does not load model libraries."""

from importlib import import_module

_EXPORTS = {
    "LLMInterface": "llm_interface", "load_model_from_config": "llm_interface",
    "setup_logger": "logger", "get_experiment_logger": "logger",
    **{name: "helpers" for name in ["set_random_seed", "load_yaml", "save_yaml", "load_json",
                                    "save_json", "ensure_dir", "chunk_text", "format_prompt", "get_device"]},
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"utils.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
