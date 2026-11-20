"""Dataset interfaces with lazy processor imports."""

from importlib import import_module
from data.dataset_loader import DatasetLoader, LegalExample, get_dataset_statistics

__all__ = ["DatasetLoader", "LegalExample", "get_dataset_statistics",
           "ContractNLIProcessor", "CUADProcessor", "PrivacyQAProcessor"]


def __getattr__(name):
    modules = {"ContractNLIProcessor": "contractnli_processor",
               "CUADProcessor": "cuad_processor", "PrivacyQAProcessor": "privacyqa_processor"}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(f"data.{modules[name]}"), name)
    globals()[name] = value
    return value
