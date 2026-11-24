"""Retrieval module."""

from retrieval.chunking import DocumentChunker, FixedLengthChunker, ClauseLevelChunker, get_chunker

# Avoid circular import: Don't import baselines at module level
# They will be imported when needed

__all__ = [
    'DocumentChunker',
    'FixedLengthChunker',
    'ClauseLevelChunker',
    'get_chunker',
    'DirectPromptingBaseline',
    'StandardRAGBaseline',
]


def __getattr__(name):
    """Preserve public baseline exports without eager circular imports."""
    from importlib import import_module
    modules = {"DirectPromptingBaseline": "baseline_direct", "StandardRAGBaseline": "baseline_rag"}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(f"retrieval.{modules[name]}"), name)
    globals()[name] = value
    return value
