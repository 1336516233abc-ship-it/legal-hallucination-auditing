"""Retrieval module."""

from retrieval.chunking import DocumentChunker, FixedLengthChunker, ClauseLevelChunker, get_chunker

# Avoid circular import: Don't import baselines at module level
# They will be imported when needed

__all__ = [
    'DocumentChunker',
    'FixedLengthChunker',
    'ClauseLevelChunker',
    'get_chunker',
    # 'DirectPromptingBaseline',  # Import directly when needed
    # 'StandardRAGBaseline',       # Import directly when needed
]

