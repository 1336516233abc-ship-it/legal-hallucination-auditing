"""
Document chunking strategies for retrieval.
Implements fixed-length and clause-level chunking.

Author: Qin Mengmeng (3036843769)
"""

from typing import List, Dict, Optional
import re
import nltk
from nltk.tokenize import sent_tokenize
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class DocumentChunker:
    """Base class for document chunking strategies."""

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk document into segments.

        Args:
            text: Input document text
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dicts with 'text', 'start', 'end', 'metadata'
        """
        raise NotImplementedError


class FixedLengthChunker(DocumentChunker):
    """Fixed-length chunking with overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        """
        Initialize fixed-length chunker.

        Args:
            chunk_size: Target chunk size in tokens (approximate)
            overlap: Overlap between chunks in tokens (approximate)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """Chunk text into fixed-length segments with overlap."""
        # Split into words (approximate tokens)
        words = text.split()
        chunks = []

        start_idx = 0
        char_position = 0

        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))

            # Get chunk text
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)

            # Calculate character positions
            start_char = text.find(chunk_words[0], char_position)
            end_char = start_char + len(chunk_text)

            chunk_dict = {
                'text': chunk_text,
                'start': start_char,
                'end': end_char,
                'chunk_id': len(chunks),
                'metadata': metadata or {}
            }
            chunks.append(chunk_dict)

            # Move to next chunk with overlap
            start_idx += (self.chunk_size - self.overlap)
            char_position = start_char + len(chunk_words[0])

        logger.info(f"Created {len(chunks)} fixed-length chunks")
        return chunks


class ClauseLevelChunker(DocumentChunker):
    """Clause-level chunking using sentence boundaries."""

    def __init__(
        self,
        max_chunk_size: int = 1024,
        min_chunk_size: int = 50,
        use_sentence_boundaries: bool = True
    ):
        """
        Initialize clause-level chunker.

        Args:
            max_chunk_size: Maximum chunk size in tokens
            min_chunk_size: Minimum chunk size in tokens
            use_sentence_boundaries: Whether to respect sentence boundaries
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.use_sentence_boundaries = use_sentence_boundaries

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """Chunk text by clauses or sentences."""
        if self.use_sentence_boundaries:
            # Use NLTK sentence tokenizer
            sentences = sent_tokenize(text)
        else:
            # Split by common clause delimiters
            sentences = self._split_by_clauses(text)

        chunks = []
        current_chunk = []
        current_length = 0
        char_start = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            # If single sentence is too long, split it
            if sentence_length > self.max_chunk_size:
                # Flush current chunk if exists
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'start': char_start,
                        'end': char_start + len(chunk_text),
                        'chunk_id': len(chunks),
                        'metadata': metadata or {}
                    })
                    char_start += len(chunk_text) + 1

                # Split long sentence
                long_chunks = self._split_long_sentence(sentence)
                for long_chunk in long_chunks:
                    chunks.append({
                        'text': long_chunk,
                        'start': text.find(long_chunk, char_start),
                        'end': text.find(long_chunk, char_start) + len(long_chunk),
                        'chunk_id': len(chunks),
                        'metadata': metadata or {}
                    })
                    char_start = text.find(long_chunk, char_start) + len(long_chunk)

                current_chunk = []
                current_length = 0

            # If adding this sentence exceeds max, flush current chunk
            elif current_length + sentence_length > self.max_chunk_size:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'start': char_start,
                        'end': char_start + len(chunk_text),
                        'chunk_id': len(chunks),
                        'metadata': metadata or {}
                    })
                    char_start += len(chunk_text) + 1

                current_chunk = [sentence]
                current_length = sentence_length

            # Add sentence to current chunk
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Flush remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start': char_start,
                'end': char_start + len(chunk_text),
                'chunk_id': len(chunks),
                'metadata': metadata or {}
            })

        # Filter out chunks that are too small
        chunks = [c for c in chunks if len(c['text'].split()) >= self.min_chunk_size]

        logger.info(f"Created {len(chunks)} clause-level chunks")
        return chunks

    def _split_by_clauses(self, text: str) -> List[str]:
        """Split text by legal clause delimiters."""
        # Common legal clause delimiters
        delimiters = r'[;.]\s+(?=[A-Z])|(?:\n\s*\n)|(?:\n\s*\([a-z]\))'

        clauses = re.split(delimiters, text)
        return [c.strip() for c in clauses if c.strip()]

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a long sentence into smaller parts."""
        # Split by commas or semicolons
        parts = re.split(r'[,;]\s+', sentence)

        chunks = []
        current = []
        current_length = 0

        for part in parts:
            part_length = len(part.split())

            if current_length + part_length > self.max_chunk_size:
                if current:
                    chunks.append(' '.join(current))
                current = [part]
                current_length = part_length
            else:
                current.append(part)
                current_length += part_length

        if current:
            chunks.append(' '.join(current))

        return chunks


def get_chunker(strategy: str, **kwargs) -> DocumentChunker:
    """
    Factory function to get chunker by strategy name.

    Args:
        strategy: Chunking strategy ("fixed_length" or "clause_level")
        **kwargs: Additional arguments for chunker

    Returns:
        DocumentChunker instance
    """
    if strategy == "fixed_length":
        return FixedLengthChunker(**kwargs)
    elif strategy == "clause_level":
        return ClauseLevelChunker(**kwargs)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
