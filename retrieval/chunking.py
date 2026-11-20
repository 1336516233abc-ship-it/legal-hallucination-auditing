"""Offset-preserving document chunkers. Sizes are whitespace-delimited words."""

import re
from typing import Dict, List, Optional


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _chunk(text, start, end, chunk_id, metadata):
    result = {
        "text": text[start:end], "start": start, "end": end,
        "chunk_id": chunk_id, "metadata": dict(metadata or {}),
    }
    if metadata and "doc_id" in metadata:
        result["doc_id"] = metadata["doc_id"]
    return result


class DocumentChunker:
    """Return original text slices with half-open character offsets."""

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        raise NotImplementedError


class FixedLengthChunker(DocumentChunker):
    """Split word spans into overlapping windows without normalizing text."""

    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        _positive_integer(chunk_size, "chunk_size")
        if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        words = list(re.finditer(r"\S+", text))
        chunks = []
        for start in range(0, len(words), self.chunk_size - self.overlap):
            end = min(start + self.chunk_size, len(words))
            chunks.append(_chunk(text, words[start].start(), words[end - 1].end(), len(chunks), metadata))
            if end == len(words):
                break
        return chunks


class ClauseLevelChunker(DocumentChunker):
    """Pack heuristic sentence/clause spans while preserving all nonblank text.

    Sentence boundaries use punctuation followed by whitespace and a capital
    letter, or blank lines. Clause mode also recognizes semicolons and numbered
    paragraph starts. This deterministic heuristic requires no downloaded data;
    it is not a legal clause parser. min_chunk_size is a merge target, never a
    reason to discard text. Oversized sentences are split at word boundaries.
    """

    def __init__(self, max_chunk_size: int = 1024, min_chunk_size: int = 50,
                 use_sentence_boundaries: bool = True):
        _positive_integer(max_chunk_size, "max_chunk_size")
        _positive_integer(min_chunk_size, "min_chunk_size")
        if min_chunk_size > max_chunk_size:
            raise ValueError("min_chunk_size must not exceed max_chunk_size")
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.use_sentence_boundaries = use_sentence_boundaries

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        boundary = r"(?<=[.!?])\s+(?=[A-Z(])|\n\s*\n"
        if not self.use_sentence_boundaries:
            boundary += r"|(?<=;)\s+|\n(?=\s*(?:\([a-z0-9]+\)|\d+\.))"
        positions = [0] + [match.end() for match in re.finditer(boundary, text)] + [len(text)]
        units = []
        for left, right in zip(positions, positions[1:]):
            words = list(re.finditer(r"\S+", text[left:right]))
            for offset in range(0, len(words), self.max_chunk_size):
                end = min(offset + self.max_chunk_size, len(words))
                units.append((left + words[offset].start(), left + words[end - 1].end(), end - offset))
        packed = []
        for start, end, count in units:
            if packed and packed[-1][2] + count <= self.max_chunk_size:
                prev_start, _, prev_count = packed[-1]
                packed[-1] = (prev_start, end, prev_count + count)
            else:
                packed.append((start, end, count))
        if len(packed) > 1 and packed[-1][2] < self.min_chunk_size:
            prev, tail = packed[-2:]
            if prev[2] + tail[2] <= self.max_chunk_size:
                packed[-2:] = [(prev[0], tail[1], prev[2] + tail[2])]
        return [_chunk(text, start, end, i, metadata) for i, (start, end, _) in enumerate(packed)]


def get_chunker(strategy: str, **kwargs) -> DocumentChunker:
    if strategy == "fixed_length":
        return FixedLengthChunker(**kwargs)
    if strategy == "clause_level":
        return ClauseLevelChunker(**kwargs)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
