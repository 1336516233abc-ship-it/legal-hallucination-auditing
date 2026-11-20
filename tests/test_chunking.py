"""Verify source invariants, coverage and chunk size boundaries."""

import re
import pytest
from retrieval.chunking import FixedLengthChunker, ClauseLevelChunker, get_chunker
from retrieval.common import load_examples


@pytest.mark.parametrize("size,overlap", [(1, 0), (2, 0), (3, 1), (5, 4), (512, 128)])
@pytest.mark.parametrize("text", ["", "   \n", "A A A A A A", " Alpha  beta\nGamma\t delta. ", "a " * 80])
def test_fixed_preserves_source_and_covers_words(size, overlap, text):
    chunks = FixedLengthChunker(size, overlap).chunk(text, {"doc_id": "A"})
    for index, chunk in enumerate(chunks):
        assert text[chunk["start"]:chunk["end"]] == chunk["text"]
        assert len(chunk["text"].split()) <= size
        assert chunk["chunk_id"] == index and chunk["doc_id"] == "A"
    for word in re.finditer(r"\S+", text):
        assert any(c["start"] <= word.start() and word.end() <= c["end"] for c in chunks)
    assert not chunks or chunks[-1]["end"] == len(text.rstrip())


@pytest.mark.parametrize("options", [dict(chunk_size=0), dict(chunk_size=-1), dict(chunk_size=True),
    dict(chunk_size=2, overlap=2), dict(overlap=-1), dict(overlap=1.5)])
def test_fixed_rejects_invalid_steps(options):
    with pytest.raises(ValueError):
        FixedLengthChunker(**options)


@pytest.mark.parametrize("sentence_mode", [True, False])
@pytest.mark.parametrize("text", ["Payment is due.", "A A A A A A", "  First clause.\n\nSecond; Third.  ",
                                  "x " * 100, " ", "First! Next? Another."])
def test_clause_preserves_all_text_and_bounds(sentence_mode, text):
    chunks = ClauseLevelChunker(7, 3, sentence_mode).chunk(text)
    assert all(text[c["start"]:c["end"]] == c["text"] and len(c["text"].split()) <= 7 for c in chunks)
    for word in re.finditer(r"\S+", text):
        assert any(c["start"] <= word.start() and word.end() <= c["end"] for c in chunks)


@pytest.mark.parametrize("options", [dict(max_chunk_size=0), dict(min_chunk_size=0),
    dict(max_chunk_size=5, min_chunk_size=6), dict(min_chunk_size=True)])
def test_clause_rejects_invalid_sizes(options):
    with pytest.raises(ValueError):
        ClauseLevelChunker(**options)


def test_factory_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        get_chunker("missing")


@pytest.mark.parametrize("dataset", ["contractnli", "cuad"])
def test_all_supplied_contracts_have_exact_source_slices(dataset):
    contexts = {example.context for example in load_examples(dataset)}
    for chunker in [FixedLengthChunker(), ClauseLevelChunker(), ClauseLevelChunker(use_sentence_boundaries=False)]:
        for context in contexts:
            chunks = chunker.chunk(context)
            assert chunks
            assert all(context[c["start"]:c["end"]] == c["text"] for c in chunks)
            if isinstance(chunker, ClauseLevelChunker):
                expected = context.split()
                actual = [word for chunk in chunks for word in chunk["text"].split()]
                assert actual == expected
