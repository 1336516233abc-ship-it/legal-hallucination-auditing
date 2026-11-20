"""Ranking semantics with deterministic embeddings and real BM25 scoring."""

import json
import math
import numpy as np
import pytest
from retrieval.retrievers.retriever import BM25Retriever, DenseRetriever, HybridRetriever, get_retriever


class Encoder:
    def encode(self, texts, **kwargs):
        lookup = {"law": [1, 0], "payment": [0, 1], "zero": [0, 0], "negative": [-1, 0]}
        return np.array([lookup.get(text, [0, 0]) for text in texts])


@pytest.fixture(params=["bm25", "dense", "hybrid"])
def retriever(request):
    return get_retriever(request.param, **({"encoder": Encoder()} if request.param != "bm25" else {}))


def test_index_required_and_empty_index_is_valid(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("law")
    retriever.index([])
    assert retriever.retrieve("law") == []


@pytest.mark.parametrize("top_k", [-1, True, 1.5])
def test_invalid_top_k(retriever, top_k):
    retriever.index([{"text": "law"}])
    with pytest.raises(ValueError):
        retriever.retrieve("law", top_k)


def test_scope_boundaries_finite_scores_and_json(retriever):
    docs = [{"text": "law", "doc_id": "A"}, {"text": "payment", "metadata": {"doc_id": "B"}}, {"text": "zero", "doc_id": "A"}]
    retriever.index(docs)
    assert retriever.retrieve("law", 0) == []
    assert retriever.retrieve("law", 5, doc_id="missing") == []
    results = retriever.retrieve("law", 20, doc_id="A")
    assert len(results) == 2
    assert all(doc["doc_id"] == "A" and type(score) is float and math.isfinite(score) for doc, score in results)
    json.dumps(results, allow_nan=False)
    assert results == retriever.retrieve("law", 20, doc_id="A")


def test_bm25_tokenization_and_reindex():
    r = BM25Retriever()
    r.index([{"text": "Governing law: Hong Kong."}, {"text": "Payment due"}, {"text": "Notice required"}])
    assert r.retrieve("LAW?", 1)[0][0]["text"].startswith("Governing")
    r.index([{"text": "New contract"}])
    assert len(r.retrieve("law", 10)) == 1
    r.index([{"text": "!!!"}])
    assert r.retrieve("?", 1)[0][1] == 0.0


def test_dense_zero_vector_and_negative_dot():
    for metric in ["cosine", "dot"]:
        r = DenseRetriever(similarity_metric=metric, encoder=Encoder())
        r.index([{"text": "zero"}, {"text": "negative"}, {"text": "law"}])
        assert r.retrieve("law", 1)[0][0]["text"] == "law"
        assert all(math.isfinite(s) for _, s in r.retrieve("zero"))


def test_hybrid_zero_scores_ties_and_sparse_only_without_model():
    r = HybridRetriever(encoder=Encoder())
    docs = [{"text": "zero"}, {"text": "zero"}]
    r.index(docs)
    assert [d for d, _ in r.retrieve("unknown")] == docs
    assert all(s == 0 for _, s in r.retrieve("unknown"))
    sparse = HybridRetriever(dense_weight=0)
    sparse.index(docs)
    assert len(sparse.retrieve("law")) == 2
    assert sparse.dense_retriever.encoder is None


@pytest.mark.parametrize("weights", [(-1, 1), (0, 0), (float("nan"), 1)])
def test_invalid_hybrid_weights(weights):
    with pytest.raises(ValueError):
        HybridRetriever(*weights)


def test_invalid_factory_and_similarity():
    with pytest.raises(ValueError):
        get_retriever("missing")
    with pytest.raises(ValueError):
        DenseRetriever(similarity_metric="missing")


def test_failed_dense_reindex_cannot_reuse_old_embeddings():
    class FailingEncoder(Encoder):
        def encode(self, texts, **kwargs):
            if "bad" in texts:
                raise RuntimeError("embedding failure")
            return super().encode(texts, **kwargs)
    r = DenseRetriever(encoder=FailingEncoder())
    r.index([{"text": "law"}])
    with pytest.raises(RuntimeError):
        r.index([{"text": "bad"}])
    with pytest.raises(ValueError):
        r.retrieve("law")
