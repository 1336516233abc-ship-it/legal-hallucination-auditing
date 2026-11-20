"""Deterministic sparse, dense and hybrid ranking with optional document scope."""

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


def _tokens(text):
    return re.findall(r"\w+(?:['’-]\w+)*", text.casefold())


def _normalize(scores):
    if len(scores) == 0:
        return scores
    minimum, maximum = float(np.min(scores)), float(np.max(scores))
    if maximum == minimum:
        return np.ones_like(scores) if maximum > 0 else np.zeros_like(scores)
    return (scores - minimum) / (maximum - minimum)


class Retriever:
    """Index chunks; retrieve returns (original chunk, finite Python float) pairs."""

    def __init__(self):
        self.documents = []
        self.indexed = False

    def index(self, documents: List[Dict]):
        for document in documents:
            if not isinstance(document.get("text"), str):
                raise ValueError("Each indexed document must contain a string text field")
        self.documents = list(documents)
        self.indexed = True

    def _scores(self, query):
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 5, doc_id: Optional[str] = None) -> List[Tuple[Dict, float]]:
        if not self.indexed:
            raise ValueError("Must call index() before retrieve()")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if top_k == 0 or not self.documents:
            return []
        candidates = [i for i, document in enumerate(self.documents)
                      if doc_id is None or document.get("doc_id", document.get("metadata", {}).get("doc_id")) == doc_id]
        if not candidates:
            return []
        scores = np.asarray(self._scores(query), dtype=float)
        if scores.shape != (len(self.documents),) or not np.isfinite(scores).all():
            raise ValueError("Retriever produced invalid scores")
        ranked = sorted(candidates, key=lambda i: (-scores[i], i))[:top_k]
        return [(self.documents[i], float(scores[i])) for i in ranked]


class BM25Retriever(Retriever):
    """BM25 with punctuation-aware tokenization and stable tie breaking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__()
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.k1, self.b = k1, b
        self.bm25 = None
        self.tokenized_docs = []

    def index(self, documents: List[Dict]):
        super().index(documents)
        self.tokenized_docs = [_tokens(doc["text"]) for doc in self.documents]
        self.bm25 = (BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)
                     if any(self.tokenized_docs) else None)

    def _scores(self, query):
        if self.bm25 is None:
            return np.zeros(len(self.documents))
        return self.bm25.get_scores(_tokens(query))


class DenseRetriever(Retriever):
    """Load SentenceTransformer on first use; encoder injection supports tests."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 similarity_metric: str = "cosine", encoder=None):
        super().__init__()
        if similarity_metric not in {"cosine", "dot"}:
            raise ValueError("similarity_metric must be cosine or dot")
        self.model_name, self.similarity_metric = model_name, similarity_metric
        self.encoder = encoder
        self.embeddings = None

    def _encode(self, texts):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name)
        embeddings = np.asarray(self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False), dtype=float)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts) or not np.isfinite(embeddings).all():
            raise ValueError("Encoder must return a finite 2D embedding matrix")
        return embeddings

    def index(self, documents: List[Dict]):
        self.indexed = False
        self.embeddings = None
        super().index(documents)
        self.indexed = False
        self.embeddings = self._encode([d["text"] for d in self.documents]) if self.documents else np.empty((0, 0))
        self.indexed = True

    def _scores(self, query):
        query_embedding = self._encode([query])[0]
        scores = np.dot(self.embeddings, query_embedding)
        if self.similarity_metric == "cosine":
            norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            scores = np.divide(scores, norms, out=np.zeros_like(scores), where=norms > 0)
            scores = np.clip(scores, -1.0, 1.0)
        return scores


class HybridRetriever(Retriever):
    """Weighted min-max normalized scores over the complete indexed candidate set."""

    def __init__(self, bm25_weight: float = 0.5, dense_weight: float = 0.5,
                 dense_model: str = "sentence-transformers/all-MiniLM-L6-v2", encoder=None,
                 k1: float = 1.5, b: float = 0.75):
        super().__init__()
        if (not all(math.isfinite(w) and w >= 0 for w in [bm25_weight, dense_weight])
                or bm25_weight + dense_weight <= 0):
            raise ValueError("Hybrid weights must be finite, nonnegative and have a positive sum")
        total = bm25_weight + dense_weight
        self.bm25_weight, self.dense_weight = bm25_weight / total, dense_weight / total
        self.bm25_retriever = BM25Retriever(k1=k1, b=b)
        self.dense_retriever = DenseRetriever(model_name=dense_model, encoder=encoder)

    def index(self, documents: List[Dict]):
        self.indexed = False
        super().index(documents)
        self.indexed = False
        self.bm25_retriever.index(self.documents)
        if self.dense_weight:
            self.dense_retriever.index(self.documents)
        self.indexed = True

    def _scores(self, query):
        bm25 = _normalize(self.bm25_retriever._scores(query)) if self.bm25_weight else np.zeros(len(self.documents))
        dense = _normalize(self.dense_retriever._scores(query)) if self.dense_weight else np.zeros(len(self.documents))
        return self.bm25_weight * bm25 + self.dense_weight * dense


def get_retriever(retriever_type: str, **kwargs) -> Retriever:
    factories = {"bm25": BM25Retriever, "dense": DenseRetriever, "hybrid": HybridRetriever}
    if retriever_type not in factories:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
    return factories[retriever_type](**kwargs)
