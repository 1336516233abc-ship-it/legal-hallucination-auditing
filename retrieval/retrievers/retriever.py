"""
Retriever implementations: BM25 and Dense retrieval.

Author: Qin Mengmeng (3036843769)
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class Retriever:
    """Base class for retrievers."""

    def index(self, documents: List[Dict]):
        """Index documents for retrieval."""
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Retrieve top-k documents for query.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of (document, score) tuples
        """
        raise NotImplementedError


class BM25Retriever(Retriever):
    """BM25 sparse retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.

        Args:
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []

    def index(self, documents: List[Dict]):
        """Index documents using BM25."""
        self.documents = documents

        # Tokenize documents
        self.tokenized_docs = [doc['text'].lower().split() for doc in documents]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)

        logger.info(f"Indexed {len(documents)} documents with BM25")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve using BM25 scoring."""
        if self.bm25 is None:
            raise ValueError("Must call index() before retrieve()")

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]

        # Return documents with scores
        results = [(self.documents[i], scores[i]) for i in top_indices]

        return results


class DenseRetriever(Retriever):
    """Dense retrieval using sentence transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_metric: str = "cosine"
    ):
        """
        Initialize dense retriever.

        Args:
            model_name: Sentence transformer model name
            similarity_metric: Similarity metric ("cosine" or "dot")
        """
        self.model_name = model_name
        self.similarity_metric = similarity_metric

        # Load sentence transformer
        logger.info(f"Loading sentence transformer: {model_name}")
        self.encoder = SentenceTransformer(model_name)

        self.documents = []
        self.embeddings = None

    def index(self, documents: List[Dict]):
        """Index documents by computing embeddings."""
        self.documents = documents

        # Extract texts
        texts = [doc['text'] for doc in documents]

        # Encode documents
        logger.info(f"Encoding {len(documents)} documents...")
        self.embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        logger.info(f"Indexed {len(documents)} documents with dense retrieval")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve using dense similarity."""
        if self.embeddings is None:
            raise ValueError("Must call index() before retrieve()")

        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)[0]

        # Compute similarities
        if self.similarity_metric == "cosine":
            # Cosine similarity
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
        else:  # dot product
            similarities = np.dot(self.embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return documents with scores
        results = [(self.documents[i], similarities[i]) for i in top_indices]

        return results


class HybridRetriever(Retriever):
    """Hybrid retriever combining BM25 and dense retrieval."""

    def __init__(
        self,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_weight: Weight for BM25 scores
            dense_weight: Weight for dense scores
            dense_model: Dense retrieval model
        """
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

        self.bm25_retriever = BM25Retriever()
        self.dense_retriever = DenseRetriever(model_name=dense_model)

    def index(self, documents: List[Dict]):
        """Index with both retrievers."""
        self.bm25_retriever.index(documents)
        self.dense_retriever.index(documents)

        logger.info(f"Indexed {len(documents)} documents with hybrid retrieval")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve using hybrid scores."""
        # Get results from both retrievers
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k * 2)

        # Normalize scores to [0, 1]
        bm25_scores = {id(doc): score for doc, score in bm25_results}
        dense_scores = {id(doc): score for doc, score in dense_results}

        if bm25_scores:
            max_bm25 = max(bm25_scores.values())
            bm25_scores = {k: v / max_bm25 for k, v in bm25_scores.items()}

        if dense_scores:
            max_dense = max(dense_scores.values())
            min_dense = min(dense_scores.values())
            if max_dense > min_dense:
                dense_scores = {
                    k: (v - min_dense) / (max_dense - min_dense)
                    for k, v in dense_scores.items()
                }

        # Combine scores
        all_docs = {id(doc): doc for doc, _ in bm25_results + dense_results}
        combined_scores = {}

        for doc_id, doc in all_docs.items():
            bm25_score = bm25_scores.get(doc_id, 0)
            dense_score = dense_scores.get(doc_id, 0)

            combined_scores[doc_id] = (
                self.bm25_weight * bm25_score +
                self.dense_weight * dense_score
            )

        # Sort by combined score
        sorted_docs = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top-k
        results = [(all_docs[doc_id], score) for doc_id, score in sorted_docs[:top_k]]

        return results


def get_retriever(retriever_type: str, **kwargs) -> Retriever:
    """
    Factory function to get retriever by type.

    Args:
        retriever_type: Type of retriever ("bm25", "dense", "hybrid")
        **kwargs: Additional arguments for retriever

    Returns:
        Retriever instance
    """
    if retriever_type == "bm25":
        return BM25Retriever(**kwargs)
    elif retriever_type == "dense":
        return DenseRetriever(**kwargs)
    elif retriever_type == "hybrid":
        return HybridRetriever(**kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
