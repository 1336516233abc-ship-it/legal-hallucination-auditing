"""
Dataset loader utilities for ContractNLI, CUAD, and PrivacyQA.
Author: Li Xinru (3036807959)
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LegalExample:
    """Standard format for all legal QA examples."""
    id: str
    query: str
    context: str
    answer: Optional[str] = None
    label: Optional[str] = None
    evidence_spans: Optional[List[Tuple[int, int]]] = None
    metadata: Optional[Dict] = None


class DatasetLoader:
    """Base class for loading legal datasets."""

    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load(self, split: str = "test") -> List[LegalExample]:
        """Load dataset split."""
        raise NotImplementedError

    def save_processed(self, examples: List[LegalExample], filename: str):
        """Save processed examples to JSON."""
        filepath = os.path.join(self.data_dir, filename)

        data = []
        for ex in examples:
            data.append({
                "id": ex.id,
                "query": ex.query,
                "context": ex.context,
                "answer": ex.answer,
                "label": ex.label,
                "evidence_spans": ex.evidence_spans,
                "metadata": ex.metadata
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(examples)} examples to {filepath}")

    def load_processed(self, filename: str) -> List[LegalExample]:
        """Load processed examples from JSON."""
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Processed file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        examples = []
        for item in data:
            examples.append(LegalExample(
                id=item["id"],
                query=item["query"],
                context=item["context"],
                answer=item.get("answer"),
                label=item.get("label"),
                evidence_spans=item.get("evidence_spans"),
                metadata=item.get("metadata")
            ))

        logger.info(f"Loaded {len(examples)} examples from {filepath}")
        return examples


def get_dataset_statistics(examples: List[LegalExample]) -> Dict:
    """Compute statistics for a dataset."""
    stats = {
        "total_examples": len(examples),
        "avg_query_length": sum(len(ex.query.split()) for ex in examples) / len(examples),
        "avg_context_length": sum(len(ex.context.split()) for ex in examples) / len(examples),
    }

    # Label distribution
    if examples and examples[0].label:
        label_counts = {}
        for ex in examples:
            label_counts[ex.label] = label_counts.get(ex.label, 0) + 1
        stats["label_distribution"] = label_counts

    # Evidence statistics
    examples_with_evidence = [ex for ex in examples if ex.evidence_spans]
    if examples_with_evidence:
        stats["examples_with_evidence"] = len(examples_with_evidence)
        stats["avg_evidence_spans"] = sum(len(ex.evidence_spans) for ex in examples_with_evidence) / len(examples_with_evidence)

    return stats
