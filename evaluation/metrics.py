"""
Evaluation metrics for legal hallucination auditing.
Implements all metrics from the proposal.

Metrics:
- Label accuracy
- Macro-F1
- Evidence recall@k
- Hallucination rate
- Answer completeness
- Abstention precision
- Latency
- Token cost

Author: Li Xinru (3036807959)
"""

import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    label_accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    micro_f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    evidence_recall_at_k: Optional[float] = None
    evidence_hit_rate: Optional[float] = None
    hallucination_rate: Optional[float] = None
    answer_completeness: Optional[float] = None
    abstention_precision: Optional[float] = None
    avg_latency: Optional[float] = None
    total_tokens: Optional[int] = None
    per_class_metrics: Optional[Dict] = None


class MetricsCalculator:
    """Calculate evaluation metrics for legal QA systems."""

    def __init__(self):
        self.label_map = {
            "entailment": 0,
            "contradiction": 1,
            "neutral": 2,
            "present": 0,
            "absent": 1,
        }

    def compute_all_metrics(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        retrieved_evidence: Optional[List[List[str]]] = None,
        latencies: Optional[List[float]] = None,
        token_counts: Optional[List[int]] = None,
    ) -> EvaluationResult:
        """
        Compute all evaluation metrics.

        Args:
            predictions: List of prediction dicts with 'label', 'answer', 'evidence', etc.
            ground_truth: List of ground truth dicts with 'label', 'answer', 'evidence_spans', etc.
            retrieved_evidence: List of retrieved evidence for each example
            latencies: List of response latencies
            token_counts: List of token counts for each example

        Returns:
            EvaluationResult object
        """
        result = EvaluationResult()

        # Label-based metrics (for ContractNLI)
        if self._has_labels(ground_truth):
            result.label_accuracy = self.compute_label_accuracy(predictions, ground_truth)
            result.macro_f1 = self.compute_macro_f1(predictions, ground_truth)
            result.micro_f1 = self.compute_micro_f1(predictions, ground_truth)
            result.per_class_metrics = self.compute_per_class_metrics(predictions, ground_truth)

        # Evidence-based metrics
        if retrieved_evidence and self._has_evidence(ground_truth):
            result.evidence_recall_at_k = self.compute_evidence_recall(
                retrieved_evidence, ground_truth
            )
            result.evidence_hit_rate = self.compute_evidence_hit_rate(
                retrieved_evidence, ground_truth
            )

        # Hallucination metrics
        if self._has_verification_info(predictions):
            result.hallucination_rate = self.compute_hallucination_rate(predictions)
            result.abstention_precision = self.compute_abstention_precision(predictions, ground_truth)

        # Answer completeness
        if self._has_answers(predictions) and self._has_answers(ground_truth):
            result.answer_completeness = self.compute_answer_completeness(predictions, ground_truth)

        # Performance metrics
        if latencies:
            result.avg_latency = np.mean(latencies)

        if token_counts:
            result.total_tokens = sum(token_counts)

        return result

    def compute_label_accuracy(
        self, predictions: List[Dict], ground_truth: List[Dict]
    ) -> float:
        """Compute label classification accuracy."""
        pred_labels = [self._normalize_label(p.get("label", "neutral")) for p in predictions]
        true_labels = [self._normalize_label(g.get("label", "neutral")) for g in ground_truth]

        return accuracy_score(true_labels, pred_labels)

    def compute_macro_f1(
        self, predictions: List[Dict], ground_truth: List[Dict]
    ) -> float:
        """Compute macro-averaged F1 score."""
        pred_labels = [self._normalize_label(p.get("label", "neutral")) for p in predictions]
        true_labels = [self._normalize_label(g.get("label", "neutral")) for g in ground_truth]

        return f1_score(true_labels, pred_labels, average="macro")

    def compute_micro_f1(
        self, predictions: List[Dict], ground_truth: List[Dict]
    ) -> float:
        """Compute micro-averaged F1 score."""
        pred_labels = [self._normalize_label(p.get("label", "neutral")) for p in predictions]
        true_labels = [self._normalize_label(g.get("label", "neutral")) for g in ground_truth]

        return f1_score(true_labels, pred_labels, average="micro")

    def compute_per_class_metrics(
        self, predictions: List[Dict], ground_truth: List[Dict]
    ) -> Dict:
        """Compute precision, recall, F1 for each class."""
        pred_labels = [self._normalize_label(p.get("label", "neutral")) for p in predictions]
        true_labels = [self._normalize_label(g.get("label", "neutral")) for g in ground_truth]

        labels = sorted(set(true_labels + pred_labels))
        metrics = {}

        for label in labels:
            label_name = self._denormalize_label(label)
            metrics[label_name] = {
                "precision": precision_score(true_labels, pred_labels, labels=[label], average="macro", zero_division=0),
                "recall": recall_score(true_labels, pred_labels, labels=[label], average="macro", zero_division=0),
                "f1": f1_score(true_labels, pred_labels, labels=[label], average="macro", zero_division=0),
            }

        return metrics

    def compute_evidence_recall(
        self, retrieved_evidence: List[List[str]], ground_truth: List[Dict], k: int = 5
    ) -> float:
        """
        Compute evidence recall@k: proportion of gold evidence spans retrieved.

        Args:
            retrieved_evidence: List of lists of retrieved evidence strings
            ground_truth: Ground truth with evidence_spans
            k: Top-k retrieved documents to consider

        Returns:
            Average recall across examples
        """
        recalls = []

        for retrieved, gt in zip(retrieved_evidence, ground_truth):
            gold_spans = gt.get("evidence_spans", [])
            if not gold_spans:
                continue

            # Get top-k retrieved evidence
            retrieved_k = retrieved[:k]

            # Check overlap with gold spans
            # This is simplified - in practice, need fuzzy matching
            hits = 0
            for gold_start, gold_end in gold_spans:
                gold_text = gt.get("context", "")[gold_start:gold_end]
                for ret_text in retrieved_k:
                    if gold_text in ret_text or ret_text in gold_text:
                        hits += 1
                        break

            recall = hits / len(gold_spans) if gold_spans else 0
            recalls.append(recall)

        return np.mean(recalls) if recalls else 0.0

    def compute_evidence_hit_rate(
        self, retrieved_evidence: List[List[str]], ground_truth: List[Dict]
    ) -> float:
        """
        Compute hit rate: proportion of queries with at least one relevant evidence.

        Args:
            retrieved_evidence: List of lists of retrieved evidence
            ground_truth: Ground truth with evidence_spans

        Returns:
            Hit rate
        """
        hits = 0
        total = 0

        for retrieved, gt in zip(retrieved_evidence, ground_truth):
            gold_spans = gt.get("evidence_spans", [])
            if not gold_spans:
                continue

            total += 1

            # Check if any retrieved evidence matches gold
            for gold_start, gold_end in gold_spans:
                gold_text = gt.get("context", "")[gold_start:gold_end]
                for ret_text in retrieved:
                    if gold_text in ret_text or ret_text in gold_text:
                        hits += 1
                        break
                if hits > 0:
                    break

        return hits / total if total > 0 else 0.0

    def compute_hallucination_rate(self, predictions: List[Dict]) -> float:
        """
        Compute hallucination rate: proportion of claims that are unsupported or contradicted.

        Args:
            predictions: Predictions with verification_results

        Returns:
            Hallucination rate
        """
        total_claims = 0
        hallucinated_claims = 0

        for pred in predictions:
            verification = pred.get("verification_results", [])

            for claim_verification in verification:
                total_claims += 1
                status = claim_verification.get("status", "supported")

                if status in ["contradicted", "not_enough_info"]:
                    hallucinated_claims += 1

        return hallucinated_claims / total_claims if total_claims > 0 else 0.0

    def compute_answer_completeness(
        self, predictions: List[Dict], ground_truth: List[Dict]
    ) -> float:
        """
        Compute answer completeness: how much of the ground truth answer is covered.

        Args:
            predictions: Predictions with answers
            ground_truth: Ground truth with answers

        Returns:
            Average completeness score
        """
        completeness_scores = []

        for pred, gt in zip(predictions, ground_truth):
            pred_answer = pred.get("answer", "").lower()
            gt_answer = gt.get("answer", "").lower()

            if not gt_answer:
                continue

            # Simple token overlap
            pred_tokens = set(pred_answer.split())
            gt_tokens = set(gt_answer.split())

            if gt_tokens:
                overlap = len(pred_tokens & gt_tokens) / len(gt_tokens)
                completeness_scores.append(overlap)

        return np.mean(completeness_scores) if completeness_scores else 0.0

    def compute_abstention_precision(
        self, predictions: List[Dict], ground_truth: List[Dict]
    ) -> float:
        """
        Compute abstention precision: when model abstains, is it correct to do so?

        Args:
            predictions: Predictions with abstention flags
            ground_truth: Ground truth

        Returns:
            Abstention precision
        """
        abstained = []
        should_abstain = []

        for pred, gt in zip(predictions, ground_truth):
            did_abstain = pred.get("abstained", False) or pred.get("label") == "not_enough_info"

            # Should abstain if ground truth is neutral or no evidence
            should = gt.get("label") == "neutral" or not gt.get("evidence_spans")

            if did_abstain:
                abstained.append(1 if should else 0)

        return np.mean(abstained) if abstained else 0.0

    def _normalize_label(self, label: str) -> int:
        """Convert label string to numeric."""
        return self.label_map.get(label.lower(), 2)  # Default to neutral/2

    def _denormalize_label(self, label_id: int) -> str:
        """Convert numeric label back to string."""
        reverse_map = {v: k for k, v in self.label_map.items()}
        return reverse_map.get(label_id, "neutral")

    def _has_labels(self, data: List[Dict]) -> bool:
        """Check if data has labels."""
        return any("label" in item for item in data)

    def _has_evidence(self, data: List[Dict]) -> bool:
        """Check if data has evidence spans."""
        return any("evidence_spans" in item and item["evidence_spans"] for item in data)

    def _has_verification_info(self, data: List[Dict]) -> bool:
        """Check if predictions have verification results."""
        return any("verification_results" in item for item in data)

    def _has_answers(self, data: List[Dict]) -> bool:
        """Check if data has answers."""
        return any("answer" in item and item["answer"] for item in data)

    def format_results(self, result: EvaluationResult) -> str:
        """Format evaluation results as a readable string."""
        lines = ["=" * 60, "EVALUATION RESULTS", "=" * 60, ""]

        if result.label_accuracy is not None:
            lines.append(f"Label Accuracy:        {result.label_accuracy:.4f}")
        if result.macro_f1 is not None:
            lines.append(f"Macro F1:              {result.macro_f1:.4f}")
        if result.micro_f1 is not None:
            lines.append(f"Micro F1:              {result.micro_f1:.4f}")

        lines.append("")

        if result.evidence_recall_at_k is not None:
            lines.append(f"Evidence Recall@k:     {result.evidence_recall_at_k:.4f}")
        if result.evidence_hit_rate is not None:
            lines.append(f"Evidence Hit Rate:     {result.evidence_hit_rate:.4f}")

        lines.append("")

        if result.hallucination_rate is not None:
            lines.append(f"Hallucination Rate:    {result.hallucination_rate:.4f}")
        if result.answer_completeness is not None:
            lines.append(f"Answer Completeness:   {result.answer_completeness:.4f}")
        if result.abstention_precision is not None:
            lines.append(f"Abstention Precision:  {result.abstention_precision:.4f}")

        lines.append("")

        if result.avg_latency is not None:
            lines.append(f"Avg Latency:           {result.avg_latency:.2f}s")
        if result.total_tokens is not None:
            lines.append(f"Total Tokens:          {result.total_tokens:,}")

        if result.per_class_metrics:
            lines.append("")
            lines.append("Per-Class Metrics:")
            for label, metrics in result.per_class_metrics.items():
                lines.append(f"  {label}:")
                lines.append(f"    Precision: {metrics['precision']:.4f}")
                lines.append(f"    Recall:    {metrics['recall']:.4f}")
                lines.append(f"    F1:        {metrics['f1']:.4f}")

        lines.append("=" * 60)

        return "\n".join(lines)
