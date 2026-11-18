"""
Error analysis utilities for deep-dive into model failures.

Author: Li Xinru (3036807959)
"""

from typing import List, Dict, Tuple
from collections import defaultdict
import json
import logging

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """Analyze errors and generate detailed reports."""

    def __init__(self):
        self.errors = []

    def collect_errors(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Collect all prediction errors.

        Args:
            predictions: Model predictions
            ground_truth: Ground truth
            threshold: Confidence threshold for filtering

        Returns:
            List of error cases
        """
        errors = []

        for pred, gt in zip(predictions, ground_truth):
            pred_label = pred.get("label", "")
            true_label = gt.get("label", "")

            # Check if prediction is incorrect
            if pred_label != true_label:
                error = {
                    "example_id": gt.get("id", ""),
                    "query": gt.get("query", ""),
                    "context": gt.get("context", "")[:500],  # First 500 chars
                    "predicted_label": pred_label,
                    "true_label": true_label,
                    "predicted_answer": pred.get("answer", ""),
                    "true_answer": gt.get("answer", ""),
                    "confidence": pred.get("confidence", None),
                    "retrieved_evidence": pred.get("retrieved_evidence", []),
                    "metadata": gt.get("metadata", {}),
                }
                errors.append(error)

        self.errors = errors
        logger.info(f"Collected {len(errors)} errors")
        return errors

    def analyze_by_label(self) -> Dict:
        """Analyze errors by true and predicted labels."""
        confusion = defaultdict(lambda: defaultdict(int))

        for error in self.errors:
            true = error["true_label"]
            pred = error["predicted_label"]
            confusion[true][pred] += 1

        return dict(confusion)

    def analyze_by_query_length(self, bins: List[int] = [0, 10, 20, 50, 100]) -> Dict:
        """Analyze errors by query length."""
        length_errors = defaultdict(list)

        for error in self.errors:
            query_len = len(error["query"].split())

            # Find appropriate bin
            for i in range(len(bins) - 1):
                if bins[i] <= query_len < bins[i + 1]:
                    length_errors[f"{bins[i]}-{bins[i+1]}"].append(error)
                    break
            else:
                length_errors[f"{bins[-1]}+"].append(error)

        return {k: len(v) for k, v in length_errors.items()}

    def analyze_by_context_length(self, bins: List[int] = [0, 500, 1000, 2000, 5000]) -> Dict:
        """Analyze errors by context length."""
        length_errors = defaultdict(list)

        for error in self.errors:
            context_len = len(error["context"].split())

            # Find appropriate bin
            for i in range(len(bins) - 1):
                if bins[i] <= context_len < bins[i + 1]:
                    length_errors[f"{bins[i]}-{bins[i+1]}"].append(error)
                    break
            else:
                length_errors[f"{bins[-1]}+"].append(error)

        return {k: len(v) for k, v in length_errors.items()}

    def analyze_retrieval_quality(self) -> Dict:
        """Analyze whether retrieval contributed to errors."""
        stats = {
            "no_evidence_retrieved": 0,
            "evidence_retrieved": 0,
            "avg_evidence_count": 0,
        }

        evidence_counts = []

        for error in self.errors:
            retrieved = error.get("retrieved_evidence", [])
            if not retrieved:
                stats["no_evidence_retrieved"] += 1
            else:
                stats["evidence_retrieved"] += 1
                evidence_counts.append(len(retrieved))

        if evidence_counts:
            stats["avg_evidence_count"] = sum(evidence_counts) / len(evidence_counts)

        return stats

    def get_hardest_examples(self, n: int = 10) -> List[Dict]:
        """
        Get the n hardest examples (those with highest confidence but still wrong).

        Args:
            n: Number of examples to return

        Returns:
            List of hardest error cases
        """
        # Sort by confidence (highest confidence errors are most interesting)
        errors_with_conf = [e for e in self.errors if e.get("confidence") is not None]
        sorted_errors = sorted(
            errors_with_conf,
            key=lambda x: x["confidence"],
            reverse=True
        )

        return sorted_errors[:n]

    def export_errors(self, filepath: str):
        """Export errors to JSON file for manual inspection."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.errors, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(self.errors)} errors to {filepath}")

    def generate_error_report(self) -> str:
        """Generate a comprehensive error analysis report."""
        lines = [
            "=" * 70,
            "ERROR ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Total Errors: {len(self.errors)}",
            "",
        ]

        # Confusion analysis
        lines.append("Confusion Matrix:")
        confusion = self.analyze_by_label()
        for true_label, pred_dict in confusion.items():
            for pred_label, count in pred_dict.items():
                lines.append(f"  True: {true_label:15s} -> Pred: {pred_label:15s}: {count:4d}")
        lines.append("")

        # Query length analysis
        lines.append("Errors by Query Length:")
        query_length_errors = self.analyze_by_query_length()
        for length_bin, count in sorted(query_length_errors.items()):
            lines.append(f"  {length_bin:15s}: {count:4d}")
        lines.append("")

        # Context length analysis
        lines.append("Errors by Context Length:")
        context_length_errors = self.analyze_by_context_length()
        for length_bin, count in sorted(context_length_errors.items()):
            lines.append(f"  {length_bin:15s}: {count:4d}")
        lines.append("")

        # Retrieval quality
        lines.append("Retrieval Quality:")
        retrieval_stats = self.analyze_retrieval_quality()
        lines.append(f"  No evidence retrieved:  {retrieval_stats['no_evidence_retrieved']:4d}")
        lines.append(f"  Evidence retrieved:     {retrieval_stats['evidence_retrieved']:4d}")
        lines.append(f"  Avg evidence count:     {retrieval_stats['avg_evidence_count']:.2f}")
        lines.append("")

        # Hardest examples
        lines.append("Top 5 Hardest Examples (high confidence, but wrong):")
        hardest = self.get_hardest_examples(n=5)
        for i, error in enumerate(hardest, 1):
            lines.append(f"  {i}. Query: {error['query'][:60]}...")
            lines.append(f"     True: {error['true_label']}, Pred: {error['predicted_label']}, Conf: {error.get('confidence', 'N/A')}")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


def compare_systems(
    system1_predictions: List[Dict],
    system2_predictions: List[Dict],
    ground_truth: List[Dict],
    system1_name: str = "System 1",
    system2_name: str = "System 2"
) -> str:
    """
    Compare two systems and identify where one performs better.

    Args:
        system1_predictions: Predictions from system 1
        system2_predictions: Predictions from system 2
        ground_truth: Ground truth
        system1_name: Name of system 1
        system2_name: Name of system 2

    Returns:
        Comparison report string
    """
    system1_correct = 0
    system2_correct = 0
    both_correct = 0
    both_wrong = 0
    only_system1_correct = []
    only_system2_correct = []

    for pred1, pred2, gt in zip(system1_predictions, system2_predictions, ground_truth):
        true_label = gt.get("label", "")
        pred1_label = pred1.get("label", "")
        pred2_label = pred2.get("label", "")

        s1_correct = pred1_label == true_label
        s2_correct = pred2_label == true_label

        if s1_correct:
            system1_correct += 1
        if s2_correct:
            system2_correct += 1

        if s1_correct and s2_correct:
            both_correct += 1
        elif not s1_correct and not s2_correct:
            both_wrong += 1
        elif s1_correct and not s2_correct:
            only_system1_correct.append(gt)
        elif s2_correct and not s1_correct:
            only_system2_correct.append(gt)

    total = len(ground_truth)

    report = [
        "=" * 70,
        f"SYSTEM COMPARISON: {system1_name} vs {system2_name}",
        "=" * 70,
        "",
        f"{system1_name} Accuracy: {system1_correct}/{total} ({system1_correct/total*100:.2f}%)",
        f"{system2_name} Accuracy: {system2_correct}/{total} ({system2_correct/total*100:.2f}%)",
        "",
        f"Both correct:           {both_correct:4d} ({both_correct/total*100:.2f}%)",
        f"Both wrong:             {both_wrong:4d} ({both_wrong/total*100:.2f}%)",
        f"Only {system1_name} correct: {len(only_system1_correct):4d} ({len(only_system1_correct)/total*100:.2f}%)",
        f"Only {system2_name} correct: {len(only_system2_correct):4d} ({len(only_system2_correct)/total*100:.2f}%)",
        "",
        "=" * 70,
    ]

    return "\n".join(report)
