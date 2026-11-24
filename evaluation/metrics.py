"""Task-aware metrics for ContractNLI and CUAD.

Author: Li Xinru (3036807959)
"""

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Optional

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from retrieval.common import LABELS
from retrieval.evidence import aggregate_evidence, score_evidence


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
    answer_word_set_recall: Optional[float] = None
    initial_verifier_unsupported_rate: Optional[float] = None
    initial_verifier_claims: int = 0
    initial_verifier_failed_claims: int = 0


def _label(value):
    return value.strip().lower() if isinstance(value, str) else None


def _label_inputs(predictions, ground_truth, dataset):
    pairs = list(zip(predictions, ground_truth, strict=True))
    if not pairs:
        return [], [], []
    true = [_label(gold.get("label")) for _, gold in pairs]
    if dataset is None:
        datasets = [task for task, labels in LABELS.items() if set(true) <= labels]
        if len(datasets) != 1:
            raise ValueError("Ground truth must use labels from one supported dataset")
        dataset = datasets[0]
    if dataset not in LABELS or not set(true) <= LABELS[dataset]:
        raise ValueError("Ground truth labels do not match the selected dataset")
    labels = sorted(LABELS[dataset])
    predicted = []
    for row, _ in pairs:
        label = _label(row.get("label"))
        if label not in labels or row.get("parse_error") or row.get("format_pass") is False:
            label = "invalid"
        predicted.append(label)
    return true, predicted, labels


def _evidence_metrics(retrieved_evidence, ground_truth, k):
    if type(k) is not int or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if len(retrieved_evidence) != len(ground_truth):
        raise ValueError("Evidence and references must be aligned")
    pairs = zip(retrieved_evidence, ground_truth, strict=True)
    scores = []
    for chunks, gold in pairs:
        if not gold.get("evidence_spans"):
            continue
        if (not isinstance(gold.get("context"), str) or chunks is None
                or any(not isinstance(chunk, dict) or chunk.get("start") is None or chunk.get("end") is None
                       or not chunk.get("doc_id", (chunk.get("metadata") or {}).get("doc_id"))
                       for chunk in chunks[:k])):
            # A partial corpus must not be reported with the full-corpus denominator.
            return aggregate_evidence([])
        scores.append(score_evidence(chunks, gold, k=k))
    return aggregate_evidence(scores)


class MetricsCalculator:
    """Calculate task metrics and separately scoped verifier diagnostics."""

    def compute_all_metrics(self, predictions, ground_truth, retrieved_evidence=None,
                            latencies=None, token_counts=None, *, dataset=None, k=5):
        """Evaluate aligned predictions and references from one task."""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and references must be aligned")
        result = EvaluationResult()
        if any("label" in row for row in ground_truth):
            true, predicted, labels = _label_inputs(predictions, ground_truth, dataset)
            result.label_accuracy = float(accuracy_score(true, predicted))
            result.macro_f1 = float(f1_score(true, predicted, labels=labels, average="macro", zero_division=0))
            result.micro_f1 = result.label_accuracy
            result.per_class_metrics = self.compute_per_class_metrics(predictions, ground_truth, dataset=dataset)
        if retrieved_evidence is not None:
            scores = _evidence_metrics(retrieved_evidence, ground_truth, k)
            result.evidence_recall_at_k = scores["evidence_recall_at_k"]
            result.evidence_hit_rate = scores["evidence_hit_rate"]
        (result.initial_verifier_unsupported_rate, result.initial_verifier_claims,
         result.initial_verifier_failed_claims) = self.compute_initial_verifier_metrics(predictions)
        result.answer_word_set_recall = self.compute_answer_word_set_recall(predictions, ground_truth)
        result.abstention_precision = self.compute_abstention_precision(predictions, ground_truth)
        if latencies is not None and len(latencies):
            result.avg_latency = float(mean(latencies))
        if token_counts is not None and len(token_counts):
            result.total_tokens = int(sum(token_counts))
        return result

    def compute_label_accuracy(self, predictions, ground_truth, *, dataset=None):
        true, predicted, _ = _label_inputs(predictions, ground_truth, dataset)
        return float(accuracy_score(true, predicted)) if true else None

    def compute_macro_f1(self, predictions, ground_truth, *, dataset=None):
        true, predicted, labels = _label_inputs(predictions, ground_truth, dataset)
        return float(f1_score(true, predicted, labels=labels, average="macro", zero_division=0)) if true else None

    def compute_micro_f1(self, predictions, ground_truth, *, dataset=None):
        """Micro-F1 equals accuracy for single-label rows including invalid predictions."""
        return self.compute_label_accuracy(predictions, ground_truth, dataset=dataset)

    def compute_per_class_metrics(self, predictions, ground_truth, *, dataset=None):
        true, predicted, labels = _label_inputs(predictions, ground_truth, dataset)
        if not true:
            return {}
        precision, recall, f1, _ = precision_recall_fscore_support(
            true, predicted, labels=labels, zero_division=0)
        return {label: {"precision": float(p), "recall": float(r), "f1": float(f)}
                for label, p, r, f in zip(labels, precision, recall, f1, strict=True)}

    def compute_evidence_recall(self, retrieved_evidence, ground_truth, k=5):
        return _evidence_metrics(retrieved_evidence, ground_truth, k)["evidence_recall_at_k"]

    def compute_evidence_hit_rate(self, retrieved_evidence, ground_truth, k=5):
        return _evidence_metrics(retrieved_evidence, ground_truth, k)["evidence_hit_rate"]

    def compute_hallucination_rate(self, predictions):
        """Compatibility entry point. Independent final-answer labels are scored by the review pipeline."""
        return None

    def compute_initial_verifier_metrics(self, predictions):
        """Return unsupported rate among valid checks, claims attempted, and claims failed."""
        total = failed = unsupported = 0
        unsupported_statuses = {"contradicted", "not_enough_info"}
        known_statuses = unsupported_statuses | {"supported"}
        for row in predictions:
            for claim in (row.get("verification") or {}).get("claims") or ():
                total += 1
                status = claim.get("status")
                if claim.get("parse_error") or status not in known_statuses:
                    failed += 1
                else:
                    unsupported += status in unsupported_statuses
        valid = total - failed
        return (unsupported / valid if valid else None), total, failed

    def compute_answer_word_set_recall(self, predictions, ground_truth):
        """Mean exact word-set overlap with nonempty human reference answers."""
        scores = []
        for prediction, gold in zip(predictions, ground_truth, strict=True):
            reference = gold.get("answer")
            reference_words = set(reference.lower().split()) if isinstance(reference, str) else set()
            if not reference_words:
                continue
            answer = prediction.get("final_answer", prediction.get("answer"))
            if prediction.get("parse_error") or prediction.get("format_pass") is False:
                answer = None
            words = set(answer.lower().split()) if isinstance(answer, str) else set()
            scores.append(len(words & reference_words) / len(reference_words))
        return mean(scores) if scores else None

    def compute_answer_completeness(self, predictions, ground_truth):
        """Compatibility alias for the lexical proxy, not a semantic assessment."""
        return self.compute_answer_word_set_recall(predictions, ground_truth)

    def compute_abstention_precision(self, predictions, ground_truth):
        """Score abstention only for explicit predictions with complete reference judgments."""
        judgments = [gold.get("should_abstain") for prediction, gold in zip(predictions, ground_truth, strict=True)
                     if prediction.get("abstained") is True and not prediction.get("parse_error")
                     and prediction.get("format_pass") is not False]
        if not judgments or any(type(value) is not bool for value in judgments):
            return None
        return mean(judgments)

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
        if result.initial_verifier_claims:
            lines.append(f"Initial Verifier Claims: {result.initial_verifier_claims:,}")
            lines.append(f"Initial Verifier Failed Claims: {result.initial_verifier_failed_claims:,}")
        if result.initial_verifier_unsupported_rate is not None:
            lines.append(f"Initial Verifier Unsupported Rate: {result.initial_verifier_unsupported_rate:.4f}")
        if result.answer_word_set_recall is not None:
            lines.append(f"Answer Word-Set Recall: {result.answer_word_set_recall:.4f}")
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
