import pytest

pytest.importorskip("sklearn")

from evaluation.metrics import MetricsCalculator
from retrieval.common import document_id


def test_invalid_and_cross_task_labels_cannot_match_neutral_or_entailment():
    calculator = MetricsCalculator()
    predictions = [{"label": "invalid"}, {"label": "present"}, {"label": "neutral", "parse_error": "bad"}]
    gold = [{"label": "neutral"}, {"label": "entailment"}, {"label": "neutral"}]
    result = calculator.compute_all_metrics(predictions, gold)
    assert result.label_accuracy == result.micro_f1 == 0
    assert set(result.per_class_metrics) == {"neutral", "entailment", "contradiction"}


def test_cuad_macro_f1_averages_only_legal_classes():
    calculator = MetricsCalculator()
    predictions = [{"label": "present"}, {"label": "invalid"}]
    gold = [{"label": "present"}, {"label": "absent"}]
    assert calculator.compute_macro_f1(predictions, gold, dataset="cuad") == 0.5
    with pytest.raises(ValueError):
        calculator.compute_label_accuracy(predictions, gold, dataset="contractnli")


def test_evidence_hit_rate_independent_and_shared_top_k():
    calculator = MetricsCalculator()
    context = "alpha beta"
    gold = [{"context": context, "evidence_spans": [(0, 5)]},
            {"context": context, "evidence_spans": [(0, 5), (6, 10)]}]
    chunks = [{"doc_id": document_id(context), "text": context[start:end], "start": start, "end": end}
              for start, end in [(0, 5), (6, 10)]]
    result = calculator.compute_all_metrics([{}, {}], gold, [[chunks[0]], [chunks[1]]], k=1)
    assert (result.evidence_hit_rate, result.evidence_recall_at_k) == (1, 0.75)
    assert calculator.compute_evidence_hit_rate([[chunks[0]], [chunks[1]]], gold, k=0) == 0
    assert calculator.compute_evidence_recall([["alpha"], ["beta"]], gold) is None


def test_verifier_failure_excluded_from_internal_rate_and_independent_rate_unavailable():
    calculator = MetricsCalculator()
    predictions = [{"verification": {"claims": [
        {"status": "supported"}, {"status": "contradicted"},
        {"status": "not_enough_info", "parse_error": "bad answer"},
    ]}}]
    result = calculator.compute_all_metrics(predictions, [{}])
    assert result.initial_verifier_unsupported_rate == 0.5
    assert (result.initial_verifier_claims, result.initial_verifier_failed_claims) == (3, 1)
    assert result.hallucination_rate is None
    assert calculator.compute_hallucination_rate(predictions) is None


def test_word_recall_ignores_null_reference_and_scores_withheld_answer_as_empty():
    calculator = MetricsCalculator()
    predictions = [{"answer": "ignored"}, {"answer": "alpha beta", "final_answer": ""},
                   {"answer": "alpha"}]
    gold = [{"answer": None}, {"answer": "alpha beta"}, {"answer": "alpha beta"}]
    result = calculator.compute_all_metrics(predictions, gold)
    assert result.answer_word_set_recall == 0.25
    assert result.answer_completeness is None


def test_abstention_requires_explicit_reference_and_nonzero_denominator():
    calculator = MetricsCalculator()
    assert calculator.compute_abstention_precision([{"label": "neutral"}], [{}]) is None
    assert calculator.compute_abstention_precision([{"abstained": True}], [{"label": "neutral"}]) is None
    assert calculator.compute_abstention_precision([{"abstained": True}], [{"should_abstain": True}]) == 1
