"""Regression tests for missing context and partial/overlapping evidence."""

import pytest
from retrieval.common import document_id
from retrieval.evidence import score_evidence, aggregate_evidence


def chunk(text, start, end):
    return {"text": text[start:end], "start": start, "end": end, "doc_id": document_id(text)}


def test_union_coverage_and_actual_k():
    gt = {"context": "abcdefghij", "evidence_spans": [[0, 6], [8, 10]]}
    chunks = [chunk(gt["context"], 0, 4), chunk(gt["context"], 3, 6), chunk(gt["context"], 8, 10)]
    first = score_evidence(chunks, gt, 1)
    assert first["evidence_recall_at_k"] == 0 and first["evidence_character_recall"] == .5
    assert score_evidence(chunks, gt, 2)["evidence_recall_at_k"] == .5
    assert score_evidence(chunks, gt, 3)["evidence_recall_at_k"] == 1
    assert score_evidence(chunks, gt, 0)["evidence_hit_rate"] == 0


def test_no_evidence_excluded_and_hits_are_per_example():
    gt = {"context": "alpha beta", "evidence_spans": [[0, 5], [6, 10]]}
    scores = [score_evidence([chunk(gt["context"], 0, 5)], gt),
              score_evidence([chunk(gt["context"], 6, 10)], gt),
              score_evidence([], {"context": "nothing", "evidence_spans": None})]
    result = aggregate_evidence(scores)
    assert result["evidence_hit_rate"] == 1
    assert result["evaluated_examples"] == 2 and result["excluded_no_evidence"] == 1
    assert aggregate_evidence([])["evidence_recall_at_k"] is None


def test_reject_missing_context_wrong_document_bad_offsets_and_changed_text():
    with pytest.raises(ValueError):
        score_evidence([], {"evidence_spans": [[0, 5]]})
    gt = {"context": "alpha", "evidence_spans": [[0, 5]]}
    good = chunk("alpha", 0, 5)
    for bad in [{**good, "doc_id": "other"}, {**good, "start": -1}, {**good, "text": "changed"}]:
        with pytest.raises(ValueError):
            score_evidence([bad], gt)
    with pytest.raises(ValueError):
        score_evidence([], {**gt, "evidence_spans": [[0, 100]]})


def test_whitespace_gaps_do_not_create_false_evidence_misses():
    gt = {"context": "alpha  beta", "evidence_spans": [[0, 11]]}
    chunks = [chunk(gt["context"], 0, 5), chunk(gt["context"], 7, 11)]
    result = score_evidence(chunks, gt)
    assert result["evidence_recall_at_k"] == 1 and result["evidence_character_recall"] == 1
