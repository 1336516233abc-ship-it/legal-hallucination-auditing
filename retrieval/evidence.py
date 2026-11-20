"""Strict offset-based retrieval evaluation, separate from legacy QA metrics."""

from retrieval.common import document_id, example_dataset, example_dict


def _merge(spans):
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _span(span, context):
    if (not isinstance(span, (list, tuple)) or len(span) != 2
            or not all(type(value) is int for value in span)
            or not 0 <= span[0] < span[1] <= len(context)):
        raise ValueError(f"Invalid source character span: {span}")
    return tuple(span)


def score_evidence(chunks, ground_truth, k=5):
    """Measure full gold-span coverage and gold-character coverage at k.

    A gold span is recalled only when all its non-whitespace characters are
    covered by retrieved intervals from the same source document. No-evidence
    examples are excluded (None), rather than scored as zero or perfect.
    Invalid or missing contexts and inconsistent source slices are rejected.
    """
    if type(k) is not int or k < 0:
        raise ValueError("k must be a nonnegative integer")
    ground_truth = example_dict(ground_truth)
    if not isinstance(ground_truth.get("context"), str):
        raise ValueError("Ground truth must contain the original context")
    context = ground_truth["context"]
    dataset = example_dataset(ground_truth, "contractnli")
    expected_id = document_id(context, ground_truth.get("metadata"), dataset)
    retrieved = []
    for chunk in chunks[:k]:
        if chunk.get("doc_id", chunk.get("metadata", {}).get("doc_id")) != expected_id:
            raise ValueError("Retrieved evidence belongs to a different or unspecified document")
        start, end = _span((chunk.get("start"), chunk.get("end")), context)
        if chunk.get("text") != context[start:end]:
            raise ValueError("Retrieved evidence text does not match its source offsets")
        retrieved.append((start, end))
    gold = [_span(span, context) for span in ground_truth.get("evidence_spans") or []]
    if not gold:
        return None
    retrieved = _merge(retrieved)
    meaningful = lambda start, end: sum(not char.isspace() for char in context[start:end])
    covered = lambda start, end: sum(meaningful(max(start, left), min(end, right))
                                   for left, right in retrieved if max(start, left) < min(end, right))
    if any(meaningful(start, end) == 0 for start, end in gold):
        raise ValueError("Gold evidence must contain non-whitespace characters")
    recalled = sum(covered(start, end) == meaningful(start, end) for start, end in gold)
    gold_union = _merge(gold)
    total_characters = sum(meaningful(start, end) for start, end in gold_union)
    return {"gold_spans": len(gold), "evidence_recall_at_k": recalled / len(gold),
            "evidence_hit_rate": float(recalled > 0),
            "evidence_character_recall": sum(covered(start, end) for start, end in gold_union) / total_characters}


def aggregate_evidence(scores):
    valid = [score for score in scores if score is not None]
    return {"evaluated_examples": len(valid), "excluded_no_evidence": len(scores) - len(valid),
            **{metric: sum(score[metric] for score in valid) / len(valid) if valid else None
               for metric in ["evidence_recall_at_k", "evidence_hit_rate", "evidence_character_recall"]}}
