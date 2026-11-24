"""Boundary tests use synthetic inputs only; they do not supply human judgments."""

import copy
import json
from pathlib import Path
import pytest

from local_demo import LocalReviewDemo
from score_reviews import score


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(text)


class SequenceModel:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []
        self.tokenizer = Tokenizer()

    def generate(self, prompt, **options):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        result = {"text": response, "tokens": 7}
        self.calls.append({"prompt": prompt, **result})
        return result


def test_changed_contract_uses_current_source_and_counts_all_calls():
    model = SequenceModel("notice", "Label: entailment\nAnswer: Notice is thirty days [1].",
                          "Status: SUPPORTED\nEvidence: [1]", "notice",
                          "Label: contradiction\nAnswer: Notice is sixty days [1].",
                          "Status: SUPPORTED\nEvidence: [1]")
    demo = LocalReviewDemo(model)
    demo.analyze("Notice is thirty days.", "Notice is thirty days.", "contractnli")
    answer, evidence, status, metadata = demo.analyze("Notice is sixty days.", "Notice is thirty days.", "contractnli")
    assert "sixty" in evidence and "thirty" not in evidence
    assert "矛盾" in answer
    assert json.loads(metadata)["generated_tokens_all_calls"] == 21


@pytest.mark.parametrize("response, reason", [("Label: true\nAnswer: notice", "missing or invalid task label"),
                                               ("Label: entailment\nAnswer: Notice required [99].", "outside the model input")])
def test_invalid_answer_is_stopped_with_a_reason(response, reason):
    model = SequenceModel("notice", response)
    demo = LocalReviewDemo(model)
    result = demo.analyze("Written notice is required.", "Notice is required.", "contractnli")
    assert "已停止展示" in result[0] and reason in result[0]
    assert len(demo.last_record["model_calls"]) == 2 and not model.calls


def test_unavailable_model_is_reported_without_retry():
    demo = LocalReviewDemo(SequenceModel(RuntimeError("simulated unavailable")))
    result = demo.analyze("Written notice is required.", "Notice is required.", "contractnli")
    assert "本次处理失败" in result[0] and "未自动重试" in result[2]
    assert demo.last_record["error"] == "RuntimeError: simulated unavailable"


def test_empty_request_makes_no_calls():
    model = SequenceModel()
    assert "请同时填写" in LocalReviewDemo(model).analyze(" ", "Question", "contractnli")[0]
    assert not model.calls


def synthetic_review():
    packet = {"scope": "synthetic_test", "review_target": "test", "contracts": {"c": "A😀B"},
              "items": [{"blind_id": "TEST", "contract_id": "c", "format_pass": True, "display_suppressed": False}]}
    review = [{"blind_id": "TEST", "annotator": "synthetic_test_only", "completed": True,
               "response_type": "substantive", "all_claims_reviewed": True, "needs_adjudication": False,
               "claims": [{"claim": "A is in the test text", "rationale": "Synthetic boundary test", "verdict": "supported",
                           "evidence_spans": [{"start": 1, "end": 2, "quote": "😀"}]}]}]
    return packet, review


def test_unreviewed_real_packet_keeps_metrics_null():
    root = Path(__file__).parent / "human_review/reviewer"
    result = score(json.loads((root / "packet.json").read_text()), json.loads((root / "reviewer_A_blank.json").read_text()))
    assert result["complete_without_pending_adjudication"] == 0 and result["outputs"] == 60
    assert result["claim_level_unsupported_rate"] is None and result["answer_level_unsupported_rate"] is None


def test_unicode_offsets_and_pending_adjudication():
    packet, reviews = synthetic_review()
    assert score(packet, reviews)["claim_denominator"] == 1
    reviews[0]["needs_adjudication"] = True
    result = score(packet, reviews)
    assert result["pending_adjudication"] == 1 and result["claim_level_unsupported_rate"] is None


@pytest.mark.parametrize("fault", ["quote", "duplicate", "missing", "verdict", "incomplete_claims"])
def test_bad_annotations_are_rejected(fault):
    packet, reviews = synthetic_review()
    if fault == "quote":
        reviews[0]["claims"][0]["evidence_spans"][0]["quote"] = "invented"
    elif fault == "duplicate":
        reviews.append(copy.deepcopy(reviews[0]))
    elif fault == "missing":
        reviews = []
    elif fault == "verdict":
        reviews[0]["claims"][0]["verdict"] = None
    else:
        reviews[0]["all_claims_reviewed"] = False
    with pytest.raises(ValueError):
        score(packet, reviews)
