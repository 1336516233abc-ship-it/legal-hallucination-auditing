"""Synthetic regression tests; these do not constitute real model or human evaluation."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from ui.app import ContractReviewUI
from ui.backends import BoundedChatModel, ThreadBoundModel, load_backend
from ui.service import ReviewService


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(text)


class SequenceModel:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.tokenizer = Tokenizer()
        self.count = 0

    def generate(self, prompt, **options):
        self.count += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return {"text": response, "tokens": 7}


def test_swapping_contracts_drops_old_evidence_and_counts_every_call():
    model = SequenceModel("notice", "Label: entailment\nAnswer: Notice is thirty days [1].",
                          "Status: SUPPORTED\nEvidence: [1]", "notice",
                          "Label: contradiction\nAnswer: Notice is sixty days [1].",
                          "Status: SUPPORTED\nEvidence: [1]")
    ui = ReviewService(model)
    first = ui.review("Notice is thirty days.", "Notice is thirty days.")
    second = ui.review("Notice is sixty days.", "Notice is thirty days.")
    assert "矛盾" in second["answer"]
    assert "sixty" in second["evidence"] and "thirty" not in second["evidence"]
    assert first["record"]["contract"] == "Notice is thirty days."
    assert second["metadata"]["generated_tokens_all_calls"] == 21
    assert len(second["record"]["model_calls"]) == 3
    assert first["metadata"]["contract_sha256"] != second["metadata"]["contract_sha256"]
    for call in second["record"]["model_calls"][1:]:
        assert "Notice is thirty days." not in call["prompt"].split("Question:")[0]


@pytest.mark.parametrize("response,fragment", [
    ("Label: true\nAnswer: untrusted", "有效结果"),
    ("Label: entailment\nAnswer: untrusted [99].", "不存在的证据编号"),
])
def test_invalid_output_is_withheld_but_retained_in_request_trace(response, fragment):
    result = ReviewService(SequenceModel("notice", response)).review("Notice required.", "Notice required.")
    assert result["answer"] == "" and fragment in result["status"]
    assert result["record"]["trace"]["raw_response"] == response
    assert result["metadata"]["model_calls"] == 2


def test_cuad_paraphrase_cannot_be_displayed_as_an_extracted_quote():
    model = SequenceModel("notice", '{"label":"present","extracted_spans":["Give a warning."],"citations":[1]}')
    result = ReviewService(model).review("Written notice is required.", "What notice is needed?", "cuad")
    assert result["answer"] == "" and "原文或引用检查" in result["status"]
    assert result["record"]["trace"]["grounding_error"]


def test_cuad_absent_is_a_task_conclusion_not_an_error_or_abstention():
    model = SequenceModel("notice", '{"label":"absent","extracted_spans":[],"citations":[]}')
    result = ReviewService(model).review("New York law applies.", "What notice is needed?", "cuad", False, True)
    assert "未找到相关条款" in result["answer"]
    assert result["record"]["trace"]["parse_error"] is None
    assert result["metadata"]["revision_enabled"] is False and model.count == 2


def test_revision_keeps_initial_verdict_separate_from_final_answer():
    model = SequenceModel("notice", "Label: entailment\nAnswer: It says 30 days [1].",
                          "Status: CONTRADICTED\nReasoning: It says 15 days.\nEvidence: [1]",
                          "Label: contradiction\nAnswer: It says 15 days [1].")
    result = ReviewService(model).review("It says 15 days notice.", "It says 30 days notice.")
    assert "矛盾" in result["answer"] and "15 days" in result["evidence"]
    assert "最终回答的语义尚未再次验证" in result["verification"]
    assert result["metadata"]["generated_tokens_all_calls"] == 28
    assert result["record"]["trace"]["initial"]["label"] == "entailment"


def test_verifier_parse_failure_is_not_shown_as_verification_success():
    model = SequenceModel("notice", "Label: entailment\nAnswer: Notice is required [1].", "invalid verifier output")
    result = ReviewService(model).review("Notice is required.", "Notice is required.", revise=False)
    assert "检查结果格式无效" in result["verification"]
    assert "未启用修订" in result["verification"]


def test_runtime_failure_is_recorded_and_next_request_can_recover():
    model = SequenceModel(RuntimeError("synthetic failure"), "notice", "Label: entailment\nAnswer: Notice required [1].")
    ui = ReviewService(model)
    failure = ui.review("Notice required.", "Notice required.")
    assert failure["answer"] == failure["evidence"] == ""
    assert "没有自动重试" in failure["status"]
    assert failure["record"]["model_calls"][0]["error"] == "RuntimeError: synthetic failure"
    success = ui.review("Notice required.", "Notice required.", verify=False)
    assert "得到支持" in success["answer"]
    assert success["metadata"]["model_calls"] == 2


@pytest.mark.parametrize("source,query,dataset", [(" ", "Q", "contractnli"),
                                                  ("C", "", "cuad"),
                                                  ("C", "Q", "unsupported"),
                                                  ("C" * 200001, "Q", "cuad"),
                                                  ("C", "Q" * 2001, "contractnli")])
def test_bad_inputs_return_empty_results_without_model_calls(source, query, dataset):
    model = SequenceModel()
    result = ReviewService(model).review(source, query, dataset)
    assert model.count == 0
    assert result["record"] is None and result["metadata"] is None
    assert result["answer"] == result["evidence"] == ""


def test_concurrent_requests_keep_their_own_source_and_trace():
    class EchoModel:
        tokenizer = Tokenizer()

        def generate(self, prompt, **options):
            if "Retrieved Evidence:" in prompt:
                text = "Alpha" if "Alpha" in prompt else "Beta"
                return {"text": f"Label: entailment\nAnswer: {text} notice required [1].", "tokens": 7}
            return {"text": "notice", "tokens": 3}

    ui = ReviewService(EchoModel())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(ui.review, f"{name} notice required.", "Notice required.", "contractnli", False)
                   for name in ("Alpha", "Beta")]
        results = [f.result() for f in futures]
    for name, other, result in zip(("Alpha", "Beta"), ("Beta", "Alpha"), results):
        assert name in result["answer"] and other not in result["evidence"]
        assert all(other not in call["prompt"] for call in result["record"]["model_calls"])
        assert result["metadata"]["generated_tokens_all_calls"] == 10


def test_old_python_callback_keeps_its_four_outputs():
    model = SequenceModel("notice", "Label: entailment\nAnswer: Notice required [1].")
    outputs = ContractReviewUI(model=model).analyze_contract("Notice required.", "Notice required.", False, False)
    assert len(outputs) == 4 and json.loads(outputs[3])["model_calls"] == 2


def test_all_backend_calls_use_chat_formatting_and_a_budget():
    class ChatModel(SequenceModel):
        def generate_for_baseline(self, prompt, **options):
            return self.generate(prompt, **options)

    backend = ChatModel("ok")
    bounded = BoundedChatModel(backend, 10)
    assert bounded.generate("short")["text"] == "ok"
    with pytest.raises(ValueError, match="input token budget"):
        bounded.generate_json_for_baseline("x" * 11)
    assert backend.count == 1


def test_model_directory_must_match_selected_family(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type":"mistral"}')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="model family"):
        load_backend("llama3", "mlx", tmp_path, "auto", 4096)


def test_thread_affine_backend_loads_and_generates_on_the_same_worker():
    class AffineBackend:
        def __init__(self):
            self.creator = threading.get_ident()

        def generate_for_baseline(self, prompt, **options):
            assert threading.get_ident() == self.creator
            return {"text": prompt, "tokens": 1, "worker": self.creator}

    model = ThreadBoundModel(AffineBackend)
    try:
        with ThreadPoolExecutor(max_workers=2) as clients:
            outputs = list(clients.map(model.generate_for_baseline, ["A", "B"]))
        assert [row["text"] for row in outputs] == ["A", "B"]
        assert outputs[0]["worker"] == outputs[1]["worker"] != threading.get_ident()
    finally:
        model.close()


def test_queued_page_request_restores_submitted_inputs_before_showing_results():
    model = SequenceModel("notice", "Label: entailment\nAnswer: Notice required [1].")
    page = ContractReviewUI(model=model).create_interface()
    handler = next(block.fn for block in page.fns.values() if block.api_name == "review")
    stream = handler("Notice required.", "Notice required.", "contractnli", False, True)
    pending = next(stream)
    assert model.count == 0 and pending[:2] == ("", "")
    assert [update["value"] for update in pending[6:11]] == [
        "Notice required.", "Notice required.", "contractnli", False, True]
    assert all(update["interactive"] is False for update in pending[6:])
    completed = next(stream)
    assert "得到支持" in completed[0]
    assert completed[5]["contract"] == pending[6]["value"]
    assert all(update["interactive"] is True for update in completed[6:])
