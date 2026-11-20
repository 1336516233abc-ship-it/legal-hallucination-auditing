"""Regression tests for the scored Agent path and its failure handling."""

import pytest
from conftest import CharacterTokenizer

pytest.importorskip("torch")
pytest.importorskip("transformers")
from agent.agentic_rag import AgenticRAG
from agent.verifier.verifier import EvidenceVerifier


class ScriptedModel:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.tokenizer = CharacterTokenizer()
        self.prompts = []

    def generate(self, prompt, **options):
        self.prompts.append(prompt)
        return {"text": next(self.responses), "tokens": 5}


def test_invalid_initial_label_stops_without_fabricating_a_prediction():
    model = ScriptedModel("notice", "Label: true\nAnswer: A notice is required.")
    result = AgenticRAG(model).answer("Does it require notice?", context="Written notice is required.",
                                       dataset="contractnli", max_input_tokens=2000)

    assert result["label"] == "invalid"
    assert result["parse_error"] == "missing or invalid task label"
    assert result["verification"] is None and result["revision"] is None
    assert len(model.prompts) == 2


def test_revision_label_replaces_initial_label_and_uses_one_contract():
    model = ScriptedModel(
        "notice", "Label: entailment\nAnswer: It says 30 days.",
        "Status: CONTRADICTED\nReasoning: The contract says 15 days.\nEvidence: [1]",
        "Label: contradiction\nAnswer: It says 15 days [1].",
        "law", "Label: true\nAnswer: California law applies.",
    )
    agent = AgenticRAG(model)
    first, second = agent.batch_answer([
        {"id": "a", "query": "Does it say 30 days notice?", "context": "It says 15 days notice."},
        {"id": "b", "query": "Does it choose California law?", "context": "New York law applies."},
    ], dataset="contractnli", max_input_tokens=2000)

    assert first["initial"]["label"] == "entailment"
    assert first["label"] == "contradiction"
    assert first["revision"]["revision_type"] == "correction"
    assert first["invalid_citations"] == []
    assert second["id"] == "b" and second["label"] == "invalid"
    assert [chunk["text"] for chunk in second["retrieved_chunks"]] == ["New York law applies."]
    assert "15 days" not in model.prompts[-1]


def test_invalid_verifier_and_revision_keep_failures_visible_after_reretrieval():
    model = ScriptedModel(
        "notice", '{"label":"absent","answer":"","extracted_spans":[],"citations":[]}',
        "I cannot emit a verification status.", '{"label":"present","answer":',
    )
    result = AgenticRAG(model).answer("Is 15 days notice required?", context="Notice is required.",
                                       dataset="cuad", max_input_tokens=2000)

    assert result["verification"]["claims"][0]["parse_error"] == "missing or invalid verifier status"
    assert result["revision"]["revision_type"] == "reretrieval"
    assert result["revision"]["prompt_evidence"][0]["text"] == "Notice is required."
    assert result["label"] == "invalid" and result["parse_error"] is not None
    assert result["answer_is_extractive"] is None
    assert len(model.prompts) == 4


def test_verifier_rejects_citation_trimmed_from_prompt():
    claim, first = "Notice is required.", "Written notice is required."
    model = ScriptedModel("Status: SUPPORTED\nEvidence: [1]", "Status: SUPPORTED\nEvidence: [2]")
    verifier = EvidenceVerifier(model)
    allowed = verifier.verify([claim], [first], strict_evidence=True)
    budget = len(model.prompts[0])
    result = verifier.verify([claim], [first, "EVIDENCE OUTSIDE BUDGET"],
                             max_input_tokens=budget, strict_evidence=True)

    assert allowed["overall_status"] == "supported"
    assert "EVIDENCE OUTSIDE BUDGET" not in model.prompts[-1]
    verdict = result["claims"][0]
    assert verdict["available_evidence_count"] == 1
    assert verdict["invalid_evidence_ids"] == [2] and verdict["supporting_evidence"] == []
    assert verdict["status"] == "not_enough_info" and result["hallucination_detected"]


@pytest.mark.parametrize("evidence_ids", ["[-1]", "[0]", "[true]", "[]", "1", "[1] or [2]"])
def test_verifier_does_not_accept_invalid_or_missing_support(evidence_ids):
    model = ScriptedModel(f"Status: SUPPORTED\nEvidence: {evidence_ids}")
    result = EvidenceVerifier(model).verify(["Written notice required."], ["Written notice required."],
                                            strict_evidence=True)
    assert result["claims"][0]["parse_error"] is not None
    assert result["claims"][0]["status"] == "not_enough_info"


@pytest.mark.parametrize("excerpt, citations", [("Notice in writing is needed.", "[1]"), ("Written notice is required.", "[2]")])
def test_agent_withholds_ungrounded_cuad_answer_but_keeps_label(excerpt, citations):
    model = ScriptedModel("notice", f'{{"label":"present","extracted_spans":["{excerpt}"],"citations":{citations}}}')
    result = AgenticRAG(model).answer("What notice is needed?", "Written notice is required.", dataset="cuad",
                                       max_input_tokens=2000)

    assert result["label"] == "present" and result["parse_error"] is None
    assert result["answer"] == excerpt and result["grounding_error"]
    assert result["final_answer"] == "" and result["verification"] is None
