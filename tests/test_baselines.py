"""Contract isolation, task schemas, source citations and prompt budgets."""

import json
from types import SimpleNamespace
import pytest
from conftest import RecordingLLM, CharacterTokenizer
from retrieval import DirectPromptingBaseline, StandardRAGBaseline
from retrieval.common import completed_json_object, parse_response, select_examples, load_examples, task_question


def test_direct_preserves_tail_and_multiline_response():
    llm = RecordingLLM("Answer: First line.\nSecond line.\nLabel: Entailment")
    result = DirectPromptingBaseline(llm).answer("query", "x" * 2200 + "TAIL_EVIDENCE")
    assert "TAIL_EVIDENCE" in llm.prompts[0]
    assert result["label"] == "entailment" and "Second line" in result["answer"]
    assert not result["input_budget"]["truncated"]


@pytest.mark.parametrize("dataset,label", [("contractnli", "contradiction"), ("cuad", "present"), ("cuad", "absent")])
def test_task_labels(dataset, label):
    answer = "contract clause" if label != "absent" else ""
    parsed = parse_response(json.dumps({"answer": answer, "label": label}), dataset)
    assert parsed["label"] == label and parsed["parse_error"] is None
    if dataset == "cuad":
        assert not parsed["abstained"]


@pytest.mark.parametrize("response", ["I would choose entailment", 'Label: unknown\nAnswer: no',
    '{"answer": "yes", "label": "present"}', 'Answer: yes\nLabel: entailment\nLabel: neutral'])
def test_invalid_nli_responses_are_visible(response):
    result = parse_response(response, "contractnli")
    assert result["label"] == "invalid" and result["parse_error"]


def test_cuad_malformed_output_and_nli_label_not_silently_mapped():
    assert parse_response('Answer: yes\nLabel: entailment', 'cuad')["label"] == "invalid"
    assert parse_response('{"answer":null,"label":"present"}', 'cuad')["parse_error"]
    assert parse_response('{"answer":"found","label":"absent"}', 'cuad')["parse_error"]


@pytest.mark.parametrize("baseline_type", [DirectPromptingBaseline, StandardRAGBaseline])
def test_cuad_verbatim_checks_use_shown_source_and_check_legacy_answer(baseline_type):
    context = "The agreement ends on 3 July 2027."
    compact = '{"label":"present","extracted_spans":["3 July 2027"],"citations":[]}'
    good = baseline_type(RecordingLLM(compact), dataset="cuad").answer("When does it end?", context)
    assert good["answer"] == "3 July 2027" and good["answer_is_extractive"] is True
    assert good["extraction_error"] is None

    inconsistent = '{"label":"present","answer":"July 3, 2027","extracted_spans":["3 July 2027"],"citations":[]}'
    bad = baseline_type(RecordingLLM(inconsistent), dataset="cuad").answer("When does it end?", context)
    assert bad["parse_error"] is None and bad["label"] == "present"
    assert bad["answer_is_extractive"] is False and bad["extraction_error"]


def test_cuad_question_uses_task_information_without_legal_review_boilerplate():
    original = ('Highlight the parts (if any) of this contract related to "Agreement Date" '
                'that should be reviewed by a lawyer. Details: The date of the contract')
    assert task_question(original, "cuad") == "Find the contract text for the field 'Agreement Date'. The date of the contract"
    assert task_question("When was it signed?", "cuad") == "When was it signed?"
    assert task_question(original, "contractnli") == original


def test_json_boundary_waits_for_closing_brace_outside_strings():
    object_text = json.dumps({"extracted_spans": ['clause } with a "quote"']})
    assert completed_json_object(object_text[:-1]) is None
    assert completed_json_object(f"Note: {object_text}") is None
    assert completed_json_object(f"  {object_text} extra") == object_text


def test_mlx_json_generation_stops_stream_and_records_boundary():
    from experiments.run_baseline_pilot import MLXPilotModel

    llm = MLXPilotModel.__new__(MLXPilotModel)
    llm.model = llm.sampler = None
    llm.tokenizer = CharacterTokenizer()
    llm._format = lambda text: text
    llm.record_calls, llm.calls = True, []
    closed = []

    def stream(*args, **kwargs):
        try:
            yield SimpleNamespace(text='{"label":"absent"', generation_tokens=5, finish_reason=None)
            yield SimpleNamespace(text='} Extra', generation_tokens=6, finish_reason=None)
            pytest.fail("Generation continued beyond the JSON boundary")
        finally:
            closed.append(True)

    llm.stream_generate = stream
    assert llm.generate_json_for_baseline("test")["text"] == '{"label":"absent"}'
    assert llm.calls[0]["finish_reason"] == "json_object"
    assert closed == [True]


def test_document_scope_switching_ambiguity_and_conflicting_ids(llm):
    rag = StandardRAGBaseline(llm)
    rag.index_documents(["A contract says apples.", "B contract says zebras."], ["A", "B"])
    with pytest.raises(ValueError):
        rag.answer("zebras")
    a = rag.answer("zebras", doc_id="A")
    assert all(c["doc_id"] == "A" for c in a["retrieved_chunks"])
    assert "zebras" not in a["retrieved_evidence"][0]
    b = rag.answer("apples", doc_id="B")
    assert b["retrieved_chunks"][0]["doc_id"] == "B"
    changed = rag.answer("new", context="Replacement contract", doc_id="B")
    assert changed["retrieved_evidence"] == ["Replacement contract"]
    with pytest.raises(ValueError):
        rag.index_documents(["x", "y"], ["same", "same"])


def test_batch_preserves_order_reuses_documents_and_metadata(llm, monkeypatch):
    rag = StandardRAGBaseline(llm, chunk_size=3, overlap=1, k1=1.2)
    examples = [{"id": "a1", "query": "law", "context": "A has governing law."},
                {"id": "b", "query": "law", "context": "B has payment terms."},
                {"id": "a2", "query": "notice", "context": "A has governing law."}]
    calls = []
    original = rag.retriever.index
    def index(chunks):
        calls.append(chunks)
        original(chunks)
    monkeypatch.setattr(rag.retriever, "index", index)
    result = rag.batch_answer(examples)
    assert [p["id"] for p in result] == ["a1", "b", "a2"]
    assert len(calls) == 2
    for ex, pred in zip(examples, result):
        for chunk in pred["retrieved_chunks"]:
            assert ex["context"][chunk["start"]:chunk["end"]] == chunk["text"]
    assert rag.batch_answer([]) == []
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("baseline_type", [DirectPromptingBaseline, StandardRAGBaseline])
def test_token_budget_covers_full_prompt_and_reports_truncation(baseline_type, llm):
    baseline = baseline_type(llm)
    context = "long contract text " * 200
    if baseline_type is DirectPromptingBaseline:
        overhead = len(baseline._create_prompt("query", ""))
    else:
        overhead = len(baseline._create_rag_prompt("query", [{"text": ""}]))
    baseline.max_input_tokens = overhead + 30
    result = baseline.answer("query", context)
    assert len(result["prompt"]) <= baseline.max_input_tokens
    assert result["input_budget"]["truncated"]
    if baseline_type is StandardRAGBaseline:
        assert all(context[c["start"]:c["end"]] == c["text"] for c in result["prompt_evidence"])
    baseline.max_input_tokens = 1
    with pytest.raises(ValueError):
        baseline.answer("query", context)


def test_budget_requires_tokenizer_and_zero_evidence_is_allowed(llm):
    llm.tokenizer = None
    with pytest.raises(ValueError):
        DirectPromptingBaseline(llm, max_input_tokens=1000).answer("query", "text")
    rag = StandardRAGBaseline(llm, top_k=0)
    assert rag.answer("query", "text")["retrieved_chunks"] == []
    assert rag.answer("query", "")["retrieved_chunks"] == []


def test_configuration_is_separated_and_unknown_options_rejected(llm):
    rag = StandardRAGBaseline(llm, chunker_kwargs={"chunk_size": 10, "overlap": 2}, retriever_kwargs={"b": .4})
    assert rag.chunker.chunk_size == 10 and rag.retriever.b == .4
    with pytest.raises(TypeError):
        StandardRAGBaseline(llm, unknown=1)


def test_sampling_is_reproducible_and_full_mode_is_full():
    rows = list(range(100))
    assert select_examples(rows, 10, 42) == select_examples(rows, 10, 42)
    assert select_examples(rows, 10, 42) != select_examples(rows, 10, 43)
    assert select_examples(rows) == rows
    assert select_examples(rows, 0) == []
    with pytest.raises(ValueError):
        select_examples(rows, -1)


@pytest.mark.parametrize("name", ["contractnli", "cuad"])
def test_real_data_small_batch_for_both_baselines(name, llm):
    examples = select_examples(load_examples(name), 10)
    for baseline in [DirectPromptingBaseline(llm, dataset=name), StandardRAGBaseline(llm, dataset=name)]:
        predictions = baseline.batch_answer(examples)
        assert [p["id"] for p in predictions] == [ex.id for ex in examples]
        assert all(p["parse_error"] is None and p["dataset"] == name for p in predictions)
        json.dumps(predictions, allow_nan=False)


def test_label_without_answer_is_a_format_error():
    assert parse_response("Label: contradiction", "contractnli")["parse_error"]


def test_schema_errors_cannot_count_as_a_valid_classification():
    parsed = parse_response('{"label":"entailment","answer":null}', "contractnli")
    assert parsed["label"] == "invalid" and parsed["parse_error"] and not parsed["abstained"]


@pytest.mark.parametrize("wrap", [lambda obj: "The response follows.\n\n" + obj,
                                  lambda obj: "Result:\n\x60\x60\x60json\n" + obj + "\n\x60\x60\x60"])
def test_parse_one_json_object_after_prose(wrap):
    result = parse_response(wrap('{"label":"absent","extracted_spans":[]}'), "cuad")
    assert result["answer"] == "" and result["label"] == "absent" and result["parse_error"] is None


def test_numbered_citation_objects_are_normalized_without_using_their_text():
    raw = '{"label":"entailment","answer":"Payment is required.","citations":[{"number":2,"text":"Other copy."},1]}'
    result = parse_response(raw, "contractnli")
    assert result["citations"] == [2, 1] and result["parse_error"] is None


@pytest.mark.parametrize("raw", [
    '{"label":"neutral","answer":"An unescaped "quote"."}',
    '{"label":"neutral","answer":"unfinished',
    '{"label":"neutral","answer":"No date."} {"label":"entailment","answer":"A date."}',
    '{"label":"neutral","label":"entailment","answer":"A date."}',
    '{"label":"neutral","answer":"No date.","citations":[{"text":"No number"}]}',
    '{"label":"neutral","answer":"No date.","citations":[{"number":true}]}',
])
def test_malformed_or_ambiguous_response_remains_invalid(raw):
    result = parse_response(raw, "contractnli")
    assert result["label"] == "invalid" and result["parse_error"]


def test_cuad_spans_derive_answer_and_keep_exact_punctuation():
    spans = ['The "Agreement" expires on 31 December.', 'The renewal term is 12 months.']
    result = parse_response(json.dumps({"label": "present", "extracted_spans": spans}), "cuad")
    assert result["answer"] == "; ".join(spans)
    assert result["extracted_spans"] == spans and result["parse_error"] is None
    assert parse_response('{"label":"present","extracted_spans":[]}', "cuad")["label"] == "invalid"


def test_label_first_answer_preserves_multiline_text_and_citation():
    result = parse_response("Label: entailment\nAnswer: Payment is required.\nSee [2].", "contractnli")
    assert result["answer"] == "Payment is required.\nSee [2]." and result["citations"] == [2]


@pytest.mark.parametrize("baseline_type", [DirectPromptingBaseline, StandardRAGBaseline])
def test_invalid_cuad_response_does_not_claim_extractive_answer(baseline_type):
    baseline = baseline_type(RecordingLLM("No structured answer"), dataset="cuad")
    result = baseline.answer("What is the notice period?", "Give seven days' notice.")
    assert result["answer_is_extractive"] is None and result["parse_error"]
