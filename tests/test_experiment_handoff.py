"""Check task selection, checkpoint safety, call accounting, and report audits."""

import copy
import json
import sqlite3

import pytest
from conftest import CharacterTokenizer
from experiments.run_experiments import run
from experiments.summarize_results import audit, validate_group


class TaskModel:
    def __init__(self, **kwargs):
        self.tokenizer = CharacterTokenizer()
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        if "SUPPORTED" in prompt:
            text = "Status: SUPPORTED\nReasoning: Supported.\nEvidence: [1]"
        elif '"extracted_spans"' in prompt:
            text = '{"label":"absent","answer":"","extracted_spans":[],"citations":[]}'
        elif "Label:" in prompt or '"label"' in prompt:
            text = "Label: neutral\nAnswer: Not specified."
        else:
            text = "payment"
        return {"text": text, "tokens": 7}


@pytest.fixture
def data(tiny_data):
    rows = json.loads((tiny_data / "contractnli_test.json").read_text())
    for row in rows:
        row.update(id=row["id"].replace("nli", "cuad"), label="absent", evidence_spans=[])
        row["metadata"]["dataset"] = "cuad"
    (tiny_data / "cuad_test.json").write_text(json.dumps(rows))
    return tiny_data


def cli(data, output):
    return ["--dataset", "cuad", "--model", "llama3", "--sample_size", "2", "--seed", "17",
            "--data_dir", str(data), "--output", str(output), "--max_input_tokens", "4000"]


def test_dry_run_honors_dataset_sample_seed_without_loading_model(data, tmp_path):
    output = tmp_path / "dry"
    def forbidden(**kwargs):
        raise AssertionError("Dry runs must not load models")
    first = run(cli(data, output) + ["--dry_run", "--ablations", "agent"], forbidden)
    assert set(first["datasets"]) == {"cuad"}
    assert first["datasets"]["cuad"]["count"] == 2
    assert first["seed"] == 17
    second = run(cli(data, output) + ["--dry_run", "--ablations", "agent"], forbidden)
    assert first == second
    assert not output.exists()
    assert {v["name"] for v in first["variants"]} >= {"agent_no_verification", "agent_no_revision"}


def test_task_labels_and_resume_keep_exact_completed_rows(data, tmp_path):
    output = tmp_path / "real"
    models = []
    def factory(**kwargs):
        model = TaskModel(**kwargs); models.append(model); return model
    results = run(cli(data, output), factory)
    assert len(results) == 3
    assert all(r["completed"] == r["format_pass"] == 2 for r in results)
    assert all(r["label_accuracy"] == 1 for r in results)
    assert all(r["independent_hallucination_rate"] is None for r in results)
    for result in results:
        assert result["total_generated_tokens"] == 7 * result["model_calls"]
        path = output / "llama3/cuad" / result["variant"]["name"] / "predictions.sqlite3"
        with sqlite3.connect(path) as db:
            traces = [json.loads(row[0]) for row in db.execute("SELECT trace FROM predictions")]
        assert all(t["label"] == "absent" and t["dataset"] == "cuad" for t in traces)
    run(cli(data, output) + ["--resume"], factory)
    assert models[-1].calls == 0
    with pytest.raises(FileExistsError):
        run(cli(data, output), factory)
    with pytest.raises(ValueError, match="manifest changed"):
        run(cli(data, output) + ["--resume", "--seed", "18"], factory)


def test_interruption_retains_checkpoint_and_resumes_remaining_rows(data, tmp_path):
    output = tmp_path / "interrupted"
    class FailOnce(TaskModel):
        def generate(self, prompt, **kwargs):
            if self.calls == 1:
                raise RuntimeError("backend disconnected")
            return super().generate(prompt, **kwargs)
    args = cli(data, output) + ["--modes", "direct_prompting"]
    with pytest.raises(RuntimeError, match="disconnected"):
        run(args, FailOnce)
    status = json.loads((output / "llama3/cuad/direct_prompting/status.json").read_text())
    assert status["status"] == "interrupted" and status["completed"] == 1
    model = TaskModel()
    result = run(args + ["--resume"], lambda **kwargs: model)
    assert model.calls == 1 and result[0]["completed"] == 2


def test_invalid_output_is_counted_as_failure(data, tmp_path):
    class Invalid(TaskModel):
        def generate(self, prompt, **kwargs):
            return {"text": "not a task response", "tokens": 4}
    result = run(cli(data, tmp_path / "invalid") + ["--modes", "direct_prompting"], Invalid)[0]
    assert result["completed"] == 2 and result["format_pass"] == 0
    assert result["label_accuracy"] == result["macro_f1"] == 0
    assert result["validation_status"] == "has_parse_errors"
    assert result["independent_hallucination_rate"] is None


def test_checked_in_result_audit_detects_legacy_dataset_overwrite():
    result = audit()
    assert len(result["main"]) == 12 and len(result["retrieval"]) == 36
    legacy = next(r for r in result["legacy_agent_audit"] if r["file"].endswith("/agentic_rag_full_predictions.json"))
    assert legacy["actual_dataset_counts"] == {"cuad": 20}
    assert "dataset_id_mismatch" in legacy["issues"]
    assert "published_summary_mismatch:total_tokens" in legacy["issues"]
    group = copy.deepcopy(result["main"][0]); group["label_accuracy"] = 1
    with pytest.raises(ValueError, match="confusion matrix"):
        validate_group(group)


def test_agent_ablation_flags_change_executed_calls(data, tmp_path):
    args = cli(data, tmp_path / "ablations") + ["--mode", "agent", "--ablations", "agent", "--sample_size", "1"]
    groups = run(args, TaskModel)
    by_name = {g["variant"]["name"]: g for g in groups}
    assert len(groups) == 3
    assert by_name["agent_no_verification"]["model_calls"] == 2
    assert by_name["agent_no_revision"]["model_calls"] == 3
    assert by_name["agentic_rag"]["model_calls"] == 3


def test_review_export_preserves_blinding_pairing_and_missing_judgments(data, tmp_path):
    import hashlib
    from experiments.prepare_support_review import prepare
    from experiments.run_full_mistral import open_checkpoint, record_prediction
    rows = json.loads((data / "cuad_test.json").read_text())[:2]
    settings = {"datasets": {"cuad": {"count": 2, "sha256": hashlib.sha256((data / "cuad_test.json").read_bytes()).hexdigest()}}}
    db_path = tmp_path / "review.sqlite3"
    db, _ = open_checkpoint(db_path, settings)
    for row in rows:
        for method in ["direct_prompting", "standard_rag", "verified_agentic_rag"]:
            record_prediction(db, ("cuad", method, row["id"]), {"format_pass": True, "parse_error": None},
                              {"label": "absent", "answer": "", "raw_response": "{}"})
    db.close()
    output = tmp_path / "review"
    assert prepare(db_path, output, "llama3", sample_size=1, data_dir=data) == 3
    exported = [json.loads(s) for s in (output / "blinded_review.jsonl").read_text().splitlines()]
    assert len({r["query"] for r in exported}) == 1
    assert all("method" not in r and "model" not in r for r in exported)
    assert all(not r["review"]["all_claims_reviewed"] and r["review"]["annotator"] is None for r in exported)
    assert json.loads((output / "selection.json").read_text())["independent_hallucination_rate"] is None
    with pytest.raises(FileExistsError):
        prepare(db_path, output, "llama3", sample_size=1, data_dir=data)
