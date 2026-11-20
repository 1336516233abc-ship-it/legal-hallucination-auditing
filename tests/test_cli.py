"""Exercise public CLI configuration, artifacts and failure checkpoints."""

import json
from pathlib import Path
import subprocess
import sys
import pytest
from conftest import RecordingLLM
from retrieval.cli import run_baseline
from retrieval.common import ROOT
from retrieval.evaluate_retrieval import evaluate


@pytest.mark.parametrize("mode", ["direct_prompting", "standard_rag"])
def test_cli_dry_run_and_full_selection(mode, tiny_data):
    manifest = run_baseline(mode, ["--data_dir", str(tiny_data), "--dry_run"])
    assert manifest["example_count"] == 12 and manifest["status"] == "dry_run"
    sample = run_baseline(mode, ["--data_dir", str(tiny_data), "--sample_size", "3", "--dry_run"])
    assert sample["example_count"] == 3
    if mode == "standard_rag":
        result = run_baseline(mode, ["--data_dir", str(tiny_data), "--top_k", "3", "--chunk_size", "20", "--overlap", "2", "--dry_run"])
        assert result["retrieval"]["top_k"] == 3 and result["retrieval"]["chunker_kwargs"]["chunk_size"] == 20


@pytest.mark.parametrize("mode", ["direct_prompting", "standard_rag"])
def test_cli_predictions_metadata_checkpoint_and_overwrite_guard(mode, tiny_data, tmp_path):
    output = tmp_path / "predictions.json"
    argv = ["--data_dir", str(tiny_data), "--sample_size", "3", "--output", str(output)]
    predictions = run_baseline(mode, argv, llm_factory=RecordingLLM)
    assert json.loads(output.read_text()) == predictions
    assert len(output.with_suffix(".jsonl").read_text().splitlines()) == 3
    meta = json.loads(output.with_suffix(".meta.json").read_text())
    assert meta["status"] == "completed" and meta["parse_error_count"] == 0
    assert [p["id"] for p in predictions] == meta["selected_ids"]
    with pytest.raises(FileExistsError):
        run_baseline(mode, argv, llm_factory=RecordingLLM)


def test_cli_failure_preserves_checkpoint(tiny_data, tmp_path):
    class FailingLLM(RecordingLLM):
        def generate(self, prompt):
            if self.prompts:
                raise RuntimeError("injected model failure")
            return super().generate(prompt)
    output = tmp_path / "failed.json"
    with pytest.raises(RuntimeError):
        run_baseline("direct_prompting", ["--data_dir", str(tiny_data), "--output", str(output)], llm_factory=FailingLLM)
    assert not output.exists()
    assert len(output.with_suffix(".jsonl").read_text().splitlines()) == 1
    assert json.loads(output.with_suffix(".meta.json").read_text())["status"] == "failed"


def test_offline_imports_and_script_help_without_model_libraries():
    result = subprocess.run([sys.executable, "-c",
        "import sys; import retrieval; from data import ContractNLIProcessor; import utils.logger; "
        "assert 'torch' not in sys.modules; assert 'datasets' not in sys.modules; "
        "assert 'sentence_transformers' not in sys.modules"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for script in ["retrieval/baseline_direct.py", "retrieval/baseline_rag.py", "retrieval/evaluate_retrieval.py"]:
        result = subprocess.run([sys.executable, str(ROOT / script), "--help"], cwd=ROOT.parent, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_retrieval_matrix_produces_real_metrics(tiny_data):
    results = evaluate(["contractnli"], ["bm25"], ["fixed_length", "clause_level"], [0, 1, 3],
                       ROOT / "configs/retrieval_config.yaml", sample_size=3, data_dir=tiny_data)
    assert len(results["experiments"]) == 6
    assert results["experiments"][0]["metrics"]["evidence_recall_at_k"] == 0
    assert results["experiments"][1]["metrics"]["evidence_recall_at_k"] == 1
    json.dumps(results, allow_nan=False)


def test_overwrite_failure_does_not_leave_a_stale_success_file(tiny_data, tmp_path):
    output = tmp_path / "old.json"
    output.write_text('[{"old": true}]')
    def failing_factory(**kwargs):
        raise RuntimeError("failed to load model")
    with pytest.raises(RuntimeError):
        run_baseline("direct_prompting", ["--data_dir", str(tiny_data), "--output", str(output), "--overwrite"], llm_factory=failing_factory)
    assert not output.exists()
    assert json.loads(output.with_suffix(".meta.json").read_text())["status"] == "failed"
