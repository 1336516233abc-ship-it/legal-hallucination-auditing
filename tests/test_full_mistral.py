"""Check that a stopped comparison resumes without resubmitting finished rows."""

import json
import sqlite3
from threading import Barrier
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from experiments import run_full_mistral as full


def test_resume_keeps_completed_predictions_and_rejects_mixed_settings(tmp_path, monkeypatch):
    settings = {"datasets": {name: {"count": 1} for name in full.DATASETS}, "revision": "frozen"}
    examples = {name: [SimpleNamespace(id=f"{name}-1", query="Question")] for name in full.DATASETS}
    calls = []
    fail_once = True

    def assess(row, dataset, prediction):
        label = "neutral" if dataset == "contractnli" else "absent"
        return {**dict.fromkeys(full.METRIC_KEYS), "gold_label": label, "prediction": label,
                "format_pass": True, "label_correct": True, "latency_seconds": 1,
                "gold_evidence_coverage": None, "model_calls": len(prediction["generation_trace"])}

    monkeypatch.setattr(full, "assess_prediction", assess)

    def make_methods():
        model = SimpleNamespace(calls=[])

        def predict(dataset, mode, row):
            nonlocal fail_once
            key = (dataset, mode, row.id)
            calls.append(key)
            if len(calls) == 2 and fail_once:
                fail_once = False
                raise KeyboardInterrupt
            model.calls.extend([{}] * (2 if mode == "verified_agentic_rag" else 1))
            return {"label": "neutral", "latency": 1, "verification": None, "revision": None}

        from functools import partial

        return model, lambda: {name: {mode: partial(predict, name, mode) for mode in full.METHODS}
                       for name in full.DATASETS}

    report = tmp_path / "published.json"
    with pytest.raises(KeyboardInterrupt):
        full.run_predictions(tmp_path, settings, examples, make_methods, report)
    assert json.loads((tmp_path / "progress.json").read_text())["completed"] == 1
    assert not report.exists()

    full.run_predictions(tmp_path, settings, examples, make_methods, report)
    assert calls.count(calls[0]) == 1
    published = json.loads(report.read_text())
    assert published["status"] == "completed" and published["completed"] == 6
    with sqlite3.connect(tmp_path / "predictions.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 6

    with pytest.raises(ValueError, match="settings changed"):
        full.run_predictions(tmp_path, {**settings, "revision": "other"}, examples, make_methods)


def test_parallel_predictions_keep_request_ids_and_replenish_workers():
    barrier = Barrier(2, timeout=5)

    def predict(job):
        if job < 2:
            barrier.wait()
        return job, job ** 2

    assert sorted(full.parallel_predictions(range(7), predict, workers=2)) == [
        (job, job ** 2) for job in range(7)
    ]
