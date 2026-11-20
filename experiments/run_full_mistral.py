"""Run all three methods on both test splits with a pinned model."""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
from itertools import zip_longest
import json
import os
from pathlib import Path
import platform
import signal
import sqlite3
from threading import local
import time

import yaml

from agent.agentic_rag import AgenticRAG
from experiments.run_baseline_pilot import (DATASETS, INPUT_TOKENS, OUTPUT_TOKENS, REVISIONS, SEED,
                                            MLXPilotModel, attach_generation_trace, cohort, summarize)
from retrieval.baseline_direct import DirectPromptingBaseline
from retrieval.baseline_rag import StandardRAGBaseline
from retrieval.cli import component_config
from retrieval.common import LABELS, ROOT, PromptBudgetError, document_id, load_examples, write_json
from retrieval.evidence import score_evidence


METHODS = ("direct_prompting", "standard_rag", "verified_agentic_rag")
PACKAGES = ("mlx", "mlx-lm", "torch", "transformers", "huggingface-hub", "rank-bm25", "numpy", "pyyaml")
OUTPUT = ROOT / "experiments/results/mistral_full"
METRIC_KEYS = ("gold_label", "prediction", "format_pass", "label_correct", "parse_error", "latency_seconds",
               "answer_is_extractive", "invalid_citations", "grounding_error", "revision_type")


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def open_checkpoint(path, settings):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    with db:
        db.execute("CREATE TABLE IF NOT EXISTS run (id INTEGER PRIMARY KEY CHECK (id = 1), settings TEXT, started_at TEXT)")
        db.execute("""CREATE TABLE IF NOT EXISTS predictions (
            dataset TEXT NOT NULL, method TEXT NOT NULL, example_id TEXT NOT NULL,
            assessment TEXT NOT NULL, trace TEXT NOT NULL,
            PRIMARY KEY (dataset, method, example_id))""")
        db.execute("INSERT OR IGNORE INTO run VALUES (1, ?, ?)", (json.dumps(settings, sort_keys=True), timestamp()))
    saved, started_at = db.execute("SELECT settings, started_at FROM run WHERE id = 1").fetchone()
    if json.loads(saved) != settings:
        db.close()
        raise ValueError("Checkpoint settings changed; resume from the frozen workspace or choose a new output directory")
    return db, started_at


def record_prediction(db, key, assessment, prediction):
    with db:
        db.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?)", (*key,
                   json.dumps(assessment, ensure_ascii=False, allow_nan=False),
                   json.dumps(prediction, ensure_ascii=False, allow_nan=False)))


def build_methods(llm, dataset, examples, budget, config):
    retrieval = config["retrieval"]
    components = {"chunker_kwargs": config["chunker_kwargs"], "retriever_kwargs": config["retriever_kwargs"]}
    direct = DirectPromptingBaseline(llm, dataset=dataset, max_input_tokens=budget)
    rag = StandardRAGBaseline(llm, dataset=dataset, max_input_tokens=budget, **components,
                              top_k=retrieval["top_k"], min_score_threshold=retrieval["min_score_threshold"])
    agent = AgenticRAG(llm, **components)
    doc_ids = {row.id: document_id(row.context, row.metadata, dataset) for row in examples}
    rag.index_documents((row.context for row in examples), (doc_ids[row.id] for row in examples))
    return {
        "direct_prompting": lambda row: direct.answer(row.query, row.context),
        "standard_rag": lambda row: rag.answer(row.query, doc_id=doc_ids[row.id]),
        "verified_agentic_rag": lambda row: agent.answer(row.query, row.context, dataset=dataset, max_input_tokens=budget),
    }


def assess_prediction(row, dataset, prediction):
    assessment = summarize([row], [prediction])["rows"][0]
    result = {key: assessment.get(key) for key in METRIC_KEYS}
    result["model_calls"] = len(prediction["generation_trace"])
    expected_id = document_id(row.context, row.metadata, dataset)
    if "prompt_evidence" in prediction:
        chunks = []
        allowed_ids = {expected_id, document_id(row.context)}
        for chunk in prediction["prompt_evidence"]:
            if chunk["doc_id"] not in allowed_ids:
                raise ValueError("A prompt evidence chunk belongs to a different document")
            chunks.append({**chunk, "doc_id": expected_id})
    else:
        end = prediction["input_budget"]["context_end"]
        chunks = [{"text": row.context[:end], "start": 0, "end": end, "doc_id": expected_id}] if end else []
    evidence = score_evidence(chunks, row, k=len(chunks))
    result["gold_evidence_coverage"] = None if evidence is None else evidence["evidence_recall_at_k"]
    return result


def group_summary(rows, planned, labels):
    n = len(rows)
    confusion = Counter((row["gold_label"], row["prediction"]) for row in rows)
    f1 = []
    for label in sorted(labels):
        predicted = sum(count for (_, pred), count in confusion.items() if pred == label)
        actual = sum(count for (gold, _), count in confusion.items() if gold == label)
        f1.append(2 * confusion[label, label] / (predicted + actual) if predicted + actual else 0.0)
    passed = sum(row["format_pass"] for row in rows)
    correct = sum(row["label_correct"] for row in rows)
    with_gold = [row for row in rows if row["gold_evidence_coverage"] is not None]
    covered = [row for row in with_gold if row["gold_evidence_coverage"] == 1.0]
    positives = [row for row in rows if row["gold_label"] == "present"]
    return {
        "planned": planned, "completed": n, "format_pass": passed, "label_correct": correct,
        "format_pass_rate": passed / n if n else None, "label_accuracy": correct / n if n else None,
        "macro_f1": sum(f1) / len(f1) if n else None,
        "confusion": {gold: {pred: count for (target, pred), count in sorted(confusion.items()) if target == gold}
                      for gold in sorted(labels)},
        "mean_latency_seconds": sum(row["latency_seconds"] for row in rows) / n if n else None,
        "model_calls": sum(row["model_calls"] for row in rows),
        "format_or_execution_errors": dict(Counter(row["parse_error"] for row in rows if row["parse_error"])),
        "grounding_errors": sum(bool(row["grounding_error"]) for row in rows),
        "rows_with_out_of_range_citations": sum(bool(row["invalid_citations"]) for row in rows),
        "revisions": sum(bool(row["revision_type"]) for row in rows),
        "gold_evidence_rows": len(with_gold), "fully_covered_gold_rows": len(covered),
        "label_accuracy_when_gold_fully_covered": (sum(r["label_correct"] for r in covered) / len(covered)
                                                     if covered else None),
        "gold_positive_count": len(positives),
        "verbatim_positive_correct": sum(r["label_correct"] and r["answer_is_extractive"] is True
                                         and not r["invalid_citations"] and not r["grounding_error"] for r in positives),
    }


def progress_report(assessments, settings, started_at, status, error=None, throughput=None):
    groups = [{"dataset": dataset, "method": method, **group_summary(rows, settings["datasets"][dataset]["count"],
                                                                     LABELS[dataset])}
              for (dataset, method), rows in assessments.items()]
    histories = [g for g in groups if g["completed"] >= 10 or g["completed"] == g["planned"]]
    eta = (sum((g["planned"] - g["completed"]) * g["mean_latency_seconds"] for g in groups
               if g["completed"] < g["planned"])
           if len(histories) == len(groups) else None)
    completed, planned = sum(g["completed"] for g in groups), sum(g["planned"] for g in groups)
    if settings.get("workers", 1) > 1:
        eta = (planned - completed) / throughput if throughput else None
    return {"status": status, "started_at": started_at, "updated_at": timestamp(), "pid": os.getpid(),
            "completed": completed, "planned": planned, "predictions_per_second": throughput,
            "model_calls": sum(g["model_calls"] for g in groups), "estimated_remaining_seconds": eta,
            "settings": settings, "groups": groups, "error": error}


def parallel_predictions(jobs, predict, workers):
    jobs = iter(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(predict, job) for job in (next(jobs, None) for _ in range(workers)) if job is not None}
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                yield future.result()
                job = next(jobs, None)
                if job is not None:
                    pending.add(pool.submit(predict, job))


def run_predictions(output, settings, examples, make_methods, summary_path=None):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db, started_at = open_checkpoint(output / "predictions.sqlite3", settings)
        with closing(db):
            assessments = {(dataset, mode): [] for dataset in DATASETS for mode in METHODS}
            done = set()
            for dataset, mode, example_id, assessment in db.execute(
                    "SELECT dataset, method, example_id, assessment FROM predictions"):
                done.add((dataset, mode, example_id))
                assessments[dataset, mode].append(json.loads(assessment))

            initial_count, inference_start = len(done), None

            def publish(status, error=None):
                throughput = ((len(done) - initial_count) / (time.perf_counter() - inference_start)
                              if inference_start is not None and len(done) > initial_count else None)
                report = progress_report(assessments, settings, started_at, status, error, throughput)
                write_json(output / "progress.json", report)
                return report

            publish("running")
            try:
                total = sum(len(rows) for rows in examples.values()) * len(METHODS)
                if len(done) < total:
                    llm, worker_methods = make_methods()
                    worker = local()

                    def predict(job):
                        dataset, mode, row = job
                        if not hasattr(worker, "methods"):
                            worker.methods = worker_methods()
                        start = time.perf_counter()
                        try:
                            prediction = worker.methods[dataset][mode](row)
                        except PromptBudgetError as exc:
                            prediction = {"query": row.query, "label": "invalid",
                                          "parse_error": f"prompt budget: {exc}", "raw_response": "", "tokens": 0,
                                          "latency": time.perf_counter() - start,
                                          "input_budget": {"input_tokens": None, "truncated": True, "context_end": 0},
                                          "prompt_evidence": [], "generation_trace": llm.calls.copy()}
                            llm.calls.clear()
                        else:
                            attach_generation_trace(prediction, mode, llm.calls)
                        prediction.update(id=row.id, dataset=dataset)
                        return (dataset, mode, row.id), assess_prediction(row, dataset, prediction), prediction

                    jobs = ((dataset, mode, row) for batch in zip_longest(*(examples[d] for d in DATASETS))
                            for dataset, row in zip(DATASETS, batch, strict=True) if row is not None
                            for mode in METHODS if (dataset, mode, row.id) not in done)
                    workers = settings.get("workers", 1)
                    inference_start = time.perf_counter()
                    results = map(predict, jobs) if workers == 1 else parallel_predictions(jobs, predict, workers)
                    for key, assessment, prediction in results:
                        dataset, mode, example_id = key
                        record_prediction(db, key, assessment, prediction)
                        done.add(key)
                        assessments[dataset, mode].append(assessment)
                        print(f"{len(done)}/{total} {dataset} {mode} {example_id} {prediction['label']} "
                              f"{prediction['latency']:.1f}s", flush=True)
                        if len(done) <= 6 or len(done) % 10 == 0:
                            publish("running")
                if len(done) != total:
                    raise ValueError(f"Checkpoint has {len(done)} results for {total} planned predictions")
                final = publish("completed")
                if summary_path is not None:
                    write_json(summary_path, final)
            except KeyboardInterrupt:
                publish("paused")
                raise
            except Exception as exc:
                publish("failed", f"{type(exc).__name__}: {exc}")
                raise


def run(output, summary_path, *, vllm_url=None, workers=1, pilot=False, model_key="mistral"):
    from huggingface_hub import snapshot_download

    examples = cohort()[1] if pilot else {dataset: load_examples(dataset) for dataset in DATASETS}
    retrieval, chunker, retriever = component_config(ROOT / "configs/retrieval_config.yaml", "bm25", "fixed_length")
    config = {"retrieval": retrieval["retrieval"], "chunker_kwargs": chunker, "retriever_kwargs": retriever}
    model = yaml.safe_load((ROOT / "configs/model_config.yaml").read_text())["model"][model_key]["name"]
    revision = REVISIONS[model_key]
    source_files = [path for directory in ("agent", "data", "experiments", "retrieval", "utils")
                    for path in (ROOT / directory).rglob("*.py") if "results" not in path.relative_to(ROOT).parts]
    settings = {
        "checkpoint": model, "revision": revision, "backend": "vllm" if vllm_url else "mlx",
        "dtype": "bfloat16", "python": platform.python_version(), "quantized": False,
        "workers": workers, "pilot": pilot,
        "temperature": 0, "seed": SEED, "input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS,
        "methods": list(METHODS), "retrieval": config,
        "packages": {name: version(name) for name in
                     (("vllm", "requests") + PACKAGES[2:] if vllm_url else PACKAGES)},
        "datasets": {d: {"count": len(rows), "sha256": sha256(ROOT / "data/processed" / f"{d}_test.json")}
                     for d, rows in examples.items()},
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(source_files)},
    }
    if vllm_url:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("vLLM full inference requires a CUDA GPU")
        settings["gpu"] = torch.cuda.get_device_name()
    else:
        settings["prefill_step_size"] = 512
    for dataset, rows in examples.items():
        if not rows or len({row.id for row in rows}) != len(rows) or any(row.label not in LABELS[dataset] for row in rows):
            raise ValueError(f"Invalid or duplicate full test rows in {dataset}")

    def make_methods():
        snapshot = snapshot_download(model, revision=revision, local_files_only=True,
                                     allow_patterns=["model-*.safetensors", "*.json", "tokenizer.model", "chat_template.jinja"])
        if vllm_url:
            from experiments.vllm_mistral import VLLMMistral

            llm = VLLMMistral(Path(snapshot), vllm_url, INPUT_TOKENS, OUTPUT_TOKENS, SEED,
                              served_name=f"{model_key}-audit")
        else:
            llm = MLXPilotModel(snapshot, record_calls=True)
        if llm.config["torch_dtype"] != settings["dtype"]:
            raise ValueError("Cached checkpoint dtype does not match the run settings")
        budget = min(llm.config["max_position_embeddings"], llm.tokenizer.model_max_length) - OUTPUT_TOKENS
        if budget < INPUT_TOKENS:
            raise ValueError("Cached checkpoint cannot support the locked prompt and generation budget")
        return llm, lambda: {d: build_methods(llm, d, rows, INPUT_TOKENS, config) for d, rows in examples.items()}

    run_predictions(output, settings, examples, make_methods, summary_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--model", choices=REVISIONS, default="mistral")
    parser.add_argument("--vllm-url", help="Local vLLM API base URL, including /v1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pilot", action="store_true", help="Run the locked 10-case cohort instead of the full splits")
    args = parser.parse_args()
    if args.workers < 1 or (args.workers != 1 and not args.vllm_url):
        parser.error("Multiple workers require the local vLLM server")
    if (args.vllm_url or args.pilot or args.model != "mistral") and args.output == OUTPUT:
        parser.error("Specify a separate --output for a vLLM run, a pilot, or another model")

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    run(args.output, args.summary_output, vllm_url=args.vllm_url, workers=args.workers, pilot=args.pilot,
        model_key=args.model)


if __name__ == "__main__":
    main()
