"""Reproduce the pre-fix review from Git revision 0166fe9 in a temporary directory.

Run with Python 3.11 and numpy, scikit-learn, rank-bm25, datasets installed.
Classes used for logic probes are compiled unchanged from their AST definitions;
package initializers and heavyweight model imports are deliberately bypassed.
This is a diagnostic harness, not an end-to-end installation or inference test.
Optional official release ZIPs are read from --source-dir; nothing is downloaded.
"""

import argparse
import ast
import collections
import hashlib
import importlib.util
import importlib.metadata
import json
import logging
import subprocess
import sys
import time
import tempfile
import typing
import zipfile
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
REVIEW_REVISION = "0166fe94c23f6e599ab54d98bbac6b49404046d8"
SOURCE_ROOT = ROOT


def load_file(relative, name):
    spec = importlib.util.spec_from_file_location(name, SOURCE_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def classes(relative, **extra):
    tree = ast.parse((SOURCE_ROOT / relative).read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    scope = dict(vars(typing), np=np, BM25Okapi=BM25Okapi,
                 logger=logging.getLogger("audit"), time=time, **extra)
    exec(compile(tree, relative, "exec"), scope)
    return scope


def dataset_summary(rows):
    spans = [(r, span) for r in rows for span in r["evidence_spans"] or []]
    invalid = [(r["id"], span) for r, span in spans
               if not (len(span) == 2 and all(type(v) is int for v in span)
                       and 0 <= span[0] < span[1] <= len(r["context"]))]
    return {
        "examples": len(rows), "unique_ids": len({r["id"] for r in rows}),
        "distinct_contexts": len({r["context"] for r in rows}),
        "labels": dict(collections.Counter(r["label"] for r in rows)),
        "evidence_spans": len(spans), "invalid_spans": len(invalid),
        "examples_with_evidence": sum(bool(r["evidence_spans"]) for r in rows),
        "null_answers": sum(r["answer"] is None for r in rows),
        "empty_contexts": sum(not r["context"] for r in rows),
        "document_count": len({r["metadata"].get("document_id",
                              r["metadata"].get("document_title")) for r in rows}),
    }


def compare_official(datasets, source_dir):
    results = {}
    nli_zip = source_dir / "contract-nli.zip"
    cuad_zip = source_dir / "cuad-data.zip"
    if nli_zip.exists():
        with zipfile.ZipFile(nli_zip) as archive:
            raw = json.loads(archive.read("contract-nli/test.json"))
        documents = {d["id"]: d for d in raw["documents"]}
        mismatches = collections.Counter()
        mapping = {"Entailment": "entailment", "Contradiction": "contradiction",
                   "NotMentioned": "neutral"}
        for row in datasets["contractnli"]:
            doc = documents[row["metadata"]["document_id"]]
            hyp = row["metadata"]["hypothesis_id"]
            annotation = doc["annotation_sets"][0]["annotations"][hyp]
            checks = {
                "context": row["context"] == doc["text"],
                "label": row["label"] == mapping[annotation["choice"]],
                "hypothesis": row["metadata"]["original_hypothesis"] == raw["labels"][hyp]["hypothesis"],
                "evidence": (row["evidence_spans"] or []) == [doc["spans"][i] for i in annotation["spans"]],
            }
            mismatches.update(key for key, ok in checks.items() if not ok)
        results["contractnli"] = {"compared": len(datasets["contractnli"]),
                                  "mismatches": dict(mismatches)}
    if cuad_zip.exists():
        with zipfile.ZipFile(cuad_zip) as archive:
            raw = json.loads(archive.read("test.json"))
        lookup = {(doc["title"], qa["question"]): (para["context"], qa)
                  for doc in raw["data"] for para in doc["paragraphs"]
                  for qa in para["qas"]}
        mismatches = collections.Counter()
        for row in datasets["cuad"]:
            context, qa = lookup[(row["metadata"]["document_title"], row["query"])]
            answers = qa["answers"]
            checks = {
                "context": row["context"] == context,
                "label": row["label"] == ("present" if answers else "absent"),
                "answer": row["answer"] == ("; ".join(a["text"] for a in answers) if answers else None),
                "evidence": (row["evidence_spans"] or []) == [[a["answer_start"], a["answer_start"] + len(a["text"])] for a in answers],
            }
            mismatches.update(key for key, ok in checks.items() if not ok)
        results["cuad"] = {"compared": len(datasets["cuad"]),
                           "mismatches": dict(mismatches)}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    global SOURCE_ROOT
    historical_source = tempfile.TemporaryDirectory(prefix="legal-audit-original-")
    SOURCE_ROOT = Path(historical_source.name)
    historical_paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", REVIEW_REVISION], cwd=ROOT, text=True).splitlines()
    for relative in historical_paths:
        if relative.endswith(".py"):
            destination = SOURCE_ROOT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(subprocess.check_output(
                ["git", "show", f"{REVIEW_REVISION}:{relative}"], cwd=ROOT))
    report = {"review_date": "2026-11-20",
              "head": REVIEW_REVISION,
              "scope": "Historical source snapshot diagnostics; current implementation is tested with pytest",
              "python": sys.version.split()[0], "datasets": {}, "probes": {}}
    datasets = {}
    for name in ["contractnli", "cuad"]:
        path = ROOT / "data" / "processed" / f"{name}_test.json"
        datasets[name] = json.loads(path.read_text())
        report["datasets"][name] = dataset_summary(datasets[name])
        report["datasets"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.source_dir:
        report["official_release_comparison"] = compare_official(datasets, args.source_dir)
        report["source_archives"] = {
            name: hashlib.sha256((args.source_dir / name).read_bytes()).hexdigest()
            for name in ["contract-nli.zip", "cuad-data.zip"]
            if (args.source_dir / name).exists()}
    report["audit_dependencies"] = {
        name: importlib.metadata.version(name)
        for name in ["numpy", "scikit-learn", "rank-bm25", "datasets"]}

    probes = report["probes"]
    check_import = subprocess.run(
        [sys.executable, "-c", "from data.dataset_loader import LegalExample"],
        cwd=SOURCE_ROOT, capture_output=True, text=True)
    probes["normal_data_package_import"] = {
        "returncode": check_import.returncode,
        "last_error": check_import.stderr.strip().splitlines()[-1] if check_import.stderr else None}

    metrics = load_file("evaluation/metrics.py", "audit_metrics").MetricsCalculator()
    loader = load_file("data/dataset_loader.py", "audit_loader")
    probes["json_to_LegalExample"] = {
        name: len([loader.LegalExample(**r) for r in rows]) for name, rows in datasets.items()}
    probes["classification_sanity"] = {
        "accuracy": float(metrics.compute_label_accuracy(
            [{"label": "entailment"}, {"label": "neutral"}],
            [{"label": "entailment"}, {"label": "contradiction"}]))}
    gt = [{"context": "alpha", "evidence_spans": [[0, 5]]},
          {"context": "wrong gold", "evidence_spans": [[0, 5], [6, 10]]}]
    probes["evidence_hit_rate_after_previous_hit"] = {
        "actual": float(metrics.compute_evidence_hit_rate([["alpha"], ["gold"]], gt)),
        "expected": 1.0}
    probes["evidence_recall_without_context"] = {
        "actual": float(metrics.compute_evidence_recall([["unrelated"]], [{"evidence_spans": [[0, 5]]}])),
        "expected": "reject missing context instead of returning 1.0"}
    probes["nli_per_class_names"] = list(metrics.compute_per_class_metrics(
        [{"label": x} for x in ["entailment", "contradiction", "neutral"]],
        [{"label": x} for x in ["entailment", "contradiction", "neutral"]]))
    try:
        metrics.compute_all_metrics(
            [{"answer": "SUPPLY CONTRACT", "label": "present"}, {"answer": "absent", "label": "absent"}],
            [{"answer": "SUPPLY CONTRACT", "label": "present"}, {"answer": None, "label": "absent"}])
        probes["cuad_null_answer"] = "no error"
    except Exception as exc:
        probes["cuad_null_answer"] = f"{type(exc).__name__}: {exc}"
    agent_shape = [{"final_answer": "claim", "verification": {"claims": [{"status": "contradicted"}]}}]
    probes["agent_verification_metric_adapter"] = {
        "actual": metrics.compute_all_metrics(agent_shape, [{}]).hallucination_rate,
        "expected": 1.0}

    chunkers = classes("retrieval/chunking.py", re=__import__("re"))
    Fixed = chunkers["FixedLengthChunker"]
    sample = "Alpha  beta\nGamma delta."
    chunks = Fixed(chunk_size=3, overlap=1).chunk(sample)
    probes["fixed_chunk_offsets"] = {
        "input": sample,
        "chunks": [{**c, "slice_matches": sample[c["start"]:c["end"]] == c["text"]} for c in chunks]}
    repeated = Fixed(chunk_size=2, overlap=0).chunk("A A A A A A")
    probes["repeated_word_offsets"] = {"actual_starts": [c["start"] for c in repeated],
                                         "expected_starts": [0, 4, 8]}
    probes["fixed_offsets_on_supplied_contracts"] = {}
    for name, rows in datasets.items():
        count, mismatched = 0, 0
        for context in {row["context"] for row in rows}:
            for chunk in Fixed().chunk(context):
                count += 1
                mismatched += context[chunk["start"]:chunk["end"]] != chunk["text"]
        probes["fixed_offsets_on_supplied_contracts"][name] = {
            "chunks": count, "slice_mismatches": mismatched}
    probes["clause_short_text"] = {
        "chunks": chunkers["ClauseLevelChunker"](use_sentence_boundaries=False).chunk("Payment is due."),
        "expected": "nonempty evidence"}

    retrievers = classes("retrieval/retrievers/retriever.py")
    class FakeLLM:
        def generate(self, prompt, **kwargs):
            return {"text": "Answer: ok\nLabel: Entailment", "tokens": 1}
    baselines = classes("retrieval/baseline_rag.py", LLMInterface=FakeLLM,
                        get_chunker=lambda *a, **kw: Fixed(**kw),
                        get_retriever=lambda *a, **kw: retrievers["BM25Retriever"](**kw))
    baseline = baselines["StandardRAGBaseline"](FakeLLM())
    baseline.index_documents(["alpha irrelevant terms", "zebra specific clause", "other common text"])
    probes["cross_contract_retrieval"] = {
        "query_document": 0,
        "retrieved_document": baseline.retriever.retrieve("zebra specific", top_k=1)[0][0]["metadata"]["doc_id"]}
    try:
        baselines["StandardRAGBaseline"](FakeLLM(), chunk_size=100)
        probes["shared_kwargs"] = "no error"
    except Exception as exc:
        probes["shared_kwargs"] = f"{type(exc).__name__}: {exc}"
    direct = classes("retrieval/baseline_direct.py", LLMInterface=FakeLLM)["DirectPromptingBaseline"](FakeLLM())
    probes["cuad_label_parser"] = {"input": "Label: present",
        "actual": direct._parse_response("Answer: yes\nLabel: present")["label"], "expected": "present"}
    probes["direct_context_truncation"] = {
        "tail_survives": "TAIL_EVIDENCE" in direct._create_prompt("question", "x" * 2000 + "TAIL_EVIDENCE")}
    probes["bm25_top_k_zero"] = {"returned": len(baseline.retriever.retrieve("alpha", top_k=0)), "expected": 0}
    py_files = [SOURCE_ROOT / p for p in historical_paths if p.endswith(".py")]
    for path in py_files:
        compile(path.read_text(), str(path), "exec")
    probes["python_syntax"] = {"files": len(py_files), "status": "passed"}
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)
    historical_source.cleanup()


if __name__ == "__main__":
    main()
