"""Run a retrieval-only ablation matrix without calling a generation model."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import platform
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.baseline_rag import StandardRAGBaseline
from retrieval.cli import component_config
from retrieval.common import ROOT, document_id, example_dict, load_examples, select_examples, write_json
from retrieval.evidence import aggregate_evidence, score_evidence


def evaluate(datasets, retrievers, chunkings, top_ks, config_path, sample_size=None, seed=42, data_dir=None):
    if not top_ks or any(type(k) is not int or k < 0 for k in top_ks):
        raise ValueError("top_ks must contain nonnegative integers")
    results = {"schema_version": 1, "kind": "retrieval_only", "seed": seed,
               "sample_size": sample_size, "created_at": datetime.now(timezone.utc).isoformat(),
               "metric_definition": "macro full-gold-span union coverage over non-whitespace characters; no-evidence examples excluded",
               "datasets": {}, "experiments": []}
    results["runtime"] = {"python": platform.python_version(), "packages": {}}
    for package in ["numpy", "rank-bm25", "sentence-transformers", "torch", "transformers"]:
        try:
            results["runtime"]["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    for name in datasets:
        rows = [example_dict(row) for row in select_examples(load_examples(name, data_dir), sample_size, seed)]
        path = Path(data_dir or ROOT / "data/processed") / f"{name}_test.json"
        results["datasets"][name] = {"examples": len(rows), "selected_ids": [r["id"] for r in rows],
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        doc_ids = [document_id(row["context"], row.get("metadata"), name) for row in rows]
        groups = {}
        for row, doc_id in zip(rows, doc_ids):
            groups.setdefault(doc_id, []).append(row)
        for chunking in chunkings:
            for retriever in retrievers:
                config, chunk_options, retriever_options = component_config(config_path, retriever, chunking)
                baseline = StandardRAGBaseline(None, dataset=name, retriever_type=retriever,
                    chunking_strategy=chunking, top_k=max(top_ks), chunker_kwargs=chunk_options,
                    retriever_kwargs=retriever_options,
                    min_score_threshold=config["retrieval"].get("min_score_threshold"))
                baseline.index_documents([row["context"] for row in rows], doc_ids)
                scores = {k: [] for k in top_ks}
                failures = {k: [] for k in top_ks}
                start_time = time.perf_counter()
                for doc_id, group in groups.items():
                    for row in group:
                        chunks = [{**chunk, "score": score} for chunk, score in baseline.retrieve_evidence(row["query"], doc_id=doc_id)]
                        for k in top_ks:
                            score = score_evidence(chunks, row, k)
                            scores[k].append(score)
                            if score is not None and score["evidence_recall_at_k"] < 1 and len(failures[k]) < 3:
                                failures[k].append({"id": row["id"], "doc_id": doc_id, "query": row["query"],
                                    "gold_spans": row["evidence_spans"], "score": score,
                                    "retrieved_spans": [{key: chunk[key] for key in ["start", "end", "chunk_id", "score"]} for chunk in chunks[:k]]})
                elapsed = time.perf_counter() - start_time
                dense = baseline.retriever if retriever == "dense" else getattr(baseline.retriever, "dense_retriever", None)
                encoder = getattr(dense, "encoder", None)
                first_module = next(iter(getattr(encoder, "_modules", {}).values()), None)
                encoder_info = None if encoder is None else {
                    "model_name": dense.model_name,
                    "max_seq_length": getattr(encoder, "max_seq_length", None),
                    "revision": getattr(getattr(getattr(first_module, "auto_model", None), "config", None), "_commit_hash", None),
                    "long_chunk_policy": "encoder truncation at max_seq_length",
                }
                for k in top_ks:
                    results["experiments"].append({"dataset": name, "retriever": retriever,
                        "chunking": chunking, "top_k": k, "chunker_kwargs": chunk_options,
                        "retriever_kwargs": retriever_options, "encoder": encoder_info, "metrics": aggregate_evidence(scores[k]),
                        "shared_retrieval_seconds": elapsed, "failure_examples": failures[k]})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["contractnli", "cuad"], default=["contractnli", "cuad"])
    parser.add_argument("--retrievers", nargs="+", choices=["bm25", "dense", "hybrid"], default=["bm25"])
    parser.add_argument("--chunkings", nargs="+", choices=["fixed_length", "clause_level"], default=["fixed_length", "clause_level"])
    parser.add_argument("--top_ks", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--sample_size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--retrieval_config", type=Path, default=ROOT / "configs/retrieval_config.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError("Output exists; choose a new path or use --overwrite")
    results = evaluate(args.datasets, args.retrievers, args.chunkings, args.top_ks,
                       args.retrieval_config, args.sample_size, args.seed, args.data_dir)
    write_json(args.output, results)
    for run in results["experiments"]:
        print(run["dataset"], run["retriever"], run["chunking"], run["top_k"], run["metrics"])


if __name__ == "__main__":
    main()
