"""Export blinded, paired review forms from a full-run SQLite checkpoint."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.common import ROOT, load_examples, write_json


def prepare(checkpoint, output, model_alias, sample_size=50, seed=42, data_dir=None):
    """Keep missing/invalid outputs visible and never generate support judgments."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if output.exists():
        raise FileExistsError("Choose a new output directory to protect review annotations")
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    with sqlite3.connect(checkpoint.as_uri() + "?mode=ro", uri=True) as db:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Checkpoint failed SQLite integrity check")
        frozen = json.loads(db.execute("SELECT settings FROM run WHERE id=1").fetchone()[0])
        rows = db.execute("SELECT dataset, method, example_id, assessment, trace FROM predictions").fetchall()
    hashes = frozen.get("datasets", frozen.get("suite", {}).get("datasets", {}))
    by_dataset = defaultdict(lambda: defaultdict(dict))
    for dataset, method, example_id, assessment, trace in rows:
        by_dataset[dataset][method][example_id] = (json.loads(assessment), json.loads(trace))
    output_rows, key, selection = [], {}, {}
    for dataset, methods in sorted(by_dataset.items()):
        directory = Path(data_dir) if data_dir else ROOT / "data/processed"
        path = directory / f"{dataset}_test.json"
        if dataset not in hashes or hashlib.sha256(path.read_bytes()).hexdigest() != hashes[dataset]["sha256"]:
            raise ValueError("Source dataset differs from checkpoint; obtain its frozen processed data")
        examples = {row.id: row for row in load_examples(dataset, directory)}
        cohorts = [set(predictions) for predictions in methods.values()]
        if any(ids != cohorts[0] for ids in cohorts):
            raise ValueError("Methods have unequal cohorts; complete or explicitly reconcile the checkpoint first")
        if len(cohorts[0]) != hashes[dataset]["count"]:
            raise ValueError("Checkpoint is incomplete relative to its frozen cohort")
        ids = sorted(cohorts[0])
        selected = sorted(random.Random(seed).sample(ids, min(sample_size, len(ids))))
        selection[dataset] = {"population": len(ids), "sampled": len(selected), "example_ids": selected}
        for example_id in selected:
            row = examples[example_id]
            for method, predictions in sorted(methods.items()):
                assessment, trace = predictions[example_id]
                identity = f"{model_alias}/{dataset}/{method}/{example_id}"
                blind_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
                key[blind_id] = dict(model=model_alias, dataset=dataset, method=method, example_id=example_id)
                output_rows.append({"blind_id": blind_id, "dataset": dataset, "query": row.query,
                                    "source_contract": row.context, "predicted_label": trace.get("label"),
                                    "displayed_answer": trace.get("final_answer", trace.get("answer", "")),
                                    "raw_response": trace.get("raw_response"), "format_pass": assessment["format_pass"],
                                    "parse_error": assessment.get("parse_error"),
                                    "review": {"annotator": None, "response_type": None, "claims": [],
                                               "all_claims_reviewed": False, "needs_adjudication": None, "notes": ""}})
    random.Random(seed).shuffle(output_rows)
    output.mkdir(parents=True)
    with (output / "blinded_review.jsonl").open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output / "private_key.json", key)
    write_json(output / "selection.json", {"seed": seed, "sampling": "paired simple random sample within each dataset",
                                           "checkpoint": str(checkpoint), "selection": selection,
                                           "reviewed": False, "independent_hallucination_rate": None})
    return len(output_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sample_size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", type=Path)
    args = p.parse_args()
    n = prepare(args.checkpoint, args.output, args.model, args.sample_size, args.seed, args.data_dir)
    print(f"Exported {n} unreviewed outputs. No hallucination scores have been assigned.")


if __name__ == "__main__":
    main()
