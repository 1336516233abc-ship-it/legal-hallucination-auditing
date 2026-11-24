"""Recompute archived task scores and token usage from the supplied JSONL exports."""

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import file_digest
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.metrics import MetricsCalculator

MODELS = ("llama3", "mistral")


def verify(directory):
    calculator = MetricsCalculator()
    source = json.loads((directory / "export-manifest.json").read_text())
    files = {item["name"]: item for model in source["models"] for item in model["files"]}
    methods = defaultdict(Counter)
    model_reports = {}
    all_ids = None
    for model in MODELS:
        path = directory / f"{model}-predictions.jsonl.gz"
        with path.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
        if digest != files[path.name]["sha256"]:
            raise ValueError(f"Checksum mismatch: {path.name}")
        expected = json.loads((ROOT / f"docs/validation/{model}_autodl_full_summary.json").read_text())
        groups = defaultdict(lambda: {"predictions": [], "gold": [], "traces": [], "tokens": Counter()})
        ids = set()
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = (row["dataset"], row["method"], row["example_id"])
                if key in ids or row["model"] != expected["settings"]["checkpoint"]:
                    raise ValueError(f"Duplicate sample key or different model: {key}")
                ids.add(key)
                group = groups[key[:2]]
                trace = row["trace"]
                group["predictions"].append({"label": trace["label"], "parse_error": trace.get("parse_error"),
                                            "format_pass": row["assessment"]["format_pass"]})
                group["gold"].append({"label": row["assessment"]["gold_label"]})
                group["traces"].append({"verification": trace.get("verification")})
                tokens = group["tokens"]
                for call in trace["generation_trace"]:
                    tokens["model_calls"] += 1
                    tokens["input_tokens"] += call["input_tokens"]
                    tokens["generated_tokens"] += call["tokens"]
        if all_ids is not None and ids != all_ids:
            raise ValueError("Model outputs have different sample keys")
        all_ids = ids
        if len(groups) != len(expected["groups"]):
            raise ValueError(f"Missing or extra group in {model}")
        reports = []
        for archived in expected["groups"]:
            dataset, method = archived["dataset"], archived["method"]
            g = groups[dataset, method]
            count = len(g["gold"])
            accuracy = calculator.compute_label_accuracy(g["predictions"], g["gold"], dataset=dataset)
            f1 = calculator.compute_macro_f1(g["predictions"], g["gold"], dataset=dataset)
            rate, claims, failed = calculator.compute_initial_verifier_metrics(g["traces"])
            passes = sum(row["format_pass"] for row in g["predictions"])
            tokens = g["tokens"]
            if (count != archived["completed"] or passes != archived["format_pass"]
                    or tokens["model_calls"] != archived["model_calls"]
                    or not math.isclose(accuracy, archived["label_accuracy"], abs_tol=1e-12)
                    or not math.isclose(f1, archived["macro_f1"], abs_tol=1e-12)):
                raise ValueError(f"Archived metrics differ for {model}/{dataset}/{method}")
            tokens["total_tokens"] = tokens["input_tokens"] + tokens["generated_tokens"]
            tokens["predictions"] = count
            methods[method].update(tokens)
            reports.append({"dataset": dataset, "method": method, "accuracy": accuracy, "macro_f1": f1,
                            "format_pass": passes, **tokens,
                            "initial_verifier_unsupported_rate": rate, "initial_verifier_claims": claims,
                            "initial_verifier_failed_claims": failed, "independent_hallucination_rate": None})
        model_reports[model] = {"archive_sha256": digest, "groups": reports}
    totals = Counter()
    for values in methods.values():
        totals.update(values)
    return {"verified_groups": sum(len(report["groups"]) for report in model_reports.values()),
            "paired_keys": len(all_ids), "methods": methods, "totals": totals, "models": model_reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing the two JSONL.gz files and export manifest")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(verify(args.directory), indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(report)
    else:
        print(report, end="")
