"""Run paired Agent component ablations on the existing locked 10-case pilot."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sqlite3
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_baseline_pilot import cohort, MLXPilotModel, REVISIONS, INPUT_TOKENS, OUTPUT_TOKENS, SEED
from experiments.run_experiments import TracedModel, run_trial
from retrieval.cli import component_config
from retrieval.common import ROOT, write_json

MISTRAL_WEIGHT_HASHES = {
    "model-00001-of-00003.safetensors": "63654d601820b88b1fa8b4a98df5714f700fbc5b3df2cc4ecbabdced35096d31",
    "model-00002-of-00003.safetensors": "a42716540ecb2385d371f2109835921ff535406cac8fe8ff28f2f0b5fc7895bd",
    "model-00003-of-00003.safetensors": "5f86e15cb3ed9078e30ae6e72445e109d0e337d9cde59b9aeea4ce8e44e54a5d",
}


def build_settings(alias, snapshot):
    cohort_manifest, examples = cohort()
    _, chunks, retriever = component_config(ROOT / "configs/retrieval_config.yaml", "bm25", "fixed_length")
    config = json.loads((snapshot / "config.json").read_text())
    if "quantization" in config or "quantization_config" in config:
        raise ValueError("This experiment requires the original unquantized checkpoint")
    weights = sorted(snapshot.glob("model-*.safetensors"))
    if not weights:
        raise ValueError("Expected sharded original model weights")
    print("Hashing model files before inference", flush=True)
    weight_hashes = {}
    for path in weights:
        with path.open("rb") as f:
            weight_hashes[path.name] = hashlib.file_digest(f, "sha256").hexdigest()
    if alias == "llama3":
        from scripts.fetch_verified_llama3 import FILES
        if weight_hashes != {name: digest for name, (_, digest) in FILES.items() if name.endswith(".safetensors")}:
            raise ValueError("Llama 3 weights differ from the pinned original hashes")
    elif snapshot.name != REVISIONS[alias] or weight_hashes != MISTRAL_WEIGHT_HASHES:
        raise ValueError("Mistral snapshot must be the pinned Hugging Face revision")
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for package in ("agent", "retrieval", "experiments", "utils", "data") for path in sorted((ROOT/package).rglob("*.py"))}
    settings = {"kind": "agent_component_ablation_pilot", "model": alias,
                "checkpoint": "mistralai/Mistral-7B-Instruct-v0.2" if alias == "mistral" else "meta-llama/Meta-Llama-3-8B-Instruct",
                "revision": REVISIONS[alias], "backend": "mlx-lm", "quantized": False,
                "dtype": config.get("torch_dtype", config.get("dtype")), "seed": SEED, "temperature": 0,
                "input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS, "cohort": cohort_manifest,
                "weight_sha256": weight_hashes, "source_sha256": source_hashes, "python": platform.python_version(),
                "configuration_sha256": {name: hashlib.sha256((snapshot/name).read_bytes()).hexdigest()
                                         for name in ("config.json", "generation_config.json", "tokenizer.json",
                                                      "tokenizer_config.json", "model.safetensors.index.json")},
                "platform": platform.system(), "architecture": platform.machine(),
                "packages": {p: version(p) for p in ("mlx-lm", "mlx", "transformers", "numpy")},
                "datasets": {dataset: {"count": len(rows), "sha256": cohort_manifest["datasets"][dataset]["source_sha256"],
                                       "selected_ids": [row.id for row in rows]} for dataset, rows in examples.items()}}
    variants = [dict(name=name, mode="agentic_rag", retriever="bm25", chunking="fixed_length", top_k=None,
                     chunker_kwargs=chunks, retriever_kwargs=retriever, verification=verification, revision=revision)
                for name, verification, revision in [("full", True, True), ("no_verification", False, False), ("no_revision", True, False)]]
    return settings, examples, variants


def run(alias, snapshot, output):
    settings, examples, variants = build_settings(alias, snapshot)
    frozen = output / "manifest.json"
    if frozen.exists() and json.loads(frozen.read_text()) != settings:
        raise ValueError("Manifest changed; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(frozen, settings)
    backend = MLXPilotModel(snapshot, record_calls=False)
    model = TracedModel(backend)
    if min(backend.config["max_position_embeddings"], backend.tokenizer.model_max_length) < INPUT_TOKENS + OUTPUT_TOKENS:
        raise ValueError("Model context cannot support the frozen budget")
    summary = {"status": "running", "kind": settings["kind"], "settings": settings, "groups": [],
               "started_at": datetime.now(timezone.utc).isoformat(), "planned": 30, "completed": 0}
    write_json(output / "summary.json", summary)
    try:
        for dataset, rows in examples.items():
            for variant in variants:
                print(f'Starting {alias} {dataset} {variant["name"]}: {len(rows)} examples', flush=True)
                directory = output / dataset / variant["name"]
                result = run_trial(model, rows, dataset, variant, INPUT_TOKENS, directory, {**settings, "variant": variant})
                summary["groups"].append(result)
                summary["completed"] += result["completed"]
                write_json(output / "summary.json", summary)
                # Publish portable complete traces as well as the resume database.
                with sqlite3.connect(directory / "predictions.sqlite3") as db:
                    data = db.execute("SELECT example_id, assessment, trace FROM predictions ORDER BY example_id").fetchall()
                with (directory / "predictions.jsonl").open("w", encoding="utf-8") as f:
                    for example_id, assessment, trace in data:
                        f.write(json.dumps({"example_id": example_id, "assessment": json.loads(assessment), "trace": json.loads(trace)}, ensure_ascii=False) + "\n")
                print(f'Completed {alias} {dataset} {variant["name"]}: {result["label_correct"]}/{result["completed"]} correct; {result["format_pass"]} parsed', flush=True)
        summary["status"] = "completed"
    except BaseException as exc:
        summary.update(status="interrupted", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "summary.json", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=REVISIONS, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.model, args.snapshot, args.output)
