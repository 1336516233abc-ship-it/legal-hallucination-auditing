"""Run the fixed, stratified pilot with one official checkpoint loaded at a time."""

import argparse
from collections import Counter
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
from statistics import mean
import time

import yaml

from retrieval.baseline_direct import DirectPromptingBaseline
from retrieval.baseline_rag import StandardRAGBaseline
from retrieval.cli import component_config
from retrieval.common import LABELS, ROOT, completed_json_object, load_examples, write_json


DATASETS = ("contractnli", "cuad")
SEED = 42
PER_LABEL = 2
INPUT_TOKENS = 4096
OUTPUT_TOKENS = 512
REVISIONS = {
    "llama3": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
    "mistral": "63a8b081895390a26e140280378bc85ec8bce07a",
}
ARTIFACTS = ROOT / "docs/validation/official_baseline_pilot"
RAW_OUTPUT = ROOT / "experiments/results/official_baseline_pilot"
COMPARISON_ARTIFACTS = ROOT / "docs/validation/verified_agentic_rag_pilot"
COMPARISON_RAW_OUTPUT = ROOT / "experiments/results/verified_agentic_rag_pilot"


class MLXPilotModel:
    """Generate locally from the original, unquantized checkpoint on Apple silicon."""

    def __init__(self, snapshot, record_calls=False):
        import mlx.core as mx
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(SEED)
        mx.set_cache_limit(512 * 1024**2)
        self.model, self.tokenizer, self.config = mlx_lm.load(snapshot, lazy=True, return_config=True)
        generation_config = json.loads((Path(snapshot) / "generation_config.json").read_text())
        eos = generation_config.get("eos_token_id")
        if eos is not None:
            self.tokenizer.eos_token_ids = [eos] if isinstance(eos, int) else eos
        self.baseline_tokenizer = self
        self.stream_generate = mlx_lm.stream_generate
        self.sampler = make_sampler(temp=0.0)
        self.backend_version = mlx_lm.__version__
        self.record_calls = record_calls
        self.calls = []

    def _format(self, text):
        return self.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                  tokenize=False, add_generation_prompt=True)

    def encode(self, text, add_special_tokens=True):
        return self.tokenizer.encode(self._format(text), add_special_tokens=False)

    def generate_for_baseline(self, text, max_tokens=OUTPUT_TOKENS, temperature=0, *, stop_at_json=False):
        # The pilot uses greedy decoding for every call, including Agent stages.
        prompt = self._format(text)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(inputs) > INPUT_TOKENS:
            raise ValueError(f"Pilot prompt exceeds {INPUT_TOKENS} tokens")
        output, completed = "", None
        stream = self.stream_generate(self.model, self.tokenizer, inputs, max_tokens=max_tokens,
                                      sampler=self.sampler, prefill_step_size=512)
        with closing(stream):
            for response in stream:
                output += response.text
                if stop_at_json:
                    completed = completed_json_object(output)
                    if completed is not None:
                        break
        result = {"text": completed if completed is not None else output.strip(),
                  "tokens": response.generation_tokens, "model_prompt": prompt}
        if self.record_calls:
            self.calls.append({"input_tokens": len(inputs), "max_output_tokens": max_tokens,
                               "finish_reason": "json_object" if completed is not None else response.finish_reason,
                               **result})
        return result

    generate = generate_for_baseline

    def generate_json_for_baseline(self, prompt):
        return self.generate_for_baseline(prompt, stop_at_json=True)


def cohort(datasets=DATASETS, split="test", data_dir=None):
    """Lock two random samples per gold label and keep source order."""
    data_dir = Path(data_dir) if data_dir else ROOT / "data/processed"
    manifest = {"seed": SEED, "per_label": PER_LABEL, "split": split, "datasets": {}}
    selected = {}
    for dataset in datasets:
        source = data_dir / f"{dataset}_{split}.json"
        examples = load_examples(dataset, data_dir=data_dir, split=split)
        rng = random.Random(SEED)
        pools = {label: [i for i, row in enumerate(examples) if row.label == label]
                 for label in sorted(LABELS[dataset])}
        indices = sorted(i for pool in pools.values() for i in rng.sample(pool, PER_LABEL))
        selected[dataset] = [examples[i] for i in indices]
        manifest["datasets"][dataset] = {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "population": len(examples),
            "rows": [{"id": row.id, "gold_label": row.label} for row in selected[dataset]],
        }
    return manifest, selected


def summarize(examples, predictions):
    if [row.id for row in examples] != [p["id"] for p in predictions]:
        raise ValueError("Pilot predictions must match the frozen cohort in source order")
    rows = []
    for example, prediction in zip(examples, predictions):
        format_pass = prediction["parse_error"] is None and prediction["label"] in LABELS[prediction["dataset"]]
        row = {
            "id": example.id, "gold_label": example.label,
            "prediction": prediction["label"], "format_pass": format_pass,
            "label_correct": format_pass and prediction["label"] == example.label,
            "parse_error": prediction["parse_error"], "raw_response": prediction["raw_response"],
            "input_tokens": prediction["input_budget"]["input_tokens"],
            "input_truncated": prediction["input_budget"]["truncated"],
            "generated_tokens": prediction["tokens"], "latency_seconds": prediction["latency"],
            "answer_is_extractive": prediction.get("answer_is_extractive"),
            "invalid_citations": prediction.get("invalid_citations"),
        }
        if "generation_trace" in prediction:
            row["generation_trace"] = [{k: v for k, v in step.items() if k != "model_prompt"}
                                       for step in prediction["generation_trace"]]
            row["total_generated_tokens"] = sum(step["tokens"] for step in row["generation_trace"])
        for key in ("extraction_error", "grounding_error"):
            if key in prediction:
                row[key] = prediction[key]
        if "initial" in prediction:
            row["initial_label"] = prediction["initial"]["label"]
            row["initial_parse_error"] = prediction["initial"]["parse_error"]
            row["query_analysis"] = prediction["query_analysis"]
            row["verification"] = prediction["verification"]
            revision = prediction["revision"]
            row["revision_type"] = revision["revision_type"] if revision else None
            row["revision_query"] = revision["refined_query"] if revision else None
            row["evidence_offsets"] = [{key: chunk[key] for key in ("doc_id", "start", "end", "chunk_id", "score")}
                                       for chunk in prediction["prompt_evidence"]]
        rows.append(row)
    count = len(rows)
    passed = sum(row["format_pass"] for row in rows)
    correct = sum(row["label_correct"] for row in rows)
    group = {"examples": count, "format_pass": passed, "label_correct": correct,
             "format_pass_rate": passed / count, "label_accuracy": correct / count,
             "label_counts": dict(Counter(row["gold_label"] for row in rows)),
             "mean_latency_seconds": mean(row["latency_seconds"] for row in rows), "rows": rows}
    if predictions[0]["dataset"] == "cuad":
        positives = [row for row in rows if row["gold_label"] == "present"]
        group["verbatim_positive_correct"] = sum(
            row["label_correct"] and row["answer_is_extractive"] is True
            and not row.get("invalid_citations") and not row.get("grounding_error") for row in positives)
        group["gold_positive_count"] = len(positives)
    return group


def lock_cohort(output, manifest):
    path = output / "cohort.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f"Dataset or sampling settings changed since the cohort was locked: {path}")
    else:
        write_json(path, manifest)


def attach_generation_trace(row, mode, calls):
    stages = ["generation"]
    if mode == "verified_agentic_rag":
        stages = ["query_refinement", "initial_generation"]
        if row["verification"] is not None:
            stages.extend(["claim_verification"] * len(row["verification"]["claims"]))
        if row["revision"] is not None:
            stages.append("revision")
    row["generation_trace"] = [{**step, "stage": stage} for step, stage in zip(calls, stages, strict=True)]
    calls.clear()


def run(alias, device, output, raw_output, replace=False, include_agent=False, *,
        datasets=DATASETS, split="test", data_dir=None):
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from utils.llm_interface import LLMInterface

    if include_agent and device != "mlx":
        raise ValueError("The verified Agent pilot uses the shared MLX backend")
    manifest, examples = cohort(datasets, split, data_dir)
    report_path = output / f"{alias}.json"
    if report_path.exists() and not replace and json.loads(report_path.read_text())["status"] == "completed":
        raise FileExistsError(f"Pilot already completed: {report_path}")
    config = yaml.safe_load((ROOT / "configs/model_config.yaml").read_text())["model"][alias]
    retrieval, chunker_options, retriever_options = component_config(
        ROOT / "configs/retrieval_config.yaml", "bm25", "fixed_length")
    report = {
        "status": "running", "checkpoint": config["name"], "revision": REVISIONS[alias],
        "device": device, "python": platform.python_version(),
        "torch": str(torch.__version__), "transformers": transformers.__version__,
        "input_token_limit": INPUT_TOKENS, "generation_token_limit": OUTPUT_TOKENS,
        "temperature": 0, "seed": SEED, "groups": [],
        "retrieval": {"type": "bm25", "chunking": "fixed_length",
                      "top_k": retrieval["retrieval"]["top_k"],
                      "min_score_threshold": retrieval["retrieval"].get("min_score_threshold"),
                      "chunker_options": chunker_options, "retriever_options": retriever_options},
    }
    lock_cohort(output, manifest)
    if include_agent:
        report["agent"] = {"verification": True, "revision": True, "revision_limit": 1,
                           "revision_reverified": False, "reretrieval_top_k": 3,
                           "query_refinement_max_tokens": 100, "claim_verification_max_tokens": 200,
                           "document_scope": "one source contract per query"}
    write_json(report_path, report)
    try:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS device is unavailable")
        transformers.set_seed(SEED)
        snapshot = snapshot_download(config["name"], revision=REVISIONS[alias],
                                     allow_patterns=["model-*.safetensors", "*.json", "tokenizer.model", "chat_template.jinja"],
                                     max_workers=4)
        model_start = time.perf_counter()
        if device == "mlx":
            llm = MLXPilotModel(snapshot, record_calls=include_agent)
            context_limit = llm.config["max_position_embeddings"]
            report.update(backend="mlx-lm", backend_version=llm.backend_version,
                          dtype=llm.config["torch_dtype"], quantized=False, prefill_step_size=512)
        else:
            llm = LLMInterface(snapshot, device=device, max_tokens=OUTPUT_TOKENS, temperature=0)
            context_limit = llm.model.config.max_position_embeddings
            report.update(backend="transformers", dtype=str(next(llm.model.parameters()).dtype), quantized=False)
        report["model_load_seconds"] = time.perf_counter() - model_start
        report["effective_max_input_tokens"] = min(
            INPUT_TOKENS, llm.tokenizer.model_max_length - OUTPUT_TOKENS,
            context_limit - OUTPUT_TOKENS)
        for dataset, sample in examples.items():
            opts = {"dataset": dataset, "max_input_tokens": report["effective_max_input_tokens"]}
            methods = [
                ("direct_prompting", DirectPromptingBaseline(llm, **opts)),
                ("standard_rag", StandardRAGBaseline(
                    llm, top_k=retrieval["retrieval"]["top_k"], chunker_kwargs=chunker_options,
                    retriever_kwargs=retriever_options, min_score_threshold=retrieval["retrieval"].get("min_score_threshold"),
                    **opts)),
            ]
            if include_agent:
                from agent.agentic_rag import AgenticRAG

                methods.append(("verified_agentic_rag", AgenticRAG(llm, chunker_kwargs=chunker_options,
                                                                  retriever_kwargs=retriever_options)))
            for mode, method in methods:
                path = raw_output / alias / f"{dataset}_{mode}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as checkpoint:
                    def record(row):
                        if include_agent:
                            attach_generation_trace(row, mode, llm.calls)
                        checkpoint.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        checkpoint.flush()
                        print(alias, dataset, mode, row["id"], row["label"],
                              row["parse_error"], f"{row['latency']:.2f}s", flush=True)

                    options = opts if mode == "verified_agentic_rag" else {}
                    predictions = method.batch_answer([asdict(row) for row in sample], on_prediction=record, **options)
                group = summarize(sample, predictions)
                report["groups"].append({"dataset": dataset, "mode": mode, **group})
                write_json(report_path, report)
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "blocked" if type(exc).__name__ == "GatedRepoError" else "failed"
        report["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        raise
    finally:
        write_json(report_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=REVISIONS)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu", "mlx"), default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw_output", type=Path)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--include-agent", action="store_true", help="Run the three-method MLX comparison")
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=DATASETS)
    parser.add_argument("--split", choices=("test", "dev"), default="test")
    parser.add_argument("--data_dir", type=Path)
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Select each dataset at most once")
    if (tuple(args.datasets) != DATASETS or args.split != "test" or args.data_dir) and (
            args.output is None or (args.model is not None and args.raw_output is None)):
        parser.error("A custom cohort requires an explicit --output and, for inference, --raw_output")
    output = args.output or (COMPARISON_ARTIFACTS if args.include_agent else ARTIFACTS)
    raw_output = args.raw_output or (COMPARISON_RAW_OUTPUT if args.include_agent else RAW_OUTPUT)
    cohort_options = {"datasets": args.datasets, "split": args.split, "data_dir": args.data_dir}
    if args.model is None:
        manifest, _ = cohort(**cohort_options)
        lock_cohort(output, manifest)
    else:
        run(args.model, args.device, output, raw_output, replace=args.replace, include_agent=args.include_agent,
            **cohort_options)


if __name__ == "__main__":
    main()
