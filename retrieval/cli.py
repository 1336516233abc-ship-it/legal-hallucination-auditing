"""Shared command-line runner for reproducible baseline generation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import random
import sys

from retrieval.common import ROOT, load_examples, select_examples, write_json


def component_config(path, retriever, chunking):
    import yaml
    config = yaml.safe_load(Path(path).read_text())
    retriever_options = dict(config["retrievers"][retriever])
    retriever_options.pop("type", None)
    return config, dict(config["chunking"][chunking]), retriever_options


def build_parser(mode):
    parser = argparse.ArgumentParser(description=f"Run {mode} on processed legal examples")
    parser.add_argument("--model", default="llama3", help="Model alias from model config")
    parser.add_argument("--model_name", help="Override with a HuggingFace identifier or local model path")
    parser.add_argument("--dataset", choices=["contractnli", "cuad"], default="contractnli")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data/processed", help="Processed JSON directory")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/model_config.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample_size", type=int, help="Number of examples; omitted means the full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_tokens", type=int, default=4096, help="Full prompt budget, before generation")
    parser.add_argument("--temperature", type=float, help="Override generation temperature; zero is greedy")
    parser.add_argument("--max_tokens", type=int, help="Maximum generated tokens")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--dry_run", action="store_true", help="Validate configuration and selection without loading models")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output and checkpoint files")
    if mode == "standard_rag":
        parser.add_argument("--retrieval_config", type=Path, default=ROOT / "configs/retrieval_config.yaml")
        parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="bm25")
        parser.add_argument("--chunking", choices=["fixed_length", "clause_level"], default="fixed_length")
        parser.add_argument("--top_k", type=int, help="Override configured top-k")
        parser.add_argument("--chunk_size", type=int)
        parser.add_argument("--overlap", type=int)
    return parser


def run_baseline(mode, argv=None, llm_factory=None):
    import yaml
    args = build_parser(mode).parse_args(argv)
    if args.max_input_tokens <= 0 or (args.max_tokens is not None and args.max_tokens <= 0):
        raise ValueError("Input and generation token budgets must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model_config = yaml.safe_load(args.config.read_text())["model"][args.model].copy()
    if args.model_name:
        model_config["name"] = args.model_name
    if args.max_tokens is not None:
        model_config["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        if args.temperature < 0:
            raise ValueError("temperature must be nonnegative")
        model_config["temperature"] = args.temperature
    examples = select_examples(load_examples(args.dataset, args.data_dir, args.split), args.sample_size, args.seed)
    options = {}
    if mode == "standard_rag":
        retrieval_config, chunker_options, retriever_options = component_config(
            args.retrieval_config, args.retriever, args.chunking)
        if args.chunk_size is not None or args.overlap is not None:
            if args.chunking != "fixed_length":
                raise ValueError("--chunk_size and --overlap require fixed_length chunking")
            if args.chunk_size is not None:
                chunker_options["chunk_size"] = args.chunk_size
            if args.overlap is not None:
                chunker_options["overlap"] = args.overlap
        options = dict(retriever_type=args.retriever, chunking_strategy=args.chunking,
                       top_k=args.top_k if args.top_k is not None else retrieval_config["retrieval"]["top_k"],
                       chunker_kwargs=chunker_options, retriever_kwargs=retriever_options,
                       min_score_threshold=retrieval_config["retrieval"].get("min_score_threshold"))
        from retrieval.baseline_rag import StandardRAGBaseline
        StandardRAGBaseline(None, dataset=args.dataset, **options)
    selected_ids = [example.id for example in examples]
    dataset_path = args.data_dir / f"{args.dataset}_{args.split}.json"
    manifest = {
        "schema_version": 1, "mode": mode, "dataset": args.dataset, "split": args.split,
        "seed": args.seed, "sample_size": args.sample_size, "example_count": len(examples),
        "selected_ids": selected_ids, "model": model_config, "retrieval": options,
        "requested_max_input_tokens": args.max_input_tokens, "device": args.device,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "dry_run" if args.dry_run else "running",
        "argv": list(argv) if argv is not None else sys.argv[1:],
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return manifest
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "experiments/results" / f"{mode}_{args.model}_{args.dataset}_{stamp}.json"
    metadata_path = output.with_suffix(".meta.json")
    checkpoint_path = output.with_suffix(".jsonl")
    for path in [output, metadata_path, checkpoint_path]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}; use --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    if llm_factory is None:
        import numpy as np
        import torch
        from transformers import set_seed
        np.random.seed(args.seed)
        set_seed(args.seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        from utils.llm_interface import LLMInterface
        llm_factory = LLMInterface
    if args.overwrite and output.exists():
        output.unlink()
    write_json(metadata_path, manifest)
    try:
        llm = llm_factory(model_name=model_config["name"], device=args.device,
                          max_tokens=model_config["max_tokens"],
                          temperature=model_config.get("temperature", 0.7),
                          top_p=model_config.get("top_p", 0.9))
        budget = args.max_input_tokens
        limits = [getattr(getattr(llm, "tokenizer", None), "model_max_length", None),
                  getattr(getattr(getattr(llm, "model", None), "config", None), "max_position_embeddings", None)]
        limits = [limit for limit in limits if isinstance(limit, int) and 0 < limit < 10**8]
        if limits:
            budget = min(budget, min(limits) - model_config["max_tokens"])
        if budget <= 0:
            raise ValueError("Generation reserve leaves no input token budget")
        manifest["effective_max_input_tokens"] = budget
        manifest["resolved_model_revision"] = getattr(getattr(getattr(llm, "model", None), "config", None), "_commit_hash", None)
        if mode == "direct_prompting":
            from retrieval.baseline_direct import DirectPromptingBaseline
            baseline = DirectPromptingBaseline(llm, dataset=args.dataset, max_input_tokens=budget)
        else:
            from retrieval.baseline_rag import StandardRAGBaseline
            baseline = StandardRAGBaseline(llm, dataset=args.dataset, max_input_tokens=budget, **options)
        with checkpoint_path.open("w", encoding="utf-8") as checkpoint:
            def record(prediction):
                checkpoint.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                checkpoint.flush()
            predictions = baseline.batch_answer(examples, on_prediction=record)
        write_json(output, predictions)
        manifest.update(status="completed", prediction_count=len(predictions),
                        parse_error_count=sum(bool(p["parse_error"]) for p in predictions),
                        checkpoint=str(checkpoint_path), output=str(output))
        manifest["validation_status"] = "has_parse_errors" if manifest["parse_error_count"] else "passed"
        if manifest["parse_error_count"]:
            logging.warning("%d outputs failed task parsing; inspect parse_error before evaluation", manifest["parse_error_count"])
        write_json(metadata_path, manifest)
        logging.info("Saved %d predictions to %s", len(predictions), output)
        return predictions
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", checkpoint=str(checkpoint_path))
        write_json(metadata_path, manifest)
        raise
