"""Run task-aware comparisons with frozen cohorts and per-example checkpoints."""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from retrieval.common import ROOT, LABELS, load_examples, select_examples, write_json
from retrieval.cli import component_config

MODES = ("direct_prompting", "standard_rag", "agentic_rag")


class TracedModel:
    """Record every backend call, including query refinement and verification."""

    def __init__(self, model):
        self.model_interface = model
        self.calls = []

    def __getattr__(self, name):
        target = getattr(self.model_interface, name)
        if not name.startswith("generate") or not callable(target):
            return target

        def call(prompt, *args, **kwargs):
            record = {"method": name, "prompt": prompt}
            self.calls.append(record)
            try:
                response = target(prompt, *args, **kwargs)
                record.update(response)
                return response
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
        return call


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/experiment_config.yaml")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model_config.yaml")
    p.add_argument("--retrieval_config", type=Path, default=ROOT / "configs/retrieval_config.yaml")
    p.add_argument("--model", nargs="+", help="Aliases in model_config; defaults to experiment config")
    p.add_argument("--models", help="Compatibility alias: comma-separated model names")
    p.add_argument("--dataset", choices=["all", *LABELS], default="all")
    p.add_argument("--modes", nargs="+", choices=MODES)
    p.add_argument("--mode", choices=["all", "direct", "rag", "agent"], help="Compatibility mode selector")
    p.add_argument("--ablations", choices=["none", "retrieval", "agent", "all"], default="none")
    p.add_argument("--sample_size", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--data_dir", type=Path, default=ROOT / "data/processed")
    p.add_argument("--split", default="test")
    p.add_argument("--output", type=Path)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    p.add_argument("--max_input_tokens", type=int, default=4096)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def plan(args):
    """Validate configuration without importing or loading a model runtime."""
    if args.max_input_tokens <= 0 or args.max_tokens <= 0 or args.temperature < 0:
        raise ValueError("Invalid generation budget or temperature")
    config = yaml.safe_load(args.config.read_text())
    model_configs = yaml.safe_load(args.model_config.read_text())["model"]
    if args.model and args.models:
        raise ValueError("Use either --model or --models")
    aliases = args.model or ([s.strip() for s in args.models.split(",")] if args.models else config["models"])
    if len(set(aliases)) != len(aliases):
        raise ValueError("Duplicate models")
    datasets = ([args.dataset] if args.dataset != "all" else
                [name for name, options in config["datasets"].items() if options.get("enabled")])
    if not datasets or set(datasets) - set(LABELS):
        raise ValueError("Only processed ContractNLI and CUAD tasks are supported")
    if args.mode and args.modes:
        raise ValueError("Use either --mode or --modes")
    mode_map = {"direct": ["direct_prompting"], "rag": ["standard_rag"], "agent": ["agentic_rag"], "all": config["modes"]}
    modes = args.modes or mode_map.get(args.mode, config["modes"])
    if not modes or len(modes) != len(set(modes)) or set(modes) - set(MODES):
        raise ValueError("Invalid or duplicate modes")
    seed = args.seed if args.seed is not None else config["reproducibility"]["seed"]
    cohorts, dataset_info = {}, {}
    for dataset in datasets:
        size = args.sample_size if args.sample_size is not None else config["datasets"][dataset].get("sample_size")
        if size is not None and (type(size) is not int or size <= 0):
            raise ValueError("sample_size must be positive or omitted")
        rows = load_examples(dataset, args.data_dir, args.split)
        if size is not None and size > len(rows):
            raise ValueError("sample_size exceeds the available split")
        cohorts[dataset] = select_examples(rows, size, seed)
        if not cohorts[dataset] or len({row.id for row in cohorts[dataset]}) != len(cohorts[dataset]):
            raise ValueError("Empty cohort or duplicate example IDs")
        dataset_info[dataset] = {"count": len(cohorts[dataset]), "selected_ids": [r.id for r in cohorts[dataset]],
                                 "sha256": hashlib.sha256((args.data_dir / f"{dataset}_{args.split}.json").read_bytes()).hexdigest()}
    variants = [{"name": mode, "mode": mode, "retriever": "bm25", "chunking": "fixed_length", "top_k": 5,
                 "verification": True, "revision": True} for mode in modes]
    if args.ablations in {"agent", "all"}:
        for name, verification, revision in [("agent_full", True, True), ("agent_no_verification", False, False), ("agent_no_revision", True, False)]:
            if name == "agent_full" and "agentic_rag" in modes:
                continue
            variants.append(dict(name=name, mode="agentic_rag", retriever="bm25", chunking="fixed_length", top_k=None,
                                 verification=verification, revision=revision))
    if args.ablations in {"retrieval", "all"}:
        options = config["ablations"]
        for retriever, chunking, k in product(options["retriever"], options["chunking"], options["top_k"]):
            if k <= 0:
                raise ValueError("top_k must be positive")
            variants.append(dict(name=f"rag_{retriever}_{chunking}_k{k}", mode="standard_rag", retriever=retriever,
                                 chunking=chunking, top_k=k, verification=False, revision=False))
    # Agent uses the existing query analyzer's dynamic k, not a fixed-k override.
    for variant in variants:
        if variant["mode"] == "agentic_rag":
            variant["top_k"] = None
        _, chunker, retriever = component_config(args.retrieval_config, variant["retriever"], variant["chunking"])
        variant.update(chunker_kwargs=chunker, retriever_kwargs=retriever)
    models = {alias: {**model_configs[alias], "temperature": args.temperature, "max_tokens": args.max_tokens}
              for alias in aliases}
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for directory in ("agent", "retrieval", "utils", "experiments", "data")
               for p in sorted((ROOT / directory).rglob("*.py"))}
    manifest = dict(schema_version=1, seed=seed, split=args.split, datasets=dataset_info, models=models,
                    variants=variants, device=args.device, max_input_tokens=args.max_input_tokens,
                    source_sha256=sources,
                    planned_predictions=sum(d["count"] for d in dataset_info.values()) * len(models) * len(variants))
    return manifest, cohorts


def make_system(model, dataset, variant, budget):
    common = dict(retriever_type=variant["retriever"], chunking_strategy=variant["chunking"],
                  chunker_kwargs=variant["chunker_kwargs"], retriever_kwargs=variant["retriever_kwargs"])
    if variant["mode"] == "direct_prompting":
        from retrieval.baseline_direct import DirectPromptingBaseline
        return DirectPromptingBaseline(model, dataset=dataset, max_input_tokens=budget)
    if variant["mode"] == "standard_rag":
        from retrieval.baseline_rag import StandardRAGBaseline
        return StandardRAGBaseline(model, dataset=dataset, max_input_tokens=budget, top_k=variant["top_k"], **common)
    from agent.agentic_rag import AgenticRAG
    return AgenticRAG(model, enable_verification=variant["verification"], enable_revision=variant["revision"], **common)


def run_trial(model, rows, dataset, variant, budget, directory, settings):
    """Resume exact settings; commit each completed example before proceeding."""
    from experiments.run_full_mistral import open_checkpoint, record_prediction, assess_prediction, group_summary
    directory.mkdir(parents=True, exist_ok=True)
    system = make_system(model, dataset, variant, budget)
    with closing(open_checkpoint(directory / "predictions.sqlite3", settings)[0]) as db:
        done = {row[0] for row in db.execute("SELECT example_id FROM predictions")}
        try:
            for row in rows:
                if row.id in done:
                    continue
                model.calls.clear()
                if variant["mode"] == "agentic_rag":
                    result = system.answer(row.query, row.context, dataset=dataset, max_input_tokens=budget)
                else:
                    result = system.answer(row.query, row.context) if variant["mode"] == "direct_prompting" else system.answer(row.query, context=row.context)
                result.update(id=row.id, dataset=dataset, generation_trace=list(model.calls))
                assessment = assess_prediction(row, dataset, result)
                record_prediction(db, (dataset, variant["name"], row.id), assessment, result)
            status = "completed"
        except BaseException as exc:
            write_json(directory / "status.json", {"status": "interrupted", "error": f"{type(exc).__name__}: {exc}",
                                                    "completed": db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
                                                    "failed_example_id": row.id, "failed_generation_trace": model.calls})
            raise
        assessments = [json.loads(r[0]) for r in db.execute("SELECT assessment FROM predictions")]
        result = {"status": status, "dataset": dataset, "variant": variant, **group_summary(assessments, len(rows), LABELS[dataset]),
                  "independent_hallucination_rate": None, "answer_completeness": None, "abstention_precision": None}
        traces = [json.loads(r[0]) for r in db.execute("SELECT trace FROM predictions")]
        tokens = [call.get("tokens") for trace in traces for call in trace["generation_trace"]]
        result["total_generated_tokens"] = sum(tokens) if tokens and all(type(n) is int and n >= 0 for n in tokens) else None
        result["validation_status"] = "has_parse_errors" if result["format_pass"] != result["completed"] else "passed"
        write_json(directory / "summary.json", result)
        write_json(directory / "status.json", {"status": status, "completed": len(assessments)})
        return result


def run(argv=None, llm_factory=None):
    args = build_parser().parse_args(argv)
    manifest, cohorts = plan(args)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest
    if args.resume and args.output is None:
        raise ValueError("--resume requires an explicit --output directory")
    if args.resume and args.temperature != 0:
        raise ValueError("Resuming stochastic decoding is not reproducible; use temperature=0")
    output = args.output or ROOT / "experiments/results" / ("suite_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    frozen = output / "manifest.json"
    if output.exists():
        if not args.resume:
            raise FileExistsError("Output directory exists; choose another or use --resume with identical settings")
        if not frozen.exists() or json.loads(frozen.read_text()) != manifest:
            raise ValueError("Frozen manifest changed; choose a new output directory")
    else:
        output.mkdir(parents=True)
        write_json(frozen, manifest)
    random.seed(manifest["seed"])
    if llm_factory is None:
        from transformers import set_seed
        from utils.llm_interface import LLMInterface
        set_seed(manifest["seed"])
        llm_factory = LLMInterface
    results = []
    for alias, config in manifest["models"].items():
        backend = llm_factory(model_name=config["name"], device=args.device, max_tokens=config["max_tokens"],
                              temperature=config["temperature"], top_p=config.get("top_p", .9))
        model = TracedModel(backend)
        limits = [getattr(getattr(backend, "tokenizer", None), "model_max_length", None),
                  getattr(getattr(getattr(backend, "model", None), "config", None), "max_position_embeddings", None)]
        limits = [limit for limit in limits if type(limit) is int and 0 < limit < 10**8]
        budget = min(args.max_input_tokens, min(limits) - args.max_tokens) if limits else args.max_input_tokens
        if budget <= 0:
            raise ValueError("Generation reserve leaves no input budget")
        revision = getattr(getattr(getattr(backend, "model", None), "config", None), "_commit_hash", None)
        for dataset, rows in cohorts.items():
            for variant in manifest["variants"]:
                settings = {"suite": manifest, "alias": alias, "dataset": dataset, "variant": variant,
                            "effective_max_input_tokens": budget, "resolved_model_revision": revision}
                result = run_trial(model, rows, dataset, variant, budget, output / alias / dataset / variant["name"], settings)
                results.append({"model": alias, **result})
        del model, backend
    write_json(output / "summary.json", {"status": "completed", "groups": results, "manifest": manifest})
    print(f"Completed {len(results)} groups. Results: {output}")
    return results


if __name__ == "__main__":
    run()
