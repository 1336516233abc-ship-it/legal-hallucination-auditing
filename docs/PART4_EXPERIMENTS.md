# Part 4 Experiment Handoff

Owner: Li Xinru. Delivery scope includes experiment orchestration, real experiments, result analysis, and UI. The report has been organized and corrected according to traceable records, see [FINAL_REPORT.md](../FINAL_REPORT.md); incomplete independent evaluation and full component ablation are explicitly preserved in the report.

Result entry points: [Retrieval and Ablation Results Table](validation/part4_2026-11-22/RESULTS.md); [2026-11-24 Full Record Analysis and Metric Verification](validation/part4_2026-11-24/README.md). Original test data, existing full results, and team member legacy files are preserved as-is.

The official UI has been integrated into `ui/app.py` and incorporated into the demonstration requirements per latest confirmation. For operation instructions, see [UI Handoff](UI_HANDOFF.md); for actual testing records, see [Official UI Acceptance](validation/ui_2026-11-23/README.md). The demonstration uses [Current Guidelines](DEMO_RUNBOOK.md) and does not reuse unverified "100% revision success rate" from old drafts.

The "full" in this document refers to the delivered processed test split; CUAD covers the delivered 10 categories, not the official 41-class complete task set. [Split and Label Description](../data/processed/README.md)

## Delivery Scope

|Part|Scope|Records|
|---|---|---|
|New retrieval runs|ContractNLI 2,091 items, CUAD 1,020 items; BM25/Dense/Hybrid × fixed/clause × k=3/5/10, total 36 configurations|`validation/part4_2026-11-22/*full_rerun.json`|
|New Agent ablation|Fixed pilot set ContractNLI 6 items, CUAD 4 items; two official models × Full/No verification/No revision, total 60 outputs|`validation/part4_2026-11-22/agent_ablation/`|
|Archived main experiment review|Mistral/Llama 3 × two datasets × Direct/RAG/Agent, summary of 18,666 archived outputs; verified data hashes and confusion matrices|`validation/part4_2026-11-22/archived_result_audit/`|
|Independent answer support|Protocol and blind review export entry prepared, no human review results yet|[Review Protocol](ANSWER_SUPPORT_PROTOCOL.md)|

The new Agent ablation is a fixed small sample for process checking and cannot be treated as full test set ablation results. Samples were selected during the old baseline pilot phase, 2 examples per category, seed 42, with all three variants using the same samples. Disabling verification simultaneously disables revision that depends on verification trigger; disabling revision preserves verification. Without independent review, detector flags are not treated as hallucination ground truth, nor is "revision occurred" treated as "revision succeeded".

## Orchestration Fixes

- `experiments/run_experiments.py` actually applies dataset, sample_size, seed, and mode selection; ablation switches execute corresponding variants.
- Uses existing task-specific generation/parsing/scoring interfaces, CUAD uses present/absent, ContractNLI uses three-class. Invalid outputs explicitly count as errors; independent hallucination rate and other untested metrics are null.
- Each model, dataset, and variant saves independent SQLite with per-instance commits; before resuming, compare data hashes, sample IDs, code, and parameters to avoid mixing results. All model calls enter traces and token counts.
- `retrieval/__init__.py` lazy exports baseline classes; 5 Agent modules defer LLM imports only used for type hints, supporting lightweight dry run and existing tests. Did not rewrite team member's Agent decision flow.
- `summarize_results.py` verifies archives; `summarize_part4.py` rescores from new predictions and verifies group statistics, paired samples, switch execution, and total tokens.

## Reproduction

Execute from repository root, Python 3.12 environment. Dependencies install only by run target; Apple Silicon real model pilot uses MLX, CUDA generic runner uses repository `requirements.txt`.

```bash
python -m pip install -r requirements-retrieval.txt
python -m experiments.run_experiments --model mistral --dataset cuad \
  --sample_size 4 --mode agent --ablations agent --seed 42 --dry_run
python -m pytest tests -q
```

Retrieval rerun:

```bash
python -m pip install 'sentence-transformers==6.3.0' 'torch==2.15.0' 'transformers==5.24.0'
python -m retrieval.evaluate_retrieval --datasets contractnli cuad --retrievers bm25 \
  --chunkings fixed_length clause_level --top_ks 3 5 10 --seed 42 --output RUN/bm25_full_rerun.json
python -m retrieval.evaluate_retrieval --datasets contractnli cuad --retrievers dense hybrid \
  --chunkings fixed_length clause_level --top_ks 3 5 10 --seed 42 --output RUN/dense_hybrid_full_rerun.json
```

Real Agent small sample:

```bash
python -m pip install 'mlx-lm==0.32.2' 'mlx==0.32.4' 'transformers==5.24.0'
# Download the original pinned Mistral checkpoint; provide its snapshot directory below.
hf download mistralai/Mistral-7B-Instruct-v0.2 --revision 63a8b081895390a26e140280378bc85ec8bce07a
python scripts/fetch_verified_llama3.py MODEL_DIR/llama3-pinned
python -m experiments.run_agent_ablation_pilot --model mistral --snapshot MISTRAL_SNAPSHOT \
  --output RUN/agent_ablation/mistral
python -m experiments.run_agent_ablation_pilot --model llama3 --snapshot MODEL_DIR/llama3-pinned \
  --output RUN/agent_ablation/llama3
```

Load only one original model at a time. Existing directories continue incomplete samples after verifying manifest; changing code or dependencies requires using a new directory. Weights and source SHA256, configuration, sample IDs, Python/MLX/Transformers versions recorded in manifest; original unquantized, cannot switch to quantized model and still use the same comparison. Different backends and dependencies may produce different outputs. First group timing may include weights' first load; cannot derive strict speed improvement conclusions from this.

This local environment: Apple M5, 24 GiB memory, macOS 26.6.2, using Metal GPU serial inference. [Hardware and Runtime Environment](validation/part4_2026-11-22/execution_environment.json)

Summary and verification:

```bash
python -m experiments.summarize_results --output RUN/archived_result_audit --plots
python -m experiments.summarize_part4 --directory RUN
```

Generic CUDA orchestration example (another new output directory):

```bash
python -m experiments.run_experiments --model llama3 mistral --dataset all \
  --mode agent --ablations agent --sample_size 20 --seed 42 --device cuda --output RUN_CUDA
# Resume only with exactly the same arguments and frozen sources:
python -m experiments.run_experiments --model llama3 mistral --dataset all \
  --mode agent --ablations agent --sample_size 20 --seed 42 --device cuda --output RUN_CUDA --resume
```

Generic runner's model loading follows `model_config.yaml`, recording actual resolved revision; to reproduce this fixed checkpoint MLX pilot, use dedicated entry. Retrieval switch `--ablations retrieval` is retrieval configuration comparison with real generation, range taken from configuration (currently BM25/Dense); pure retrieval full includes Hybrid, use `evaluate_retrieval` above.

## Still Pending

1. AutoDL full per-instance records obtained and verified on 2026-11-24; full error statistics, pre/post revision label changes, and 20 traceable cases completed, see [Analysis Delivery](validation/part4_2026-11-24/README.md). Local blind review materials remain unannotated drafts; sample size and division of labor await team confirmation.
2. Team confirms independent answer support protocol and completes annotation, then calculates hallucination rate across three methods. Currently no such results, nor reliable "100% revision success rate".
3. If full Agent component ablation needed, expand sample on available GPU/server, preserving same data and model version for all three variants. This small sample does not replace that item.

This historical Part 4 analysis package contains source code, summaries, per-instance JSONL, and validation records, excluding model weights, virtual environments, database temporary files, or report PDF. Subsequent course submission package and official report PDF have been separately released, see [Final Submission Instructions](FINAL_SUBMISSION.md); do not treat historical analysis package as final submission package. SQLite checkpoints remain on run machine; JSONL provides complete portable predictions and call traces.

Shared evaluator fixed in subsequent main commit `5e84091`, see [Final Evaluation Fixes](FINAL_EVALUATION_FIXES.md). Old evaluator diagnostics in historical analysis preserve original timepoint meaning; new recomputation see [Final Trace Audit](validation/final_submission_trace_audit.json).
