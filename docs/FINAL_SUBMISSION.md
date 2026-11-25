# Group 55 Final Submission

Project: **Legal Hallucination Auditing with Agentic Retrieval for Contract Review**. Team members: Li Xinru, Qin Mengmeng, Yin Fengling, Wang Zhengyao.

## Running After Extraction

Run from the `legal-hallucination-auditing/` directory inside the ZIP. Use Python 3.11 or 3.12, create a new virtual environment, then install dependencies. Apple Silicon uses MLX; no remote model API needed when local model weights are available. Model weights must be obtained separately; Llama 3 may require model repository access permissions. For CUDA mode and memory requirements, see [UI Handoff](UI_HANDOFF.md).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-retrieval.txt -r requirements-ui.txt
python -m pip install 'mlx-lm==0.32.2' 'mlx==0.32.4' 'transformers==5.23.0'
python ui/app.py --model mistral --backend mlx --snapshot /path/to/mistral-snapshot --port 7860
```

Open port 7860 in your browser. For normal, modified condition, and missing input examples, see [example file](../examples/final_review_cases.json). The built-in 30-day notice, 60-day notice, and liability cap buttons on the page can also be used to reproduce. Text evidence comes from the current contract; real models may produce invalid labels and be blocked by safety mechanisms. The official UI also supports Llama 3 with appropriate snapshots. Local commands and independently deployed shared demo services are configured separately.

## Validation and Evaluation

```bash
python -m pip install 'scikit-learn>=1.3' 'torch>=2.0.0' 'requests>=2.31.0'
python -m pytest -q tests docs/validation/part4_2026-11-23/test_local_delivery.py
python -m experiments.run_experiments --model llama3 mistral --dataset cuad --sample_size 1 --mode all --dry_run
python scripts/verify_full_trace_export.py ../evaluation_traces
```

Runtime environment records for full-scale tasks can be found in the two official summary files. Complete regression tests that depend on Torch and Transformers require installing corresponding model dependencies. Fault injection test cases cover invalid labels and out-of-bounds citations; model returns in tests are explicitly marked as synthetic. For browser acceptance and service calls with real models, see [Acceptance Records](validation/ui_2026-11-23/README.md). The final ZIP will include evidence from this run.

Comparison evaluation uses task-legal labels. Format failures are included in the accuracy denominator; Macro-F1 averages only over legal labels. Evidence metrics require original text positions; initial answer verifier diagnostics preserve failure counts. Word set recall is a surface-level proxy metric. Independent final answer hallucination rate, semantic answer completeness, and formal abstention precision remain untested when annotations are absent.

## File Locations

- [Report PDF](final_report/COMP7608B-Group55-Final-Report.pdf) uses the official NeurIPS 2026 style. The same file is attached to the course ZIP root directory; LaTeX source code is saved in `docs/final_report/`.
- `evaluation_traces/` preserves two original compressed JSONL files and export manifest, totaling 18,666 predictions. The submission package retains equivalent per-instance JSONL data. The original SQLite is in a separate archive.
- Experiment results and analysis can be found in [Full-scale Trace Analysis](validation/part4_2026-11-24/README.md) and [Final Recomputation](validation/final_submission_trace_audit.json).
- The submitted version is based on the version fingerprint and per-file SHA-256 in `manifest.json` at the ZIP root directory. `acceptance.json` records post-extraction regression and real Q&A. For AI usage and available prompts, see [AI_USE.md](../AI_USE.md).
- The GitHub version provides [Fix Records](FINAL_EVALUATION_FIXES.md) and [Extraction Acceptance Records](validation/final_submission_acceptance.json).
