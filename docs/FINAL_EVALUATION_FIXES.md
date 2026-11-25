# Group 55 Final Evaluation Fix Records

**Date**: November 25, 2026. Project: Legal Hallucination Auditing with Agentic Retrieval for Contract Review. This fix started from GitHub `main` commit `ab185fa`, involving legacy evaluator fixes, full-scale trace recomputation, and final report corrections.

## Legacy Evaluator Fixes

| Issue Found | Fix and Regression Results |
| --- | --- |
| `invalid` was counted as correct on `neutral` samples | Score based on the legal label sets of ContractNLI and CUAD respectively. Invalid labels, cross-task labels, and parse failures all enter the denominator and count as errors. |
| Macro-F1 for CUAD two-class counterexample was calculated as 0.33 | Macro-F1 averages only over legal classes for the corresponding task; this counterexample corrected to 0.50. |
| Empty reference answer throws exception | Skip empty references. When reference exists but no valid output, word set recall is recorded as zero. This surface-level metric is output separately. |
| Two samples each hitting one span, old Hit Rate returned 0.50 | Reuse original text character interval scoring; Recall@k and Hit Rate use the same `k`. Hit Rate counts by sample; this counterexample corrected to 1.00. When verifiable positions are missing, evidence metrics remain untested. |

Read initial answer check records from `verification.claims`, separately counting unsupported proportion, check total, and failure count among valid checks. Abstention precision is calculated only when prediction explicitly abstains, reference provides explicit judgment, and denominator is non-zero. Independent final answer hallucination rate, semantic completeness, and semantic revision success rate still lack independent scoring. For code and regression test cases, see [Evaluator](../evaluation/metrics.py) and [Regression Tests](../tests/test_legacy_metrics.py).

## Full-scale Recomputation and Report Corrections

Recomputed **18,666 predictions and 37,834 model calls** from Llama 3 and Mistral original traces. Three methods across two models on ContractNLI and CUAD totaling **12 groups**; accuracy, Macro-F1, and format pass counts all consistent with archives. Entire process reused existing traces.

| Method | Input tokens | Generated tokens | Total |
| --- | ---: | ---: | ---: |
| Direct Prompting | 16,726,643 | 347,613 | 17,074,256 |
| Standard RAG | 17,561,928 | 409,452 | 17,971,380 |
| Verified Agentic RAG | 50,416,305 | 2,050,494 | 52,466,799 |
| Total | 84,704,876 | 2,807,559 | 87,512,435 |

Standard RAG token usage is 1.05x that of Direct; Agent is 2.92x that of Standard RAG. Small-sample ablation listed separately; actual costs based on AutoDL bills. Final report retains Llama 3 on ContractNLI improving from 42.52% to 55.52%, while also listing declines in the other three combinations. Report removed untraceable manual review claims, distinguishes task label changes, initial answer verifier judgments, and independent semantic evaluation, and replaced example hashes in the appendix. [Recomputation Results](validation/final_submission_trace_audit.json) · [Report Markdown](../FINAL_REPORT.md) · [Official Report PDF](final_report/COMP7608B-Group55-Final-Report.pdf)

## Acceptance and Reproduction

Extracted course submission package on the same Mac, ran using fresh Python 3.11 environment and existing Mistral weights. **188 automated tests passed**. Dry run of two-model experiment generated 6 scheduled tasks. Real Gradio service correctly handled 30-day notice and liability cap examples. 60-day modified condition example produced invalid label and blocked answer; empty contract was rejected. Test injection also covered out-of-bounds citations and recovery of next request after model exception. Installation and inference on other machines not validated. [Extraction Acceptance Original Records](validation/final_submission_acceptance.json)

```bash
python -m pip install -r requirements-retrieval.txt -r requirements-ui.txt
python -m pip install 'scikit-learn>=1.3' 'torch>=2.0.0' 'transformers==5.23.0' 'requests>=2.31.0'
python -m pytest -q tests docs/validation/part4_2026-11-23/test_local_delivery.py
python scripts/verify_full_trace_export.py /path/to/evaluation_traces
```

`evaluation_traces` is taken from the course submission package, containing two gzip JSONL files and export manifest. GitHub preserves this fix, recomputation results, report, and acceptance records. Course submission ZIP, original per-instance traces, and model weights are stored separately. The version fingerprint in the extraction acceptance record points to the validated course submission package. [Course Submission and Startup Instructions](FINAL_SUBMISSION.md)
