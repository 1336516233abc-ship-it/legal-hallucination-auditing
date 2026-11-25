# Experiment Results Verification Table

Main experiments: 12 groups, 18,666 system outputs. Data hash and confusion matrix verification passed; server item-by-item traces not yet reviewed.

|Dataset|Model|Method|n|Accuracy|Macro-F1|Format Pass Rate|Avg sec/item|
|---|---|---|---:|---:|---:|---:|---:|
|contractnli|mistral|Direct|2091|43.42%|0.3253|83.69%|1.93|
|contractnli|mistral|RAG|2091|41.89%|0.2707|82.16%|2.52|
|contractnli|mistral|Agent|2091|37.06%|0.2305|75.42%|10.82|
|cuad|mistral|Direct|1020|67.06%|0.7214|85.98%|1.51|
|cuad|mistral|RAG|1020|69.71%|0.7284|90.98%|1.39|
|cuad|mistral|Agent|1020|63.24%|0.6971|81.18%|6.32|
|contractnli|llama3|Direct|2091|47.01%|0.4491|100.00%|1.45|
|contractnli|llama3|RAG|2091|42.52%|0.4271|99.81%|1.85|
|contractnli|llama3|Agent|2091|55.52%|0.5283|98.09%|9.94|
|cuad|llama3|Direct|1020|77.55%|0.7851|94.71%|1.28|
|cuad|llama3|RAG|1020|68.63%|0.7346|84.90%|1.29|
|cuad|llama3|Agent|1020|62.06%|0.6834|78.92%|5.89|

Accuracy denominator retains format failures; Macro-F1 averages over task valid classes. All independent hallucination rates await assessment, not filled as 0.

## Retrieval Ablation

|Dataset|Retriever|Chunking|k|Evidence Recall|Valid Samples|
|---|---|---|---:|---:|---:|
|contractnli|bm25|fixed_length|3|90.81%|1188|
|contractnli|bm25|fixed_length|5|97.89%|1188|
|contractnli|bm25|fixed_length|10|99.97%|1188|
|contractnli|dense|fixed_length|3|89.48%|1188|
|contractnli|dense|fixed_length|5|96.33%|1188|
|contractnli|dense|fixed_length|10|99.85%|1188|
|contractnli|hybrid|fixed_length|3|92.20%|1188|
|contractnli|hybrid|fixed_length|5|98.63%|1188|
|contractnli|hybrid|fixed_length|10|99.97%|1188|
|contractnli|bm25|clause_level|3|93.76%|1188|
|contractnli|bm25|clause_level|5|98.62%|1188|
|contractnli|bm25|clause_level|10|100.00%|1188|
|contractnli|dense|clause_level|3|91.13%|1188|
|contractnli|dense|clause_level|5|97.44%|1188|
|contractnli|dense|clause_level|10|99.97%|1188|
|contractnli|hybrid|clause_level|3|94.80%|1188|
|contractnli|hybrid|clause_level|5|98.62%|1188|
|contractnli|hybrid|clause_level|10|99.97%|1188|
|cuad|bm25|fixed_length|3|51.12%|599|
|cuad|bm25|fixed_length|5|61.75%|599|
|cuad|bm25|fixed_length|10|82.31%|599|
|cuad|dense|fixed_length|3|43.11%|599|
|cuad|dense|fixed_length|5|56.45%|599|
|cuad|dense|fixed_length|10|78.02%|599|
|cuad|hybrid|fixed_length|3|48.98%|599|
|cuad|hybrid|fixed_length|5|60.47%|599|
|cuad|hybrid|fixed_length|10|81.12%|599|
|cuad|bm25|clause_level|3|51.83%|599|
|cuad|bm25|clause_level|5|64.24%|599|
|cuad|bm25|clause_level|10|82.26%|599|
|cuad|dense|clause_level|3|45.72%|599|
|cuad|dense|clause_level|5|59.41%|599|
|cuad|dense|clause_level|10|81.09%|599|
|cuad|hybrid|clause_level|3|51.65%|599|
|cuad|hybrid|clause_level|5|63.77%|599|
|cuad|hybrid|clause_level|10|82.86%|599|

## Old Agent Ablation Audit

These 6 files not used as complete, controlled 2 datasets × 3 variants conclusion.

- `experiments/results/priority4/agentic_rag_full_predictions.json`: Actual dataset {'cuad': 20}; Issues: `dataset_id_mismatch; published_summary_mismatch:total_tokens; published_summary_mismatch:avg_latency; published_summary_mismatch:hallucinations_detected; published_summary_mismatch:revisions_made; legacy_trace_not_independently_evaluated`.
- `experiments/results/priority4/agentic_rag_no_revision_predictions.json`: Actual dataset {'contractnli': 20}; Issues: `legacy_trace_not_independently_evaluated`.
- `experiments/results/priority4/agentic_rag_no_verification_predictions.json`: Actual dataset {'contractnli': 20}; Issues: `legacy_trace_not_independently_evaluated`.
- `experiments/results/priority4/cuad_agentic_rag_full_predictions.json`: Actual dataset {'cuad': 20}; Issues: `invalid_task_label_schema; legacy_trace_not_independently_evaluated`.
- `experiments/results/priority4/cuad_agentic_rag_no_revision_predictions.json`: Actual dataset {'cuad': 20}; Issues: `invalid_task_label_schema; legacy_trace_not_independently_evaluated`.
- `experiments/results/priority4/cuad_agentic_rag_no_verification_predictions.json`: Actual dataset {'cuad': 20}; Issues: `invalid_task_label_schema; legacy_trace_not_independently_evaluated`.
