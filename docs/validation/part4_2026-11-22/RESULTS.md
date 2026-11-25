# Part 4 Experiment Results Summary

This page is experiment delivery checklist and numerical tables, course report not yet written.

- New runs: 36 groups complete test set retrieval evaluation; each of two original models 30 Agent ablation outputs, total 60.
- Archived review: Two models × two datasets × three methods, 18,666 full results; verified data hashes and summary confusion matrices, server item-by-item traces not yet retrieved.
- Agent ablation uses existing fixed 10 cases (ContractNLI 6, CUAD 4); each variant uses same samples. Small-sample results only support process checking, cannot extrapolate to full test set performance.
- Independent hallucination rate, answer completeness, and abstention precision not yet assessed; missing values not written as 0.

## New Agent Component Ablation

Both models are fixed revision, unquantized original weights, MLX local inference, temperature 0, seed 42, input 4,096, single output 512 tokens. Full enables verification and revision; No verification simultaneously disables verification and revision dependent on verification; No revision retains verification.

|Model|Dataset|Variant|Label Correct|Format Pass|Macro-F1|Avg sec/item|Total Gen Tokens|Revision Count|
|---|---|---|---:|---:|---:|---:|---:|---:|
|mistral|contractnli|full|3/6|5/6|0.4444|61.60|2127|4|
|mistral|contractnli|no_verification|3/6|5/6|0.4444|22.24|751|0|
|mistral|contractnli|no_revision|3/6|5/6|0.4444|61.34|1879|0|
|mistral|cuad|full|3/4|4/4|0.7333|53.45|943|3|
|mistral|cuad|no_verification|3/4|4/4|0.7333|24.05|467|0|
|mistral|cuad|no_revision|3/4|4/4|0.7333|42.96|773|0|
|llama3|contractnli|full|6/6|6/6|1.0000|68.93|2101|3|
|llama3|contractnli|no_verification|5/6|6/6|0.8222|29.64|967|0|
|llama3|contractnli|no_revision|5/6|6/6|0.8222|55.81|1666|0|
|llama3|cuad|full|1/4|4/4|0.2000|35.74|631|1|
|llama3|cuad|no_verification|1/4|4/4|0.2000|25.84|490|0|
|llama3|cuad|no_revision|1/4|4/4|0.2000|33.83|615|0|

Format failures retained in denominator. Token count totals all model calls; revision count only indicates revision occurred, does not represent revision success. First group timing may include initial weight loading; cannot be used for rigorous cross-variant speed conclusions, nor directly compared with server vLLM timing.

## New Retrieval Evaluation

Maximum absolute difference from archived metrics across 36 groups: `5.55e-16`. Run JSON does not have fixed encoder revision; unique cached version used in offline runs separately verified and recorded in `retrieval_environment.json`; dependency versions in run JSON.

|Dataset|Retriever|Chunking|k|Evidence Recall|Valid Samples|
|---|---|---|---:|---:|---:|
|contractnli|bm25|clause_level|3|93.76%|1188|
|contractnli|bm25|clause_level|5|98.62%|1188|
|contractnli|bm25|clause_level|10|100.00%|1188|
|contractnli|bm25|fixed_length|3|90.81%|1188|
|contractnli|bm25|fixed_length|5|97.89%|1188|
|contractnli|bm25|fixed_length|10|99.97%|1188|
|contractnli|dense|clause_level|3|91.13%|1188|
|contractnli|dense|clause_level|5|97.44%|1188|
|contractnli|dense|clause_level|10|99.97%|1188|
|contractnli|dense|fixed_length|3|89.48%|1188|
|contractnli|dense|fixed_length|5|96.33%|1188|
|contractnli|dense|fixed_length|10|99.85%|1188|
|contractnli|hybrid|clause_level|3|94.80%|1188|
|contractnli|hybrid|clause_level|5|98.62%|1188|
|contractnli|hybrid|clause_level|10|99.97%|1188|
|contractnli|hybrid|fixed_length|3|92.20%|1188|
|contractnli|hybrid|fixed_length|5|98.63%|1188|
|contractnli|hybrid|fixed_length|10|99.97%|1188|
|cuad|bm25|clause_level|3|51.83%|599|
|cuad|bm25|clause_level|5|64.24%|599|
|cuad|bm25|clause_level|10|82.26%|599|
|cuad|bm25|fixed_length|3|51.12%|599|
|cuad|bm25|fixed_length|5|61.75%|599|
|cuad|bm25|fixed_length|10|82.31%|599|
|cuad|dense|clause_level|3|45.72%|599|
|cuad|dense|clause_level|5|59.41%|599|
|cuad|dense|clause_level|10|81.09%|599|
|cuad|dense|fixed_length|3|43.11%|599|
|cuad|dense|fixed_length|5|56.45%|599|
|cuad|dense|fixed_length|10|78.02%|599|
|cuad|hybrid|clause_level|3|51.65%|599|
|cuad|hybrid|clause_level|5|63.77%|599|
|cuad|hybrid|clause_level|10|82.86%|599|
|cuad|hybrid|fixed_length|3|48.98%|599|
|cuad|hybrid|fixed_length|5|60.47%|599|
|cuad|hybrid|fixed_length|10|81.12%|599|

Recall requires complete gold interval covered by union of retrieved chunks; 903 ContractNLI and 421 CUAD without gold evidence do not enter this metric denominator. Recall does not equal answer support rate.

## Archived Main Experiments

See [12-Group Main Experiments and Old Ablation Audit Table](archived_result_audit/results_tables.md) for details. Original server completed full generation; this iteration did not re-run these 18,666 predictions. Old `priority4` ablation has sample dataset mismatches and CUAD label protocol inconsistencies, cannot be used for controlled comparison; this new run records saved separately.

## Traceable Files

- `summary.json`: Machine-readable summary, paired correctness changes, and input file SHA256.
- `agent_ablation/<model>/manifest.json`: Weight hash, code hash, environment, sample IDs.
- `agent_ablation/<model>/<dataset>/<variant>/predictions.jsonl`: Item-by-item predictions, evaluation, and complete model call traces.
- `bm25_full_rerun.json`, `dense_hybrid_full_rerun.json`: 36 groups full retrieval results and failure examples.
- `tests.xml`: Automated test records, separate from real model outputs.
