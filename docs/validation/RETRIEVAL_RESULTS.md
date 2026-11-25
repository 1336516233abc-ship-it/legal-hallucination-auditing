# Retrieval Ablation Results

This round runs with fixed configurations on two complete test sets, without calling generation models. Result file is retrieval_ablation_2026-11-20.json, containing 36 configurations, data checksums, sample IDs, parameters, and failure cases.

## Metric Criteria

Evidence Recall@k is calculated per contract by gold span: only when the union of retrieval intervals covers all non-whitespace characters of that span is it counted as a hit. Continuous evidence across chunks can be merged; whitespace gaps do not constitute missing coverage. Take span recall ratio within sample, then average across samples with evidence.

Hit Rate represents the proportion of samples with at least one complete gold span covered. Character Recall represents the coverage ratio of gold non-whitespace characters, merging overlapping intervals first to avoid double counting. Missing context, document mismatch, or slice content inconsistency will error; samples without evidence do not participate in these metrics.

ContractNLI: 1,188 of 2,091 items participate in evidence metrics, 903 excluded. CUAD: 599 of 1,020 items participate, 421 excluded.

## Recall@5 Comparison

| Dataset | Retrieval Method | Fixed Length | Clause Boundary |
|---|---|---:|---:|
| contractnli | bm25 | 97.89% | 98.62% |
| contractnli | dense | 96.33% | 97.44% |
| contractnli | hybrid | 98.63% | 98.62% |
| cuad | bm25 | 61.75% | 64.24% |
| cuad | dense | 56.45% | 59.41% |
| cuad | hybrid | 60.47% | 63.77% |

## Configuration and Interpretation Boundaries

Both strategies have maximum chunk budget of 512 words; fixed length overlap=128, clause strategy min_chunk_size=50. Dense uses all-MiniLM-L6-v2, encoding limit 256 subword tokens; long chunks are truncated during encoding. Hybrid weights 0.5 each. JSON simultaneously records k=3/5/10.

Table describes current fixed configuration, did not re-select parameters based on test set. Fixed length and clause strategy have different boundaries and overlap methods; cannot attribute differences to single factor. Larger k may approach full text coverage for short contracts; high evidence recall does not equal answer accuracy or less hallucination.

Current independent metric definition differs from old evaluation/metrics.py's string containment judgment; cannot directly mix the two values in summary. Final report needs to use this unified explicit criterion, or have evaluation lead provide unified new version then recalculate.
