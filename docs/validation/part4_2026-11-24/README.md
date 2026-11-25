# Part 4: Full Record Analysis and Metric Verification (2026-11-24)

Owner: Li Xinru. This page is analysis delivery of completed experiments, not the course report.

Obtained and item-by-item verified AutoDL-exported **18,666 predictions**: two models × two datasets × three methods, total 12 groups; ContractNLI 2,091 items per group, CUAD 1,020 items per group. This iteration did not re-run models. CUAD scope is 10 classes provided by repository, not official full 41 classes.

**Completed:** File integrity, SQLite/JSONL consistency, data and sample IDs, evidence original text and positions, all group summary verification; full error statistics, pre- and post-revision label comparison, 20 traceable cases, and computable metric summary.

**Still incomplete:** Unified independent hallucination assessment of final answers across three methods. Manual annotation count is 0, corresponding hallucination rate remains null; internal initial answer verification cannot substitute for this item.

## File Entry Points

- [Download Complete Analysis Package](COMP7608B-Part4-Full-Trace-Analysis-v2.zip): Byte-identical to locally delivered v2 package.
- [Readable index of 20 cases](CASES.md); [Complete cases with contract and evidence records](analysis/CASES.jsonl).
- [Full analysis JSON](analysis/RESULTS.json); [18,666 item-by-item statistics (gzip)](analysis/row_metrics.jsonl.gz).
- [Complete metrics JSON](metrics-followup/METRICS.json): Contains denominators, internal validation statistics, and alternative metrics.
- [Full analysis HTML](analysis/SUMMARY.html) and [Metrics explanation HTML](metrics-followup/METRICS_REVIEW.html): Download and open in browser.

## Usable Task Metrics

Format failures remain in accuracy denominator and count as errors; Macro-F1 averages over valid classes for each dataset. Initial retrieval Recall@5 only counts samples with gold evidence: ContractNLI 1,188 items, CUAD 599 items; Direct has no retrieval, recorded as N/A. Final prompt evidence coverage also see complete JSON.

|Model|Dataset|Method|Accuracy|Macro-F1|Initial Recall@5|Avg Time (sec)|
|---|---|---|---:|---:|---:|---:|
|llama3|contractnli|Direct|47.01%|0.4491|N/A|1.45|
|llama3|contractnli|RAG|42.52%|0.4271|97.89%|1.85|
|llama3|contractnli|Agent|55.52%|0.5283|97.29%|9.94|
|llama3|cuad|Direct|77.55%|0.7851|N/A|1.28|
|llama3|cuad|RAG|68.63%|0.7346|61.75%|1.29|
|llama3|cuad|Agent|62.06%|0.6834|56.40%|5.89|
|mistral|contractnli|Direct|43.42%|0.3253|N/A|1.93|
|mistral|contractnli|RAG|41.89%|0.2707|97.89%|2.52|
|mistral|contractnli|Agent|37.06%|0.2305|97.11%|10.82|
|mistral|cuad|Direct|67.06%|0.7214|N/A|1.51|
|mistral|cuad|RAG|69.71%|0.7284|61.75%|1.39|
|mistral|cuad|Agent|63.24%|0.6971|57.89%|6.32|

Time from original server 8-parallel execution, does not represent serial throughput. Complete JSON separately records all calls' input and generation tokens; token count is not monetary cost.

## Full Error Statistics

Following events can overlap, cannot sum as mutually exclusive error categories. Label errors include format failures; citation overflow and extraction evidence errors listed separately. Valid CUAD absent empty extraction does not count as abnormal interception.

|Model|Dataset|Method|Label Errors|Format Failures|Citation Overflow|Evidence Check Errors|
|---|---|---|---:|---:|---:|---:|
|llama3|contractnli|Direct|1108|0|0|0|
|llama3|contractnli|RAG|1202|4|151|0|
|llama3|contractnli|Agent|930|40|211|211|
|llama3|cuad|Direct|229|54|0|0|
|llama3|cuad|RAG|320|154|143|0|
|llama3|cuad|Agent|387|215|149|271|
|mistral|contractnli|Direct|1183|341|0|0|
|mistral|contractnli|RAG|1215|373|100|0|
|mistral|contractnli|Agent|1316|514|112|112|
|mistral|cuad|Direct|336|143|0|0|
|mistral|cuad|RAG|309|92|107|0|
|mistral|cuad|Agent|375|192|128|223|

## Agent Pre- and Post-Revision

This only describes label changes within same Agent run, not equivalent to controlled component ablation, nor can be called semantic revision success rate.

|Model|Dataset|Revisions Occurred|Label Wrong→Right|Label Right→Wrong|Initial Correct/Total|Final Correct/Total|
|---|---|---:|---:|---:|---:|---:|
|llama3|contractnli|1555|262|44|943/2091|1161/2091|
|llama3|cuad|498|19|56|670/1020|633/1020|
|mistral|contractnli|1466|9|2|768/2091|775/2091|
|mistral|cuad|662|14|49|680/1020|645/1020|

## Metric Criteria and Old Implementation Issues

1. Old `MetricsCalculator` reads `verification_results`; server records as `verification.claims`. Directly calling old hallucination rate function returns 0 when uncounted; `compute_all_metrics` skips this metric. Cannot write missing measurement as zero hallucination.
2. Semantic verification in records only targets Agent initial answers, and some initial answers did not enter verification due to format or evidence errors. Direct/RAG have no similar records; revised answers have no second semantic verification. Therefore internal statistics cannot be used for three-method final hallucination rate comparison.
3. Verifier format failures are mapped to `not_enough_info`. JSON simultaneously preserves original proportion with failures, failure count, and valid check subset proportion; latter has selection bias, neither is independent ground truth.
4. Old label normalization treats invalid values as neutral; e.g., Mistral / ContractNLI / Agent inflates from correct 37.06% to 42.99%. This page reuses verified task-specific scoring. Shared evaluator not modified this iteration.
5. Word set recall only applies to CUAD's 599 non-empty reference answers, not equivalent to semantic completeness; ContractNLI reference answers all null. Abstention flags actually all correspond to neutral classification, only counted as old rule proxy statistics; zero denominator recorded as N/A.

## Validation Records

- [Source file SHA256 verification](file_integrity.json): Four database/compressed item-by-item files all consistent with export manifest.
- [Call trace verification](analysis/call_trace_checks.json): 37,834 calls, phase order and input/output record checks passed.
- [Analysis independent check](analysis/output_validation.json) and [Metrics independent check](metrics-followup/VALIDATION.json): Contains SQL comparison, missing values, and denominator checks.
- [Delivery file manifest](publication_manifest.json): SHA256 of this upload content and original analysis script hash verification.

## Offline Reproduction

Analysis scripts preserve original files and hashes at that time, do not depend on personal absolute paths; convention is analysis directory and repository directory named `legal-hallucination-ui-review` placed in same parent directory. Use Python 3.12, execute following commands in repository root, replace RUN_ROOT and ZIP paths with your own new directory and received original data package. Output directory must not yet exist.

```bash
python -m pip install -r requirements-retrieval.txt scikit-learn
RUN_ROOT=/path/to/new-trace-reproduction
mkdir -p "$RUN_ROOT/trace-audit/source"
ln -s "$PWD" "$RUN_ROOT/legal-hallucination-ui-review"
cp docs/validation/part4_2026-11-24/analyze_traces.py "$RUN_ROOT/trace-audit/"
cp docs/validation/part4_2026-11-24/calculate_available_metrics.py "$RUN_ROOT/trace-audit/"
cp docs/validation/part4_2026-11-24/file_integrity.json "$RUN_ROOT/trace-audit/"
unzip /path/to/COMP7608B-AutoDL-Full-Traces.zip -d "$RUN_ROOT/trace-audit/source"
python "$RUN_ROOT/trace-audit/analyze_traces.py"
python "$RUN_ROOT/trace-audit/calculate_available_metrics.py"
```

Reproduction uses this branch code; source code blob corresponding to original evaluator diagnostics is `4f0a612fe5fe483266c7f289955a9169a753c78d`. If team members subsequently correct shared evaluator, diagnostic output should save as new version, cannot overwrite this record.

Large original SQLite, original full JSONL, model weights not included in this submission; reproduction requires original ZIP maintained by team. 600 blank blind review materials are only local sampling draft, work allocation not yet determined, draft and model method decoding table not uploaded this iteration. This quantity is not a new task added to any single member.
