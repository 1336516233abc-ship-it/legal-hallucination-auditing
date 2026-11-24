# Legal Hallucination Auditing: Agentic RAG for Contract Analysis

**Course**: COMP7608B - Large Language Models

**Group**: 55
**Team Members**:
- Yin Fengling (3036807375) - Dataset preprocessing & evaluation metrics
- Qin Mengmeng (3036843769) - Retrieval system & baseline implementations
- Wang Zhengyao (3036795675) - Agent components (Query Analyzer, Verifier, Reviser)
- Li Xinru (3036807959) - Experiment orchestration, analysis, reporting & UI

**Submission Date**: November 25, 2026

Formal submission artifacts: [report PDF](docs/final_report/COMP7608B-Group55-Final-Report.pdf) and [submission guide](docs/FINAL_SUBMISSION.md). This extended Markdown report includes the detailed Part 4 tables. The existing packaged PDF and submission fingerprint are maintained separately.

---

## Executive Summary

This project implements and evaluates an Agentic RAG system designed to audit and reduce hallucinations in LLM-generated responses for legal contract analysis. We developed a multi-stage workflow incorporating query analysis, evidence retrieval, answer generation, claim verification, and at most one revision. Our system was evaluated on two legal datasets (ContractNLI and CUAD) using two models (Llama 3 8B and Mistral 7B), producing 18,666 predictions across three methods: Direct Prompting, Standard RAG, and Verified Agentic RAG.

**Key Results**:
- Agentic RAG improved label accuracy by 13.01 percentage points over Standard RAG on Llama 3 × ContractNLI (42.52% → 55.52%)
- Performance gains were model- and task-dependent; three other configurations showed accuracy decreases
- Llama 3 triggered 1,555 ContractNLI and 498 CUAD revisions. These counts measure workflow behavior.
- Retrieval recall@5 reached 97.89% (ContractNLI) and 61.75% (CUAD) using BM25 with fixed-length chunking

---

## 1. Users and Proposal Continuity

### 1.1 User Needs and Core Requirements

**Target Users**: Legal professionals, contract analysts, and compliance officers who need to:
- Verify factual claims in contract analyses against source documents
- Identify unsupported or contradictory statements in LLM-generated answers
- Understand what evidence supports each conclusion

**Core Requirements**:
1. **Evidence-grounded answers**: All generated claims must be traceable to specific contract text
2. **Hallucination detection**: Identify when the model makes unsupported assertions
3. **Automatic revision**: Correct detected hallucinations through targeted re-retrieval
4. **Transparency**: Provide numbered citations linking claims to evidence spans

### 1.2 Scope: Planned vs. Delivered

| Component | Planned | Delivered | Status |
|-----------|---------|-----------|--------|
| **Data Processing** | ContractNLI, CUAD, PrivacyQA | ContractNLI (2,091), CUAD (1,020, 10 categories) | ✅ Core datasets complete; PrivacyQA deferred |
| **Retrieval Systems** | BM25, Dense, Hybrid | All three with two chunking strategies | ✅ Complete with 36-configuration ablation |
| **Baselines** | Direct Prompting, Standard RAG | Both implemented and evaluated | ✅ Complete on full test sets |
| **Agentic Components** | Query Analyzer, Generator, Verifier, Reviser | All four modules integrated | ✅ Complete with workflow traces |
| **Evaluation Metrics** | Accuracy, F1, Evidence Recall, Hallucination Rate | Task metrics delivered; independent hallucination assessment pending | Automated task metrics verified; independent human scoring pending |
| **UI Prototype** | Originally optional; required in updated team scope | Pasted-text proposition checking and clause extraction | Implemented and locally accepted; model failures retained (`ui/app.py`) |
| **Experiments** | Two models, two datasets and component comparisons | 18,666 predictions across 12 configurations | Main comparison complete. Component ablations use a fixed 10-example pilot. |

### 1.3 Changes from Original Proposal

**Dataset Scope**: CUAD evaluation covers 10 representative clause categories (Document Name, Parties, Agreement Date, Effective Date, Expiration Date, Renewal Term, Notice Period to Terminate Renewal, Governing Law, Venue, and Termination for Convenience) rather than all 41 official categories. This allowed deeper analysis of core clause types within resource constraints.

**PrivacyQA**: Deferred as optional cross-domain validation after completing core legal datasets.

**Hallucination Measurement**: The proposal called for claim-level support assessment. The delivered verifier provides internal diagnostics:
- Internal verification only checks initial answers, not final revised outputs
- Verification logic itself can have false positives/negatives
- Independent human assessment is required for reliable hallucination rates across all three methods

**Methodology Refinement**: Added component ablation studies (No Verification, No Revision) to compare the behavior of related Agent components, beyond the originally planned method comparisons.

---

## 2. Design and Implementation

### 2.1 System Workflow

Our Agentic RAG system follows a five-stage pipeline:

```
Input Query + Contract
     ↓
[1] Query Analysis → refined query, query type, optimal top-k
     ↓
[2] Evidence Retrieval → ranked chunks with scores
     ↓
[3] Answer Generation → initial answer with citations
     ↓
[4] Claim Verification → supported / contradicted / not_enough_info
     ↓
[5] Answer Revision (if needed) → re-retrieve + revise, max 1 iteration
     ↓
Final Answer + Evidence + Verification Status
```

**Stage 1 - Query Analysis** (`agent/query_analyzer/analyzer.py`):
- Classifies query type (termination, liability, duration, etc.) using keyword matching
- Uses LLM query refinement in the full Agent runs: the trace audit records 3,111 query-refinement calls per model
- Sets top-k=7 for rights/obligations and liability, 3 for duration/timeline, 10 for exceptions, and 5 otherwise; the configured chunker remains fixed for the run

**Stage 2 - Evidence Retrieval** (`retrieval/baseline_rag.py`):
- Chunks contract using fixed-length (512 whitespace-separated words, 128-word overlap) or clause-level segmentation
- Indexes chunks with BM25, Dense (all-MiniLM-L6-v2), or Hybrid retrieval
- Returns top-k chunks with character-level provenance (`start`, `end`, `chunk_id`)

**Stage 3 - Answer Generation** (`agent/generator.py`):
- Formats evidence with numbered citations [1], [2], ...
- Prompts model to generate label + reasoning + answer with explicit citations
- Passes task output spans to the verifier; if no claims are supplied, the verifier extracts them from the answer (CUAD absent receives an explicit absence claim)
- ContractNLI uses entailment/contradiction/neutral; CUAD uses present/absent with extractive spans

**Stage 4 - Claim Verification** (`agent/verifier/verifier.py`):
- Decomposes answer into atomic claims
- For each claim, prompts model: "Is this claim supported by the evidence?"
- Aggregates claim-level verdicts into supported, contradicted or partially_supported; individual unsupported claims use not_enough_info
- Sets `hallucination_detected = True` if any claim is contradicted or lacks support

**Stage 5 - Answer Revision** (`agent/reviser/reviser.py`):
- Triggered only when verification detects hallucination
- In the task-aware path, contradicted claims trigger correction using the initial prompt evidence; otherwise unsupported claims trigger refined-query re-retrieval (top-3)
- Prompts the model to reassess the task label and answer using the selected evidence
- Records task revision type as correction or reretrieval; a revision is absent when not triggered
- Limited to one revision to prevent infinite loops

### 2.2 Autonomous Decisions

The system makes the following autonomous decisions without user intervention:

1. **Retrieval Parameter Selection**: Query analyzer adjusts top-k based on query complexity (3–10 chunks)
2. **Hallucination Threshold**: Verification flags output if any claim is unsupported/contradicted
3. **Revision Strategy**:
   - If grounding check fails (invalid citations), revision is skipped and final answer is withheld
   - If semantic check fails, system attempts one re-retrieval and revision
   - After revision, no second verification is performed (implementation limitation)
4. **Format Validation**: Outputs that fail to parse or contain out-of-bound citations are explicitly marked as invalid and excluded from user display

### 2.3 Models and Prompts

**Models**:
- **Llama 3 8B Instruct** (revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`)
- **Mistral 7B Instruct v0.2** (revision `63a8b081895390a26e140280378bc85ec8bce07a`)
- Inference: vLLM 0.10.2, BF16, temperature 0, seed 42, max 4,096 input tokens, max 512 output tokens per call

**Prompt Design** (schematic summaries, not verbatim source prompts; implementations in `agent/generator.py`, `agent/verifier/verifier.py`):

Generation prompt template (ContractNLI):
```
You are analyzing a contract to answer a legal question.

Evidence:
[1] {chunk_1_text}
[2] {chunk_2_text}
...

Question: {query}

Provide:
1. Label: entailment | contradiction | neutral
2. Reasoning: ...
3. Answer: ... Use citations like [1], [2].
```

Verification prompt template:
```
Evidence:
{numbered_evidence}

Claim: "{atomic_claim}"

Question: Is this claim fully supported, contradicted, or not addressed by the evidence?
Answer: supported | contradicted | not_enough_info
Explanation: ...
```

Revision prompt template:
```
Original answer: {initial_answer}
Problem: {unsupported_claims}
New evidence: {re_retrieved_chunks}

Revise the answer to be fully supported by evidence. Use citations.
```

### 2.4 Data and State Management

**Datasets**:
- **ContractNLI**: 2,091 test examples, 1,188 with gold evidence spans
- **CUAD**: 1,020 test examples (10 clause types), 599 with non-empty gold answers
- Data format: `{id, query, context, label, evidence_spans, metadata}`
- Files: `data/processed/contractnli_test.json`, `data/processed/cuad_test.json`
- SHA256 hashes recorded in experimental manifests for reproducibility

**State Persistence**:
- Experiments store predictions in SQLite with `(dataset, method, example_id)` as primary key
- Each record contains: query, final answer, label, verification status, latency, token counts, full generation trace
- Checkpoint resume: the runner compares stored settings (including source/data hashes, sample IDs and model configuration), then skips completed IDs; exported database hashes are separately checked during the trace audit
- Final outputs: JSON summaries + per-row JSONL + complete call traces (37,834 LLM calls across 18,666 predictions)

**Trace and case sources**: the AutoDL export contains complete calls; `docs/validation/part4_2026-11-24/analysis/CASES.jsonl` contains 20 selected case records, not the entire prediction set.
Each prediction records:
- Input: original query, contract text, metadata
- Retrieval: chunks retrieved, scores, character spans
- Generation: prompt, raw response, parsed fields, tokens
- Verification: per-claim checks, reasoning, overall status
- Revision: new retrieval, revised prompt, revised answer, label changes
- Final: label, answer, citations, hallucination flag, latency

### 2.5 Key Design Trade-offs

| Decision | Chosen Approach | Alternative | Justification |
|----------|----------------|-------------|---------------|
| Verification granularity | Claim-level checking | Document-level NLI | Claim-level provides specific feedback for targeted revision |
| Revision limit | Max 1 iteration | Multiple iterations until convergence | Bounds cost and preserves a finite trace; multiple revision limits were not experimentally compared |
| Verification scope | Only initial answer | Initial + revised answers | Resource constraint; noted as limitation for future work |
| Citation format | Numbered inline | Footnotes or no citations | Inline citations enable automated grounding checks |
| Retrieval index | Per-document isolation | Global cross-document index | Prevents cross-contamination in batch processing |
| Chunking | Fixed 512 words | Semantic/sentence boundaries | Fixed-length ensures consistent budget; clause-level tested as ablation |

---

## 3. Reliability and Recovery

### 3.1 Tested Failure Situations

**Malformed task output.** The parser in `retrieval/common.py` preserves the failed raw output, sets an invalid label, and records a parse error. The Agent withholds an unusable final answer. Invalid predictions remain in the accuracy denominator. This includes ContractNLI outputs with illegal labels and malformed CUAD JSON.

**Out-of-range citation or nonverbatim extract.** The check in `agent/agentic_rag.py` uses the evidence that actually reached the generator. The UI withholds the answer, displays the reason, and keeps the trace. Of the RAG and Agent groups, ContractNLI had 100–211 rows with out-of-range citations per group (4.8–10.1%), and CUAD had 107–149 (10.5–14.6%). Citation-range checks can pass when the cited passage is semantically incorrect.

**Changed conditions.** The live interface cleared results when the sample contract changed from 30 to 60 days. The next request used the new contract and returned a contradiction for the unchanged 30-day proposition. The verification trace shows an attempted revision. See the UI acceptance records and full-trace cases under `docs/validation/`.

### 3.2 Recovery Boundaries

The workflow allows at most one revision, enforced by control flow. The per-call token budget truncates prompt evidence. Model exceptions in the interactive service give an actionable status and no automatic retry. The experiment checkpoint binds settings and skips already completed keys when resuming with the same manifest.

---

### 3.3 UI Acceptance Scope

The original Apple Silicon/MLX acceptance used three constructed inputs per model: a 30-day notice proposition, a changed 60-day contract, and liability-clause extraction. Llama 3 passed all three; Mistral passed two and returned an invalid label for the changed-contract case, which was blocked. Empty inputs made no model calls. See [original UI acceptance](docs/validation/ui_2026-11-23/README.md).

The later [submission acceptance](docs/validation/final_submission_acceptance.json) records extraction into a fresh Python 3.11 environment on the same Mac, 188 passing tests, and the same Mistral success/failure pattern. The current regression rerun also passed all 188 tests. This does not establish acceptance on another machine or real CUDA/CPU/MPS inference. Automated tests and constructed demo cases are not independent human support labels.

## 4. Non-functional Requirements

### 4.1 Latency

**Target**: Complete answer within 30 seconds for interactive use; support batch processing for larger evaluations.

**Measured Performance** (Llama 3 on full test sets):

| Method | ContractNLI avg | CUAD avg | Analysis |
|--------|-----------------|----------|----------|
| Direct Prompting | 1.45s | 1.28s | Single LLM call; baseline |
| Standard RAG | 1.85s | 1.29s | +28% (NLI) due to retrieval + longer prompts |
| Agentic RAG | 9.94s | 5.89s | 5.4–6.8× slower than Direct; driven by multi-call verification |

Agent runs invoke generation, claim verification and revision when triggered. The reported means are from eight-worker vLLM runs and do not estimate cold-start latency or interactive service throughput. On the previously observed local Gradio deployment, two synthetic contracts demonstrated the multi-call behavior. No per-stage latency breakdown is claimed.

### 4.2 Token Usage and Compute Cost

| Full-test method | Predictions | Model calls | Input tokens | Generated tokens | Total tokens |
|---|---:|---:|---:|---:|---:|
| Direct Prompting | 6,222 | 6,222 | 16,726,643 | 347,613 | 17,074,256 |
| Standard RAG | 6,222 | 6,222 | 17,561,928 | 409,452 | 17,971,380 |
| Verified Agentic RAG | 6,222 | 25,390 | 50,416,305 | 2,050,494 | 52,466,799 |
| **Total** | **18,666** | **37,834** | **84,704,876** | **2,807,559** | **87,512,435** |

Totals sum input and generated tokens from every recorded model call, including query rewrites, verification and revision. RAG consumed 1.05 times the Direct baseline total. Agent consumed 2.92 times the Standard RAG total. The 60-output component pilot is reported separately in `docs/validation/part4_2026-11-22/RESULTS.md`.

The experiments used rented AutoDL compute with self-hosted inference. Token totals describe workload. Actual spend depends on the AutoDL billing record and is not estimated here.

### 4.3 Privacy and Data Handling

**Data Sources**:
- ContractNLI: Academic dataset, public contracts from EDGAR filings
- CUAD: Atticus Project, public commercial contracts
- The evaluation used public datasets and the demonstration used synthetic contracts.

**Measures**:
- Main inference used team-run model instances. The shared Gradio demo is a separate deployment.
- Model inference on team-controlled infrastructure (AutoDL, local machines)
- Synthetic examples in UI demo marked as "constructed input, not real case data"

**Limitations**:
- System not designed for production deployment with real client contracts
- The release contains public benchmark material and synthetic examples.
- The local UI binds to loopback by default. The shared demo uses separate authentication.

### 4.4 Reproducibility

- Full-run settings and export manifests record model revisions, data hashes and execution settings; the local pilot additionally records weight and source-file SHA256 values. Appendix B lists hashes recalculated from the tracked files.
- The local pilot uses temperature 0 and seed 42. The generic runner records the resolved model revision but does not automatically pin every future run to the archived checkpoint; use the documented pilot entry point and exact revision for that cohort.
- `requirements.txt` mostly specifies minimum versions, so it is **not a frozen environment**. Recorded AutoDL, local MLX and retrieval environments must be distinguished. The validated UI dependency is `gradio==6.32.0` in `requirements-ui.txt`.
- All 37,834 full-run calls were checked in the supplied export. Public analysis files contain derived statistics and selected cases; the course submission package includes the exported JSONL traces and manifest; obtain that package or the separately held raw archive from the team for a full trace-level rerun.
- The independent 36-configuration retrieval rerun differed from the archived metrics by at most `5.55e-16`. No repeated-run study establishes identical generation or one-token repeatability for the component pilot.

Commands, settings and limitations are in [Part 4 handoff](docs/PART4_EXPERIMENTS.md) and [trace-analysis reproduction](docs/validation/part4_2026-11-24/README.md). Fixed seeds reduce variability; hardware, dependency and inference-backend changes can still alter outputs.

---

## 5. Evaluation

### 5.1 Research Questions

**RQ1**: Does RAG reduce hallucinations in legal question answering compared to direct prompting?

**RQ2**: Does agentic verification and revision improve upon standard RAG?

**RQ3**: What types of contract clauses and reasoning are most prone to errors?

### 5.2 Evaluation Design

**Datasets and Metrics**:
| Dataset | Task | Size | Gold Evidence | Metrics |
|---------|------|------|---------------|---------|
| ContractNLI | NLI (entailment/contradiction/neutral) | 2,091 | 1,188 (56.8%) | Label Accuracy, Macro-F1, Evidence Recall@k |
| CUAD | Clause extraction (present/absent + spans) | 1,020 | 599 (58.7%) | Label Accuracy, Macro-F1, Evidence Recall@k, Extractive Match |

**Baselines**:
1. **Direct Prompting**: Full contract (truncated to 4,096 tokens) + query → answer
2. **Standard RAG**: BM25 retrieval (top-5, fixed 512-word chunks) → generation
3. **Verified Agentic RAG**: Query analysis → retrieval → generation → verification → revision (if needed)

**Component Ablations** (fixed 10-example pilot per model):
- **Full**: Verification + Revision enabled
- **No Verification**: Both verification and revision disabled
- **No Revision**: Verification enabled, but revision skipped

**Observed component-pilot results (MLX; 60 outputs total)**:

| Model | Dataset | Variant | Correct | Format pass | Macro-F1 | Avg seconds | Generated tokens | Revision attempts |
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

Each model uses the same 6 ContractNLI and 4 CUAD questions in all three variants. Only Llama 3 / ContractNLI changes label accuracy in this pilot (5/6 to 6/6). The cohort is too small to infer full-test effects. Generated-token totals include all calls but exclude input tokens; first-group timing can include initial weight loading. No Verification disables both checking and its dependent revision. Source: [pilot results and manifests](docs/validation/part4_2026-11-22/RESULTS.md). Old `priority4` ablations with dataset/label-protocol mismatches are excluded from this report's comparisons.

**Models**: Llama 3 8B Instruct, Mistral 7B Instruct v0.2

**Evaluation Protocol**:
- Format failures counted as errors (included in accuracy denominator)
- Label accuracy = correct predictions / total predictions
- Macro-F1 = average F1 across valid task labels (3-way for NLI, 2-way for CUAD)
- Evidence Recall@k = mean per-example proportion of gold spans fully covered by the retrieved-chunk union, restricted to examples with gold evidence (1,188 NLI; 599 CUAD). Direct has no retrieval and is N/A.
- The independently defined final-answer hallucination rate remains null. Internal Agent initial-answer verifier judgments and lexical-overlap/legacy-abstention proxies in `METRICS.json` are separately labeled; none is a substitute for human semantic evaluation.

### 5.3 Main Results

#### 5.3.1 Label Accuracy and F1 (Full Test Sets, 18,666 Predictions)

| Model | Dataset | Method | Accuracy | Macro-F1 | Format Pass Rate |
|-------|---------|--------|----------|----------|------------------|
| **Llama 3** | **ContractNLI** | Direct | 47.01% | 0.4491 | 100.00% |
| | | RAG | 42.52% | 0.4271 | 99.81% |
| | | **Agent** | **55.52%** | **0.5283** | 98.09% |
| Llama 3 | CUAD | Direct | 77.55% | 0.7851 | 94.71% |
| | | RAG | 68.63% | 0.7346 | 84.90% |
| | | Agent | 62.06% | 0.6834 | 78.92% |
| **Mistral** | **ContractNLI** | Direct | 43.42% | 0.3253 | 83.69% |
| | | RAG | 41.89% | 0.2707 | 82.16% |
| | | Agent | 37.06% | 0.2305 | 75.42% |
| Mistral | CUAD | Direct | 67.06% | 0.7214 | 85.98% |
| | | RAG | 69.71% | 0.7284 | 90.98% |
| | | Agent | 63.24% | 0.6971 | 81.18% |

**Key Findings**:

1. **RQ1 remains unanswered for hallucination reduction.** Independent final-answer support labels have not been collected. RAG versus Direct label accuracy changes range from -8.92 to +2.65 percentage points; these task scores cannot establish a change in hallucination rate. RAG retrieval Recall@5 is 97.89% on ContractNLI and 61.75% on CUAD, but there is no corresponding Direct retrieval score to compare.
2. **RQ2 shows a conditional task-accuracy benefit.** Agent versus RAG improves Llama 3 / ContractNLI by 13.01 percentage points and decreases accuracy in the other three model/dataset combinations. No significance test or general benefit for complex legal reasoning is established. Within the Llama 3 / ContractNLI Agent run, revision changes 262 labels from wrong to right and 44 from right to wrong (net +218); this within-run comparison is different from the RAG-versus-Agent comparison.
3. **RQ3 has descriptive evidence, not an independently adjudicated error taxonomy.** CUAD retrieval coverage is lower in these settings, and selected cases show that retrieved gold evidence can coexist with wrong labels. Format errors overlap with label and citation errors. Across all six groups per model, format pass rates range from **78.92–100.00% for Llama 3** and **75.42–90.98% for Mistral**; the stronger 95–100% claim does not hold for Llama 3 on CUAD. Per-clause counts are available in `analysis/RESULTS.json`, but do not on their own identify causal reasoning failures.

#### 5.3.2 Evidence Retrieval Performance (36 Configurations)

**ContractNLI Recall@5** (1,188 examples with gold evidence):

| Retriever | Chunking | Recall@3 | Recall@5 | Recall@10 |
|-----------|----------|----------|----------|-----------|
| BM25 | Clause-level | 93.76% | 98.62% | 100.00% |
| BM25 | Fixed-length | 90.81% | 97.89% | 99.97% |
| Dense | Clause-level | 91.13% | 97.44% | 99.97% |
| Dense | Fixed-length | 89.48% | 96.33% | 99.85% |
| Hybrid | Clause-level | 94.80% | 98.62% | 99.97% |
| Hybrid | Fixed-length | 92.20% | **98.63%** | 99.97% |

**CUAD Recall@5** (599 examples with gold evidence):

| Retriever | Chunking | Recall@3 | Recall@5 | Recall@10 |
|-----------|----------|----------|----------|-----------|
| BM25 | Clause-level | 51.83% | **64.24%** | 82.26% |
| BM25 | Fixed-length | 51.12% | 61.75% | 82.31% |
| Dense | Clause-level | 45.72% | 59.41% | 81.09% |
| Dense | Fixed-length | 43.11% | 56.45% | 78.02% |
| Hybrid | Clause-level | 51.65% | 63.77% | 82.86% |
| Hybrid | Fixed-length | 48.98% | 60.47% | 81.12% |

**Analysis**:
- At k=5, Hybrid/fixed-length has the highest ContractNLI score (98.63%), narrowly above BM25/clause-level and Hybrid/clause-level (98.62%). BM25/clause-level leads CUAD at k=5 (64.24%). The best configuration depends on dataset and k.
- ContractNLI coverage is high under these span-scoring rules; it does not eliminate generation errors. CUAD leaves more gold spans uncovered.
- Dense-only retrieval is below BM25 in the matched configurations here. A domain mismatch is a possible explanation, not one isolated by this experiment.

#### 5.3.3 Agent Revision Behavior (Llama 3, Full Test Sets)

| Dataset | Revisions Triggered | Label Wrong→Right | Label Right→Wrong | Initial Correct | Final Correct | Net Gain |
|---------|---------------------|-------------------|-------------------|-----------------|---------------|----------|
| ContractNLI | 1,555 / 2,091 (74.4%) | 262 | 44 | 943 / 2,091 (45.1%) | 1,161 / 2,091 (55.5%) | **+218** |
| CUAD | 498 / 1,020 (48.8%) | 19 | 56 | 670 / 1,020 (65.7%) | 633 / 1,020 (62.1%) | **-37** |

**Observations**:
- Llama 3 revision attempt rates are 48.8–74.4%; they measure workflow behavior, not the accuracy of hallucination detection
- ContractNLI: 262 wrong-to-right label changes vs. 44 right-to-wrong changes → net positive task-label change
- CUAD: 19 wrong-to-right label changes vs. 56 right-to-wrong changes → net negative task-label change
- Revision does not guarantee improvement; depends on quality of re-retrieved evidence and model's ability to correct itself

### 5.4 Failure Cases and Limitations

**Case Analysis** (20 documented cases in `docs/validation/part4_2026-11-24/CASES.md`):

| Failure Mode | Description | Example Case # | Impact |
|--------------|-------------|----------------|--------|
| **Evidence coverage ≠ correct answer** | Gold evidence is in the Direct model input but its label is wrong | Case 1 | Input evidence coverage does not guarantee task correctness |
| **Format failure** | Unparsable JSON or missing required fields | Case 3, 6 | Must retain failed outputs in denominator |
| **Revision introduces errors** | Revision changes correct answer to wrong | Case 4 | Label accuracy: contradiction→entailment (wrong) |
| **Revision corrects errors** | Revision changes wrong answer to correct | Case 5 | Label accuracy: contradiction→entailment (correct) |
| **Citation out of range** | Model cites evidence indices that don't exist | Case 2 | Grounding check failures |

**Note**: Cases are numbered 1-20 in the source document, selected by deterministic flags (e.g., `gold_covered_but_label_wrong`, `revision_correct_to_wrong`, `format_failure`). These are illustrative examples, not a random sample for error rate estimation.

**Documented Limitation**:
- Revised answers are not re-verified (implementation gap)
- Some revisions introduced new errors: Case 4 shows revision changing correct to wrong answer
- Internal verifier status cannot substitute for independent human hallucination assessment
- Cases selected by deterministic flags, not random sampling

### 5.5 Comparison with Baselines

Direct Prompting, Standard RAG and Verified Agentic RAG provide an internal comparison on the same processed datasets and model snapshots. Direct receives a token-budgeted contract; RAG receives retrieved evidence; Agent additionally uses query analysis, verification and conditional revision. Their differences are not a controlled ablation of verification alone. External hallucination-detection baselines were not evaluated, which limits comparison with prior work.

### 5.6 Output Variability and Reproducibility

Main results use the recorded AutoDL/vLLM runs; the component pilot uses MLX on Apple Silicon. Temperature 0 and fixed seed 42 do not guarantee bit-identical generations across repeated or cross-platform runs. This delivery does not measure repeated-run variance. Model snapshots are recorded (Llama 3 `8afb486c…`, Mistral `63a8b081…`); pilot weight hashes and environment details are in the per-model manifests. Interpret pilot results within that cohort and backend.

---

## 6. Contributions and Attribution

### 6.1 Individual Contributions

**Yin Fengling (3036807375)** - Dataset Preprocessing & Evaluation Metrics
- Implemented `data/contractnli_processor.py` and `data/cuad_processor.py`
- Processed 2,091 ContractNLI and 1,020 CUAD test examples with unified schema
- Developed `evaluation/metrics.py` for task labels and retrieval evidence. The final audit reports lexical answer overlap separately from semantic completeness.
- Validated dataset integrity and gold evidence spans
- Designed metric calculation protocol for format failures and missing labels

**Qin Mengmeng (3036843769)** - Retrieval System & Baselines
- Implemented chunking strategies in `retrieval/chunking.py` (fixed-length and clause-level)
- Built BM25, Dense, and Hybrid retrievers in `retrieval/retrievers/`
- Developed Direct Prompting baseline (`retrieval/baseline_direct.py`)
- Developed Standard RAG baseline (`retrieval/baseline_rag.py`)
- Executed 36-configuration retrieval ablation study
- Delivered full test set results for Mistral and Llama 3 on AutoDL (12,444 baseline predictions)

**Wang Zhengyao (3036795675)** - Agent Components
- Implemented Query Analyzer (`agent/query_analyzer/analyzer.py`)
- Implemented Answer Generator (`agent/generator.py`)
- Implemented Evidence Verifier (`agent/verifier/verifier.py`)
- Implemented Answer Reviser (`agent/reviser/reviser.py`)
- Integrated full Agentic RAG workflow (`agent/agentic_rag.py`)
- Conducted component unit tests (Priority 1–2) and integration tests (Priority 3)
- Executed full-scale Agentic RAG experiments (6,222 predictions on ContractNLI + CUAD)

**Li Xinru (3036807959)** - Experiment Orchestration, Analysis, Reporting & UI
- Developed and repaired experiment orchestration, checkpoint resume, task-aware scoring and component-ablation integration, with AI assistance disclosed below.
- Independently reran 36 retrieval configurations and ran the paired two-model component pilot (60 outputs).
- Audited the received AutoDL export of 18,666 predictions: data hashes, sample IDs, summary/confusion counts, evidence provenance and 37,834 recorded model calls. This local analysis did not regenerate the full server experiment.
- Produced full-run error summaries, before/after label transitions and 20 automatically selected traceable cases; these are not completed human support annotations.
- Implemented the formal Gradio UI, contract isolation, failure handling, all-call accounting and local acceptance checks.
- Prepared experiment handoffs, result summaries, demo instructions and corrections to report evidence and interpretation with disclosed AI assistance.

### 6.2 Code and Library Attribution

**External Libraries**:
- Hugging Face Transformers and vLLM: model loading/tokenization and server inference.
- rank-bm25 and sentence-transformers: lexical and dense retrieval.
- scikit-learn: task-label metrics and confusion matrices.
- Gradio: UI; the validated dependency is `gradio==6.32.0`.
- MLX/MLX-LM: local Apple Silicon pilot and demo.

Versions differ between the archived server environment and local pilot. Appendix C points to the recorded environments; root requirements are not a lockfile.

**Dataset Sources**:
- ContractNLI: Koreeda & Manning (2021), [GitHub](https://github.com/stanfordnlp/contract-nli)
- CUAD: Hendrycks et al. (2021), [Atticus Project](https://www.atticusprojectai.org/cuad)
- Both datasets used under academic research license

**Reused Code**:
- BM25 wrapper adapted from rank-bm25 documentation examples (~30 lines)
- Dense retrieval setup adapted from sentence-transformers examples (~40 lines)
- All other code (retrieval integration, baselines, agent workflow, evaluation, experiments) is original team implementation

**Models**:
- Meta Llama 3 8B Instruct: Meta AI, used under Llama 3 Community License
- Mistral 7B Instruct v0.2: Mistral AI, Apache 2.0 license

### 6.3 Unfinished and Mocked Functionality

**Not Implemented**:
1. **Independent Hallucination Assessment**:
   - Planned human evaluation protocol defined in `docs/ANSWER_SUPPORT_PROTOCOL.md`
   - The repository pilot material contains 60 unannotated outputs. A separate 600-answer full-run sampling draft was prepared locally and was not uploaded; neither represents completed human scoring
   - Human annotations remained incomplete at final packaging
   - Current metrics rely on task labels (accuracy, F1); hallucination rate remains unmeasured across methods

2. **Second-round Verification**:
   - Revised answers are not re-verified
   - The final answer retains structural and citation checks. Independent semantic re-verification is pending.
   - Noted as implementation limitation affecting result interpretation

3. **Full 41-category CUAD Evaluation**:
   - Evaluated 10 representative categories only
   - Remaining 31 categories deferred due to scope constraints

4. **PrivacyQA Cross-domain Validation**:
   - Code implemented but experiments not run
   - Deferred as optional stretch goal

**Mocked / Placeholder**:
- None. All reported results come from real model outputs. Synthetic examples in demo explicitly marked as "constructed input."

---

## 7. AI Assistance Declaration

The team draft records Claude and GitHub Copilot for development, debugging, documentation and report assistance. Codex was used to develop and validate retrieval and baselines, audit the experiment traces, repair the legacy evaluator, edit this report, build the submission package and verify its runtime. The model outputs in the evaluation came from the named Llama 3 and Mistral runs. Automated tests and synthetic user inputs are documented separately from independent annotation. See `AI_USE.md` for prompt excerpts and coverage limitations. Independent human evaluation is pending. Codex also assisted Part 4 orchestration, the component pilot, trace analysis and UI implementation/acceptance; this includes substantive code and analysis assistance.

## 8. Conclusions

### 8.1 Summary of Findings

1. **Agentic RAG benefits are conditional**, not universal:
   - Observed improvement (Llama 3 × ContractNLI: +13.01 percentage points)
   - Degradation in other settings (Mistral × ContractNLI: -4.8 pp; both models on CUAD: -6 to -6.6 pp)
   - Success depends on model quality, task complexity, and verifier reliability

2. **Retrieval quality varies by task**:
   - ContractNLI: 98% recall at k=5 (retrieval highly effective)
   - CUAD: 62–64% recall at k=5 (retrieval leaves room for improvement)
   - High retrieval recall does not guarantee high generation accuracy

3. **Revision is high-frequency but mixed quality**:
   - 48.8–74.4% of Llama 3 predictions triggered revision across the two tasks
   - Net positive on ContractNLI (+218 correct), net negative on CUAD (-37 correct)
   - Revision can introduce new errors (citation drift, format failures)

4. **Format compliance is a major error source**:
   - Mistral: 9.02–24.58% format failure across the six groups
   - Llama 3: 0–21.08% format failure across the six groups
   - Better prompting or constrained decoding needed for production

### 8.2 Limitations

1. **Hallucination rate unmeasured**: Task accuracy reported; independent answer support assessment incomplete
2. **Single revision limit**: System stops after one revision attempt; may miss opportunities for further correction
3. **No re-verification**: Revised answers receive structural and citation checks; semantic re-verification is pending
4. **Platform variance**: Results from vLLM (CUDA); MLX (Apple Silicon) gives different outputs
5. **Limited model coverage**: Only 7–8B models tested; larger models may behave differently
6. **Dataset scope**: CUAD covers 10/41 categories; PrivacyQA untested

### 8.3 Future Work

1. **Complete independent hallucination assessment** with human annotators across all three methods
2. **Implement second-round verification** after revision to catch newly introduced errors
3. **Test larger models** (Llama 3 70B, GPT-4) to assess scaling behavior
4. **Optimize verification efficiency** via batched claim checking or trained verifier models
5. **Extend to full CUAD** (41 categories) and cross-domain datasets (PrivacyQA)
6. **Develop constrained decoding** to reduce format failures in production
7. **Compare with external baselines** (SelfCheckGPT, RARR, ChainPoll) on hallucination detection

### 8.4 Lessons Learned

1. **Verification has a workload cost**: Agent used 2.92 times the Standard RAG token total; the aggregate does not isolate verification from the other Agent stages.
2. **More components ≠ better performance**: Agent underperformed RAG in 3/4 configurations; complexity must be justified
3. **Reproducibility requires discipline**: Pinning model revisions, logging SHA256 hashes, and preserving full traces was essential for validation
4. **Format failures remain material**: Format, citation and label-error events overlap; the counts do not establish an exclusive ranking of semantic error causes
5. **Task-dependent tuning needed**: One-size-fits-all approach failed; NLI and QA tasks require different optimization

---

## Appendices

### A. Repository and Evidence Map

| Path | Role |
|---|---|
| `data/processed/` | Processed test sets and scope documentation |
| `retrieval/` | Chunking, retrievers and Direct/RAG baselines |
| `agent/` | Query analysis, generation, verification and revision |
| `experiments/run_experiments.py` | Task-aware experiment orchestration and resume |
| `experiments/run_agent_ablation_pilot.py` | Fixed-cohort local component pilot |
| `ui/app.py`, `ui/service.py`, `ui/backends.py` | Formal UI, request isolation and model adapters |
| `docs/validation/part4_2026-11-22/` | 36 retrieval configurations and 60 component outputs |
| `docs/validation/part4_2026-11-24/` | Full-trace analysis, 20 cases and available metrics |
| `docs/validation/ui_2026-11-23/` | UI test and real-model acceptance evidence |
| `docs/PART4_EXPERIMENTS.md`, `docs/UI_HANDOFF.md` | Reproduction and operation instructions |
| `REPORT_DATA_VERIFICATION.md` | Scope, sources and limitations of this report correction |
| `docs/final_report/`, `docs/FINAL_SUBMISSION.md`, `AI_USE.md` | Published report PDF/source, package instructions and team AI disclosure |

Historical `experiments/results/priority4/` material is retained for provenance; its incompatible old component ablations are not used as valid comparisons here.

### B. Key Files for Reproducibility

SHA256 values below are recalculated from the tracked files, not placeholders. The publication manifest covers its own delivery files and is not a manifest of every repository file.

| File | SHA256 |
|---|---|
| `data/processed/contractnli_test.json` | `ba6aa63b66a354028703c3e3574a902f55cc751dc37cc1d9d2f2c11bd39bb116` |
| `data/processed/cuad_test.json` | `8b95cd181abd935c2a7d32238764c4a6d0a7818f161eea9dbc6f833e446f5104` |
| `docs/validation/llama3_autodl_full_summary.json` | `f914d023238bb0915e09ead106866056e0de10037cf2750395225a2320262338` |
| `docs/validation/mistral_autodl_full_summary.json` | `6652304a080416e84f7d58eb91d2a07b59cae464753f81743420bf28a3cf04e7` |
| `docs/validation/part4_2026-11-24/analysis/RESULTS.json` | `708bd4ba35fb314102aa62b361102dfe1e267f065961cdbbee95d87e29988ce0` |

Verify with `shasum -a 256 <file>` or Python `hashlib.sha256(Path(file).read_bytes()).hexdigest()`.

### C. Experimental Environment

**Main Experiments (AutoDL - 18,666 predictions)**:
- GPU: NVIDIA RTX 6000D (1 per model)
- OS: Linux x86-64, CUDA 12.8
- Python: 3.12.3
- vLLM: 0.10.2, Transformers: 4.56.2, PyTorch: 2.8.0+cu128
- Additional packages: huggingface-hub 0.34.4, rank-bm25 0.2.2, numpy 2.2.6
- Parallelism: 8 concurrent requests, BF16 precision, temperature 0, seed 42

**Component Ablation Pilot (Local - 60 predictions)**:
- Hardware: Apple M5, 24 GiB RAM
- OS: macOS 26.6.2 (Darwin kernel)
- Python: 3.12.14
- MLX: 0.32.4, MLX-LM: 0.32.2, Transformers: 5.24.0
- Metal GPU, sequential inference, temperature 0, seed 42

**Retrieval Ablation (Local - 36 configurations)**:
- Python: 3.12.14
- sentence-transformers: 6.3.0, torch: 2.15.0, transformers: 5.24.0, rank-bm25: 0.2.2
- CPU execution with HF_HUB_OFFLINE=1

### D. Contact Information

For questions regarding this project, contact:
- **Li Xinru** (3036807959): Experiments, analysis, reporting
- **Qin Mengmeng** (3036843769): Retrieval, baselines
- **Wang Zhengyao** (3036795675): Agent components
- **Yin Fengling** (3036807375): Data, evaluation

**Code Repository**: [GitHub](https://github.com/1336516233abc-ship-it/legal-hallucination-auditing)

---

**End of Report**
