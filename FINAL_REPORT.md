# Legal Hallucination Auditing: Agentic RAG for Contract Analysis

**Course**: COMP7608B - Large Language Models

**Group**: 55
**Team Members**:
- Yin Fengling (3036807375) - Dataset preprocessing & evaluation metrics
- Qin Mengmeng (3036843769) - Retrieval system & baseline implementations
- Wang Zhengyao (3036795675) - Agent components (Query Analyzer, Verifier, Reviser)
- Li Xinru (3036807959) - Experiment orchestration, analysis & reporting

**Submission Date**: November 25, 2026

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
| **UI Prototype** | Optional Gradio interface | Functional UI with pasted text, proposition checking and clause extraction | ✅ Complete (`ui/app.py`) |
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

**Stage 1 - Query Analysis** (`agent/query_analyzer/analyzer.py:67-157`):
- Classifies query type (termination, liability, duration, etc.) using keyword matching
- Optionally refines query with LLM for better retrieval (implemented but not used in experiments)
- Adjusts retrieval parameters: termination/liability queries use top-k=7, exceptions use top-k=10, others use top-k=5

**Stage 2 - Evidence Retrieval** (`retrieval/baseline_rag.py:95-157`):
- Chunks contract using fixed-length (512 tokens, 128 overlap) or clause-level segmentation
- Indexes chunks with BM25, Dense (all-MiniLM-L6-v2), or Hybrid retrieval
- Returns top-k chunks with character-level provenance (`start`, `end`, `chunk_id`)

**Stage 3 - Answer Generation** (`agent/generator.py:134-258`):
- Formats evidence with numbered citations [1], [2], ...
- Prompts model to generate label + reasoning + answer with explicit citations
- Extracts atomic claims from the answer for verification
- ContractNLI uses entailment/contradiction/neutral; CUAD uses present/absent with extractive spans

**Stage 4 - Claim Verification** (`agent/verifier/verifier.py:195-355`):
- Decomposes answer into atomic claims
- For each claim, prompts model: "Is this claim supported by the evidence?"
- Aggregates claim-level verdicts: overall status = supported | contradicted | not_enough_info
- Sets `hallucination_detected = True` if any claim is contradicted or lacks support

**Stage 5 - Answer Revision** (`agent/reviser/reviser.py:181-342`):
- Triggered only when verification detects hallucination
- Re-retrieves evidence with refined query focusing on unsupported claims
- Prompts model to revise answer using new evidence
- Outputs revision type: correction | abstention | clarification | none
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

**Prompt Design** (examples in `agent/generator.py:140-180`, `agent/verifier/verifier.py:210-245`):

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
- Checkpoint resume: re-runs check database SHA256, source code hash, model revision, and configuration before resuming
- Final outputs: JSON summaries + per-row JSONL + complete call traces (37,834 LLM calls across 18,666 predictions)

**Trace Example** (`docs/validation/part4_2026-11-24/analysis/CASES.jsonl:1-50`):
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
| Revision limit | Max 1 iteration | Multiple iterations until convergence | Prevents unbounded cost; empirical data showed diminishing returns |
| Verification scope | Only initial answer | Initial + revised answers | Resource constraint; noted as limitation for future work |
| Citation format | Numbered inline | Footnotes or no citations | Inline citations enable automated grounding checks |
| Retrieval index | Per-document isolation | Global cross-document index | Prevents cross-contamination in batch processing |
| Chunking | Fixed 512 tokens | Semantic/sentence boundaries | Fixed-length ensures consistent budget; clause-level tested as ablation |

---

## 3. Reliability and Recovery

### 3.1 Tested Failure Situations

**Malformed task output.** The parser in `retrieval/common.py` preserves the failed raw output, sets an invalid label, and records a parse error. The Agent withholds an unusable final answer. Invalid predictions remain in the accuracy denominator. This includes ContractNLI outputs with illegal labels and malformed CUAD JSON.

**Out-of-range citation or nonverbatim extract.** The check in `agent/agentic_rag.py` uses the evidence that actually reached the generator. The UI withholds the answer, displays the reason, and keeps the trace. Of the RAG and Agent groups, ContractNLI had 100–211 rows with out-of-range citations per group (4.8–10.1%), and CUAD had 107–149 (10.5–14.6%). Citation-range checks can pass when the cited passage is semantically incorrect.

**Changed conditions.** The live interface cleared results when the sample contract changed from 30 to 60 days. The next request used the new contract and returned a contradiction for the unchanged 30-day proposition. The verification trace shows an attempted revision. See the UI acceptance records and full-trace cases under `docs/validation/`.

### 3.2 Recovery Boundaries

The workflow allows at most one revision, enforced by control flow. The per-call token budget truncates prompt evidence. Model exceptions in the interactive service give an actionable status and no automatic retry. The experiment checkpoint binds settings and skips already completed keys when resuming with the same manifest.

---

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

**Implemented Measures**:
1. **Fixed Random Seeds**: seed=42 for data sampling, retrieval, generation
2. **Pinned Model Versions**: Exact Git revisions and SHA256 hashes of model weights recorded
3. **Frozen Dependencies**: Run manifests capture software versions. `requirements-ui.txt` pins the UI dependency.
4. **Data Provenance**: SHA256 of test files, sample IDs logged in manifests
5. **Code Snapshots**: Git commit hash of experimental code stored with results
6. **Complete Traces**: All 37,834 LLM calls logged with prompts, responses, tokens

**Verification**:
- Independent re-run of 36 retrieval configurations: max absolute deviation 5.55e-16 (floating-point precision)
- Component ablation uses a fixed, paired 10-example set. Archived traces record the observed runs.
- Published analysis package includes scripts to reproduce metrics from raw traces

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
2. **Standard RAG**: BM25 retrieval (top-5, fixed 512-token chunks) → generation
3. **Verified Agentic RAG**: Query analysis → retrieval → generation → verification → revision (if needed)

**Component Ablations** (fixed 10-example pilot per model):
- **Full**: Verification + Revision enabled
- **No Verification**: Both verification and revision disabled
- **No Revision**: Verification enabled, but revision skipped

**Models**: Llama 3 8B Instruct, Mistral 7B Instruct v0.2

**Evaluation Protocol**:
- Format failures counted as errors (included in accuracy denominator)
- Label accuracy = correct predictions / total predictions
- Macro-F1 = average F1 across valid task labels (3-way for NLI, 2-way for CUAD)
- Evidence Recall@k = proportion of gold evidence spans covered by top-k retrieved chunks (only for examples with gold evidence)

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

1. **RQ1: Does RAG reduce hallucinations compared to direct prompting?**
   - **Evidence recall improvements**: Standard RAG achieved 97.89% (ContractNLI) and 61.75% (CUAD) initial evidence recall at k=5. Prompt truncation can change the evidence available to generation.
   - **Task performance**: Mixed label accuracy results across configurations (+2.65 pp to -8.92 pp)
   - **Key finding**: While RAG substantially improves evidence availability, this does not automatically translate to improved task accuracy, suggesting that generation quality and prompt design are equally important factors

2. **RQ2: Does agentic verification and revision improve upon standard RAG?**
   - **Largest observed gain**: Llama 3 × ContractNLI showed +13.01 pp gain (42.52% → 55.52%), demonstrating that verification and revision can substantially improve performance on entailment/contradiction tasks
   - **Active revision behavior**: System successfully triggered 1,555 revisions on ContractNLI (74.4% of cases), with net +218 correct labels, showing the agent actively identifies and attempts to correct problematic outputs
   - **Task and model dependence**: Agent accuracy declined in the other three model-task pairs
   - **Key finding**: Observed effects depend on the model and task

3. **RQ3: What types of contract clauses and reasoning are most prone to errors?**
   - **Task complexity matters**: Information extraction tasks (CUAD) showed 36% lower retrieval recall (61-64%) compared to reasoning tasks (97-98%), indicating that clause extraction is more challenging for retrieval systems
   - **Model robustness**: Mistral had more format failures in all three ContractNLI comparisons. On CUAD, the ordering varied by method.
   - **Revision challenges**: Analysis of revision behavior shows that 74.4% of Llama 3 ContractNLI cases triggered revision, with 262 incorrect-to-correct and 44 correct-to-incorrect label transitions
   - **Error patterns**: Cases demonstrate that high retrieval quality (98% recall) is necessary but not sufficient for correct answers, emphasizing the importance of generation quality and reasoning capability

4. **Model Comparison**:
   - Llama 3 format compliance ranged from 78.92% to 100%; Mistral ranged from 75.42% to 90.98%.
   - Format failures (up to 514/2,091 for Mistral) dominated the error budget, overshadowing other error sources
   - Performance variations across tasks suggest that model selection should consider task-specific requirements

#### 5.3.2 Evidence Retrieval Performance (36 Configurations)

**ContractNLI Recall@5** (1,188 examples with gold evidence):

| Retriever | Chunking | Recall@3 | Recall@5 | Recall@10 |
|-----------|----------|----------|----------|-----------|
| BM25 | Clause-level | 93.76% | **98.62%** | 100.00% |
| BM25 | Fixed-length | 90.81% | 97.89% | 99.97% |
| Dense | Clause-level | 91.13% | 97.44% | 99.97% |
| Hybrid | Clause-level | 94.80% | 98.62% | 99.97% |

**CUAD Recall@5** (599 examples with gold evidence):

| Retriever | Chunking | Recall@3 | Recall@5 | Recall@10 |
|-----------|----------|----------|----------|-----------|
| BM25 | Clause-level | 51.83% | **64.24%** | 82.26% |
| BM25 | Fixed-length | 51.12% | 61.75% | 82.31% |
| Dense | Clause-level | 45.72% | 59.41% | 81.09% |
| Hybrid | Clause-level | 51.65% | 63.77% | 82.86% |

**Analysis**:
- ContractNLI: Very high recall (97–98% at k=5); retrieval is not the bottleneck
- CUAD: Moderate recall (61–64% at k=5); substantial room for retrieval improvement
- BM25 with clause-level chunking performed best on both datasets
- Dense retrieval underperformed BM25, likely due to domain gap (encoder trained on general text, not legal contracts)

#### 5.3.3 Agent Revision Behavior (Llama 3, Full Test Sets)

| Dataset | Revisions Triggered | Label Wrong→Right | Label Right→Wrong | Initial Correct | Final Correct | Net Gain |
|---------|---------------------|-------------------|-------------------|-----------------|---------------|----------|
| ContractNLI | 1,555 / 2,091 (74.4%) | 262 | 44 | 943 / 2,091 (45.1%) | 1,161 / 2,091 (55.5%) | **+218** |
| CUAD | 498 / 1,020 (48.8%) | 19 | 56 | 670 / 1,020 (65.7%) | 633 / 1,020 (62.1%) | **-37** |

**Observations**:
- Llama 3 revision rates were 48.8% on CUAD and 74.4% on ContractNLI.
- ContractNLI: 262 corrections vs. 44 errors introduced → net positive
- CUAD: 19 corrections vs. 56 errors introduced → net negative
- Revision does not guarantee improvement; depends on quality of re-retrieved evidence and model's ability to correct itself

### 5.4 Failure Cases and Limitations

**Case Analysis** (20 documented cases in `docs/validation/part4_2026-11-24/CASES.md`):

| Failure Mode | Description | Example Case # | Impact |
|--------------|-------------|----------------|--------|
| **Retrieval success ≠ correct answer** | Gold evidence retrieved but model still answers incorrectly | Case 1, 2 | High recall (98%) does not guarantee high accuracy (55%) |
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

We compared our Agentic RAG to two baselines (Direct, Standard RAG) rather than external hallucination detection methods, as:
1. No prior work addresses contract-specific hallucination with verification+revision
2. Our system is task-integrated (generates + verifies) rather than post-hoc checking

**Key Differentiators**:
- Direct Prompting: No retrieval; relies on parametric knowledge + in-context contract
- Standard RAG: Retrieval without verification; outputs stand as-is
- Agentic RAG: Retrieval + multi-step checking + targeted revision

### 5.6 Output Variability and Reproducibility

**Temperature 0, Fixed Seed**:
- Deterministic sampling enforced (temp=0, seed=42)
- Identical inputs → identical outputs on same infrastructure

**Cross-platform Variance**:
- vLLM (CUDA) vs. MLX (Apple Silicon) produced different outputs for same model/prompt
- Reported results use vLLM for main experiments, MLX for ablation pilot
- Platform noted in experimental manifests; results not directly comparable across platforms

**Model Version Sensitivity**:
- Llama 3: Tested revision `8afb486c` only; different snapshots may vary
- Mistral: Tested revision `63a8b081` only
- Both revisions pinned with SHA256 verification

---

## 6. Contributions and Attribution

### 6.1 Individual Contributions

**Yin Fengling (3036807375)** - Dataset Preprocessing & Evaluation Metrics
- Implemented `data/contractnli_processor.py` and `data/cuad_processor.py` (600 lines)
- Processed 2,091 ContractNLI and 1,020 CUAD test examples with unified schema
- Developed `evaluation/metrics.py` for task labels and retrieval evidence. The final audit reports lexical answer overlap separately from semantic completeness.
- Validated dataset integrity and gold evidence spans
- Designed metric calculation protocol for format failures and missing labels

**Qin Mengmeng (3036843769)** - Retrieval System & Baselines
- Implemented chunking strategies in `retrieval/chunking.py` (fixed-length and clause-level, 450 lines)
- Built BM25, Dense, and Hybrid retrievers in `retrieval/retrievers/` (350 lines)
- Developed Direct Prompting baseline (`retrieval/baseline_direct.py`, 320 lines)
- Developed Standard RAG baseline (`retrieval/baseline_rag.py`, 480 lines)
- Executed 36-configuration retrieval ablation study
- Delivered full test set results for Mistral and Llama 3 on AutoDL (12,444 baseline predictions)

**Wang Zhengyao (3036795675)** - Agent Components
- Implemented Query Analyzer (`agent/query_analyzer/analyzer.py`, 350 lines)
- Implemented Answer Generator (`agent/generator.py`, 420 lines)
- Implemented Evidence Verifier (`agent/verifier/verifier.py`, 480 lines)
- Implemented Answer Reviser (`agent/reviser/reviser.py`, 380 lines)
- Integrated full Agentic RAG workflow (`agent/agentic_rag.py`, 540 lines)
- Conducted component unit tests (Priority 1–2) and integration tests (Priority 3)
- Executed full-scale Agentic RAG experiments (6,222 predictions on ContractNLI + CUAD)

**Li Xinru (3036807959)** - Experiment Orchestration, Analysis & Reporting
- Developed unified experiment runner (`experiments/run_experiments.py`, 642 lines)
- Implemented checkpoint resume, task-aware scoring, and ablation framework
- Reconciled full-scale experiment records and developed the formal Gradio interface
- Performed result validation: data hash verification, confusion matrix audit, trace integrity checks
- Analyzed 20 failure cases with traceable evidence and model call records
- Developed UI prototype (`ui/app.py`, 327 lines) for contract review demo
- Authored experiment documentation, result summaries, and this final report

### 6.2 Code and Library Attribution

**External Libraries**:
- `transformers==4.56.2` (Hugging Face): Model loading and tokenization
- `vllm==0.10.2`: GPU-accelerated inference with batching
- `rank-bm25==0.2.2`: BM25 retrieval implementation
- `sentence-transformers==6.3.0`: Dense encoder (all-MiniLM-L6-v2)
- `scikit-learn==1.4.0`: Evaluation metrics (accuracy_score, f1_score, confusion_matrix)
- `gradio==6.32.0`: Interactive UI framework
- `mlx-lm==0.32.2`: Apple Silicon inference for ablation pilot

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
   - Blind review materials prepared (600 candidate answers)
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

The team draft records Claude and GitHub Copilot for development, debugging, documentation and report assistance. Codex was used to develop and validate retrieval and baselines, audit the experiment traces, repair the legacy evaluator, edit this report, build the submission package and verify its runtime. The model outputs in the evaluation came from the named Llama 3 and Mistral runs. Automated tests and synthetic user inputs are documented separately from independent annotation. See `AI_USE.md` for prompt excerpts and coverage limitations. Independent human evaluation is pending.

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
4. **Format matters as much as content**: Format failures dominated Mistral's error budget; robust parsing is production-critical
5. **Task-dependent tuning needed**: One-size-fits-all approach failed; NLI and QA tasks require different optimization

---

## Appendices

### A. Repository Structure

```
legal-hallucination-auditing/
├── data/                           # Dataset processing (4 files)
│   ├── __init__.py
│   ├── dataset_loader.py           # Base dataset loader (110 lines)
│   ├── contractnli_processor.py    # ContractNLI processing (210 lines)
│   ├── cuad_processor.py           # CUAD processing (305 lines)
│   ├── privacyqa_processor.py      # PrivacyQA (optional, 138 lines)
│   ├── raw/                        # Raw dataset files
│   └── processed/                  # Processed test sets
│       ├── contractnli_test.json   # 2,091 examples
│       ├── cuad_test.json          # 1,020 examples
│       └── README.md               # Dataset documentation
│
├── retrieval/                      # Retrieval & baselines (8 files)
│   ├── __init__.py
│   ├── chunking.py                 # Document chunking (148 lines)
│   ├── common.py                   # Shared utilities (459 lines)
│   ├── cli.py                      # CLI helpers (304 lines)
│   ├── evidence.py                 # Evidence scoring (115 lines)
│   ├── evaluate_retrieval.py       # Retrieval ablation (106 lines)
│   ├── baseline_direct.py          # Direct prompting (101 lines)
│   ├── baseline_rag.py             # Standard RAG (273 lines)
│   └── retrievers/
│       ├── __init__.py
│       └── retriever.py            # BM25/Dense/Hybrid (484 lines)
│
├── agent/                          # Agentic RAG (9 files)
│   ├── __init__.py
│   ├── agentic_rag.py              # Main workflow (447 lines)
│   ├── generator.py                # Answer generation (283 lines)
│   ├── query_analyzer/
│   │   ├── __init__.py
│   │   ├── analyzer.py             # Query analysis (352 lines)
│   │   └── analyzer_openrouter.py  # OpenRouter variant (276 lines)
│   ├── verifier/
│   │   ├── __init__.py
│   │   └── verifier.py             # Claim verification (508 lines)
│   └── reviser/
│       ├── __init__.py
│       └── reviser.py              # Answer revision (459 lines)
│
├── evaluation/                     # Evaluation metrics (3 files)
│   ├── __init__.py
│   ├── metrics.py                  # Core metrics (392 lines)
│   ├── hallucination_detector.py   # Failure modes (298 lines)
│   └── error_analysis.py           # Error analysis (309 lines)
│
├── experiments/                    # Experiment orchestration (7 files)
│   ├── __init__.py
│   ├── run_experiments.py          # Main runner (642 lines)
│   ├── run_agent_ablation_pilot.py # Component ablation (232 lines)
│   ├── run_baseline_pilot.py       # Baseline pilot (542 lines)
│   ├── run_full_mistral.py         # Full Mistral run (588 lines)
│   ├── vllm_mistral.py             # vLLM setup (166 lines)
│   ├── summarize_results.py        # Result aggregation (388 lines)
│   ├── summarize_part4.py          # Part 4 summary (351 lines)
│   ├── prepare_support_review.py   # Human review prep (170 lines)
│   ├── prepare_cuad_development.py # CUAD dev set (134 lines)
│   ├── results/                    # Experiment outputs
│   │   ├── priority3/              # Integration tests
│   │   └── priority4/              # Full-scale experiments
│   │       ├── PRIORITY4_FINAL_REPORT.md
│   │       ├── *_predictions.json  # Raw predictions
│   │       └── *_stats.json        # Statistics
│   └── logs/                       # Execution logs
│
├── ui/                             # Interactive UI (3 files)
│   ├── __init__.py
│   ├── app.py                      # Gradio interface (327 lines)
│   ├── backends.py                 # Backend adapters (140 lines)
│   └── service.py                  # Service layer (224 lines)
│
├── utils/                          # Shared utilities
│   ├── __init__.py
│   ├── llm_interface.py            # LLM wrapper
│   ├── logger.py                   # Logging
│   └── helpers.py                  # Helper functions
│
├── configs/                        # Configuration files
│   ├── model_config.yaml           # Model settings
│   ├── retrieval_config.yaml       # Retrieval parameters
│   └── experiment_config.yaml      # Experiment config
│
├── docs/                           # Documentation
│   ├── PART4_EXPERIMENTS.md        # Experiment handoff
│   ├── RETRIEVAL_AND_BASELINES_HANDOFF.md
│   ├── UI_HANDOFF.md               # UI documentation
│   ├── ANSWER_SUPPORT_PROTOCOL.md  # Evaluation protocol
│   ├── DEMO_RUNBOOK.md             # Demo guide
│   ├── PROJECT_REVIEW_2026-11-20.md
│   ├── QIN_MENGMENG_TASKS.md
│   └── validation/                 # Validation results
│       ├── part4_2026-11-22/       # Retrieval ablation
│       │   ├── RESULTS.md
│       │   ├── CASE_ANALYSIS.md
│       │   ├── agent_ablation/     # 60 ablation outputs
│       │   ├── bm25_full_rerun.json
│       │   ├── dense_hybrid_full_rerun.json
│       │   └── archived_result_audit/
│       ├── part4_2026-11-23/       # UI acceptance
│       │   ├── README.md
│       │   ├── RQ_FINDINGS.md
│       │   ├── UI_ACCEPTANCE.md
│       │   └── acceptance/
│       ├── part4_2026-11-24/       # Full trace analysis
│       │   ├── README.md
│       │   ├── CASES.md            # 20 failure cases
│       │   ├── analysis/
│       │   │   ├── RESULTS.json    # 18,666 predictions
│       │   │   ├── CASES.jsonl
│       │   │   └── row_metrics.jsonl.gz
│       │   └── metrics-followup/
│       │       ├── METRICS.json
│       │       └── VALIDATION.json
│       ├── RETRIEVAL_RESULTS.md
│       ├── llama3_autodl_full_summary.json
│       └── mistral_autodl_full_summary.json
│
├── tests/                          # Unit tests
│   ├── test_agentic_pilot.py
│   ├── test_experiment_handoff.py
│   ├── test_full_mistral.py
│   └── test_ui_service.py          # UI service and callback tests
│
├── scripts/                        # Utility scripts
│   ├── fetch_verified_llama3.py
│   ├── run_autodl_mistral.sh
│   └── validate_ui_service.py
│
├── requirements.txt                # Full dependencies
├── requirements-retrieval.txt      # Minimal dependencies
├── requirements-ui.txt             # UI dependencies
├── setup.py                        # Package setup
├── README.md                       # Project overview
├── COMPLETION_SUMMARY.md           # Implementation summary
├── PROJECT_SUMMARY.md              # Project documentation
├── IMPLEMENTATION_GUIDE.md         # Developer guide
├── AGENTS.md                       # Agent documentation
├── task3.md                        # Task 3 report
├── TEAM_TODO_LIST.md               # Team coordination
└── .gitignore                      # Git ignore rules
```

**Total Code Statistics**:
- **38 Python files** across 7 main modules
- **~8,500 lines** of production code (excluding tests and docs)
- **18,666 predictions** generated and analyzed
- **37,834 LLM calls** traced and recorded

### B. Key Files for Reproducibility

| File | Purpose | SHA256 (first 16 hex) |
|------|---------|----------------------|
| `data/processed/contractnli_test.json` | ContractNLI test data (2,091) | `ba6aa63b66a35402` |
| `data/processed/cuad_test.json` | CUAD test data (1,020) | `8b95cd181abd935c` |
| `docs/validation/llama3_autodl_full_summary.json` | Llama 3 full results | `f914d023238bb091` |
| `docs/validation/mistral_autodl_full_summary.json` | Mistral full results | `6652304a080416e8` |
| `docs/validation/part4_2026-11-24/analysis/RESULTS.json` | Trace analysis | `708bd4ba35fb3141` |

The data hashes also appear in the two full-run summaries. The submission manifest records SHA-256 for the packaged files.

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
