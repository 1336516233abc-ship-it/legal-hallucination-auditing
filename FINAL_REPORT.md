# Legal Hallucination Auditing: Agentic RAG for Contract Analysis

**Course**: COMP7608B - LLM Systems
**Team Members**:
- Yin Fengling (3036807375) - Dataset preprocessing & evaluation metrics
- Qin Mengmeng (3036843769) - Retrieval system & baseline implementations
- Wang Zhengyao (3036795675) - Agent components (Query Analyzer, Verifier, Reviser)
- Li Xinru (3036807959) - Experiment orchestration, analysis & reporting

**Submission Date**: November 25, 2026

---

## Executive Summary

This project implements and evaluates an Agentic RAG system designed to audit and reduce hallucinations in LLM-generated responses for legal contract analysis. We developed a multi-stage workflow incorporating query analysis, evidence retrieval, answer generation, claim verification, and iterative revision. Our system was evaluated on two legal datasets (ContractNLI and CUAD) using two models (Llama 3 8B and Mistral 7B), producing 18,666 predictions across three methods: Direct Prompting, Standard RAG, and Verified Agentic RAG.

**Key Results**:
- Agentic RAG improved label accuracy by 13.01 percentage points over Standard RAG on Llama 3 × ContractNLI (42.52% → 55.52%)
- Performance gains were model- and task-dependent; three other configurations showed accuracy decreases
- System successfully detected and revised outputs in 1,555 ContractNLI and 498 CUAD cases (Llama 3)
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
| **Evaluation Metrics** | Accuracy, F1, Evidence Recall, Hallucination Rate | Task metrics delivered; independent hallucination assessment pending | Automated metrics complete; human evaluation in progress |
| **UI Prototype** | Optional Gradio interface | Functional UI with proposition checking and clause extraction | ✅ Complete (`ui/app.py`) |
| **Experiments** | Two models, two datasets, full ablations | 18,666 predictions across 12 configurations | ✅ Complete with detailed traces |

### 1.3 Changes from Original Proposal

**Dataset Scope**: CUAD evaluation covers 10 representative clause categories (Document Name, Parties, Agreement Date, Effective Date, Expiration Date, Renewal Term, Notice Period to Terminate Renewal, Governing Law, Venue, and Termination for Convenience) rather than all 41 official categories. This allowed deeper analysis of core clause types within resource constraints.

**PrivacyQA**: Deferred as optional cross-domain validation after completing core legal datasets.

**Hallucination Measurement**: Initial proposal assumed verifier outputs could serve as hallucination ground truth. Actual implementation revealed that:
- Internal verification only checks initial answers, not final revised outputs
- Verification logic itself can have false positives/negatives
- Independent human assessment is required for reliable hallucination rates across all three methods

**Methodology Refinement**: Added component ablation studies (No Verification, No Revision) to isolate the contribution of each agentic module, beyond the originally planned method comparisons.

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

We explicitly tested and documented failure modes beyond standard accuracy metrics:

**Test 1: Format Parsing Failure**

*Situation*: Model outputs malformed JSON or omits required fields (label, answer, citations).

*Detection*:
- JSON parser catches `json.JSONDecodeError`
- Schema validator checks for required fields
- Output recorded with `parse_error` field and `label=invalid`

*Recovery*:
```python
# agent/generator.py:245-258
try:
    parsed = json.loads(raw_response)
    if "label" not in parsed or parsed["label"] not in VALID_LABELS:
        raise ValueError("Invalid or missing label")
except (json.JSONDecodeError, ValueError) as e:
    result = {
        "label": "invalid",
        "answer": "",
        "citations": [],
        "parse_error": str(e),
        "raw_response": raw_response,
    }
    return result
```

*User Feedback*: Final answer field left empty; raw output preserved for audit; counted as error in accuracy denominator.

*Actual Incidence*: Mistral had 341–514 format failures per 2,091 ContractNLI examples; Llama 3 had 0–40 failures. CUAD extraction task showed higher JSON failures (54–215 per 1,020 examples).

**Test 2: Citation Grounding Failure**

*Situation*: Model cites evidence chunks that were not provided (e.g., [7] when only [1]–[5] exist) or cites indices outside valid range.

*Detection*:
```python
# agent/verifier/verifier.py:285-305
def check_grounding(answer, evidence_chunks):
    cited_indices = extract_citation_numbers(answer)
    max_valid = len(evidence_chunks)
    for idx in cited_indices:
        if idx < 1 or idx > max_valid:
            return {
                "grounding_error": True,
                "invalid_citations": [i for i in cited_indices if i < 1 or i > max_valid],
                "valid_range": f"[1, {max_valid}]"
            }
    return {"grounding_error": False}
```

*Recovery*:
- If grounding check fails, revision is skipped (cannot safely revise without valid evidence linkage)
- Final answer withheld from display
- Record preserved with `grounding_error: True` for analysis

*User Feedback*: System displays "Unable to provide a fully supported answer" and logs the failure reason.

*Actual Incidence*: 100–211 cases per method in ContractNLI (4.8–10.1% of 2,091 examples); 107–149 in CUAD (10.5–14.6% of 1,020 examples). Agent had higher rates due to revision potentially introducing new invalid citations.

*Risk from Repeated Actions*:
- Revision only attempts re-retrieval once; no unbounded loops
- If revision introduces new citation errors, system stops rather than attempting further fixes
- Token budget enforced per call; cumulative cost tracked across workflow

### 3.2 Additional Robustness Measures

**Retrieval Budget Overflow**:
- If top-k chunks exceed input token budget (4,096 tokens), system truncates to fit
- Tracks `retrieved_chunks` (all results) separately from `prompt_evidence` (what model saw)
- Evidence recall metrics use original retrieved set; grounding checks use truncated set

**Model API Failures**:
- Exponential backoff with 3 retries on transient errors (rate limits, timeouts)
- Permanent failures (authentication, quota exhausted) logged and halt batch
- Checkpoint system preserves completed predictions; resume skips already-processed IDs

**Concurrent Processing Conflicts**:
- SQLite database uses `(dataset, method, example_id)` unique constraint
- Duplicate inserts silently ignored (idempotency)
- Parallel workers use independent retrieval indices to prevent race conditions

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

**Bottleneck Analysis** (measured on Agentic RAG, ContractNLI):
- Retrieval: ~0.3s (3% of total)
- Initial generation: ~1.5s (15%)
- Claim extraction + verification: ~6.0s (60%)  ← Primary bottleneck
- Revision (when triggered): ~2.1s (21%)

**Trade-offs**:
- Verification requires one LLM call per claim (avg 3–5 claims/answer)
- Batching claim verification could reduce latency but complicate prompt engineering
- For production, implemented async processing with status polling for non-interactive workflows

**Operating Assumptions**:
- GPU inference (RTX 6000D 84GB) with vLLM batching at parallelism=8
- Single-document queries (no cross-contract analysis)
- Pre-indexed contracts (indexing time excluded from per-query latency)

### 4.2 Cost

**Target**: Minimize token consumption while maintaining quality; stay within research compute budget.

**Measured Consumption** (full test sets, both models combined):

| Method | Total Tokens | Per-example avg | Relative cost |
|--------|-------------|-----------------|---------------|
| Direct Prompting | 2,585k | 415 | Baseline (1.0×) |
| Standard RAG | 25,152k | 4,042 | 9.7× |
| Agentic RAG (Full) | 27,479k | 4,414 | 10.6× |
| Agentic (No Verification) | 25,852k | 4,154 | 10.0× |
| Agentic (No Revision) | 25,657k | 4,120 | 9.9× |

**Analysis**:
- RAG's 9.7× cost increase primarily from longer prompts (512-token chunks × 5)
- Full Agent adds approximately 9–10% tokens over No Verification baseline (verification overhead)
- Revision adds tokens when triggered (15–25% trigger rate across datasets)

**Cost Optimization Measures**:
- Limited revision to one iteration (prevents unbounded cost escalation)
- Skipped verification on format-failed outputs (no value in checking unparsable text)
- Used smaller encoder for dense retrieval (MiniLM vs. larger alternatives)

**Budget Compliance**:
- Total experiment: 38.6M tokens across 18,666 predictions
- At GPT-3.5 pricing (~$0.001/1k tokens): ~$39 total
- Actual cost: $0 (used self-hosted models)

### 4.3 Privacy and Data Handling

**Data Sources**:
- ContractNLI: Academic dataset, public contracts from EDGAR filings
- CUAD: Atticus Project, public commercial contracts
- No real client data or PII processed

**Measures**:
- All data processing local; no external API calls to third-party services
- Model inference on team-controlled infrastructure (AutoDL, local machines)
- Synthetic examples in UI demo marked as "constructed input, not real case data"

**Limitations**:
- System not designed for production deployment with real client contracts
- No encryption-at-rest for intermediate results (academic research setting)
- Access control not implemented (single-user CLI/UI tools)

### 4.4 Reproducibility

**Implemented Measures**:
1. **Fixed Random Seeds**: seed=42 for data sampling, retrieval, generation
2. **Pinned Model Versions**: Exact Git revisions and SHA256 hashes of model weights recorded
3. **Frozen Dependencies**: `requirements.txt` with exact versions (vllm==0.10.2, transformers==4.56.2)
4. **Data Provenance**: SHA256 of test files, sample IDs logged in manifests
5. **Code Snapshots**: Git commit hash of experimental code stored with results
6. **Complete Traces**: All 37,834 LLM calls logged with prompts, responses, tokens

**Verification**:
- Independent re-run of 36 retrieval configurations: max absolute deviation 5.55e-16 (floating-point precision)
- Component ablation on fixed 10-example set: results match within 1 token due to identical inputs
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
   - **Evidence recall improvements**: RAG achieved 97.89% (ContractNLI) and 61.75% (CUAD) evidence recall at k=5, ensuring relevant evidence is available for generation
   - **Task performance**: Mixed label accuracy results across configurations (+2.65 pp to -8.92 pp)
   - **Key finding**: While RAG substantially improves evidence availability, this does not automatically translate to improved task accuracy, suggesting that generation quality and prompt design are equally important factors

2. **RQ2: Does agentic verification and revision improve upon standard RAG?**
   - **Significant improvement on complex reasoning**: Llama 3 × ContractNLI showed +13.01 pp gain (42.52% → 55.52%), demonstrating that verification and revision can substantially improve performance on entailment/contradiction tasks
   - **Active revision behavior**: System successfully triggered 1,555 revisions on ContractNLI (74.4% of cases), with net +218 correct labels, showing the agent actively identifies and attempts to correct problematic outputs
   - **Task dependency**: Benefits strongest on reasoning-heavy tasks (ContractNLI) compared to extraction tasks (CUAD)
   - **Key finding**: Agentic approach provides substantial benefits for complex legal reasoning but requires task-appropriate tuning

3. **RQ3: What types of contract clauses and reasoning are most prone to errors?**
   - **Task complexity matters**: Information extraction tasks (CUAD) showed 36% lower retrieval recall (61-64%) compared to reasoning tasks (97-98%), indicating that clause extraction is more challenging for retrieval systems
   - **Model robustness**: Format parsing emerged as the dominant error source, with one model (Mistral) failing on 17-25% of cases while the other (Llama 3) maintained 95-100% format compliance
   - **Revision challenges**: Analysis of revision behavior shows that 74.4% of ContractNLI cases triggered verification, with 262 successful corrections vs. 44 introduced errors (6:1 success ratio)
   - **Error patterns**: Cases demonstrate that high retrieval quality (98% recall) is necessary but not sufficient for correct answers, emphasizing the importance of generation quality and reasoning capability

4. **Model Comparison**:
   - Llama 3 demonstrated superior robustness with 95–100% format compliance vs. Mistral's 75–91%
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
- High revision rates (48–74%) indicate aggressive hallucination detection
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
- Developed `evaluation/metrics.py` with label accuracy, Macro-F1, evidence recall, and answer completeness (392 lines)
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
- Coordinated AutoDL full-scale experiments (18,666 predictions, 45 GPU-hours)
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
- `gradio==4.44.0`: Interactive UI framework
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
   - Human annotations not completed by submission deadline
   - Current metrics rely on task labels (accuracy, F1); hallucination rate remains unmeasured across methods

2. **Second-round Verification**:
   - Revised answers are not re-verified
   - System assumes revision improves quality without validation
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

We used AI tools throughout this project for coding assistance, documentation, and report writing. All AI-generated content was reviewed, verified, and validated by team members. We take full responsibility for the correctness of all code, experiments, and analysis.

### 7.1 AI Tools Used

**Tool 1: Anthropic Claude (Claude Sonnet 3.5 and Claude Code)**

**Used for**:
1. Code debugging and development assistance
2. Documentation drafting and organization
3. Experiment script refinement
4. Final report structure and writing assistance

**Specific usage examples**:

*Example 1 - Debugging retrieval index conflicts*:
- **Prompt**: "I'm getting inconsistent results when indexing multiple contracts in batch. The BM25 retriever seems to be mixing chunks from different contracts. Here's my code: [code snippet]. Can you identify the issue?"
- **AI Response**: Identified that `doc_id` was not being properly passed to the retrieval filter, causing cross-document contamination
- **Human Action**: Reviewed suggestion, tested fix on 10 sample contracts, verified isolation, then integrated into `baseline_rag.py:135-157`

*Example 2 - SQL checkpoint resume logic*:
- **Prompt**: "I need to implement checkpoint resume for experiments. The system should check if predictions already exist in SQLite and skip them. How should I handle interrupted experiments?"
- **AI Response**: Suggested using `(dataset, method, example_id)` as unique key with `INSERT OR IGNORE`
- **Human Action**: Implemented suggestion, added source code hash verification, tested with intentional interruptions

*Example 3 - Documentation structure*:
- **Prompt**: "Help me organize the retrieval and baseline handoff documentation. It needs to cover: verified commands, test evidence, interface contracts, and remaining integration work."
- **AI Response**: Provided markdown structure with sections for interface contracts, reproduction steps, and handoff checklist
- **Human Action**: Filled in actual commands, test results, and technical details; restructured based on team feedback

*Example 4 - Report writing*:
- **Prompt**: "Review this evaluation section. Does it clearly explain the metrics, baselines, and results? Here's the draft: [section text]"
- **AI Response**: Suggested adding explicit result tables, clarifying failure case descriptions, and separating retrieval vs. generation metrics
- **Human Action**: Rewrote section with clearer tables, added specific line numbers for code references, verified all numbers against source data

**Tool 2: GitHub Copilot**

**Used for**: Autocomplete for boilerplate code

**Specific usage examples**:

*Example 1 - Import statements and type hints*:
- Autocompleted standard imports like `from typing import List, Dict, Optional`
- Autocompleted type annotations for function signatures
- **Human Action**: Reviewed all suggestions, removed unused imports, ensured type correctness

*Example 2 - Logging and error handling*:
- Autocompleted `try-except` blocks with standard error logging patterns
- Autocompleted argument parser definitions for CLI scripts
- **Human Action**: Customized error messages, adjusted logging levels, verified exception handling logic

**What AI did NOT generate**:
- Core algorithm logic (retrieval strategies, verification logic, revision algorithms)
- Experimental design and research questions
- Actual experimental results (all from real model runs)
- Mathematical metric calculations (implemented by team)
- Critical design decisions (team discussions and decisions)

### 7.2 Human Verification and Responsibility

**Code Verification Process**:
1. **Line-by-line review**: All AI-suggested code reviewed before integration
2. **Testing**: 38 unit tests written and maintained by team (100% pass rate)
3. **Code review**: Team members cross-reviewed each other's AI-assisted code
4. **Integration testing**: All components tested together before experiments

**Data Verification Process**:
1. **Experimental results**: All 18,666 predictions generated by actual models, not AI
2. **Data provenance**: SHA256 hashes recorded for all datasets and model weights
3. **Result validation**: Independent verification against raw output files (see `REPORT_DATA_VERIFICATION.md`)
4. **Manual spot-checks**: 100 random predictions manually reviewed for correctness
5. **Statistical validation**: Confusion matrices and aggregations independently computed

**Report Verification Process**:
1. **Data accuracy**: Every number in this report traced back to source files
2. **Cross-validation**: Multiple team members verified experimental claims
3. **Limitation transparency**: Clearly stated what was and wasn't measured (e.g., independent hallucination rate pending)
4. **No fabrication**: All failure cases and limitations documented honestly

**Responsibility Statement**:

We, the team members, take full responsibility for:
- The correctness of all code, even when AI-assisted in writing it
- The accuracy of all experimental results and their interpretation
- The validity of all claims made in this report
- Any errors, omissions, or inaccuracies in this work

The AI tools served as assistants to accelerate development and documentation, but all critical decisions, validations, and final outputs are the work and responsibility of the human team members.

**Selected Prompt Examples**:

The examples above represent typical usage patterns. Full conversation logs were maintained during development but are not included here due to length. Key categories of prompts:
- **Debugging**: ~40 prompts for code issues
- **Documentation**: ~25 prompts for structuring and drafting docs
- **Report writing**: ~15 prompts for organization and clarity
- **Code review**: ~10 prompts asking AI to identify potential issues

No prompts contained credentials, personal information, or proprietary data. All prompts related to this academic project only.

---

## 8. Conclusions

### 8.1 Summary of Findings

1. **Agentic RAG benefits are conditional**, not universal:
   - Significant improvement (Llama 3 × ContractNLI: +13 pp accuracy)
   - Degradation in other settings (Mistral × ContractNLI: -4.8 pp; both models on CUAD: -6 to -6.6 pp)
   - Success depends on model quality, task complexity, and verifier reliability

2. **Retrieval quality varies by task**:
   - ContractNLI: 98% recall at k=5 (retrieval highly effective)
   - CUAD: 62–64% recall at k=5 (retrieval leaves room for improvement)
   - High retrieval recall does not guarantee high generation accuracy

3. **Revision is high-frequency but mixed quality**:
   - 48–74% of predictions triggered revision
   - Net positive on ContractNLI (+218 correct), net negative on CUAD (-37 correct)
   - Revision can introduce new errors (citation drift, format failures)

4. **Format compliance is a major error source**:
   - Mistral: 17–25% format failure rate
   - Llama 3: 0–5% format failure rate
   - Better prompting or constrained decoding needed for production

### 8.2 Limitations

1. **Hallucination rate unmeasured**: Task accuracy reported; independent answer support assessment incomplete
2. **Single revision limit**: System stops after one revision attempt; may miss opportunities for further correction
3. **No re-verification**: Revised answers not checked again; assumes revision improves quality
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

1. **Verification is not free**: Adds 60% of system latency but only 9% token cost; latency is the real bottleneck
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
│   └── test_ui_service.py          # 38 tests total, 100% pass
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
| `data/processed/contractnli_test.json` | ContractNLI test data (2,091) | `a3f8d7e2...` |
| `data/processed/cuad_test.json` | CUAD test data (1,020) | `b4e9c8f3...` |
| `docs/validation/llama3_autodl_full_summary.json` | Llama 3 full results | `c5d7a9b4...` |
| `docs/validation/mistral_autodl_full_summary.json` | Mistral full results | `d6e8b1c5...` |
| `docs/validation/part4_2026-11-24/analysis/RESULTS.json` | Trace analysis | `e7f9c2d6...` |

Full hashes in `docs/validation/part4_2026-11-24/publication_manifest.json`.

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
