# Qin Mengmeng Responsibilities and Acceptance Checklist

This document preserves original tasks and acceptance requirements, implementation status per [Retrieval and Baseline Development Handoff](RETRIEVAL_AND_BASELINES_HANDOFF.md).

Name: Qin Mengmeng; Student ID: 3036843769. Organization date: November 20, 2026; Project deadline: November 25, 2026.

Your responsibility is to make retrieval/ into verifiable, reproducible retrieval and baseline modules, delivering Direct Prompting, Standard RAG execution results and retrieval ablation analysis. Existing code already has framework, but has defects affecting experiment correctness, prioritize completing P0 fixes below, then run complete experiments.

Responsibility sources: Team Members in README.md, personal checklist starting from TEAM_TODO_LIST.md:135, Team Task Assignment in PROJECT_SUMMARY.md. First two team members' data files already available for development; loader import defects need synchronous fixes.

## Files and Deliverables You Are Responsible For

| File or Artifact | Content You Need to Complete | Acceptance Criteria |
|---|---|---|
| `retrieval/chunking.py` | Fixed length, clause-level chunking | Preserve original text and offsets; do not miss short contracts; explicit error on invalid parameters |
| `retrieval/retrievers/retriever.py` | BM25, Dense, Hybrid | Reliable sorting; correct document scope; explicit behavior for empty index, zero scores, top_k boundaries |
| `retrieval/baseline_direct.py` | Direct Prompting baseline | Generate and parse answers by dataset; explicit input token budget; support small samples and full scale |
| `retrieval/baseline_rag.py` | Standard RAG baseline | Retrieve within current contract; preserve evidence provenance; support retrieval, chunking, top_k configuration |
| Retrieval configuration and interface documentation | Allow Agent and experiment framework to reuse your components | Parameters actually take effect; example calls and output format consistent |
| Component tests and small sample records | Prove post-fix behavior meets task requirements | Record commands, input ranges, output and failure handling |
| Baseline results and logs in `experiments/results/` | For Li Xinru to aggregate | Sample IDs aligned, correct quantities, traceable config, model, seed |
| Retrieval analysis and report materials | Evidence Recall@k, strategy comparison, failure cases | Use verified metrics, explain handling of no-evidence samples |

Original file header with your name does not mean these modules completed by you or tested. Currently no baseline execution results available to confirm your experiment portion complete.

## Priority 1: Fix Issues Affecting Correctness

- [ ] **Restrict retrieval by contract.** Each query retrieves only from corresponding contract, using per-contract index or reliable filtering; reuse index for multiple questions from same contract, avoid mixing entire test set. Sync with Wang on same defect in Agent.
- [ ] **Correct chunk original text positions.** Generate chunks with real character ranges satisfying `context[start:end] == text`. Cover multiple spaces, newlines, repeated words, overlapping regions, head/tail chunks; do not infer positions by first-word search.
- [ ] **Handle clause-level boundaries.** Preserve short contracts and tail chunks; long sentence splits must still correspond to original text; validate `chunk_size > 0` and `0 <= overlap < chunk_size`.
- [ ] **Preserve complete provenance.** Results contain stable doc_id, chunk_id, start, end, original text, retrieval score, and can trace back to example_id.
- [ ] **Remove hardcoded 10-item limit.** Add actually effective sample_size and full-scale mode; use fixed seed sampling and preserve sample IDs. Two datasets have different sample sizes, full mode needs to check output quantities.
- [ ] **Separate component configurations.** Pass chunker_kwargs and retriever_kwargs separately; make declared parameters in retrieval_config actually enter instances.
- [ ] **Clarify Direct input budget.** Replace unexplained first 2,000 character truncation; set token budget by model context limit, record whether truncated, and explain together with RAG evidence budget.

Reproduction, source line numbers and module attribution for above issues see [Project Review](PROJECT_REVIEW_2026-11-20.md). Currently do not treat existing "full-scale commands" as already usable; baseline `--sample_size` written in TODO not yet implemented, experiment runner's same-named parameter also not taking effect.

## Priority 2: Component Acceptance and Baseline Execution

After completing chunking fixes, first do retrieval evaluation not requiring LLM. Default reference configuration is fixed_length=512, overlap=128, BM25, top_k=5; length here is approximate word count, cannot be treated as precise token count from model tokenizer.

- [ ] Verify chunk original text slices and offsets on two real datasets.
- [ ] BM25, Dense, Hybrid return specified number of chunks and serializable scores on same batch of query/document; handle top_k=0, top_k greater than chunk count, empty text and zero score situations.
- [ ] Verify querying contract A does not return contract B's chunks; remains correct during consecutive calls and context switching.
- [ ] Check ClauseLevel sentence splitting resources and overlong sentence behavior.
- [ ] Run 5-10 samples per dataset across labels and documents first, confirm output compatible with evaluation entry.
- [ ] Check CUAD no-answer samples do not crash evaluation, ContractNLI empty reference answers not treated as answer generation failure.
- [ ] Full-scale run Direct and Standard RAG, save raw responses, parsed results, retrieval provenance, duration, token statistics and execution configuration.

Expected prediction count for full scale: per model per system ContractNLI 2,091, CUAD 1,020. By repository plan of two models, two baselines, two datasets, baseline main experiments total **12,444 predictions**, not including Agent and retrieval ablation. First test actual duration with small samples, then arrange complete run.

## Retrieval Ablation and Analysis

Your original scope includes comparison of retriever, chunking and top_k, main line is BM25 vs Dense, fixed length vs clause-level, top_k=3/5/10; TEAM_TODO additionally lists Hybrid. Model main experiment plan is Llama3 and Mistral, uniformly arranged by Li Xinru.

Recommend first delivering two baselines with fixed configuration, then organizing following comparisons. Specific run matrix should remain consistent with Li's experiment framework:

| Comparison | Changed Factor | Fixed Factors |
|---|---|---|
| Retrieval method | BM25 / Dense; supplement Hybrid | Data, chunking, top_k, generation model |
| Chunking strategy | fixed_length / clause_level | Data, retrieval method, top_k, generation model |
| Retrieval quantity | 3 / 5 / 10 | Data, retrieval method, chunking, generation model |

Retrieval metrics themselves can be calculated offline first, no need to call generation model for each item. Avoid changing multiple factors at once; when parameter selection needed use dev set or pre-fixed scheme, test set for final comparison.

- [ ] Define with Yin the hit criteria for Recall@k, including evidence split by multiple chunks, partial coverage and no gold evidence cases.
- [ ] Confirm with Yin and Li actual k passed to metrics; current compute_all_metrics evidence recall defaults to calculating by 5.
- [ ] Output Recall/Hit Rate tables by dataset, strategy and k, including evaluable sample count.
- [ ] Provide Li with specific examples of retrieval failure and generation failure, help answer RQ1 and RQ3.
- [ ] Write methods, hyperparameters, result analysis and limitations materials for retrieval and baselines in final report.

## Handoffs and Collaboration You Need

| Contact | What You Need From Them | What You Provide To Them |
|---|---|---|
| Yin Fengling | Data package import, actual conversion scripts, evaluation defect fixes; unify NLI/CUAD task format | Chunk original text and positions, retrieval result examples, Recall test cases |
| Wang Zhengyao | Agent per-contract retrieval, final answer output alignment | Stable chunker/retriever interface, evidence location format, call examples |
| Li Xinru | Runner dataset/sample_size taking effect, ablation actual execution, ground truth containing context, unified hallucination evaluation | Baseline predictions, parameters, logs, duration and retrieval analysis |

Shared interface can retain existing `chunk(text, metadata)`, `index(chunks)`, `retrieve(query, top_k)`, but must explain where document scope is guaranteed. Output chunk retains at least text, start, end, chunk_id and stable doc_id; recommend also retaining query, dataset, example_id and config in saved predictions for cross-system alignment.

Data preprocessing, metric implementation, Agent verification and revision, complete experiment aggregation and final report consolidation each have responsible persons. You need to provide interfaces and experiment materials, and jointly solve handoff issues; no need to take over all modules just because current framework has gaps.

## Schedule Advancing From Current Date

Following is execution order organized per original deadline, actual duration needs measurement with small samples:

| Date | Your Focus | Deliverable Check That Day |
|---|---|---|
| November 20 | P0 fixes, BM25 and data interface acceptance | Position check, contract isolation and small sample baseline pass |
| November 21 | Two baseline main experiments, Dense and clause-level acceptance | At least one model's two dataset results can be aggregated |
| November 22 | Complete agreed matrix and ablation | Raw results, complete config and logs ready |
| November 23-24 | Retrieval analysis, cases, methods and results writing | Tables and report materials for Li, reproduction commands |
| November 25 | Final review and gap filling | Prediction quantities, citation positions, results reproducible |

## When Your Part Is Complete It Should Satisfy

- [ ] Raw data loadable, consistent with your component output format.
- [ ] Original text, evidence positions and retrieval scope for same contract verifiable.
- [ ] Direct, Standard RAG small sample and agreed full-scale experiments on two datasets all have real records.
- [ ] Ablation comparisons actually run, have raw output and parameter documentation.
- [ ] Recall@k uses fixed evaluation entry, contains correct context, doc_id and actual k.
- [ ] Wang can reuse retrieval interface, Li can align and aggregate results by example_id.
- [ ] Already delivered retrieval and baseline report materials, not just source code and checked lists.

`experiments/results/` and `experiments/logs/` excluded by .gitignore by default. During handoff explicitly commit streamlined results, release archive or shared storage location, and ensure teammates can access; cannot only save on personal computer.
