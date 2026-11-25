# Project Review and Yin Fengling Deliverable Acceptance

This is a historical review before fixes. Current implementation, acceptance status and teammate handoff instructions see [Retrieval and Baseline Development Handoff](RETRIEVAL_AND_BASELINES_HANDOFF.md). Diagnostic scripts will read historical source code from the Git version below, and will not re-judge already fixed code as original defects.

Review date: November 20, 2026. Repository version: `0166fe94c23f6e599ab54d98bbac6b49404046d8` (main).

**Conclusion: Yin Fengling's two test sets pass data content verification and can serve as inputs for subsequent development; her data loading, conversion process reproduction and evaluation modules have not yet reached full acceptance conditions. Qin Mengmeng can begin retrieval and baseline fixes without continuing to wait for data files. Project currently has framework and data, but does not yet have complete experiment results to answer research questions.**

## Project Overview and Current Division of Labor

Have read all 34 version-controlled Python files, 3 YAML configs, 6 root-directory documents, data delivery documentation and commit history; performed full structural check on two JSON test sets and line-by-line comparison with official release packages. All GitHub status PRs and issue list were empty at review time; remote branches only have main.

Project compares three contract analysis systems: Direct Prompting, Standard RAG, Agentic RAG. The third system workflow is query analysis → chunked retrieval → answer generation → item-by-item verification → re-retrieval and revision when necessary. Experiment goal is to compare accuracy, evidence recall and hallucination rate, and analyze failure types.

| Member | Current Responsibility | Main Interface |
|---|---|---|
| Yin Fengling | `data/` and `evaluation/` | Unified data format, evidence annotation, reliable metrics |
| Qin Mengmeng | `retrieval/` | Chunking, BM25/Dense/Hybrid, Direct and Standard RAG |
| Wang Zhengyao | `agent/` | Query analysis, generation, verification, revision and workflow |
| Li Xinru | `experiments/`, `ui/`, result analysis and final report | Experiment scheduling, aggregation, integration |

Division basis is `README.md`, `TEAM_TODO_LIST.md` and `PROJECT_SUMMARY.md` updated on November 18 at `e798106`. `COMPLETION_SUMMARY.md` and Author comments in some code still retain old division, cannot use this to reallocate responsibility, nor treat file header names as personal completion records.

Repository has 3 commits: initial framework `17a2fef`, division document adjustment `e798106`, Yin Fengling's data commit `0166fe9`. Last commit only adds two JSONs and data README; data processors and evaluation code all from initial commit. "PROJECT COMPLETE" and check marks in documents do not equal execution acceptance.

## Yin Fengling's Completion Status

### Delivered and Verified Data

| Check Item | ContractNLI | CUAD |
|---|---:|---:|
| Sample count and unique sample ID count | 2,091 | 1,020 |
| Contract count, by source document identifier | 123 | 102 |
| Labels | entailment 968; contradiction 220; neutral 903 | present 599; absent 421 |
| Samples with evidence | 1,188 | 599 |
| Total evidence ranges | 2,404 | 1,116 |
| Out-of-bounds or empty evidence ranges | 0 | 0 |
| Empty contract original text | 0 | 0 |
| Samples with `answer: null` | 2,091 | 421 |

Both files can be mapped to `LegalExample`. ContractNLI's original text, labels, original hypotheses, evidence ranges all consistent with official test full set. CUAD's questions, original text, labels, answers and evidence ranges all consistent with selected 10 categories in official test full set; no-answer evidence `null` and empty list treated with same meaning.

Source data verification uses [ContractNLI official release package](https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip) and [CUAD official repository release package](https://github.com/The-Atticus-Project/cuad/blob/main/data.zip). ContractNLI evidence index meaning also conforms to [official format specification](https://stanfordnlp.github.io/contract-nli/). Comparison results, input files and release package SHA256 saved in `audit_results_2026-11-20.json`.

ContractNLI has 123 document IDs, 122 distinct original texts, this is consistent with official release package and should not be misjudged as missing one contract. During retrieval should use stable document identifier; for duplicate original text and multiple questions per contract should avoid repeatedly building same index.

CUAD only covers 10 categories, 102 entries each: Document Name, Parties, Agreement Date, Effective Date, Expiration Date, Termination For Convenience, Notice Period To Terminate Renewal, Governing Law, Most Favored Nation, Non-Compete. It is not the complete 41-category CUAD evaluation set, nor the five categories written in experiment config like termination, liability, payment, etc.

ContractNLI has no free-text reference answers, CUAD's absent samples have no answers, both compatible with task definition; evaluation functions must explicitly handle these cases.

### Deliverables Still Needing Completion

| Original Work | Acceptance Judgment | Need to Complete |
|---|---|---|
| ContractNLI, CUAD test data | Completed, content verification passed | Retain current files as baseline |
| Data documentation and sources | Available | Add conversion commands, versions and verification methods |
| Standard package import and loading | Did not pass | Fix imports in three processors |
| Regenerate results from official source data | Missing matching process in repository | Submit actual conversion scripts; retain document IDs, hypothesis IDs, original questions and offsets |
| Evaluation metric verification and unit tests | Did not pass acceptance | Fix defects listed below and submit boundary tests |
| Hallucination detection, error analysis, metric reports | Only framework, missing verification results | Joint debug unified output, add execution records |
| Subsequent experiment assistance and evaluation report section | Pending experiment phase | Continue advancing per November 21-25 division |

Cannot determine from repository whether she configured environment or ran uncommitted tests on personal computer. Current conclusion addresses delivered materials, does not negate work not checked in; subsequent tasks in experiment phase also should not be judged overdue on November 20.

## Confirmed Issues and Responsibility Attribution

All locations correspond to original version at review time. P0 should be resolved before formal experiments, P1 should be resolved before corresponding deliverable acceptance.

| Priority | Issue and Evidence | Consequence | Primary Responsible Person |
|---|---|---|---|
| P0 | `data/contractnli_processor.py:19`, `cuad_processor.py:18`, `privacyqa_processor.py:17` use `from dataset_loader import ...` | Importing `data.dataset_loader` from project root immediately reports `ModuleNotFoundError`; reproducible even after datasets installed | Yin |
| P0 | `evaluation/metrics.py:279` calls `.lower()` on empty reference answer | Evaluation mixing CUAD present/absent triggers `AttributeError` | Yin |
| P0 | `experiments/run_experiments.py:203` and two other places omit `context` when constructing ground truth; `metrics.py:191` defaults to empty string | Irrelevant evidence will get Recall=1.0 due to empty string containment, may produce false perfect scores | Li fix parameter passing, Yin add validation |
| P0 | `retrieval/baseline_rag.py:188` indexes contracts from all samples aggregated, queries don't filter by document; Agent does the same | Questions from one contract can hit another contract. Real BM25 minimal reproduction already returns wrong documents | Qin, Wang sync fix Agent |
| P0 | Both baseline prompt/parser only support NLI three labels | CUAD present/absent have no correct task output; `Label: present` parsed as neutral. Cannot mechanically equate absent with NLI contradiction | Qin; Wang, Yin align |
| P0 | `retrieval/chunking.py:66` reorganizes whitespace, `:71` only searches position by first word | Chunk text inconsistent with original text slice, repeated words will also locate incorrectly. Will break citations and evidence evaluation | Qin |
| P0 | `agent/agentic_rag.py:163` outputs `verification`, `metrics.py:251` reads `verification_results` | Existing contradicted declarations may still have no hallucination rate; real Agent structure input returns null | Wang, Yin, Li |
| P1 | `metrics.py:232` uses cumulative hits to judge whether current sample stops | Constructed case where both samples retrieve hits, Hit Rate gets 0.5 instead of 1.0 | Yin |
| P1 | `metrics.py:327` reverses label mapping with duplicate values | NLI per-category report shows present/absent/neutral, category names incorrect | Yin |
| P1 | Both baseline CLIs hardcode `examples[:10]`, have no `--sample_size` | Small sample parameter in TODO not usable; "full test set" command still only runs 10 entries | Qin |
| P1 | `baseline_direct.py:71` only retains first 2,000 characters of original text | Later evidence completely invisible; need to clarify model token budget and truncation rules, otherwise comparison contains input budget differences | Qin, align with Li |
| P1 | `baseline_rag.py:48` and `:51` pass same kwargs to two types of components | Setting chunk_size will pass to BM25 constructor and report TypeError; config cannot extend normally | Qin |
| P1 | `baseline_rag.py:115` results only save evidence text and score | Lost document ID, chunk ID and start/end positions, cannot reliably accept citations and retrieval scope | Qin |
| P1 | `experiments/run_experiments.py:266` ablation function only outputs logs; `:381` dataset, sample_size parsed but not applied | Configured ablation but no results; so-called small sample may still enter full experiment | Li, Qin provide baseline config support |

Fixed-length chunking original text consistency check uses default 512/128 parameters, processes each distinct original text once: ContractNLI 598 of 601 chunks do not satisfy `context[start:end] == chunk['text']`; CUAD 1,924 of 1,943 chunks do not satisfy. Constructed text `A A A A A A`, chunk_size=2, overlap=0, returns start points `[0, 2, 4]`, should be `[0, 4, 8]`.

Other boundaries needing inclusion in subsequent acceptance: ClauseLevel by default discards short text under 50 words; chunk_size/overlap not validated, illegal step size may cause loops; BM25 top_k=0 returns entire index; Hybrid BM25 normalization does not prevent maximum score being zero. Last two cases top_k=0 already dynamically reproduced, zero score normalization and illegal step size from code inspection.

## Cross-Module Experiment Validity Issues

Direct and Standard RAG have no verification results, currently cannot directly calculate their claimed hallucination rates. Should apply same evaluation process to final answers from three systems, and unify claim extraction, evidence sources and judgment criteria. Agent built-in verification can be used for revision, but cross-system conclusions cannot rely solely on its internal judgment.

Agent after revision still returns initial label, and saves initial answer's verification results; no re-verification of final answer. Token aggregation omits consumption of query rewriting and verification steps. Should be fixed by Wang, then align statistical caliber with Yin and Li.

Hallucination detector reads evidence from claim's `evidence`, while Verifier outputs `supporting_evidence`, error cases also don't retain clause_type, causing evidence and per-clause analysis loss. `error_analysis.py` saved context truncates to 500 characters first then counts length, cannot use this to analyze impact of original contract length on errors. These are joint debugged by Yin and Li.

`utils/llm_interface.py` API generation still raises `NotImplementedError`, Mac MPS branch will fall back to CPU, temperature=0 will be replaced by `or`. GPU deployment documentation's API example returns string, incompatible with project-required `{text, tokens, latency}`; its `--use_api` also not implemented. This review did not verify price, quota and duration estimates in documentation, should not be directly treated as resource commitments.

## Review Method and Reproduction

This review performed syntax compilation check on 34 original Python files, structural check on 3,111 data entries, line-by-line comparison of two official test sources, and actually ran minimal reproductions of metrics, chunking and BM25. Full results see [audit output](audit_results_2026-11-20.json), diagnostic scripts see [audit_project.py](../scripts/audit_project.py).

To avoid downloading large models, logic diagnosis loads unmodified class definitions from original file AST, bypassing package initialization and heavy model imports; LLM replaced with fixed return values. Data package import errors reproduced in separate process with real package imports. This time did not run real Llama/Mistral inference, Dense/Hybrid models or complete experiment suite, therefore do not claim end-to-end has passed.

Using independent temporary Python 3.11 environment, not changing machine default Python:

```bash
uv venv --python python3.11 /tmp/legal-hallucination-audit-20261120
uv pip install --python /tmp/legal-hallucination-audit-20261120/bin/python numpy scikit-learn rank-bm25 datasets
/tmp/legal-hallucination-audit-20261120/bin/python scripts/audit_project.py --output docs/audit_results_local.json
```

Official source comparison additionally needs to name above release packages as `contract-nli.zip` and `cuad-data.zip` and place in independent directory, add `--source-dir /path/to/releases`. This time's downloaded copies located at `/tmp/legal-hallucination-audit-20261120/`, did not overwrite post-processed data in repository.

This review adds audit documentation, diagnostic scripts and output, did not modify original implementation, did not run paid APIs, did not commit or push to GitHub. Subsequent fix and experiment arrangements see [Qin Mengmeng Responsibilities and Acceptance Checklist](QIN_MENGMENG_TASKS.md).
