# Retrieval and Baseline Development Handoff

Baseline owner is Qin Mengmeng, baseline delivered on November 20, 2026; this document was supplemented on November 22, 2026 with Li Xinru's Part 4 handoff. This document is for team members to reproduce retrieval and baseline experiments, integrate Agent, and compile reports.

**2026-11-22, Li Xinru's Part 4 continuation:** Generic runner has integrated task interface, actual ablation, item-by-item checkpoints, and strict recovery; `retrieval` baseline export and Agent type import received minimal integration fixes. Newly ran 36 groups of full retrieval and dual-model fixed small-sample Agent ablation, archiving main experiments for separate verification. Execution method, applicable scope, and remaining gaps see [Part 4 Experiment Handoff](PART4_EXPERIMENTS.md), actual values and original files see [Current Results](validation/part4_2026-11-22/RESULTS.md). This iteration does not continue writing the course report.

- [Retrieval and Baseline Full Results](#retrieval-and-baseline-full-results)
- [Delivered Implementation and Interface](#delivered-implementation-and-interface)
- [Reproduction Steps](#reproduction-steps)
- [Development Validation and Failure Cases](#development-validation-and-failure-cases)
- [Team Member Follow-up Items](#team-member-follow-up-items)
- [GitHub Delivery Files](#github-delivery-files)

**2026-11-24, Part 4 Analysis Continuation:** Complete server item-by-item output obtained, verifying 18,666 predictions and 37,834 calls, completing full error statistics, pre- and post-revision label comparison, 20 traceable cases, and computable metrics. See [Comprehensive Analysis and Criteria Verification](validation/part4_2026-11-24/README.md). This iteration only adds analysis delivery; independent final answer hallucination assessment and manual annotation not yet complete.

## Retrieval and Baseline Full Results

Mistral and Llama 3 AutoDL full runs have all completed. Each model runs three methods on the same 2,091 ContractNLI and 1,020 CUAD test samples. Two baselines total 12,444 predictions, Verified Agentic RAG comparison totals 6,222, grand total **18,666**. Two SQLite databases separately verified 9,333 records each and database integrity. [Test Data Description](../data/processed/README.md)

Corresponding model versions, test file SHA256, environment, and per-group statistics are saved in two official summaries.

| Model | Fixed revision | Format Pass | Label Correct | Full Summary |
| --- | --- | ---: | ---: | --- |
| Mistral 7B Instruct v0.2 | `63a8b081895390a26e140280378bc85ec8bce07a` | 7,678/9,333 | 4,599/9,333 | [Mistral](validation/mistral_autodl_full_summary.json) |
| Meta Llama 3 8B Instruct | `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` | 8,866/9,333 | 5,157/9,333 | [Llama 3](validation/llama3_autodl_full_summary.json) |

Each model uses one RTX 6000D 84 GB, vLLM 0.10.2, unquantized BF16, temperature 0, seed 42, parallelism 8. Input maximum 4,096 tokens, single task answer maximum 512 tokens. Standard RAG uses BM25 top 5 within current contract, fixed chunking of 512 words with 128-word overlap. Official summaries taken from AutoDL full runs. Locked trial runs, Alibaba Cloud API connectivity tests, and 83 Mistral predictions during local pauses are separately archived.

Format pass requires output to be parsable and label to belong to corresponding task. Label accuracy uses all predictions within group as denominator, format failures count as errors. Macro-F1 is equal-weight average across valid classes for the task. Per-item average time includes steps actually executed in corresponding workflow.

| Dataset | Model | Method | Format Pass Rate | Label Accuracy | Macro-F1 | Avg Time (sec) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ContractNLI | Mistral | Direct Prompting | 1,750/2,091 (83.69%) | 908/2,091 (43.42%) | 0.3253 | 1.93 |
| ContractNLI | Mistral | Standard RAG | 1,718/2,091 (82.16%) | 876/2,091 (41.89%) | 0.2707 | 2.52 |
| ContractNLI | Mistral | Verified Agentic RAG | 1,577/2,091 (75.42%) | 775/2,091 (37.06%) | 0.2305 | 10.82 |
| ContractNLI | Llama 3 | Direct Prompting | 2,091/2,091 (100.00%) | 983/2,091 (47.01%) | 0.4491 | 1.45 |
| ContractNLI | Llama 3 | Standard RAG | 2,087/2,091 (99.81%) | 889/2,091 (42.52%) | 0.4271 | 1.85 |
| ContractNLI | Llama 3 | Verified Agentic RAG | 2,051/2,091 (98.09%) | 1,161/2,091 (55.52%) | 0.5283 | 9.94 |
| CUAD | Mistral | Direct Prompting | 877/1,020 (85.98%) | 684/1,020 (67.06%) | 0.7214 | 1.51 |
| CUAD | Mistral | Standard RAG | 928/1,020 (90.98%) | 711/1,020 (69.71%) | 0.7284 | 1.39 |
| CUAD | Mistral | Verified Agentic RAG | 828/1,020 (81.18%) | 645/1,020 (63.24%) | 0.6971 | 6.32 |
| CUAD | Llama 3 | Direct Prompting | 966/1,020 (94.71%) | 791/1,020 (77.55%) | 0.7851 | 1.28 |
| CUAD | Llama 3 | Standard RAG | 866/1,020 (84.90%) | 700/1,020 (68.63%) | 0.7346 | 1.29 |
| CUAD | Llama 3 | Verified Agentic RAG | 805/1,020 (78.92%) | 633/1,020 (62.06%) | 0.6834 | 5.89 |

- **Two baselines.** On ContractNLI, Standard RAG compared to Direct Prompting has accuracy drops of 1.53 and 4.50 percentage points on Mistral and Llama 3 respectively. On CUAD, Mistral improves by 2.65 percentage points, Llama 3 drops by 8.92 percentage points. Differences calculated from unrounded accuracy.
- **Agent comparison.** On Llama 3's ContractNLI, Verified Agentic RAG reaches 55.52%, 13.01 percentage points higher than Standard RAG, with per-item average time rising from 1.85 to 9.94 seconds. In the other three model-dataset combinations, Agent accuracy is lower than Standard RAG.
- **Format and evidence.** ContractNLI format failures are all missing or invalid task labels: Mistral has 1,228, Llama 3 has 44. Llama 3 on CUAD has 220 that did not return a complete single JSON object, 197 with citation arrays not meeting positive integer numbering requirements. Of CUAD's 599 gold-standard positive examples, predictions simultaneously meeting correct label, verbatim extraction, and not triggering citation overflow or evidence errors: Mistral Standard RAG has 179, Llama 3 Direct Prompting has 396.

Full JSON also includes confusion matrices, format failure reasons, citation overflow, and gold evidence coverage in actual prompts. Cross-method hallucination rates still require unified answer support assessment. Item-by-item output and generation traces exported from AutoDL on 2026-11-24 and completed consistency verification; team maintains original ZIP, this submission provides [analysis results, item-by-item statistics, and cases](validation/part4_2026-11-24/README.md). Report-cited values can use verified task metrics; independent hallucination rate remains untested.

## Delivered Implementation and Interface

| Module | Behavior Team Members Need to Know |
| --- | --- |
| Chunking and Retrieval | Fixed window and clause-level chunking preserve original text and character positions. BM25, Dense, Hybrid support stable top-k ranking. Low-level retrieval calls must restrict `doc_id` or only index current contract. |
| Direct Prompting | Prefixes current contract to model within token budget. Generates, parses, and records truncation by dataset. |
| Standard RAG | After indexing multiple contracts, locates via `doc_id`; queries only retrieve from that contract. Batch processing reuses index by contract, returns in input order. |
| Verified Agentic RAG | Evaluation uses path of passed `dataset`. When initial answer format error or evidence check fails, preserves trace and skips model verification. When evidence insufficient, can re-retrieve within same contract, maximum one revision, scores by final explicit label. |
| Independent Evaluation Entry | Retrieval ablation and official full runs maintain independent runners; generic `experiments/run_experiments.py` supports task scoring, sample selection, actual ablation, and strict checkpoint recovery. Independent hallucination rate still awaits manual review. |

`chunk(text, metadata=None)` returns chunks with `text`, `start`, `end`, `chunk_id`. `start` and `end` are half-open character intervals in original text, satisfying `source[start:end] == chunk["text"]`. `chunk_id` together with `doc_id` identifies chunk. Retriever interface is `index(chunks)` and `retrieve(query, top_k=5, doc_id=None)`, returning `(chunk, score)` list. Chunk budget counted by words, 4,096 input budget counted by generation model tokenizer.

```python
from retrieval.baseline_rag import StandardRAGBaseline

rag = StandardRAGBaseline(
    llm=llm,
    dataset="contractnli",
    retriever_type="bm25",
    chunking_strategy="fixed_length",
    top_k=5,
    chunker_kwargs={"chunk_size": 512, "overlap": 128},
    retriever_kwargs={"k1": 1.5, "b": 0.75},
    max_input_tokens=4096,
)
rag.index_documents([contract_a, contract_b], doc_ids=["A", "B"])
result = rag.answer(query, doc_id="A")
```

Also available via `rag.answer(query, context=contract_text)` to directly specify contract. Component integration can use `StandardRAGBaseline(llm=None).retrieve_evidence(query, context=contract_text, doc_id=source_doc_id)` to retrieve separately. Agent entry for three-method comparison is `AgenticRAG.answer(query, context, dataset="contractnli", max_input_tokens=4096)`; for CUAD pass `dataset="cuad"`. Batch processing preserves sample ID and `metadata` for convenient downstream alignment and source location.

**Output and Scoring Conventions**

- ContractNLI uses `entailment`, `contradiction`, `neutral`; prompts include `Label:` and `Answer:` examples. CUAD uses `present`, `absent`, outputs JSON. Positive examples require extracting original text; negative examples have empty answer and extraction lists.
- Format errors preserve `raw_response` and `parse_error`, classified as `label=invalid`. CUAD's `answer_is_extractive` records whether extraction fragments appear verbatim in model input. Agent additionally records `grounding_error`. When final format or evidence check fails, Agent leaves display-use `final_answer` empty; original label and original response remain auditable.
- `retrieved_chunks` saves original top-k results. Computing retrieval Recall uses this field. Auditing actual content seen by model uses budget-trimmed `prompt_evidence`. Citation numbers correspond to evidence chunks actually used in that call.
- `prompt` saves task content, `model_prompt` saves actual input after applying chat template. Two baselines' `tokens` records generation token count, input token count in `input_budget`. Agent's stage calls also see `generation_trace`. When Python API configures `max_input_tokens`, must provide tokenizer or have model adapter provide it.

## Reproduction Steps

### Local Testing and Two Baselines

Execute in repository root. Minimal environment for local logic, BM25, and dry run. Full environment and Dense encoder or generation model weights install as needed.

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements-retrieval.txt
python -m pytest -q
python -m retrieval.baseline_direct --dataset contractnli --sample_size 10 --dry_run
python -m retrieval.baseline_rag --dataset cuad --sample_size 10 --retriever bm25 --top_k 5 --dry_run
```

For generation, additionally install `requirements.txt`. Following examples require CUDA and readable model weights. Omitting `--sample_size` reads complete split. Official comparison uses subsequent fixed-revision full entry.

```bash
python -m retrieval.baseline_direct --model mistral --dataset contractnli \
  --sample_size 10 --device cuda --temperature 0 --output experiments/results/direct_mistral_contractnli.json
python -m retrieval.baseline_rag --model mistral --dataset contractnli \
  --sample_size 10 --device cuda --temperature 0 --output experiments/results/rag_mistral_contractnli.json
```

Independent baseline CLI outputs final JSON, item-by-item JSONL, and `.meta.json`. Protects existing files by default. JSONL for troubleshooting. Full runner saves SQLite checkpoints, supports resume from breakpoint. Check `validation_status` and `parse_error_count` when summarizing.

### Retrieval Ablation

```bash
python -m retrieval.evaluate_retrieval --datasets contractnli cuad \
  --retrievers bm25 dense hybrid --chunkings fixed_length clause_level \
  --top_ks 3 5 10 --output experiments/results/retrieval_ablation.json
```

36 groups from two datasets, three retrievers, two chunking strategies, and three k values. When running only BM25, minimal environment can be used. Only samples with gold evidence enter evidence metric denominator: ContractNLI 1,188 items, CUAD 599 items. Fixed chunking is 512 words with 128-word overlap; clause-level chunking maximum 512 words, merge target 50 words. Dense uses all-MiniLM-L6-v2, encoding limit 256 subword tokens. Hybrid has 0.5 weight on each side. Detailed criteria, Recall comparison, and retrieval failure examples see [Retrieval Ablation Report](validation/RETRIEVAL_RESULTS.md) and [Complete Results](validation/retrieval_ablation_2026-11-20.json).

### AutoDL Dual-Model Full Run

Run in frozen directory containing source code, configs, and two repository test files. Following dependencies based on Linux x86-64, Python 3.12, PyTorch 2.8.0+cu128, CUDA 12.8 AutoDL image. Completed two-group run parameters see full JSON's `settings`.

```bash
python3.12 -m venv --system-site-packages venv
source venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  'vllm==0.10.2' 'transformers==4.56.2' 'huggingface-hub==0.34.4' \
  'openai==1.109.1' 'rank-bm25==0.2.2' 'pyyaml==6.0.3' 'requests==2.32.5'
```

Launch script loads fixed revision. When downloading Llama 3 directly from Hugging Face in new environment, execute `hf auth login` with account that has opened access to that model. When complete cache exists, can add `HF_HUB_OFFLINE=1` to launch command. Llama 3 weights used in official experiments from ModelScope fixed version; before loading verified four weight shards and six config, tokenizer, and index files. Alibaba Cloud PAI only used for API connectivity testing. [Image Verification Script](../scripts/fetch_verified_llama3.py)

```bash
bash scripts/run_autodl_mistral.sh pilot mistral
bash scripts/run_autodl_mistral.sh full mistral
bash scripts/run_autodl_mistral.sh pilot llama3
bash scripts/run_autodl_mistral.sh full llama3
```

Run in model order on same instance. Each model first verifies 30-item pilot, then launches that model's full run. Each run saves `progress.json`, `predictions.sqlite3`, `vllm.log` in `experiments/results/<model>_autodl_<phase>/`, generates `summary.json` upon completion. Accept with `status=completed`, six-group sample counts, and `PRAGMA quick_check`. SQLite deduplicates by `(dataset, method, example_id)`. Resume must reuse same output directory, frozen code, model, and config. When needing to resume official checkpoint, use that run's own `source_sha256` as reference.

## Development Validation and Failure Cases

- **Part 4 Integration Validation (2026-11-22).** This round complete test `151 passed`, including existing retrieval tests in real dependency environment and new orchestration checkpoint, sample selection, ablation switches, error denominator, token accounting, and blind review export tests. [This Round JUnit](validation/part4_2026-11-22/tests.xml). These automated tests and real model outputs are recorded separately; real retrieval re-runs and Agent pilot runs detailed in Part 4 results table.
- **Automated Acceptance.** Re-ran full environment during handoff organization, 144 items passed. Minimal environment 123 items passed, 4 skipped. Both baselines' dry runs successfully selected 10 items. Records in repository correspond to earlier format fix phase. [JUnit Summary](validation/test_summary.json) · [Environment Record](validation/environment.json)
- **Qwen Format Investigation.** First round only passed 1/8. After re-reading original responses with new parser, passed 7/8; re-ran with fixed prompts passed 8/8. In old results, 4 CUAD items had explanations before complete JSON, 2 ContractNLI Standard RAG used objects to represent citations, 1 ContractNLI Direct Prompting generated unescaped quotes within JSON. These 7 all did not exhaust generation budget. Quote case improved with two-line prompt with examples; first two types handled by compatible parsing and citation normalization. [Retest Record](validation/qwen_format_retest)
- **Official Model First Pilot.** Mistral completed 20 baseline predictions locally with MLX, of which 11 format pass, 6 label correct. Llama 3's Hugging Face weights at that time were access-restricted, produced no predictions. Effective dual-model comparison later completed on AutoDL. [Mistral First Round Results](validation/official_baseline_pilot/mistral.json) · [Llama 3 First Round Record](validation/official_baseline_pilot/llama3.json)
- **CUAD Protocol Revision.** Development samples from CUAD official training split, isolated from test set by contract title and text hash. Final prompt removed lawyer review wording from question stem, requires structured extraction. Generator stops after outputting first complete JSON object. On locked 4 CUAD test samples, Direct Prompting, Standard RAG, and Verified Agentic RAG all format pass 4/4, label correct counts 2/4, 2/4, 3/4 respectively. [Development Sample Source](validation/cuad_output_development/source.json) · [Development Retest](validation/cuad_output_development/final/mistral.json) · [Locked Test Retest](validation/cuad_output_development/held_out/mistral.json)
- **AutoDL Locked Pilot.** Two models share 6 ContractNLI and 4 CUAD for three-method check. Mistral format 27/30, label correct 15/30; Llama 3 format 30/30, label correct 21/30. [Mistral Pilot](validation/mistral_autodl_pilot_summary.json) · [Llama 3 Pilot](validation/llama3_autodl_pilot_summary.json)

### Traceable Failure Cases

| Sample or Contract | Observation |
| --- | --- |
| `contractnli_test_523_nda-8` | Mistral all three methods output `Label: true`, natural stop. Invalid label causes both format and accuracy to count as errors. |
| `contractnli_test_41_nda-1` | Annotated as `contradiction`. Agent's initial and post-revision prompts both covered annotated evidence, final still judged as `neutral`. |
| CUAD PACIRA licensing distribution agreement | Agent retest after re-retrieval within same contract covered gold renewal notice clause, final still judged as `absent`. Standard RAG covered gold from first prompt, label likewise incorrect. |
| CUAD HPIL cooperation agreement | Early Agent judged date and label correctly but rewrote original sentence, verbatim check failed. Final retest Agent copied original date, Standard RAG label correct but citation number out of bounds. |
| CUAD N2K sponsorship agreement | Early Direct Prompting exhausted 512 tokens without closing JSON. |

Case sources are [Early Mistral Three-Method Item-by-Item Results](validation/verified_agentic_rag_pilot/mistral.json) and [Locked Test Retest](validation/cuad_output_development/held_out/mistral.json). When citing in report, note pilot phase used.

## Team Member Follow-up Items

| Owner | Handoff Items |
| --- | --- |
| Yin Fengling | Complete reproducible conversion process from official raw data to current JSON. Shared evaluator distinguishes labels, empty answers, and `parse_error` for two datasets; statistics evidence hits by actual retrieval k. |
| Wang Zhengyao | Demo entry explicitly passes `dataset`, displays `final_answer`, verification results, and interception reasons. Post-revision currently only has deterministic format and evidence checks; if adding second model verification, needs separate recording. |
| Li Xinru | Has integrated generic runner, verified archived summaries, and provided new experiment results and blind review export entry. Server item-by-item traces obtained, comprehensive analysis completed; subsequently needs to confirm independent answer support method and manual annotation division of labor with evaluation lead; new Agent ablation is small-sample, not equivalent to full component ablation. Official UI entry and acceptance see [UI Handoff](UI_HANDOFF.md). |
| Qin Mengmeng | Assist extracting item-by-item execution traces and analyzing false negatives. If continuing Dense or Hybrid generation comparison, keep model, test samples, and input budget fixed; use dev set for parameter selection. |

## GitHub Delivery Files

| Content | Repository Location |
| --- | --- |
| Chunking, retrievers, task formats, two baselines, and independent ablation | [`retrieval/`](../retrieval/) · [Retrieval Config](../configs/retrieval_config.yaml) |
| Agent implementation connecting three-method comparison | [`agent/`](../agent/) |
| Dual-model full run, locked pilot, vLLM interface, and CUAD dev set construction | [`experiments/`](../experiments/) · [AutoDL Launch Script](../scripts/run_autodl_mistral.sh) |
| Data import, lazy dependencies, and model budget related integration changes | [`data/`](../data/) · [`utils/`](../utils/) · [Full Dependencies](../requirements.txt) · [Minimal Dependencies](../requirements-retrieval.txt) |
| Automated test cases and official, diagnostic results | [`tests/`](../tests/) · [`docs/validation/`](validation/) · [Validation Summary Script](../scripts/update_validation_report.py) |
| Work scope and history verification | [Original Responsibilities and Acceptance Checklist](QIN_MENGMENG_TASKS.md) · [Pre-Fix Project Review](PROJECT_REVIEW_2026-11-20.md) |

Two original test JSONs already tracked in repository, this round keeps as-is. Delivers above tests and summarized experimental artifacts with code. Model weights, virtual environments, credentials, and item-by-item run artifacts under `experiments/results/` remain in work environment. If subsequently modifying interfaces, data descriptions, or experiment results, synchronously update this document and corresponding validation files.

## Final Delivery Review

Old evaluation entry scores ContractNLI and CUAD separately. Invalid labels and format failures count as errors; retrieval recall and hits use original text intervals and unified k. Initial answer verifier diagnostics listed separately; semantic metrics remain untested when independent annotation absent. Use two compressed item-by-item exports to verify 12 groups' accuracy, Macro-F1, and format counts, and recalculate all 37,834 calls' 87,512,435 tokens. [Experiment Review and Launch Instructions](FINAL_SUBMISSION.md)

[Final Evaluation Fix Record](FINAL_EVALUATION_FIXES.md) summarizes four fixed counterexamples, report corrections, 188 regression tests, and real Q&A acceptance.
