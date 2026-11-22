# LLM-Legal Hallucination Auditing Agent

An Agentic RAG system designed to audit and reduce hallucinations in LLM-generated responses for legal contract analysis.

## Project Overview

This project implements an agentic workflow that:
1. Analyzes legal queries to understand intent
2. Retrieves relevant evidence from contracts
3. Generates initial answers grounded in evidence
4. Verifies claims against retrieved evidence
5. Re-retrieves and revises answers when hallucinations are detected

## Team Members

- **Yin Fengling (3036807375)**: Dataset preprocessing & evaluation metrics
- **Qin Mengmeng (3036843769)**: Retrieval system & baseline implementations
- **Wang Zhengyao (3036795675)**: Agent components (Query Analyzer, Verifier, Reviser)
- **Li Xinru (3036807959)**: Experiment orchestration, analysis & reporting

## Project Structure

```
legal-hallucination-auditing/
├── data/                       # Dataset processing module
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Processed test datasets
│   ├── dataset_loader.py      # Dataset loading utilities
│   ├── contractnli_processor.py
│   ├── cuad_processor.py
│   └── privacyqa_processor.py (optional)
├── retrieval/                  # Retrieval & baseline systems
│   ├── retrievers/            # BM25 and dense retrievers
│   ├── chunking.py            # Document chunking strategies
│   ├── baseline_direct.py     # Direct prompting baseline
│   └── baseline_rag.py        # Standard RAG baseline
├── agent/                      # Agentic RAG components
│   ├── query_analyzer/        # Query classification & analysis
│   ├── generator.py           # Answer generator
│   ├── verifier/              # Evidence verification
│   └── reviser/               # Answer revision
├── evaluation/                 # Evaluation metrics
│   ├── metrics.py             # All evaluation metrics
│   ├── hallucination_detector.py
│   └── error_analysis.py
├── experiments/                # Experiment orchestration
│   ├── run_experiments.py     # Main experiment runner
│   ├── results/               # Experiment outputs
│   └── logs/                  # Execution logs
├── configs/                    # Configuration files
│   ├── model_config.yaml
│   ├── retrieval_config.yaml
│   └── experiment_config.yaml
├── utils/                      # Shared utilities
│   ├── llm_interface.py       # LLM API wrapper
│   ├── logger.py
│   └── helpers.py
├── ui/                         # Optional UI prototype
│   └── app.py
├── outputs/                    # Results and analysis
├── requirements.txt
└── setup.py
```

## Retrieval and baseline development

The current retrieval and baseline interfaces, verified commands, test evidence,
and remaining integration tasks are documented in
[Retrieval and baselines handoff (Chinese)](docs/RETRIEVAL_AND_BASELINES_HANDOFF.md).
The common experiment runner now supports task-aware scoring, frozen cohorts,
component ablations, and checkpoint resume. Retrieval-only development and tests need
only `requirements-retrieval.txt`; Dense retrieval and model generation need the
full model dependencies.

See [Qin Mengmeng's full-test results](docs/RETRIEVAL_AND_BASELINES_HANDOFF.md#检索与基线全量结果)
for Mistral and Llama 3 on ContractNLI and CUAD, and the
[retrieval ablation results](docs/validation/RETRIEVAL_RESULTS.md) for the
retriever, chunking, and top-k comparisons.

Li Xinru's [Part 4 experiment handoff](docs/PART4_EXPERIMENTS.md) contains
reproduction commands and the [new run results](docs/validation/part4_2026-11-22/RESULTS.md).
The new runs separate 36 full-test retrieval configurations from a paired
10-example Agent component pilot on each formal model. Archived full-test
generation results are audited separately; independent hallucination scores
remain unmeasured.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

## Quick Start

### 1. Prepare Datasets

```bash
# Download and process ContractNLI
python data/contractnli_processor.py --download --process

# Download and process CUAD
python data/cuad_processor.py --download --process
```

### 2. Run Baselines

```bash
# Direct prompting baseline
python retrieval/baseline_direct.py --model llama3 --dataset contractnli

# Standard RAG baseline
python retrieval/baseline_rag.py --model llama3 --retriever bm25 --top_k 5
```

### 3. Run Agentic RAG

```bash
# Full agentic workflow
python experiments/run_experiments.py --mode agent --model llama3
```

### 4. Run All Experiments

```bash
# Run complete experiment suite
python experiments/run_experiments.py --mode all --models llama3,mistral
```

## Research Questions

**RQ1**: Does RAG reduce hallucinations in legal question answering compared to direct prompting?

**RQ2**: Does agentic verification and revision improve upon standard RAG?

**RQ3**: What types of contract clauses and reasoning are most prone to hallucinations?

## Evaluation Metrics

### Primary Metrics
- **Label Accuracy**: Classification correctness (entailment/contradiction/neutral)
- **Macro-F1**: Balanced F1 across all classes
- **Evidence Recall@k**: Proportion of gold evidence spans retrieved
- **Hallucination Rate**: Percentage of unsupported/contradicted claims

### Secondary Metrics
- **Answer Completeness**: Coverage of required information
- **Abstention Precision**: Accuracy when model abstains from answering
- **Latency**: End-to-end response time
- **Token Cost**: Total tokens consumed

## Datasets

### ContractNLI
- Task: Natural Language Inference on contracts
- Labels: Entailment, Contradiction, Neutral
- Evaluation: Test set only

### CUAD (Contract Understanding Atticus Dataset)
- Task: Clause extraction and retrieval
- 41 clause types from legal contracts
- Evaluation: Representative clause categories

### PrivacyQA (Optional)
- Cross-domain stress test
- Privacy policy question answering

## Reproducibility

All experiments are fully reproducible with:
- Fixed random seeds
- Versioned model checkpoints
- Documented hyperparameters
- Saved prompts and configurations

See `experiments/experiment_config.yaml` for complete settings.

## Models

- **Llama3** (primary)
- **Mistral** (comparison)

Both models accessed via HuggingFace Transformers or API endpoints.

## Ablation Studies

- Retrieval method: BM25 vs Dense retrieval
- Chunking strategy: Fixed-length vs Clause-level
- Top-k values: 3, 5, 10
- Model comparison: Llama3 vs Mistral

## Output Format

All system outputs include:
- Generated answer
- Evidence citations with span references
- Hallucination risk labels
- Verification details (for Agentic RAG)

## License

Academic project for LLM course, November 2026.

## Deadline

November 25, 2026

## Contact

For questions, contact team members via their student IDs.
