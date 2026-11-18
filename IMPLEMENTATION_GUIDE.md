# Legal Hallucination Auditing Agent - Implementation Guide

## Quick Start Guide

### 1. Installation

```bash
cd legal-hallucination-auditing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 2. Download NLTK Data

```python
python -c "import nltk; nltk.download('punkt')"
```

### 3. Prepare Datasets

#### ContractNLI
```bash
python data/contractnli_processor.py --download --process --split test
```

#### CUAD
```bash
python data/cuad_processor.py --download --process --split test
```

### 4. Run Experiments

#### Run all experiments
```bash
python experiments/run_experiments.py --mode all
```

#### Run specific baseline
```bash
# Direct prompting
python retrieval/baseline_direct.py --model llama3 --dataset contractnli

# Standard RAG
python retrieval/baseline_rag.py --model llama3 --retriever bm25 --top_k 5

# Agentic RAG
python agent/agentic_rag.py --model llama3
```

#### Run with specific model
```bash
python experiments/run_experiments.py --models llama3,mistral
```

### 5. Launch UI (Optional)

```bash
python ui/app.py --model llama3 --port 7860
```

## Project Workflow

### Phase 1: Data Preparation (Days 1-2)
**Owner: Li Xinru**

1. Download datasets:
```bash
python data/contractnli_processor.py --download
python data/cuad_processor.py --download
```

2. Process and validate:
```bash
python data/contractnli_processor.py --process --split test
python data/cuad_processor.py --process --split test
```

3. Verify processed data:
```python
from data.dataset_loader import get_dataset_statistics
from data.contractnli_processor import ContractNLIProcessor

processor = ContractNLIProcessor()
examples = processor.load(split='test')
stats = get_dataset_statistics(examples)
print(stats)
```

### Phase 2: Baseline Implementation (Days 1-3)
**Owner: Qin Mengmeng**

1. Test retrieval components:
```python
from retrieval.chunking import FixedLengthChunker
from retrieval.retrievers.retriever import BM25Retriever

chunker = FixedLengthChunker(chunk_size=512)
retriever = BM25Retriever()

chunks = chunker.chunk(contract_text)
retriever.index(chunks)
results = retriever.retrieve(query, top_k=5)
```

2. Run baselines:
```bash
# Test on small sample first
python retrieval/baseline_direct.py --model llama3 --dataset contractnli
python retrieval/baseline_rag.py --model llama3 --retriever bm25
```

### Phase 3: Agent Implementation (Days 1-3)
**Owner: Yin Fengling**

1. Test individual components:
```python
from agent.query_analyzer.analyzer import QueryAnalyzer
from agent.generator import AnswerGenerator
from agent.verifier.verifier import EvidenceVerifier
from agent.reviser.reviser import AnswerReviser

# Test query analyzer
analyzer = QueryAnalyzer()
result = analyzer.analyze("Can this contract be terminated?")
print(result)

# Test full pipeline
from agent.agentic_rag import AgenticRAG
agent = AgenticRAG(llm, enable_verification=True, enable_revision=True)
```

2. Run agentic system:
```bash
python agent/agentic_rag.py --model llama3 --dataset contractnli
```

### Phase 4: Experiments & Evaluation (Days 4-5)
**Owner: Wang Zhengyao**

1. Run complete experiments:
```bash
python experiments/run_experiments.py --mode all --models llama3,mistral
```

2. Analyze results:
```python
from evaluation.metrics import MetricsCalculator
from evaluation.hallucination_detector import HallucinationDetector
from evaluation.error_analysis import ErrorAnalyzer

# Load results
results = load_json('experiments/results/complete_results.json')

# Analyze hallucinations
detector = HallucinationDetector()
analysis = detector.analyze_failure_patterns()
```

### Phase 5: Analysis & Reporting (Days 6-7)

1. Generate comparison reports:
```python
from evaluation.error_analysis import compare_systems

comparison = compare_systems(
    direct_predictions,
    agentic_predictions,
    ground_truth,
    "Direct Prompting",
    "Agentic RAG"
)
```

2. Create visualizations and final report

## Configuration

### Model Configuration (`configs/model_config.yaml`)
```yaml
model:
  llama3:
    name: "meta-llama/Meta-Llama-3-8B-Instruct"
    max_tokens: 512
    temperature: 0.7
```

### Retrieval Configuration (`configs/retrieval_config.yaml`)
```yaml
retrievers:
  bm25:
    k1: 1.5
    b: 0.75

chunking:
  fixed_length:
    chunk_size: 512
    overlap: 128
```

### Experiment Configuration (`configs/experiment_config.yaml`)
```yaml
modes:
  - direct_prompting
  - standard_rag
  - agentic_rag

models:
  - llama3
  - mistral
```

## Evaluation Metrics

### Primary Metrics
- **Label Accuracy**: Classification correctness
- **Macro-F1**: Balanced F1 score
- **Evidence Recall@k**: Retrieval quality
- **Hallucination Rate**: Unsupported/contradicted claims

### Secondary Metrics
- **Answer Completeness**: Coverage of ground truth
- **Abstention Precision**: Accuracy when abstaining
- **Latency**: Response time
- **Token Cost**: Total tokens used

## Research Questions

### RQ1: Does RAG reduce hallucinations?
Compare Direct Prompting vs Standard RAG on hallucination rate.

### RQ2: Does verification+revision help?
Compare Standard RAG vs Agentic RAG on accuracy and hallucination rate.

### RQ3: What causes hallucinations?
Analyze failure modes by:
- Contract clause type
- Query type
- Reasoning complexity

## Troubleshooting

### Issue: Out of memory
**Solution**: Use smaller batch sizes or reduce chunk size
```python
# In configs/model_config.yaml
resources:
  batch_size: 1  # Reduce from 4
```

### Issue: Slow retrieval
**Solution**: Use BM25 instead of dense retrieval for faster results
```bash
python retrieval/baseline_rag.py --retriever bm25
```

### Issue: Model download fails
**Solution**: Set HuggingFace cache directory
```bash
export HF_HOME=/path/to/large/drive
```

### Issue: Dataset download fails
**Solution**: Manual download
- ContractNLI: https://stanfordnlp.github.io/contract-nli/
- CUAD: https://www.atticusprojectai.org/cuad

## Expected Outputs

### Results Directory Structure
```
experiments/results/
├── direct_prompting_llama3_contractnli.json
├── standard_rag_llama3_contractnli.json
├── agentic_rag_llama3_contractnli.json
├── complete_results_20261118_153000.json
└── summary_report_20261118_153000.txt
```

### Output Format
```json
{
  "id": "contractnli_test_0",
  "query": "Can the contract be terminated?",
  "final_answer": "Yes, with 30 days notice [1]",
  "label": "entailment",
  "hallucination_detected": false,
  "latency": 2.34,
  "tokens": 523
}
```

## Team Communication

### Daily Sync Checklist
- [ ] What did you complete today?
- [ ] Any blockers?
- [ ] What's next?
- [ ] Dependencies on other team members?

### Integration Points
1. **Data → Retrieval**: Ensure processed data format is correct
2. **Retrieval → Agent**: Verify chunk format and retriever interface
3. **Agent → Evaluation**: Confirm output format includes verification results
4. **All → Experiments**: Test end-to-end pipeline early

## Deadline: November 25, 2026

### Final Checklist
- [ ] All datasets processed
- [ ] All baselines run successfully
- [ ] Agentic RAG working end-to-end
- [ ] All evaluation metrics computed
- [ ] Hallucination analysis complete
- [ ] Error analysis and comparisons done
- [ ] Final report written
- [ ] Code documented and cleaned
- [ ] README updated
- [ ] Reproducibility verified

## Contact

- Li Xinru (3036807959): Data & Evaluation
- Qin Mengmeng (3036843769): Retrieval & Baselines
- Yin Fengling (3036807375): Agent Components
- Wang Zhengyao (3036795675): Experiments & Analysis
