# Processed test datasets

These files use the project's `LegalExample` format. The contract text is
preserved, and evidence offsets are half-open character ranges `[start, end)`
into that text.

| File | Source split | Examples | Labels |
|---|---|---:|---|
| `contractnli_test.json` | Official ContractNLI test split, 123 NDAs | 2,091 | 968 entailment, 220 contradiction, 903 neutral |
| `cuad_test.json` | Official CUAD v1 test split, 102 contracts and 10 selected clause types | 1,020 | 599 present, 421 absent |

- **ContractNLI** was released by Hitachi America, Ltd. Cite [Yuta Koreeda and Christopher D. Manning, Findings of EMNLP 2021](https://aclanthology.org/2021.findings-emnlp.164/). We converted contract and hypothesis pairs into individual examples and resolved evidence span indices to character offsets. See the [official dataset and terms](https://stanfordnlp.github.io/contract-nli/).
- **CUAD v1** was curated by The Atticus Project, Inc. Cite [Dan Hendrycks, Collin Burns, Anya Chen, and Spencer Ball, NeurIPS 2021](https://arxiv.org/abs/2103.06268). We selected 10 clause types and flattened the QA annotations while retaining the original questions and answer offsets. See the [official dataset](https://www.atticusprojectai.org/cuad/).

Both source releases are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
