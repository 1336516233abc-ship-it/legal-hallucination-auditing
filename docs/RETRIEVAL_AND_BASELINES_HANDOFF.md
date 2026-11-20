# 检索与基线开发交接

负责人为 Qin Mengmeng。更新于 2026 年 11 月 20 日。本文供组员复现检索与基线实验、接入 Agent 和汇总报告使用。

- [检索与基线全量结果](#检索与基线全量结果)
- [已交付实现与接口](#已交付实现与接口)
- [复现步骤](#复现步骤)
- [开发验证与失败案例](#开发验证与失败案例)
- [组员接续事项](#组员接续事项)
- [GitHub 交付文件](#github-交付文件)

## 检索与基线全量结果

Mistral 和 Llama 3 的 AutoDL 全量均已完成。每个模型在同一份 ContractNLI 的 2,091 条及 CUAD 的 1,020 条测试数据上运行三种方法。两条基线共 12,444 条预测，Verified Agentic RAG 对照共 6,222 条，总计 **18,666 条**。两套 SQLite 分别核对了 9,333 条记录和数据库完整性。[测试数据说明](../data/processed/README.md)

对应的模型版本、测试文件 SHA256、环境和各组统计保存在两份正式汇总中。

| 模型 | 固定 revision | 格式通过 | 标签正确 | 全量汇总 |
| --- | --- | ---: | ---: | --- |
| Mistral 7B Instruct v0.2 | `63a8b081895390a26e140280378bc85ec8bce07a` | 7,678/9,333 | 4,599/9,333 | [Mistral](validation/mistral_autodl_full_summary.json) |
| Meta Llama 3 8B Instruct | `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` | 8,866/9,333 | 5,157/9,333 | [Llama 3](validation/llama3_autodl_full_summary.json) |

每个模型使用一张 RTX 6000D 84 GB，vLLM 0.10.2，未量化 BF16，温度 0，seed 42，并行度 8。输入最多 4,096 token，单次任务回答最多 512 token。Standard RAG 在当前合同内用 BM25 取 top 5，固定分块为 512 词、重叠 128 词。正式汇总取自 AutoDL 全量任务。锁定试跑、阿里云 API 连通测试和本机暂停时的 83 条 Mistral 预测均另行留档。

格式通过要求输出可解析且标签属于相应任务。标签准确率以组内全部预测为分母，格式失败计错。Macro-F1 对任务的合法类别等权平均。单条平均耗时包含对应工作流实际执行的步骤。

| 数据集 | 模型 | 方法 | 格式通过率 | 标签准确率 | Macro-F1 | 平均耗时（秒） |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ContractNLI | Mistral | Direct Prompting | 1,750/2,091（83.69%） | 908/2,091（43.42%） | 0.3253 | 1.93 |
| ContractNLI | Mistral | Standard RAG | 1,718/2,091（82.16%） | 876/2,091（41.89%） | 0.2707 | 2.52 |
| ContractNLI | Mistral | Verified Agentic RAG | 1,577/2,091（75.42%） | 775/2,091（37.06%） | 0.2305 | 10.82 |
| ContractNLI | Llama 3 | Direct Prompting | 2,091/2,091（100.00%） | 983/2,091（47.01%） | 0.4491 | 1.45 |
| ContractNLI | Llama 3 | Standard RAG | 2,087/2,091（99.81%） | 889/2,091（42.52%） | 0.4271 | 1.85 |
| ContractNLI | Llama 3 | Verified Agentic RAG | 2,051/2,091（98.09%） | 1,161/2,091（55.52%） | 0.5283 | 9.94 |
| CUAD | Mistral | Direct Prompting | 877/1,020（85.98%） | 684/1,020（67.06%） | 0.7214 | 1.51 |
| CUAD | Mistral | Standard RAG | 928/1,020（90.98%） | 711/1,020（69.71%） | 0.7284 | 1.39 |
| CUAD | Mistral | Verified Agentic RAG | 828/1,020（81.18%） | 645/1,020（63.24%） | 0.6971 | 6.32 |
| CUAD | Llama 3 | Direct Prompting | 966/1,020（94.71%） | 791/1,020（77.55%） | 0.7851 | 1.28 |
| CUAD | Llama 3 | Standard RAG | 866/1,020（84.90%） | 700/1,020（68.63%） | 0.7346 | 1.29 |
| CUAD | Llama 3 | Verified Agentic RAG | 805/1,020（78.92%） | 633/1,020（62.06%） | 0.6834 | 5.89 |

- **两条基线。** 在 ContractNLI 上，Standard RAG 相比 Direct Prompting 的准确率在 Mistral 和 Llama 3 上分别下降 1.53 和 4.50 个百分点。在 CUAD 上，Mistral 提高 2.65 个百分点，Llama 3 下降 8.92 个百分点。差值按未四舍五入的准确率计算。
- **Agent 对照。** 在 Llama 3 的 ContractNLI 上，Verified Agentic RAG 达到 55.52%，较 Standard RAG 提高 13.01 个百分点，单条平均耗时从 1.85 秒升至 9.94 秒。其余三个模型与数据集组合中，Agent 的准确率均低于 Standard RAG。
- **格式与证据。** ContractNLI 的格式失败均为缺失或非法任务标签，Mistral 有 1,228 条，Llama 3 有 44 条。Llama 3 在 CUAD 上有 220 条未返回完整的单一 JSON 对象，197 条的引用数组未满足正整数编号要求。CUAD 的 599 条金标正例中，同时满足标签正确、逐字抽取且未触发引用越界或证据错误的预测，在 Mistral Standard RAG 下有 179 条，在 Llama 3 Direct Prompting 下有 396 条。

全量 JSON 还包含混淆矩阵、格式失败原因、引用越界及实际提示中的金标证据覆盖。跨方法的幻觉率仍需统一的答案支持性评估。逐条输出和生成轨迹保存在两台已关机的 AutoDL 实例磁盘上，各自运行目录中的 `predictions.sqlite3`。报告引用数值可直接使用仓库中的 JSON。需要逐条轨迹时，由负责人从实例磁盘取回。

## 已交付实现与接口

| 模块 | 接手时需要知道的行为 |
| --- | --- |
| 分块与检索 | 固定窗口和条款分块保留原文及字符位置。BM25、Dense、Hybrid 支持稳定的 top-k 排序。低层检索调用要限制 `doc_id` 或只索引当前合同。 |
| Direct Prompting | 在 token 预算内将当前合同前缀给模型。按数据集生成、解析并记录截断。 |
| Standard RAG | 注册多份合同后通过 `doc_id` 定位，查询只检索该合同。批处理按合同复用索引，返回顺序与输入一致。 |
| Verified Agentic RAG | 评估使用传入 `dataset` 的路径。初答格式错误或证据检查失败时保留轨迹，跳过模型查证。证据不足时可在同合同重检索，最多修订一次，按最终显式标签评分。 |
| 独立评估入口 | 检索消融和正式全量分别使用独立 runner。旧版 `experiments/run_experiments.py` 尚待接入。 |

`chunk(text, metadata=None)` 返回带 `text`、`start`、`end`、`chunk_id` 的分块。`start` 和 `end` 是原文半开字符区间，满足 `source[start:end] == chunk["text"]`。`chunk_id` 与 `doc_id` 一起标识分块。检索器接口为 `index(chunks)` 与 `retrieve(query, top_k=5, doc_id=None)`，返回 `(chunk, score)` 列表。分块预算按词统计，4,096 的输入预算按生成模型 tokenizer 统计。

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

也可通过 `rag.answer(query, context=contract_text)` 直接指定合同。组件集成可用 `StandardRAGBaseline(llm=None).retrieve_evidence(query, context=contract_text, doc_id=source_doc_id)` 单独检索。用于三方法对照的 Agent 入口为 `AgenticRAG.answer(query, context, dataset="contractnli", max_input_tokens=4096)`，CUAD 传 `dataset="cuad"`。批处理保留样本 ID 与 `metadata`，方便下游对齐和定位出处。

**输出与评分约定**

- ContractNLI 使用 `entailment`、`contradiction`、`neutral`，提示词包含 `Label:` 和 `Answer:` 示例。CUAD 使用 `present`、`absent`，输出 JSON。正例要求提取原文，负例的答案和抽取列表为空。
- 格式错误保留 `raw_response` 和 `parse_error`，归为 `label=invalid`。CUAD 的 `answer_is_extractive` 记录抽取片段是否逐字出现在模型输入中。Agent 会额外记录 `grounding_error`。最终格式或证据检查失败时，Agent 将展示用的 `final_answer` 留空，原标签与原响应仍可审计。
- `retrieved_chunks` 保存原始 top-k 结果。计算检索 Recall 使用该字段。审计模型实际看到的内容使用预算裁剪后的 `prompt_evidence`。引用编号对应当次调用实际使用的证据块。
- `prompt` 保存任务内容，`model_prompt` 保存实际应用聊天模板后的输入。两条基线的 `tokens` 记录生成 token 数，输入 token 数在 `input_budget`。Agent 的阶段调用另见 `generation_trace`。Python API 配置 `max_input_tokens` 时需要提供 tokenizer 或由模型适配器提供。

## 复现步骤

### 本地测试与两条基线

在仓库根目录执行。最小环境用于本地逻辑、BM25 和 dry run。完整环境和 Dense 编码器或生成模型权重按需安装。

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements-retrieval.txt
python -m pytest -q
python -m retrieval.baseline_direct --dataset contractnli --sample_size 10 --dry_run
python -m retrieval.baseline_rag --dataset cuad --sample_size 10 --retriever bm25 --top_k 5 --dry_run
```

需生成时另装 `requirements.txt`。以下示例需要 CUDA 及可读取的模型权重。省略 `--sample_size` 即读取完整划分。正式对照使用后续的固定 revision 全量入口。

```bash
python -m retrieval.baseline_direct --model mistral --dataset contractnli \
  --sample_size 10 --device cuda --temperature 0 --output experiments/results/direct_mistral_contractnli.json
python -m retrieval.baseline_rag --model mistral --dataset contractnli \
  --sample_size 10 --device cuda --temperature 0 --output experiments/results/rag_mistral_contractnli.json
```

独立基线 CLI 输出最终 JSON、逐条 JSONL 及 `.meta.json`。默认保护已有文件。JSONL 用于故障排查。全量 runner 保存 SQLite 检查点，可断点续跑。汇总时检查 `validation_status` 和 `parse_error_count`。

### 检索消融

```bash
python -m retrieval.evaluate_retrieval --datasets contractnli cuad \
  --retrievers bm25 dense hybrid --chunkings fixed_length clause_level \
  --top_ks 3 5 10 --output experiments/results/retrieval_ablation.json
```

36 组来自两个数据集、三种检索器、两种分块和三个 k。仅运行 BM25 时可使用最小环境。只有带金标证据的样本进入证据指标分母，ContractNLI 为 1,188 条，CUAD 为 599 条。固定分块为 512 词、重叠 128 词，条款分块最大 512 词、合并目标 50 词。Dense 用 all-MiniLM-L6-v2，编码上限 256 个 subword token。Hybrid 两侧权重各为 0.5。详细口径、Recall 对照和检索失败样例见[检索消融报告](validation/RETRIEVAL_RESULTS.md)与[完整结果](validation/retrieval_ablation_2026-11-20.json)。

### AutoDL 双模型全量

在包含源代码、配置和两个仓库测试文件的冻结目录里运行。以下依赖基于 Linux x86-64、Python 3.12、PyTorch 2.8.0+cu128、CUDA 12.8 的 AutoDL 镜像。完成的两组运行参数见全量 JSON 的 `settings`。

```bash
python3.12 -m venv --system-site-packages venv
source venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  'vllm==0.10.2' 'transformers==4.56.2' 'huggingface-hub==0.34.4' \
  'openai==1.109.1' 'rank-bm25==0.2.2' 'pyyaml==6.0.3' 'requests==2.32.5'
```

启动脚本加载固定 revision。在新环境直接向 Hugging Face 下载 Llama 3 时，用已开通该模型访问权限的账号执行 `hf auth login`。已有完整缓存时可为启动命令添加 `HF_HUB_OFFLINE=1`。正式实验用的 Llama 3 权重来自 ModelScope 固定版本，加载前核对了四个权重分片及六个配置、tokenizer 和索引文件。阿里云 PAI 仅用于 API 连通试验。[镜像校验脚本](../scripts/fetch_verified_llama3.py)

```bash
bash scripts/run_autodl_mistral.sh pilot mistral
bash scripts/run_autodl_mistral.sh full mistral
bash scripts/run_autodl_mistral.sh pilot llama3
bash scripts/run_autodl_mistral.sh full llama3
```

同一实例上按模型顺序运行。每个模型先复核试跑的 30 条，再启动该模型全量。每次运行在 `experiments/results/<model>_autodl_<phase>/` 保存 `progress.json`、`predictions.sqlite3`、`vllm.log`，完成后生成 `summary.json`。以 `status=completed`、六组样本数和 `PRAGMA quick_check` 验收。SQLite 用 `(dataset, method, example_id)` 去重。续跑须复用同一输出目录、冻结代码、模型和配置。需要续跑正式检查点时，以该次运行自己的 `source_sha256` 为准。

## 开发验证与失败案例

- **自动化验收。** 整理交接时重跑完整环境，144 项通过。最小环境 123 项通过、4 项跳过。两条基线的 dry run 均成功选出 10 条。仓库中的记录对应此前的格式修复阶段。[JUnit 汇总](validation/test_summary.json) · [环境记录](validation/environment.json)
- **Qwen 格式排查。** 首轮只通过 1/8。用新解析器重读原响应后通过 7/8，采用修复后的提示词重跑通过 8/8。旧结果中的 4 条 CUAD 在完整 JSON 前附有说明，2 条 ContractNLI Standard RAG 用对象表示引用，1 条 ContractNLI Direct Prompting 在 JSON 内生成了未转义引号。这 7 条均未用尽生成预算。引号案例通过带示例的两行提示改善，前两类由兼容解析与引用归一化处理。[复测记录](validation/qwen_format_retest)
- **正式模型首次试跑。** Mistral 在本机用 MLX 完成 20 条基线预测，其中 11 条格式通过、6 条标签正确。Llama 3 当时的 Hugging Face 权重请求受限，未产出预测。双模型的有效对照后来在 AutoDL 完成。[Mistral 首轮结果](validation/official_baseline_pilot/mistral.json) · [Llama 3 首轮记录](validation/official_baseline_pilot/llama3.json)
- **CUAD 协议修订。** 开发样本来自 CUAD 官方训练划分，并按合同标题和文本哈希与测试集隔离。最终提示词去掉题干中的律师审核措辞，要求结构化抽取。生成器在输出首个完整 JSON 对象后停止。在锁定的 4 条 CUAD 测试样本上，Direct Prompting、Standard RAG 和 Verified Agentic RAG 的格式均通过 4/4，标签正确数依次为 2/4、2/4、3/4。[开发样本来源](validation/cuad_output_development/source.json) · [开发复测](validation/cuad_output_development/final/mistral.json) · [锁定测试复测](validation/cuad_output_development/held_out/mistral.json)
- **AutoDL 锁定试跑。** 两个模型共用 6 条 ContractNLI 和 4 条 CUAD 做三方法检查。Mistral 格式 27/30、标签正确 15/30，Llama 3 格式 30/30、标签正确 21/30。[Mistral 试跑](validation/mistral_autodl_pilot_summary.json) · [Llama 3 试跑](validation/llama3_autodl_pilot_summary.json)

### 可回查的失败案例

| 样本或合同 | 观察 |
| --- | --- |
| `contractnli_test_523_nda-8` | Mistral 三种方法均输出 `Label: true`，自然停止。非法标签导致格式和准确率均计错。 |
| `contractnli_test_41_nda-1` | 标注为 `contradiction`。Agent 初次和修订后的提示都覆盖了标注证据，最终仍判为 `neutral`。 |
| CUAD PACIRA 许可分销协议 | Agent 复测后经同合同重检索覆盖了金标续期通知条款，最终仍判为 `absent`。Standard RAG 从首次提示已覆盖金标，标签同样有误。 |
| CUAD HPIL 合作协议 | 早期 Agent 判对日期和标签，但改写了原句，逐字校验失败。最终复测中 Agent 复制了原文日期，Standard RAG 标签正确但引用编号越界。 |
| CUAD N2K 赞助协议 | 早期 Direct Prompting 用满 512 token 后仍未闭合 JSON。 |

案例来源为[早期 Mistral 三方法逐条结果](validation/verified_agentic_rag_pilot/mistral.json)与[锁定测试复测](validation/cuad_output_development/held_out/mistral.json)。引用到报告时注明所用试跑阶段。

## 组员接续事项

| 负责人 | 接手事项 |
| --- | --- |
| Yin Fengling | 补齐从官方原始数据生成当前 JSON 的可复现转换流程。共享评估器区分两个数据集的标签、空答案和 `parse_error`，按实际检索 k 统计证据命中。 |
| Wang Zhengyao | 演示入口显式传 `dataset`，展示 `final_answer`、查证结果及拦截原因。修订后目前只有确定性格式和证据检查，若加二次模型查证需单独记录。 |
| Li Xinru | 从正式全量 JSON 取数，按模型、方法和数据集汇总。需要样本级比较时用 `example_id` 对齐。把通用 runner 接到本次已验证的接口，并给三种方法使用同一套外部答案支持性标准。 |
| Qin Mengmeng | 协助提取逐条运行轨迹与分析漏判。若继续做 Dense 或 Hybrid 生成对照，保持模型、测试样本和输入预算固定，用开发集选参。 |

## GitHub 交付文件

| 内容 | 仓库位置 |
| --- | --- |
| 分块、检索器、任务格式、两条基线与独立消融 | [`retrieval/`](../retrieval/) · [检索配置](../configs/retrieval_config.yaml) |
| 接通三方法对照的 Agent 实现 | [`agent/`](../agent/) |
| 双模型全量、锁定试跑、vLLM 接口和 CUAD 开发集构建 | [`experiments/`](../experiments/) · [AutoDL 启动脚本](../scripts/run_autodl_mistral.sh) |
| 数据导入、延迟依赖与模型预算相关集成改动 | [`data/`](../data/) · [`utils/`](../utils/) · [完整依赖](../requirements.txt) · [最小依赖](../requirements-retrieval.txt) |
| 自动化用例与正式、诊断结果 | [`tests/`](../tests/) · [`docs/validation/`](validation/) · [验证汇总脚本](../scripts/update_validation_report.py) |
| 工作范围与历史核查 | [原始职责与验收清单](QIN_MENGMENG_TASKS.md) · [修复前项目核查](PROJECT_REVIEW_2026-11-20.md) |

两个原始测试 JSON 已在仓库跟踪，本轮保持原样。随代码交付上述测试和经过汇总的实验产物。模型权重、虚拟环境、凭证和 `experiments/results/` 下的逐条运行产物留在工作环境。后续如修改接口、数据说明或实验结果，同步更新本文和对应验证文件。
