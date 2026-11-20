# 项目核查与 Yin Fengling 交付验收

本文为修复前的历史核查。当前实现、验收状态和队友接续说明见[检索与基线开发交接](RETRIEVAL_AND_BASELINES_HANDOFF.md)。诊断脚本会从下述 Git 版本读取历史源码，不会把已修复代码重新判定为原始缺陷。

核查日期：2026 年 11 月 20 日。仓库版本：`0166fe94c23f6e599ab54d98bbac6b49404046d8`（main）。

**结论：Yin Fengling 的两个测试集交付通过数据内容核查，可以作为后续开发输入；她负责的数据加载、转换流程复现及评估模块尚未达到完整验收条件。Qin Mengmeng 可以开始检索与基线修复，无需继续等待数据文件。项目当前有框架和数据，但还没有可据以回答研究问题的完整实验结果。**

## 项目全貌与当前分工

已阅读全部 34 个受版本控制的 Python 文件、3 个 YAML 配置、6 份根目录说明、数据交付说明及提交历史；对两个 JSON 测试集做全量结构检查，并与官方发布包逐条比对。GitHub 的全部状态 PR 和 issue 列表在核查时均为空；远程分支只有 main。

项目比较三套合同分析系统：Direct Prompting、Standard RAG、Agentic RAG。第三套流程为查询分析 → 分块检索 → 生成答案 → 逐项验证 → 必要时重新检索和修订。实验目标是比较准确率、证据召回和幻觉率，并分析失败类型。

| 成员 | 当前职责 | 主要接口 |
|---|---|---|
| Yin Fengling | `data/` 和 `evaluation/` | 统一数据格式、证据标注、可靠指标 |
| Qin Mengmeng | `retrieval/` | 分块、BM25/Dense/Hybrid、Direct 和 Standard RAG |
| Wang Zhengyao | `agent/` | 查询分析、生成、验证、修订与工作流 |
| Li Xinru | `experiments/`、`ui/`、结果分析及最终报告 | 实验调度、汇总、集成 |

分工依据是 11 月 18 日 `e798106` 更新后的 `README.md`、`TEAM_TODO_LIST.md` 和 `PROJECT_SUMMARY.md`。`COMPLETION_SUMMARY.md` 及部分代码的 Author 注释仍保留旧分工，不能据此重新分配责任，也不能把文件头的姓名当作个人完成记录。

仓库共 3 次提交：初始框架 `17a2fef`、分工文档调整 `e798106`、Yin Fengling 的数据提交 `0166fe9`。最后一次提交仅新增两个 JSON 和数据 README；数据处理器与评估代码均来自初始提交。文档中的“PROJECT COMPLETE”与勾选标记不等于运行验收。

## Yin Fengling 的完成情况

### 已交付并核实的数据

| 检查项 | ContractNLI | CUAD |
|---|---:|---:|
| 样本数及唯一样本 ID 数 | 2,091 | 1,020 |
| 合同数，按源文档标识 | 123 | 102 |
| 标签 | entailment 968；contradiction 220；neutral 903 | present 599；absent 421 |
| 有证据的样本 | 1,188 | 599 |
| 证据范围总数 | 2,404 | 1,116 |
| 越界或空的证据范围 | 0 | 0 |
| 空合同原文 | 0 | 0 |
| `answer: null` 的样本 | 2,091 | 421 |

两个文件都可映射成 `LegalExample`。ContractNLI 的原文、标签、原始假设、证据范围与官方 test 全量一致。CUAD 的问题、原文、标签、答案和证据范围与官方 test 中所选 10 类全量一致；无答案的证据 `null` 与空列表按相同含义处理。

源数据核对使用 [ContractNLI 官方发布包](https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip) 和 [CUAD 官方仓库发布包](https://github.com/The-Atticus-Project/cuad/blob/main/data.zip)。ContractNLI 证据索引的含义也符合[官方格式说明](https://stanfordnlp.github.io/contract-nli/)。比较结果、输入文件及发布包 SHA256 保存在 `audit_results_2026-11-20.json`。

ContractNLI 有 123 个文档 ID、122 种不同原文，这与官方发布包一致，不应误判为漏了一份合同。检索时应使用稳定文档标识；对重复原文和每份合同的多道题应避免反复建相同索引。

CUAD 只覆盖 10 个类别，每类 102 条：Document Name、Parties、Agreement Date、Effective Date、Expiration Date、Termination For Convenience、Notice Period To Terminate Renewal、Governing Law、Most Favored Nation、Non-Compete。它不是完整的 41 类 CUAD 评测集，也不是实验配置中写的 termination、liability、payment 等五类。

ContractNLI 没有自由文本参考答案、CUAD 的 absent 样本没有答案，均与任务定义兼容；评估函数必须显式处理这些情况。

### 仍需补齐的交付

| 原定工作 | 验收判断 | 需要补齐 |
|---|---|---|
| ContractNLI、CUAD 测试数据 | 已完成，内容核查通过 | 保留当前文件作为基准 |
| 数据说明与来源 | 已有 | 补转换命令、版本和校验方式 |
| 标准包导入及加载 | 未通过 | 修复三个 processor 的导入 |
| 从官方源数据重新生成结果 | 仓库内缺少匹配流程 | 提交真实转换脚本；保留文档 ID、假设 ID、原始问题与 offsets |
| 评估指标验证及单元测试 | 未通过验收 | 修复下列缺陷，并提交边界测试 |
| 幻觉检测、错误分析、指标报告 | 只有框架，缺少验证结果 | 联调统一输出，补运行记录 |
| 后续实验协助与评估报告部分 | 尚待实验阶段 | 按 11 月 21–25 日分工继续推进 |

不能从仓库判断她是否在个人电脑上配置了环境或运行过未提交的测试。当前结论针对已交付材料，不否定未入库的工作；实验阶段的后续任务也不应在 11 月 20 日就判为逾期。

## 已确认的问题与责任归属

以下位置均对应核查时的原始版本。P0 应在正式实验前解决，P1 应在对应交付验收前解决。

| 优先级 | 问题与证据 | 后果 | 主要负责人 |
|---|---|---|---|
| P0 | `data/contractnli_processor.py:19`、`cuad_processor.py:18`、`privacyqa_processor.py:17` 使用 `from dataset_loader import ...` | 从项目根目录导入 `data.dataset_loader` 即报 `ModuleNotFoundError`；已安装 datasets 后仍可复现 | Yin |
| P0 | `evaluation/metrics.py:279` 对空参考答案调用 `.lower()` | 混合 CUAD present/absent 的评估触发 `AttributeError` | Yin |
| P0 | `experiments/run_experiments.py:203` 等三处构造 ground truth 时遗漏 `context`；`metrics.py:191` 默认取空字符串 | 无关证据也会因空字符串包含关系得到 Recall=1.0，可能产生虚假的满分 | Li 修传参，Yin 加校验 |
| P0 | `retrieval/baseline_rag.py:188` 汇总所有样本的合同建索引，查询没有按文档筛选；Agent 也如此 | 某份合同的问题可命中另一份合同。真实 BM25 的最小复现已返回错误文档 | Qin，Wang 同步修 Agent |
| P0 | 两条 baseline 的 prompt/parser 均仅支持 NLI 三标签 | CUAD 的 present/absent 无正确任务输出；`Label: present` 被解析成 neutral。不可把 absent 机械等同 NLI contradiction | Qin；Wang、Yin 对齐 |
| P0 | `retrieval/chunking.py:66` 重组空白，`:71` 仅按首词搜索位置 | chunk 文本与原文切片不一致，重复词还会定位错误。会破坏引用、证据评估 | Qin |
| P0 | `agent/agentic_rag.py:163` 输出 `verification`，`metrics.py:251` 读取 `verification_results` | 已有 contradicted 声明仍可能没有幻觉率；真实 Agent 结构的输入返回 null | Wang、Yin、Li |
| P1 | `metrics.py:232` 用累计 hits 判断当前样本是否停止 | 两个样本都检索命中的构造案例，Hit Rate 得到 0.5 而非 1.0 | Yin |
| P1 | `metrics.py:327` 反转有重值的标签映射 | NLI 分类别报告显示 present/absent/neutral，类别名称错误 | Yin |
| P1 | 两条 baseline 的 CLI 均写死 `examples[:10]`，没有 `--sample_size` | 待办中的小样本参数不可用；“完整测试集”命令仍只跑 10 条 | Qin |
| P1 | `baseline_direct.py:71` 只保留原文前 2,000 个字符 | 后段证据完全不可见；需要明确模型 token 预算和截断规则，否则比较含有输入预算差异 | Qin，与 Li 对齐 |
| P1 | `baseline_rag.py:48` 和 `:51` 把同一份 kwargs 传给两种组件 | 设置 chunk_size 会传给 BM25 构造器并报 TypeError；配置无法正常扩展 | Qin |
| P1 | `baseline_rag.py:115` 结果只保存证据文本和分数 | 丢失文档 ID、chunk ID 与起止位置，无法可靠验收引用与检索范围 | Qin |
| P1 | `experiments/run_experiments.py:266` 消融函数仅输出日志；`:381` 的 dataset、sample_size 解析后未应用 | 配置了消融却没有结果；所谓小样本可能仍进入全量实验 | Li，Qin 提供基线配置支持 |

固定长度分块的原文一致性检查使用默认 512/128 参数，对不同原文各处理一次：ContractNLI 601 个块中 598 个不满足 `context[start:end] == chunk['text']`；CUAD 1,943 个块中 1,924 个不满足。构造文本 `A A A A A A`，chunk_size=2、overlap=0 时，返回起点 `[0, 2, 4]`，应为 `[0, 4, 8]`。

其他需要纳入后续验收的边界：ClauseLevel 默认会丢弃不足 50 词的短文本；chunk_size/overlap 未校验，非法步长可能造成循环；BM25 的 top_k=0 返回整个索引；Hybrid 的 BM25 归一化未防止最大分为零。后两者中 top_k=0 已动态复现，零分归一化和非法步长来自代码检查。

## 跨模块的实验有效性问题

Direct 与 Standard RAG 没有验证结果，当前不能直接计算它们的声明幻觉率。应对三种系统的最终答案应用同一评估流程，并统一声明提取、证据来源和判定标准。Agent 自带验证可用于修订，但跨系统结论不能仅依赖它的内部判断。

Agent 修订后仍返回初始 label，且保存的是初始答案的验证结果；没有最终答案的重新验证。token 汇总遗漏查询改写和验证步骤的消耗。应由 Wang 修正，再与 Yin、Li 对齐统计口径。

幻觉检测器从 claim 的 `evidence` 读取证据，而 Verifier 输出 `supporting_evidence`，错误案例也没有保留 clause_type，导致证据和分条款分析丢失。`error_analysis.py` 保存的 context 先截成 500 字符再统计长度，不能据此分析原合同长度对错误的影响。这些由 Yin 与 Li 联调。

`utils/llm_interface.py` 的 API 生成仍抛出 `NotImplementedError`，Mac MPS 分支会退回 CPU，temperature=0 会被 `or` 替换。GPU 部署文档中的 API 示例返回字符串，与项目要求的 `{text, tokens, latency}` 不兼容；其中的 `--use_api` 也未实现。本次没有验证文档中的价格、额度和耗时估算，不应直接当作资源承诺。

## 核查方法与复现

本次通过 34 个原始 Python 文件的语法编译检查、3,111 条数据的结构检查、两个官方测试源的逐条比较，并实际运行了指标、分块与 BM25 的最小复现。完整结果见 [审计输出](audit_results_2026-11-20.json)，诊断脚本见 [audit_project.py](../scripts/audit_project.py)。

为避免下载大模型，逻辑诊断从原文件 AST 中加载未修改的类定义，绕过包初始化和重型模型导入；LLM 用固定返回值替代。数据包导入错误在独立进程中用真实包导入复现。此次没有运行真实 Llama/Mistral 推理、Dense/Hybrid 模型或完整实验套件，因此不宣称端到端已通过。

使用独立的临时 Python 3.11 环境，不更改本机默认 Python：

```bash
uv venv --python python3.11 /tmp/legal-hallucination-audit-20261120
uv pip install --python /tmp/legal-hallucination-audit-20261120/bin/python numpy scikit-learn rank-bm25 datasets
/tmp/legal-hallucination-audit-20261120/bin/python scripts/audit_project.py --output docs/audit_results_local.json
```

官方源比对还需将上述发布包分别命名为 `contract-nli.zip` 和 `cuad-data.zip` 放入独立目录，增加 `--source-dir /path/to/releases`。本次下载的副本位于 `/tmp/legal-hallucination-audit-20261120/`，未覆盖仓库里的处理后数据。

本次新增审计文档、诊断脚本及输出，未修改原实现、未运行付费 API、未提交或推送 GitHub。后续修复及实验安排见 [Qin Mengmeng 职责与验收清单](QIN_MENGMENG_TASKS.md)。
