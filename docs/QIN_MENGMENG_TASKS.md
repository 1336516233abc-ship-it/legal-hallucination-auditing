# Qin Mengmeng 职责与验收清单

本文保留原任务与验收要求，实施状态以[检索与基线开发交接](RETRIEVAL_AND_BASELINES_HANDOFF.md)为准。

姓名：Qin Mengmeng；学号：3036843769。整理日期：2026 年 11 月 20 日；项目截止：2026 年 11 月 25 日。

**你的职责是把 `retrieval/` 做成可验证、可复现的检索与基线模块，交付 Direct Prompting、Standard RAG 的运行结果和检索消融分析。现有代码已经有框架，但存在影响实验正确性的缺陷，优先完成下面的 P0 修复，再运行完整实验。**

职责来源：`README.md` 的 Team Members、`TEAM_TODO_LIST.md:135` 起的个人清单、`PROJECT_SUMMARY.md` 的 Team Task Assignment。第一位组员的两个数据文件已可用于开发；加载器的导入缺陷需要同步修复。

## 你负责的文件与交付物

| 文件或产物 | 你要完成的内容 | 验收标准 |
|---|---|---|
| `retrieval/chunking.py` | 固定长度、条款级分块 | 保留原文与 offsets；不漏短合同；参数非法时明确报错 |
| `retrieval/retrievers/retriever.py` | BM25、Dense、Hybrid | 排序可靠；文档范围正确；空索引、零分、top_k 边界有明确行为 |
| `retrieval/baseline_direct.py` | Direct Prompting 基线 | 按数据集生成与解析答案；明确输入 token 预算；支持小样本及全量 |
| `retrieval/baseline_rag.py` | Standard RAG 基线 | 当前合同内检索；保留证据出处；支持检索、分块、top_k 配置 |
| 检索配置及接口说明 | 让 Agent 和实验框架复用你的组件 | 参数实际生效；示例调用与输出格式一致 |
| 组件测试和小样本记录 | 证明修复后行为符合任务要求 | 记录命令、输入范围、输出与失败处理 |
| `experiments/results/` 中的基线结果及日志 | 供 Li Xinru 汇总 | 样本 ID 对齐，数量正确，配置、模型、seed 可追溯 |
| 检索分析和报告素材 | Evidence Recall@k、策略比较、失败案例 | 使用核实后的指标，说明无证据样本的处理方式 |

原始文件头写有你的姓名，不代表这些模块已由你完成或经过测试。目前没有基线运行结果可据以确认你的实验部分完成。

## 第一优先级：修复影响正确性的问题

- [ ] **按合同限制检索。** 每次查询只检索对应合同，可使用每合同索引或可靠过滤；同一合同多道题复用索引，避免全测试集混检。与 Wang 同步 Agent 的相同缺陷。
- [ ] **修正分块的原文位置。** 用真实字符区间生成 chunk，满足 `context[start:end] == text`。覆盖多空格、换行、重复词、重叠区、首尾块；不要用首词搜索推算位置。
- [ ] **处理条款级边界。** 保留短合同和短尾块；长句拆分仍须对应原文；验证 `chunk_size > 0` 且 `0 <= overlap < chunk_size`。
- [ ] **区分两种任务。** ContractNLI 输出 entailment/contradiction/neutral；CUAD 输出条款抽取结果及 present/absent。与 Yin 确认统一标签、缺失答案和弃权定义，不能仅按数值共用标签映射。
- [ ] **保留完整出处。** 结果包含稳定 doc_id、chunk_id、start、end、原文 text、检索 score，并能追溯到 example_id。
- [ ] **移除写死的 10 条限制。** 增加实际生效的 sample_size 和全量模式；使用固定 seed 抽样并保留样本 ID。两个数据集样本量不同，完整模式需检查输出数量。
- [ ] **拆开组件配置。** chunker_kwargs 与 retriever_kwargs 分别传递；让 retrieval_config 中声明的相关参数真正进入实例。
- [ ] **明确 Direct 输入预算。** 替换无说明的前 2,000 字符截断；按模型上下文限制设 token 预算，记录是否截断，并与 RAG 的证据预算一起说明。

上述问题的复现、源码行号和模块归属见 [项目核查](PROJECT_REVIEW_2026-11-20.md)。当前不要把现成的“全量命令”当作已经可用；待办中写的 baseline `--sample_size` 目前尚未实现，实验 runner 同名参数也没有生效。

## 第二优先级：组件验收与基线运行

完成分块修复后，先做无需 LLM 的检索评测。默认参考配置为 fixed_length=512、overlap=128、BM25、top_k=5；这里的长度是近似词数，不能当作模型 tokenizer 的精确 token 数。

- [ ] 在两个真实数据集上验证 chunk 的原文切片与 offsets。
- [ ] BM25、Dense、Hybrid 在同一批 query/document 上返回规定数量的块和可序列化分数；处理 top_k=0、top_k 大于块数、空文本和零分情况。
- [ ] 验证查询 A 合同不会返回 B 合同的 chunk；连续调用并更换 context 时也保持正确。
- [ ] 核对 ClauseLevel 的句子切分资源和超长句行为。
- [ ] 每个数据集先运行 5–10 个跨标签、跨文档样本，确认输出与评估入口兼容。
- [ ] 检查 CUAD 无答案样本不会让评估崩溃，ContractNLI 的空参考答案不会被当作答案生成失败。
- [ ] 全量运行 Direct 与 Standard RAG，保存原始响应、解析结果、检索出处、耗时、token 统计和执行配置。

全量的预期预测数是：每模型每系统 ContractNLI 2,091 条，CUAD 1,020 条。按仓库计划的两个模型、两条基线、两个数据集计算，基线主实验共 **12,444 条预测**，不含 Agent 和检索消融。先以小样本测实际耗时，再安排完整运行。

## 检索消融与分析

你的原定范围包括 retriever、chunking 和 top_k 的比较，主线是 BM25 vs Dense、固定长度 vs 条款级、top_k=3/5/10；TEAM_TODO 另外列了 Hybrid。模型主实验计划为 Llama3 和 Mistral，由 Li Xinru 统一编排。

建议先交付固定配置的两条基线，再组织以下对照。具体运行矩阵应与 Li 的实验框架保持一致：

| 比较 | 改变的因素 | 固定的因素 |
|---|---|---|
| 检索方法 | BM25 / Dense；补充 Hybrid | 数据、分块、top_k、生成模型 |
| 分块策略 | fixed_length / clause_level | 数据、检索方法、top_k、生成模型 |
| 检索数量 | 3 / 5 / 10 | 数据、检索方法、分块、生成模型 |

检索本身的指标可先离线计算，不需要为每一项都调用生成模型。避免一次改变多个因素；需要选择参数时使用开发集或预先固定的方案，测试集用于最终对比。

- [ ] 与 Yin 定义 Recall@k 的命中口径，包含证据被多个块切分、部分覆盖及无 gold evidence 的情况。
- [ ] 与 Yin、Li 确认实际 k 传入指标；当前 compute_all_metrics 的证据召回默认按 5 计算。
- [ ] 输出按数据集、策略和 k 的 Recall/Hit Rate 表格，包含可评估样本数。
- [ ] 向 Li 提供检索失败与生成失败的具体例子，帮助回答 RQ1 和 RQ3。
- [ ] 撰写最终报告中检索与基线的方法、超参数、结果分析及局限性材料。

## 你需要的交接与协作

| 对接人 | 需要对方提供或修复 | 你交给对方的内容 |
|---|---|---|
| Yin Fengling | 数据包导入、真实转换脚本、评估缺陷修复；统一 NLI/CUAD 任务格式 | chunk 原文及位置、检索结果样例、Recall 测试案例 |
| Wang Zhengyao | Agent 按合同检索、最终答案输出对齐 | chunker/retriever 的稳定接口、证据定位格式、调用示例 |
| Li Xinru | runner 的 dataset/sample_size 生效、消融实际执行、ground truth 包含 context、统一幻觉评估 | 基线预测、参数、日志、耗时和检索分析 |

共享接口可保留现有 `chunk(text, metadata)`、`index(chunks)`、`retrieve(query, top_k)`，但必须说明文档范围由哪里保证。输出 chunk 至少保留 text、start、end、chunk_id 和稳定 doc_id；建议在保存后的预测中同时保留 query、dataset、example_id 和配置，便于跨系统对齐。

数据预处理、指标实现、Agent 验证与修订、完整实验汇总和最终报告统稿各有负责人。你需要提供接口和实验素材，并共同解决交接问题；不必因当前框架存在缺口就把所有模块接过来。

## 从当前日期推进的安排

以下是依据原截止日期整理的执行顺序，实际耗时需用小样本测量：

| 日期 | 你的重点 | 当天交付检查 |
|---|---|---|
| 11 月 20 日 | P0 修复，BM25 与数据接口验收 | 位置检查、合同隔离和小样本基线通过 |
| 11 月 21 日 | 两条基线主实验，Dense 与条款级验收 | 至少一模型的两数据集结果可汇总 |
| 11 月 22 日 | 完成约定矩阵与消融 | 原始结果、完整配置与日志齐全 |
| 11 月 23–24 日 | 检索分析、案例、方法与结果写作 | 给 Li 的表格与报告素材，复现命令 |
| 11 月 25 日 | 最终复查与补漏 | 预测数量、引用位置、结果可复现 |

## 完成你的部分时应满足

- [ ] 原始数据可加载，与你的组件输出格式一致。
- [ ] 同一份合同的原文、证据位置与检索范围可核实。
- [ ] Direct、Standard RAG 在两个数据集上的小样本和约定全量实验均有真实记录。
- [ ] 消融对照实际运行，有原始输出与参数说明。
- [ ] Recall@k 使用修复后的评估入口，包含正确 context、doc_id 和实际 k。
- [ ] Wang 可复用检索接口，Li 可用 example_id 对齐和汇总结果。
- [ ] 已交付检索与基线报告材料，而非只有源码和勾选清单。

`experiments/results/` 和 `experiments/logs/` 默认受 .gitignore 排除。交接时明确提交精简结果、发布归档或共享存储的位置，并保证队友能访问；不能只保存在个人电脑上。
