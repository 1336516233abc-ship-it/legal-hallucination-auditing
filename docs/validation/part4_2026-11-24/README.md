# Part 4：全量记录分析与指标核对（2026-11-24）

负责人：Li Xinru。本页是已完成实验的分析交付，不是课程报告。

已取得并逐条核对 AutoDL 导出的 **18,666 条预测**：两个模型 × 两个数据集 × 三种方法，共 12 组；ContractNLI 每组 2,091 条，CUAD 每组 1,020 条。本次没有重新运行模型。CUAD 范围为仓库提供的 10 类，不是官方全部 41 类。

**已完成：** 文件完整性、SQLite/JSONL 一致性、数据及样本 ID、证据原文与位置、全部组汇总核对；全量错误统计、修订前后标签比较、20 个可回查案例，以及可计算指标汇总。

**仍未完成：** 三种方法最终回答的统一独立幻觉评估。人工标注数为 0，对应幻觉率保持 null；内部初答验证不能代替该项。

## 文件入口

- [下载完整分析包](COMP7608B-Part4-Full-Trace-Analysis-v2.zip)：与本地已交付 v2 包逐字节相同。
- [20 个案例的可读索引](CASES.md)；[完整案例及合同、证据记录](analysis/CASES.jsonl)。
- [全量分析 JSON](analysis/RESULTS.json)；[18,666 条逐条统计（gzip）](analysis/row_metrics.jsonl.gz)。
- [完整指标 JSON](metrics-followup/METRICS.json)：含分母、内部验证统计及替代指标。
- [全量分析 HTML](analysis/SUMMARY.html) 与 [指标说明 HTML](metrics-followup/METRICS_REVIEW.html)：下载后用浏览器打开。

## 可以使用的任务指标

格式失败保留在准确率分母，并计为错误；Macro-F1 对各数据集合法类别取平均。初次检索 Recall@5 仅计有金标证据的样本：ContractNLI 1,188 条、CUAD 599 条；Direct 没有检索，记 N/A。最终提示的证据覆盖另见完整 JSON。

|模型|数据集|方法|准确率|Macro-F1|初检 Recall@5|平均耗时（秒）|
|---|---|---|---:|---:|---:|---:|
|llama3|contractnli|Direct|47.01%|0.4491|N/A|1.45|
|llama3|contractnli|RAG|42.52%|0.4271|97.89%|1.85|
|llama3|contractnli|Agent|55.52%|0.5283|97.29%|9.94|
|llama3|cuad|Direct|77.55%|0.7851|N/A|1.28|
|llama3|cuad|RAG|68.63%|0.7346|61.75%|1.29|
|llama3|cuad|Agent|62.06%|0.6834|56.40%|5.89|
|mistral|contractnli|Direct|43.42%|0.3253|N/A|1.93|
|mistral|contractnli|RAG|41.89%|0.2707|97.89%|2.52|
|mistral|contractnli|Agent|37.06%|0.2305|97.11%|10.82|
|mistral|cuad|Direct|67.06%|0.7214|N/A|1.51|
|mistral|cuad|RAG|69.71%|0.7284|61.75%|1.39|
|mistral|cuad|Agent|63.24%|0.6971|57.89%|6.32|

耗时来自原服务器 8 并行运行，不代表串行吞吐。完整 JSON 分别记录全部调用的输入和生成 token；token 数不是货币费用。

## 全量错误统计

下列事件可以重叠，不能相加当成互斥错误类别。标签错误包含格式失败；引用越界和抽取依据错误另列。合法的 CUAD absent 空抽取不计为异常拦截。

|模型|数据集|方法|标签错误|格式失败|引用越界|依据检查错误|
|---|---|---|---:|---:|---:|---:|
|llama3|contractnli|Direct|1108|0|0|0|
|llama3|contractnli|RAG|1202|4|151|0|
|llama3|contractnli|Agent|930|40|211|211|
|llama3|cuad|Direct|229|54|0|0|
|llama3|cuad|RAG|320|154|143|0|
|llama3|cuad|Agent|387|215|149|271|
|mistral|contractnli|Direct|1183|341|0|0|
|mistral|contractnli|RAG|1215|373|100|0|
|mistral|contractnli|Agent|1316|514|112|112|
|mistral|cuad|Direct|336|143|0|0|
|mistral|cuad|RAG|309|92|107|0|
|mistral|cuad|Agent|375|192|128|223|

## Agent 修订前后

这里只描述同一次 Agent 运行中标签的变化，不等于受控组件消融，也不能称为语义修订成功率。

|模型|数据集|发生修订|标签错→对|标签对→错|初答正确/总数|终答正确/总数|
|---|---|---:|---:|---:|---:|---:|
|llama3|contractnli|1555|262|44|943/2091|1161/2091|
|llama3|cuad|498|19|56|670/1020|633/1020|
|mistral|contractnli|1466|9|2|768/2091|775/2091|
|mistral|cuad|662|14|49|680/1020|645/1020|

## 指标口径与旧实现问题

1. 旧 `MetricsCalculator` 读 `verification_results`，服务器记录为 `verification.claims`。直接调用旧幻觉率函数会在未计数时返回 0；`compute_all_metrics` 则跳过这一指标。不能把缺测写成零幻觉。
2. 记录中的语义验证仅针对 Agent 初答，且部分初答因格式或依据错误没有进入验证。Direct/RAG 无同类记录，修订后答案没有二次语义验证。因此内部统计不能用于三方法最终幻觉率比较。
3. 验证器格式失败会被映射为 `not_enough_info`。JSON 同时保留含失败的原始比例、失败数量和有效检查子集比例；后者存在选择偏差，两者均不是独立真值。
4. 旧标签归一化将非法值当成 neutral；例如 Mistral / ContractNLI / Agent 会由正确的 37.06% 虚增至 42.99%。本页沿用已核对的任务专用评分。共享评估器本次未修改。
5. 词集合召回只适用于 CUAD 的 599 个非空参考答案，不等于语义完整性；ContractNLI 参考答案全为 null。弃权标志实际均对应 neutral 分类，只作为旧规则代理统计，零分母记 N/A。

## 验证记录

- [源文件 SHA256 核对](file_integrity.json)：四个数据库/压缩逐条文件均与导出清单一致。
- [调用轨迹核对](analysis/call_trace_checks.json)：37,834 次调用，阶段顺序及输入输出记录检查通过。
- [分析独立检查](analysis/output_validation.json) 与 [指标独立检查](metrics-followup/VALIDATION.json)：含 SQL 对照、缺测值和分母检查。
- [交付文件清单](publication_manifest.json)：本次上传内容的 SHA256 与原始分析脚本哈希核验。

## 离线复现

分析脚本保留当时原文件与哈希，不依赖本人的绝对路径；其约定是分析目录与名为 `legal-hallucination-ui-review` 的仓库目录同放一父目录。使用 Python 3.12，在仓库根目录执行以下命令，RUN_ROOT 和 ZIP 路径替换为自己的新目录和收到的原始数据包。输出目录必须尚不存在。

```bash
python -m pip install -r requirements-retrieval.txt scikit-learn
RUN_ROOT=/path/to/new-trace-reproduction
mkdir -p "$RUN_ROOT/trace-audit/source"
ln -s "$PWD" "$RUN_ROOT/legal-hallucination-ui-review"
cp docs/validation/part4_2026-11-24/analyze_traces.py "$RUN_ROOT/trace-audit/"
cp docs/validation/part4_2026-11-24/calculate_available_metrics.py "$RUN_ROOT/trace-audit/"
cp docs/validation/part4_2026-11-24/file_integrity.json "$RUN_ROOT/trace-audit/"
unzip /path/to/COMP7608B-AutoDL-Full-Traces.zip -d "$RUN_ROOT/trace-audit/source"
python "$RUN_ROOT/trace-audit/analyze_traces.py"
python "$RUN_ROOT/trace-audit/calculate_available_metrics.py"
```

复现使用本分支代码；原评估器诊断所对应的源码 blob 为 `4f0a612fe5fe483266c7f289955a9169a753c78d`。若组员后续修正共享评估器，诊断输出应另存新版本，不能覆盖本次记录。

大型原始 SQLite、原始全量 JSONL、模型权重未放入本次提交；复现需团队保管的原始 ZIP。600 条空白盲评材料仅为本地抽样草案，尚未确定工作分配，本次不上传草案及模型方法解码表。该数量不是新增给某一位成员的任务。
