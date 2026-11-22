# 三种方法共用的外部答案支持性评估方案

状态：Li Xinru 于 2026-11-22 准备的待团队确认方案。当前未开展独立人工评审，未产生幻觉率结果。该方案不依赖 Agent 自己的 verifier，适用于 Direct、RAG、Agent 及组件消融。

## 输入和盲评

取回两个正式全量运行目录中的 `predictions.sqlite3`，同时保留冻结代码、模型设置、`summary.json`、`progress.json` 与原处理数据。先执行 SQLite 完整性检查；再核对模型 revision、数据哈希、每种方法样本数量及 `(dataset, method, example_id)`。不要用仅含汇总的 JSON 代替逐条答案。

导出工具会检查 SQLite 完整性、冻结数据哈希、方法间相同样本集合及计划数量，保留解析失败，并随机打乱方法身份。每个数据集用 seed=42 简单随机选择最多 50 个相同 example_id，所有方法评审同一批问题。按模型分别导出时，完整数据集合相同可得到相同的抽样。样本量可显式调整；初次 50 条只是评审试点，不称为全量评估。

```bash
python -m experiments.prepare_support_review \
  --checkpoint experiments/results/mistral_autodl_full/predictions.sqlite3 \
  --model mistral --sample_size 50 --seed 42 \
  --output experiments/results/support_review/mistral
```

对 Llama 3 换成对应检查点和独立输出目录。输出包括：

- `blinded_review.jsonl`：问题、原合同、最终展示答案、预测标签、原响应、格式状态、空白评审栏。
- `private_key.json`：盲评 ID 与模型、方法、样本的映射；评分结束前不提供给评审者。
- `selection.json`：抽样范围、选中 ID 和 seed。`independent_hallucination_rate` 保持 null。

不要把原运行中的 verification、revision 或检测标记交给盲评者作为评分依据。脚本只生成表，不执行付费 API，不自动生成或猜测人工标签。若使用模型辅助评审，单独记录裁判模型、提示词、版本、温度及与人工的一致性，不把它当成已验证的人工真值。

## 统一判定规则

1. `format_failure`：输出解析失败。单独计数，不把缺失标签映射成正常类别。
2. `abstention`：明确拒绝或表示无法作答。CUAD 的 absent、NLI 的 neutral 不是自动弃权；“不存在这个条款”本身仍是可核对的任务断言。
3. `substantive`：其余实质性答案。将答案拆为最小可判断的事实断言，同时检查预测标签所表达的任务结论。
4. 每个断言选 `supported`、`contradicted` 或 `not_enough_information`；只用原合同及给定问题，不补充外部法律常识。引用原文字符区间 `[start,end)` 和原句，确认 `source_contract[start:end]` 与摘录一致。未找到支持时记录检索/审阅说明，不能制造证据区间。
5. 区分实体、日期/期限、条件/例外、义务范围、引用错误、证据遗漏等错误标签。允许多标签，但统计时说明分母。错误标签只作描述，不能由关键词自动推导事实真伪。

建议每个评审记录使用以下结构；`null` 和空列表表示尚未评审：

```json
{
  "annotator": "reviewer_A",
  "response_type": "substantive",
  "claims": [
    {
      "claim": "待填写：答案中的原子断言",
      "verdict": null,
      "evidence_spans": [],
      "quoted_evidence": [],
      "error_types": [],
      "rationale": ""
    }
  ],
  "all_claims_reviewed": false,
  "needs_adjudication": null,
  "notes": ""
}
```

## 指标与分母

- **答案级未支持率（本方案的幻觉率口径）**：在全部断言都评完的实质性答案中，至少含 1 个 contradicted 或 not_enough_information 断言的答案数 / 已完整评审的实质性答案数。同时公布两个类别分别的数量；分母为 0 时返回 null。
- **断言级未支持率**：上述两类断言数 / 已评审断言总数。不能与答案级指标混写。
- **评审覆盖率**：已完成判定的输出数 / 抽取输出数；对格式失败和弃权单独列数量与占比。报告采样设计和样本数，不把尚未评审当成 supported。
- **答案完整性**：对有金标目标证据的样本，先定义应该回答的必需事实单元，再统计正确覆盖比例。逐字复制源合同、标签正确、证据召回均不能直接替代它。
- **弃权精确度**：仅在明确的“应当弃权”人工标注规则下计算；目前两个任务的 absent/neutral 不提供该真值，先保持 null。
- **修订成功率**：按同一评审标准对比初始与最终答案；仅在初始确有错误的样本中，统计错误消除且未引入新错误的比例。`revisions_made / hallucinations_detected` 不属于这个指标。

两名评审者先独立标注同一小批校准样本，再核对分歧并冻结规范。正式报告记录一致性和裁决过程。跨方法比较使用配对样本；如果要做区间或显著性检验，应考虑同一合同多个问题的聚类，不能把所有行默认当作独立观测。此项在拿到逐条数据和实际标注前不计算。
