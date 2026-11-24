# Part 4 实验结果汇总

本页为实验交付清单与数值表，未撰写课程报告。

- 新跑：36 组完整测试集检索评测；两个原版模型各 30 条 Agent 消融输出，共 60 条。
- 归档复核：两模型 × 两数据集 × 三方法的 18,666 条全量结果；核对了数据哈希和汇总混淆矩阵，未取回服务器逐条轨迹。
- Agent 消融使用既有的固定 10 例（ContractNLI 6、CUAD 4），每个变体使用同样的样本。小样本结果只支持流程检查，不能外推为全测试集性能。
- 独立幻觉率、答案完整性和拒答精确率仍未评估，不将缺失值写成 0。

## 新跑 Agent 组件消融

两个模型均为固定 revision、未量化的原版权重，MLX 本机推理，温度 0、seed 42、输入 4,096、单次输出 512 token。Full 开启验证和修订；No verification 同时关闭验证与依赖验证的修订；No revision 保留验证。

|模型|数据集|变体|标签正确|格式通过|Macro-F1|平均秒/条|生成 token 总数|修订次数|
|---|---|---|---:|---:|---:|---:|---:|---:|
|mistral|contractnli|full|3/6|5/6|0.4444|61.60|2127|4|
|mistral|contractnli|no_verification|3/6|5/6|0.4444|22.24|751|0|
|mistral|contractnli|no_revision|3/6|5/6|0.4444|61.34|1879|0|
|mistral|cuad|full|3/4|4/4|0.7333|53.45|943|3|
|mistral|cuad|no_verification|3/4|4/4|0.7333|24.05|467|0|
|mistral|cuad|no_revision|3/4|4/4|0.7333|42.96|773|0|
|llama3|contractnli|full|6/6|6/6|1.0000|68.93|2101|3|
|llama3|contractnli|no_verification|5/6|6/6|0.8222|29.64|967|0|
|llama3|contractnli|no_revision|5/6|6/6|0.8222|55.81|1666|0|
|llama3|cuad|full|1/4|4/4|0.2000|35.74|631|1|
|llama3|cuad|no_verification|1/4|4/4|0.2000|25.84|490|0|
|llama3|cuad|no_revision|1/4|4/4|0.2000|33.83|615|0|

格式失败保留在分母。token 数合计所有模型调用；修订次数只说明修订发生，不代表修订成功。首次组的耗时可能包含权重首次加载，不能用于严谨的跨变体速度结论，也不能与服务器 vLLM 耗时直接比较。

## 新跑检索评测

36 组与归档指标的最大绝对差异：`5.55e-16`。运行 JSON 没有固定编码器 revision；离线运行所用的唯一缓存版本已另行核对并记录在 `retrieval_environment.json`，依赖版本在运行 JSON 中。

|数据集|检索器|分块|k|证据召回率|有效样本|
|---|---|---|---:|---:|---:|
|contractnli|bm25|clause_level|3|93.76%|1188|
|contractnli|bm25|clause_level|5|98.62%|1188|
|contractnli|bm25|clause_level|10|100.00%|1188|
|contractnli|bm25|fixed_length|3|90.81%|1188|
|contractnli|bm25|fixed_length|5|97.89%|1188|
|contractnli|bm25|fixed_length|10|99.97%|1188|
|contractnli|dense|clause_level|3|91.13%|1188|
|contractnli|dense|clause_level|5|97.44%|1188|
|contractnli|dense|clause_level|10|99.97%|1188|
|contractnli|dense|fixed_length|3|89.48%|1188|
|contractnli|dense|fixed_length|5|96.33%|1188|
|contractnli|dense|fixed_length|10|99.85%|1188|
|contractnli|hybrid|clause_level|3|94.80%|1188|
|contractnli|hybrid|clause_level|5|98.62%|1188|
|contractnli|hybrid|clause_level|10|99.97%|1188|
|contractnli|hybrid|fixed_length|3|92.20%|1188|
|contractnli|hybrid|fixed_length|5|98.63%|1188|
|contractnli|hybrid|fixed_length|10|99.97%|1188|
|cuad|bm25|clause_level|3|51.83%|599|
|cuad|bm25|clause_level|5|64.24%|599|
|cuad|bm25|clause_level|10|82.26%|599|
|cuad|bm25|fixed_length|3|51.12%|599|
|cuad|bm25|fixed_length|5|61.75%|599|
|cuad|bm25|fixed_length|10|82.31%|599|
|cuad|dense|clause_level|3|45.72%|599|
|cuad|dense|clause_level|5|59.41%|599|
|cuad|dense|clause_level|10|81.09%|599|
|cuad|dense|fixed_length|3|43.11%|599|
|cuad|dense|fixed_length|5|56.45%|599|
|cuad|dense|fixed_length|10|78.02%|599|
|cuad|hybrid|clause_level|3|51.65%|599|
|cuad|hybrid|clause_level|5|63.77%|599|
|cuad|hybrid|clause_level|10|82.86%|599|
|cuad|hybrid|fixed_length|3|48.98%|599|
|cuad|hybrid|fixed_length|5|60.47%|599|
|cuad|hybrid|fixed_length|10|81.12%|599|

召回率要求完整金标区间被检索块并集覆盖；无金标证据的 903 条 ContractNLI、421 条 CUAD 不进入该指标分母。召回率不等于答案支持率。

## 归档主实验

详见 [12 组主实验与旧消融审计表](archived_result_audit/results_tables.md)。原服务器已完成全量生成，本次没有重新运行这 18,666 条预测。旧 `priority4` 消融有样本数据集错配与 CUAD 标签协议不符，不能用于受控对照；本次新跑记录单独保存。

## 可回查文件

- `summary.json`：机器可读的汇总、配对正确性变化及输入文件 SHA256。
- `agent_ablation/<model>/manifest.json`：权重哈希、代码哈希、环境、样本 ID。
- `agent_ablation/<model>/<dataset>/<variant>/predictions.jsonl`：逐条预测、评估和完整模型调用轨迹。
- `bm25_full_rerun.json`、`dense_hybrid_full_rerun.json`：36 组全量检索结果和失败样例。
- `tests.xml`：自动化测试记录，与真实模型输出分开。
