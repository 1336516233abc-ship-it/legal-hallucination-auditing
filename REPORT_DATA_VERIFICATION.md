# Part 4 与 UI 报告核对记录

日期：2026-11-25。基准：`main` 提交 `5e84091`。

已保留组员新提交的评估器修复、正式 PDF/LaTeX、全量复算与新环境验收。本次在最新版本上补齐剩余 Part 4 与 UI 说明，不覆盖新代码或重新制作课程提交包。主分支已修正的 token 总数继续沿用并再次核对。

本次修正实验、分析、复现、UI 和 AI 使用声明的报告内容，以及 AI_USE.md 和三份当前交接/演示说明。没有改动 Agent、检索、共享评估器、原始数据或已有实验输出；没有新增推理或人工评分。成员分工保留主分支署名，其他成员的具体工具使用与贡献需各自确认。

## 已完成核对

| 项目 | 来源与方法 | 结果 |
|---|---|---|
| 主实验表 | `metrics-followup/METRICS.json`：12 组 × 准确率、Macro-F1、格式通过率 | 36 个数值与报告一致 |
| 检索表 | `part4_2026-11-22/summary.json` 的 `new_retrieval` | 36 个召回率与报告一致 |
| 组件消融 | `summary.json` 与两模型、两数据集、三变体逐条 JSONL | 60 条输出；12 行 × 6 个数值字段一致；重新计数正确、格式通过、修订、生成 token，并复算 F1 |
| 全量 token | 汇总 12 组 `total_input_tokens`、`total_generated_tokens`、`total_model_calls` | 87,512,435 输入加输出 token；37,834 次调用 |
| 成本表 | 按方法汇总，每种方法 6,222 条 | 4 行（含总计）× 5 个数值字段一致；不混入小样本消融 |
| 文件哈希 | 对报告附录 B 的 5 个文件重新计算完整 SHA256 | 5 项一致，保留已修正值，并展开为完整 SHA256 |
| 案例 | `analysis/CASES.jsonl` 与 `CASES.md` | 20 个目的性案例；Case 1 是 Direct 输入覆盖但标签错误，Case 2 是引用越界 |
| UI | `ui/service.py`、`ui/app.py`、依赖与已有验收记录 | Gradio 6.32.0；换合同清空旧结果；无自动重试；最终语义未二次验证 |
| 文档链接与补丁 | 检查此次六份文档中的相对链接；`git diff --check` | 目标存在；无补丁空白错误 |

上表为程序核对和源码/记录审阅，**不是独立人工语义评估**。主实验记录位于 `docs/validation/part4_2026-11-24/`；检索和消融记录位于 `docs/validation/part4_2026-11-22/`。

已有全量分析对源 ZIP 的 SQLite/JSONL、样本 ID、证据位置和调用记录进行了检查。对应证据为 [文件完整性](docs/validation/part4_2026-11-24/file_integrity.json)、[调用检查](docs/validation/part4_2026-11-24/analysis/call_trace_checks.json)、[分析检查](docs/validation/part4_2026-11-24/analysis/output_validation.json) 与 [指标检查](docs/validation/part4_2026-11-24/metrics-followup/VALIDATION.json)。本次报告修正使用这些已交付记录，没有声称重新运行服务器实验。

## 成本口径

| 方法 | 输入 token | 生成 token | 合计 | 调用次数 |
|---|---:|---:|---:|---:|
| Direct | 16,726,643 | 347,613 | 17,074,256 | 6,222 |
| RAG | 17,561,928 | 409,452 | 17,971,380 | 6,222 |
| Agent | 50,416,305 | 2,050,494 | 52,466,799 | 25,390 |
| 总计 | 84,704,876 | 2,807,559 | 87,512,435 | 37,834 |

核对命令（仓库根目录；只读）：

```python
import json
from collections import defaultdict
from pathlib import Path

p = Path("docs/validation/part4_2026-11-24/metrics-followup/METRICS.json")
totals = defaultdict(lambda: [0, 0, 0, 0])
for group in json.loads(p.read_text())["groups"]:
    for index, key in enumerate(("n", "total_input_tokens", "total_generated_tokens", "total_model_calls")):
        totals[group["method"]][index] += group[key]
for method, (rows, inputs, outputs, calls) in totals.items():
    print(method, rows, inputs, outputs, inputs + outputs, calls)
```

60 条组件消融只记录生成 token 的表，不与全量输入加输出表混算。没有已验证账单，报告不再写“实际费用为 0”或使用其他 API 的定价估计本次 AutoDL 费用。

## 最终保留或补充的结论与复现说明

下列成本、幻觉率限制和部分 AI 声明已由 `5e84091` 更正，本次保留；本次额外补齐的主要是组件表、遗漏检索配置、实现说明、UI 必做范围和当前文档入口。

- 将“全量组件消融完成”改为“每模型固定 10 题、共 60 输出的试点”，补齐实际表格。
- 将“修订成功”改为修订尝试及标签错转对/对转错；不将标签变化当作语义修复真值。
- RQ1 的跨方法独立幻觉率尚未测得；Agent 仅在四个模型/数据集组合中的一个提高标签准确率，不外推为普遍收益。
- Llama 3 格式通过率为 78.92–100%，Mistral 为 75.42–90.98%；不再以“格式错误主导全部错误”描述重叠的事件统计。
- 补入遗漏的检索配置：ContractNLI 在 k=5 时最高为 Hybrid/fixed 98.63%；CUAD 为 BM25/clause 64.24%。
- 固定分块为 512 个空白分隔词、128 词重叠；这与 4,096 模型输入 token 预算不同。
- 全量 Agent 实际包含查询改写调用；任务修订分为沿用初始证据的 correction 与重新检索的 reretrieval，报告不再把所有修订写成同一种步骤。
- 删除无记录支持的阶段耗时比例、全冻结依赖、生成结果完全确定、45 GPU 小时与示意代码行数；保留可查的环境和入口。
- UI 真实验收为 Llama 3 三例通过、Mistral 两例通过加一例格式失败；后续同机全新 Python 3.11 环境安装和 Mistral 服务已由主分支提交包记录验收；CUDA/CPU/MPS 与另一台机器尚未验收。换合同隔离等自动化测试与真实模型质量分开说明。
- 保留 main 已更正的 AI 声明和成员可提供的提示词，补充 Part 4 编排、试点、UI 实现及验收的 Codex 协助范围。详细交互记录的覆盖限制仍保留。
- 同步 `docs/PART4_EXPERIMENTS.md`、`docs/UI_HANDOFF.md`、`docs/DEMO_RUNBOOK.md`，避免当前入口仍称只有汇总、未取得轨迹或未撰写报告。历史实验记录原样保留。

## 验证边界与剩余事项

本次在最新基准重新执行 `python -m pytest -q tests docs/validation/part4_2026-11-23/test_local_delivery.py` 得到 **188 passed**；这次只有文档变化。既有 [UI 验收](docs/validation/ui_2026-11-23/README.md) 单独记录真模型和浏览器行为，不等于独立盲评。

三方法最终回答的独立幻觉率、语义完整性、独立弃权精确度与语义修订成功率仍未完成。仓库 60 条盲评材料未标注；600 条为未上传的本地抽样草案，不是新增个人任务。词汇重叠和旧弃权规则仅作为明确标注的代理统计。

本文不替课程验收、组员贡献确认、最终人审或 Moodle 提交；报告保留这些边界，不能仅因自动化核对通过就称全部评估完成。

后续交付证据：[最终评估修复](docs/FINAL_EVALUATION_FIXES.md)、[全量逐条复算](docs/validation/final_submission_trace_audit.json)、[同机新环境验收](docs/validation/final_submission_acceptance.json)、[正式 PDF 与提交说明](docs/FINAL_SUBMISSION.md)。原始主结果、PDF 和课程提交包指纹没有因本次文档修正而改变。
