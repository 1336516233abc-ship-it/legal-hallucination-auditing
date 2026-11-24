# Group 55 最终评估修复记录

**日期** 2026 年 11 月 25 日。项目为 Legal Hallucination Auditing with Agentic Retrieval for Contract Review。本次从 GitHub `main` 的 `ab185fa` 开始修复，涉及旧评估器、全量轨迹复算和最终报告。

## 旧评估器的修复

| 核对发现 | 修复与回归结果 |
| --- | --- |
| `invalid` 曾在 `neutral` 样本上计为正确 | 依据 ContractNLI、CUAD 各自的合法标签集判分。非法标签、跨任务标签和解析失败均进入分母并计错。 |
| CUAD 两类反例的 Macro-F1 曾算成 0.33 | Macro-F1 只在对应任务的合法类别上求平均，该反例修正为 0.50。 |
| 空参考答案会抛异常 | 跳过空参考。有参考而没有有效输出时，词集合召回记零。该词面指标单独输出。 |
| 两个样本各命中一段，旧 Hit Rate 返回 0.50 | 复用原文字符区间评分，Recall@k 与 Hit Rate 使用同一个 `k`。Hit Rate 按样本计数，该反例修正为 1.00。缺少可核验位置时，证据指标保持未测。 |

从 `verification.claims` 读取初答检查记录，分别统计有效检查中的不支持比例、检查总数和失败数。只有预测明确弃权、参考给出明确判断且分母非零时才计算弃权精确度。独立终答幻觉率、语义完整性和语义修订成功率仍缺独立评分。代码与回归用例分别见 [评估器](../evaluation/metrics.py) 和 [回归测试](../tests/test_legacy_metrics.py)。

## 全量复算与报告更正

从 Llama 3 和 Mistral 原始轨迹复算 **18,666 条预测、37,834 次模型调用**。两种模型在 ContractNLI 和 CUAD 上的三种方法共 **12 组**，准确率、Macro-F1 和格式通过数均与归档一致。全过程复用已有轨迹。

| 方法 | 输入 token | 生成 token | 合计 |
| --- | ---: | ---: | ---: |
| Direct Prompting | 16,726,643 | 347,613 | 17,074,256 |
| Standard RAG | 17,561,928 | 409,452 | 17,971,380 |
| Verified Agentic RAG | 50,416,305 | 2,050,494 | 52,466,799 |
| 总计 | 84,704,876 | 2,807,559 | 87,512,435 |

Standard RAG 的 token 用量为 Direct 的 1.05 倍，Agent 为 Standard RAG 的 2.92 倍。小样本消融单列，实际开支以 AutoDL 账单为准。最终报告保留 Llama 3 在 ContractNLI 上从 42.52% 提高到 55.52% 的结果，同时列出其余三个组合的下降。报告移除了无法追溯的人工复查说法，区分任务标签变化、初答 verifier 判断及独立语义评估，并替换附录中的示例哈希。[复算结果](validation/final_submission_trace_audit.json) · [报告 Markdown](../FINAL_REPORT.md) · [正式报告 PDF](final_report/COMP7608B-Group55-Final-Report.pdf)

## 验收与复现

在同一台 Mac 上解压课程提交包，使用全新 Python 3.11 环境和已有的 Mistral 权重运行。**188 项自动化测试通过**。双模型实验的 dry run 生成 6 个预定任务。真实 Gradio 服务正确处理 30 天通知和责任上限示例。60 天改条件示例产生非法标签并拦截答案，空合同被拒绝。测试注入另覆盖越界引用和模型异常后的下一次请求恢复。其他机器的安装和推理未验收。[解压验收原始记录](validation/final_submission_acceptance.json)

```bash
python -m pip install -r requirements-retrieval.txt -r requirements-ui.txt
python -m pip install 'scikit-learn>=1.3' 'torch>=2.0.0' 'transformers==5.23.0' 'requests>=2.31.0'
python -m pytest -q tests docs/validation/part4_2026-11-23/test_local_delivery.py
python scripts/verify_full_trace_export.py /path/to/evaluation_traces
```

`evaluation_traces` 取自课程提交包，包含两个 gzip JSONL 和导出清单。GitHub 保存本次修复、复算结果、报告和验收记录。课程提交 ZIP、原始逐条轨迹及模型权重另行保存。解压验收记录中的版本指纹指向已验收的课程提交包。[课程提交与启动说明](FINAL_SUBMISSION.md)
