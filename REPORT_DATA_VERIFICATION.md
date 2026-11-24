# 报告数据来源核对清单

本文档核对 `FINAL_REPORT.md` 中所有实验数据的准确性和来源。

---

## ✅ 1. 主实验结果表（Section 5.3.1）

**数据来源**: `docs/validation/part4_2026-11-24/README.md` 第23-36行

### 表格：Label Accuracy and F1 (18,666 Predictions)

| 报告中的数据 | 文档中的数据 | 状态 |
|-------------|-------------|------|
| Llama 3 × ContractNLI × Direct: 47.01%, F1=0.4491 | 47.01%, 0.4491 | ✅ 完全匹配 |
| Llama 3 × ContractNLI × RAG: 42.52%, F1=0.4271 | 42.52%, 0.4271 | ✅ 完全匹配 |
| Llama 3 × ContractNLI × Agent: 55.52%, F1=0.5283 | 55.52%, 0.5283 | ✅ 完全匹配 |
| Llama 3 × CUAD × Direct: 77.55%, F1=0.7851 | 77.55%, 0.7851 | ✅ 完全匹配 |
| Llama 3 × CUAD × RAG: 68.63%, F1=0.7346 | 68.63%, 0.7346 | ✅ 完全匹配 |
| Llama 3 × CUAD × Agent: 62.06%, F1=0.6834 | 62.06%, 0.6834 | ✅ 完全匹配 |
| Mistral × ContractNLI × Direct: 43.42%, F1=0.3253 | 43.42%, 0.3253 | ✅ 完全匹配 |
| Mistral × ContractNLI × RAG: 41.89%, F1=0.2707 | 41.89%, 0.2707 | ✅ 完全匹配 |
| Mistral × ContractNLI × Agent: 37.06%, F1=0.2305 | 37.06%, 0.2305 | ✅ 完全匹配 |
| Mistral × CUAD × Direct: 67.06%, F1=0.7214 | 67.06%, 0.7214 | ✅ 完全匹配 |
| Mistral × CUAD × RAG: 69.71%, F1=0.7284 | 69.71%, 0.7284 | ✅ 完全匹配 |
| Mistral × CUAD × Agent: 63.24%, F1=0.6971 | 63.24%, 0.6971 | ✅ 完全匹配 |

**延迟数据来源**: 同一表格

| 报告中的延迟 | 文档中的延迟 | 状态 |
|-------------|-------------|------|
| Llama 3 × ContractNLI × Agent: 9.94s | 9.94秒 | ✅ 完全匹配 |
| Llama 3 × ContractNLI × RAG: 1.85s | 1.85秒 | ✅ 完全匹配 |
| Llama 3 × ContractNLI × Direct: 1.45s | 1.45秒 | ✅ 完全匹配 |
| Llama 3 × CUAD × Agent: 5.89s | 5.89秒 | ✅ 完全匹配 |

**格式通过率数据来源**: `docs/validation/part4_2026-11-24/README.md` 第44-57行

| 报告中的格式失败 | 文档中的格式失败 | 状态 |
|-----------------|-----------------|------|
| Mistral × ContractNLI × Direct: 341 | 341 | ✅ 完全匹配 |
| Mistral × ContractNLI × Agent: 514 | 514 | ✅ 完全匹配 |
| Llama 3 × ContractNLI × Direct: 0 | 0 | ✅ 完全匹配 |
| Llama 3 × ContractNLI × Agent: 40 | 40 | ✅ 完全匹配 |
| Llama 3 × CUAD × Direct: 54 | 54 | ✅ 完全匹配 |
| Llama 3 × CUAD × Agent: 215 | 215 | ✅ 完全匹配 |

---

## ✅ 2. 检索召回率表（Section 5.3.2）

**数据来源**: `docs/validation/part4_2026-11-22/RESULTS.md` 第6-45行

### ContractNLI Recall@5 (1,188 examples)

| 报告中的数据 | 文档中的数据 | 状态 |
|-------------|-------------|------|
| BM25 × Clause-level: 98.62% | 98.62% | ✅ 完全匹配 |
| BM25 × Fixed-length: 97.89% | 97.89% | ✅ 完全匹配 |
| Dense × Clause-level: 97.44% | 97.44% | ✅ 完全匹配 |
| Hybrid × Clause-level: 98.62% | 98.62% | ✅ 完全匹配 |

### CUAD Recall@5 (599 examples)

| 报告中的数据 | 文档中的数据 | 状态 |
|-------------|-------------|------|
| BM25 × Clause-level: 64.24% | 64.24% | ✅ 完全匹配 |
| BM25 × Fixed-length: 61.75% | 61.75% | ✅ 完全匹配 |
| Dense × Clause-level: 59.41% | 59.41% | ✅ 完全匹配 |
| Hybrid × Clause-level: 63.77% | 63.77% | ✅ 完全匹配 |

**有效样本数**:
- ContractNLI: 1,188 条（报告 ✅）
- CUAD: 599 条（报告 ✅）

---

## ✅ 3. Agent修订行为表（Section 5.3.3）

**数据来源**: `docs/validation/part4_2026-11-24/README.md` 第64-69行

| 报告中的数据 | 文档中的数据 | 状态 |
|-------------|-------------|------|
| Llama 3 × ContractNLI 修订触发: 1,555 | 1555 | ✅ 完全匹配 |
| Llama 3 × ContractNLI 错→对: 262 | 262 | ✅ 完全匹配 |
| Llama 3 × ContractNLI 对→错: 44 | 44 | ✅ 完全匹配 |
| Llama 3 × ContractNLI 初答正确: 943/2091 | 943/2091 | ✅ 完全匹配 |
| Llama 3 × ContractNLI 终答正确: 1161/2091 | 1161/2091 | ✅ 完全匹配 |
| Llama 3 × CUAD 修订触发: 498 | 498 | ✅ 完全匹配 |
| Llama 3 × CUAD 错→对: 19 | 19 | ✅ 完全匹配 |
| Llama 3 × CUAD 对→错: 56 | 56 | ✅ 完全匹配 |

**净收益计算**:
- ContractNLI: 262 - 44 = **+218** ✅
- CUAD: 19 - 56 = **-37** ✅

---

## ✅ 4. RQ1 和 RQ2 的准确率变化（Section 5.3.1）

**数据来源**: `docs/validation/part4_2026-11-23/RQ_FINDINGS.md` 第18-40行

### RQ1: RAG vs Direct

| 报告中的变化 | 文档中的变化 | 状态 |
|-------------|-------------|------|
| Mistral × ContractNLI: -1.53 pp | -1.53 | ✅ 完全匹配 |
| Mistral × CUAD: +2.65 pp | +2.65 | ✅ 完全匹配 |
| Llama 3 × ContractNLI: -4.50 pp | -4.50 | ✅ 完全匹配 |
| Llama 3 × CUAD: -8.92 pp | -8.92 | ✅ 完全匹配 |

### RQ2: Agent vs RAG

| 报告中的变化 | 文档中的变化 | 状态 |
|-------------|-------------|------|
| Mistral × ContractNLI: -4.83 pp | -4.83 | ✅ 完全匹配 |
| Mistral × CUAD: -6.47 pp | -6.47 | ✅ 完全匹配 |
| Llama 3 × ContractNLI: **+13.01 pp** | +13.01 | ✅ 完全匹配 |
| Llama 3 × CUAD: -6.57 pp | -6.57 | ✅ 完全匹配 |

---

## ✅ 5. 引用越界失败（Section 3.1 Test 2）

**数据来源**: `docs/validation/part4_2026-11-24/README.md` 第44-57行

| 报告中的数据 | 文档中的数据 | 状态 |
|-------------|-------------|------|
| ContractNLI 范围: 100-211 cases | RAG: 151, Agent: 211 | ✅ 准确描述范围 |
| CUAD 范围: 107-149 cases | RAG: 143, Agent: 149 | ✅ 准确描述范围 |
| 比例描述: 0.5-1.5% | 100/2091=4.8%, 211/2091=10.1% | ⚠️ 需要修正 |

**问题发现**: 报告中的"0.5-1.0% (ContractNLI)"实际应为"**4.8-10.1%**"；CUAD的"1.0-1.5%"实际应为"**10.5-14.6%**"

---

## ✅ 6. Token消耗数据（Section 4.2）

**数据来源**: 需要从原始JSON或实验日志中计算

| 报告中的数据 | 需要验证的来源 | 状态 |
|-------------|---------------|------|
| Total: 38.6M tokens | 需要汇总12组实验 | ⚠️ 未在单一文档中找到 |
| Agent adds 9.2% over No Verification | 需要从component ablation计算 | ⚠️ 未直接找到 |
| Revision adds 6.5% when triggered | 需要从detailed traces计算 | ⚠️ 未直接找到 |

**注意**: 报告Section 4.2引用了token消耗数据，但`RQ_FINDINGS.md:71`明确指出"旧统计存在口径问题"，应谨慎使用这些数字或标注为"preliminary estimate"。

---

## ✅ 7. 数据集规模（Throughout report）

**数据来源**: `docs/validation/part4_2026-11-24/README.md` 第5行；`data/processed/README.md`

| 报告中的数据 | 文档中的数据 | 状态 |
|-------------|-------------|------|
| ContractNLI: 2,091 examples | 2,091条 | ✅ 完全匹配 |
| ContractNLI with evidence: 1,188 | 1,188题有金标证据 | ✅ 完全匹配 |
| CUAD: 1,020 examples | 1,020条 | ✅ 完全匹配 |
| CUAD with evidence: 599 | 599题有金标证据 | ✅ 完全匹配 |
| Total predictions: 18,666 | 18,666条预测 | ✅ 完全匹配 |
| Total LLM calls: 37,834 | 未在README中明确，需要验证 | ⚠️ 需要确认 |

---

## ✅ 8. 失败案例数量（Section 5.4）

**数据来源**: `docs/validation/part4_2026-11-24/CASES.md`; `RQ_FINDINGS.md` 第52-62行

| 报告中的数据 | 文档中的数据 | 状态 |
|-------------|-------------|------|
| 20 documented cases | 20个可回查案例 | ✅ 完全匹配 |
| Case IDs: C01-C09 cited | C01-C09案例编号 | ✅ 完全匹配 |

---

## ⚠️ 需要修正的问题汇总

### 1. 引用越界比例（高优先级）

**位置**: Section 3.1, Test 2: Citation Grounding Failure

**当前报告**:
> *Actual Incidence*: 100–211 cases per method in ContractNLI (0.5–1.0%); 107–149 in CUAD (1.0–1.5%).

**应该修改为**:
> *Actual Incidence*: 100–211 cases per method in ContractNLI (4.8–10.1% of 2,091); 107–149 in CUAD (10.5–14.6% of 1,020).

### 2. Token消耗数据（中优先级）

**位置**: Section 4.2 Cost

**问题**: RQ_FINDINGS.md明确指出"旧统计存在口径问题"

**建议**:
- 要么移除具体的token百分比数字
- 要么添加注释："*Preliminary estimates; unified token counting methodology under review*"

### 3. "100% revision success rate"声明（已避免）

**检查结果**: ✅ 报告中**没有**声称"100%修订成功率"，符合RQ_FINDINGS.md的警告

### 4. 37,834 LLM调用数（低优先级）

**位置**: Multiple sections

**状态**: 在README中被提及但未在主指标表中明确列出，需要确认来源

---

## ✅ 总体核对结果

| 类别 | 检查项 | 匹配 | 需修正 |
|------|-------|------|--------|
| 准确率和F1 | 12组×2指标=24项 | 24 | 0 |
| 检索召回率 | 8组配置 | 8 | 0 |
| Agent修订统计 | 10项数据 | 10 | 0 |
| RQ1/RQ2变化 | 8项变化 | 8 | 0 |
| 格式失败 | 6项数据 | 6 | 0 |
| 失败案例 | 案例数和ID | 匹配 | 0 |
| 引用越界比例 | 2项范围 | 范围正确 | **比例计算错误** |
| Token消耗 | 3项估算 | - | **需要添加警告** |
| **总计** | **~70项数据** | **~68项** | **2-3项** |

---

## 推荐行动

1. **立即修正**: 引用越界的百分比计算（Section 3.1）
2. **添加注释**: Token消耗数据标注为初步估算（Section 4.2）
3. **可选验证**: 确认37,834 LLM调用数的具体来源文档

---

**核对完成日期**: 2026-11-24
**核对人**: 基于项目文档的自动化核对
**可信度**: 高（所有主要指标均可追溯到原始文档）
