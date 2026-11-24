# 项目演示脚本

> **旧稿，已停止作为当前演示依据（2026-11-23）。** 下文的“100% 修订成功率”“显著减少幻觉”和旧耗时/token 说法未经独立评估支持，不能直接用于汇报；时长安排也须以课程通知为准。当前演示请使用 [正式演示指引](docs/DEMO_RUNBOOK.md) 和 [已核对的结果与限制](docs/validation/part4_2026-11-23/RQ_FINDINGS.md)。以下内容仅保留作历史记录。

## 演示时间：15-20分钟

---

## Part 1: 问题背景 (2分钟)

### 幻灯片1：法律AI的幻觉问题
```
场景：律师使用ChatGPT分析合同

问题："合同中提到的违约金是多少？"
❌ 错误回答："违约金为合同金额的20%"
   (实际上合同里根本没提到违约金)

风险：
- 法律建议错误 → 客户损失
- 合规性问题
- 法律责任
```

### 幻灯片2：研究问题
```
RQ1: RAG能减少幻觉吗？
RQ2: 验证+修订能进一步改进吗？⭐ (我们的创新)
RQ3: 什么类型的条款最容易产生幻觉？
```

---

## Part 2: 系统演示 (10分钟)

### 演示方式A：实时运行（推荐）

**准备工作：**
```bash
# 启动前检查环境
python verify_env.py

# 选择一个简单的测试用例
cd C:\Users\HUAWEI\Desktop\legal-hallucination-auditing
```

**演示命令（选择其一）：**

```bash
# 方案1：使用UI界面（最佳）
python ui/app.py --model llama3 --port 7860
# 然后在浏览器打开 http://localhost:7860

# 方案2：运行单个样本对比
python test_agentic_rag_priority3.py --sample_size 1 --verbose

# 方案3：使用已有测试脚本
python -c "
from agent.agentic_rag import AgenticRAG
from utils.llm_interface import LLMInterface
from retrieval.retrievers.retriever import BM25Retriever

# 初始化
llm = LLMInterface(model_name='llama3')
agent = AgenticRAG(llm=llm, enable_verification=True, enable_revision=True)

# 测试用例
query = 'Can this contract be terminated early?'
contract = '''
TERMINATION CLAUSE:
Either party may terminate this agreement with 30 days written notice.
Early termination requires mutual consent and payment of termination fee.
'''

# 运行演示
result = agent.run(query, contract)
print('=== 生成的答案 ===')
print(result['answer'])
print('\n=== 验证结果 ===')
print(result['verification'])
print('\n=== 修订历史 ===')
print(result.get('revision_history', 'No revision needed'))
"
```

### 演示方式B：展示已有结果

**展示priority4的实验结果：**

```bash
# 打开结果文件
cd experiments/results/priority4/

# 展示对比数据
cat comprehensive_summary.json | python -m json.tool
```

**重点展示：**
1. **Full Agentic RAG vs No Verification**
   - 幻觉检测率：15-25% vs 0%
   - 修订成功率：100%

2. **具体案例分析**
   - 打开 `agentic_rag_full_predictions.json`
   - 找一个检测到幻觉并成功修订的例子
   - 展示before/after对比

---

## Part 3: 实验结果 (5分钟)

### 幻灯片3：主要发现

**表格：系统性能对比**
```
| 系统配置         | 速度  | 幻觉检测 | 修订能力 | Token成本 |
|-----------------|-------|---------|---------|----------|
| 直接提示         | 最快   | ❌ 无    | ❌ 无   | 最低     |
| 标准RAG         | 快     | ❌ 无    | ❌ 无   | 中等     |
| 无验证Agentic    | 快     | ❌ 无    | ❌ 无   | 中等     |
| 完整Agentic ⭐  | 2-4x慢 | ✅ 15-25%| ✅ 100% | +9-14%  |
```

### 幻灯片4：关键数据

**ContractNLI数据集：**
- 测试样本：20个
- 检测到幻觉：3个 (15%)
- 成功修订：3个 (100%)
- 额外成本：+9% tokens

**CUAD数据集：**
- 测试样本：20个
- 检测到幻觉：5个 (25%)
- 成功修订：5个 (100%)
- 额外成本：+14% tokens

### 幻灯片5：真实案例

**案例1：检测到的幻觉**
```
问题："合同中提到的赔偿上限是多少？"

初始答案（有幻觉）：
"赔偿上限为合同金额的50%。" ❌

验证结果：
⚠️ 声明未得到证据支持

修订后答案：
"根据第12条，赔偿责任不超过最近12个月支付的费用。" ✅
```

---

## Part 4: 技术亮点 (2分钟)

### 幻灯片6：创新点

**1. 细粒度验证机制**
- 将答案拆分成独立声明
- 逐条验证是否有证据支持
- 标记：支持/矛盾/未提及

**2. 自动修订流程**
- 检测到幻觉 → 自动重新检索
- 使用更精确的查询
- 生成修正后的答案

**3. 跨数据集验证**
- ContractNLI (逻辑推理)
- CUAD (信息抽取)
- 两种任务都有效

---

## Part 5: 总结与Q&A (3分钟)

### 幻灯片7：总结

**主要贡献：**
1. ✅ 证明了Agentic RAG能显著减少幻觉
2. ✅ 100%的幻觉修订成功率
3. ✅ 合理的性能开销（仅+9-14% tokens）
4. ✅ 跨任务泛化能力

**实际应用价值：**
- 法律文档分析
- 合同审查自动化
- 合规性检查
- 风险评估

**局限性（诚实很重要）：**
- 样本量较小（每个数据集20个）
- 单一模型测试
- 速度有2-4倍开销

### Q&A准备

**预期问题：**

1. **"为什么不用更大的数据集？"**
   答：计算资源和时间限制，但在20个样本上已经展示了一致的效果。

2. **"速度慢2-4倍，实际可用吗？"**
   答：在法律场景下，准确性比速度更重要。而且token成本只增加9-14%。

3. **"能否用在其他领域？"**
   答：可以。我们验证了两种不同任务类型（NLI和QA），理论上可扩展到医疗、金融等其他高风险领域。

4. **"如何评估修订质量？"**
   答：通过对比ground truth标签，计算准确率、F1等指标。目前修订成功率100%。

---

## 演示检查清单

### 演示前准备（提前1小时）

- [ ] **环境测试**
  ```bash
  python verify_env.py
  source venv/bin/activate  # 或 venv\Scripts\activate
  ```

- [ ] **数据检查**
  ```bash
  ls data/processed/  # 确认数据存在
  ls experiments/results/priority4/  # 确认结果存在
  ```

- [ ] **UI测试（如果使用）**
  ```bash
  python ui/app.py --model llama3 --port 7860
  # 测试是否能正常访问
  ```

- [ ] **备份演示材料**
  - PPT/幻灯片
  - 代码片段
  - 结果JSON文件
  - 截图/视频录屏（备用）

### 演示时注意事项

1. **时间控制**
   - 背景：2分钟
   - 演示：10分钟（重点）
   - 结果：5分钟
   - Q&A：3分钟

2. **重点突出**
   - 强调"验证+修订"是你们的创新
   - 展示100%修订成功率
   - 展示真实案例对比

3. **备选方案**
   - 如果实时运行失败，立即切换到预先准备的结果文件
   - 准备录屏视频作为最后备份

4. **互动技巧**
   - 让观众选择一个测试问题（提前准备几个）
   - 边运行边解释每个步骤
   - 邀请观众猜测哪个答案有幻觉

---

## 技术准备

### 最小演示环境

```bash
# 必需的文件
- data/processed/contractnli_test.json
- experiments/results/priority4/
- agent/agentic_rag.py
- ui/app.py (可选)

# 必需的依赖
pip install -r requirements.txt
```

### 应急预案

**场景1：网络/API失败**
- 使用预先生成的结果文件
- 展示已保存的predictions.json

**场景2：代码运行错误**
- 切换到结果分析模式
- 展示统计数据和案例研究

**场景3：时间不足**
- 跳过技术细节
- 直接展示最终结果对比
- 重点：100%修订成功率

---

## 可选：创建演示视频

如果现场演示风险太大，可以预先录制：

```bash
# 使用OBS Studio或其他录屏工具
# 录制内容：
1. 系统启动
2. 输入测试问题
3. 展示完整工作流程
4. 显示验证和修订结果
5. 对比有/无验证的差异
```

---

**祝演示成功！🎉**
