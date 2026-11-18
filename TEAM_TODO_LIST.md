# 团队待办事项清单 (Team TODO List)

**项目名称**: Legal Hallucination Auditing Agent
**截止日期**: 2026年11月25日
**剩余时间**: 7天
**当前日期**: 2026年11月18日

---

## 📅 总体时间规划

### 第1-3天 (11月18-20日): 开发与测试
- 环境搭建
- 各模块独立测试
- 解决集成问题

### 第4-5天 (11月21-22日): 运行实验
- 批量处理所有数据集
- 收集所有指标
- 运行消融实验

### 第6-7天 (11月23-25日): 分析与报告
- 分析结果
- 回答研究问题
- 撰写最终报告
- 验证可复现性

---

## 👤 Yin Fengling (3036807375)

### 负责模块
- `data/` - 数据处理模块
- `evaluation/` - 评估指标模块

### 📋 待办事项

#### 优先级 1 (第1天完成 - 11月18日)

**1.1 环境搭建**
- [ ] 克隆/进入项目目录
- [ ] 创建虚拟环境
  ```bash
  cd legal-hallucination-auditing
  python -m venv venv
  venv\Scripts\activate  # Windows
  ```
- [ ] 安装依赖
  ```bash
  pip install -r requirements.txt
  pip install -e .
  ```
- [ ] 下载NLTK数据
  ```bash
  python -c "import nltk; nltk.download('punkt')"
  ```

**1.2 数据集下载与处理**
- [ ] 下载ContractNLI数据集
  ```bash
  python data/contractnli_processor.py --download
  ```
- [ ] 处理ContractNLI测试集
  ```bash
  python data/contractnli_processor.py --process --split test
  ```
- [ ] 下载CUAD数据集
  ```bash
  python data/cuad_processor.py --download
  ```
- [ ] 处理CUAD测试集
  ```bash
  python data/cuad_processor.py --process --split test
  ```

**1.3 数据验证**
- [ ] 验证处理后的数据格式
  ```python
  from data.dataset_loader import get_dataset_statistics
  from data.contractnli_processor import ContractNLIProcessor

  processor = ContractNLIProcessor()
  examples = processor.load(split='test')
  stats = get_dataset_statistics(examples)
  print(stats)
  ```
- [ ] 检查数据文件是否在 `data/processed/` 目录
- [ ] 确认数据统计信息合理（样本数、标签分布等）

#### 优先级 2 (第2天完成 - 11月19日)

**2.1 实现评估指标**
- [ ] 测试 `metrics.py` 中的所有指标
  - Label Accuracy（标签准确率）
  - Macro-F1 和 Micro-F1
  - Evidence Recall@k（证据召回率）
  - Hallucination Rate（幻觉率）
  - Answer Completeness（答案完整度）
  - Abstention Precision（弃权精确度）
- [ ] 编写单元测试验证指标计算正确性
- [ ] 测试幻觉检测器 `hallucination_detector.py`

**2.2 准备评估脚本**
- [ ] 确保 `error_analysis.py` 可以正常运行
- [ ] 准备用于比较不同系统的分析脚本

#### 优先级 3 (第4-5天 - 11月21-22日)

**3.1 协助实验运行**
- [ ] 监控实验进度，确保评估指标正确计算
- [ ] 收集并整理所有实验的评估结果

#### 优先级 4 (第6-7天 - 11月23-25日)

**4.1 结果分析**
- [ ] 生成详细的指标报告
- [ ] 创建对比表格（Direct Prompting vs RAG vs Agentic RAG）
- [ ] 分析幻觉失败模式
- [ ] 协助撰写报告的"评估"部分

### ⚠️ 注意事项
- 如果数据集下载失败，可以手动从官网下载：
  - ContractNLI: https://stanfordnlp.github.io/contract-nli/
  - CUAD: https://www.atticusprojectai.org/cuad
- 确保处理后的数据格式与 `LegalExample` 类定义一致
- 与 Qin Mengmeng 协调，确认数据格式满足检索模块需求

### 🔗 依赖关系
- **无前置依赖** - 可以立即开始
- **被依赖** - Qin Mengmeng 和 Wang Zhengyao 需要你的数据处理结果
- **协作** - 与 Li Xinru 协作进行结果分析

---

## 👤 Qin Mengmeng (3036843769)

### 负责模块
- `retrieval/` - 检索系统与基线实现

### 📋 待办事项

#### 优先级 1 (第1天完成 - 11月18日)

**1.1 环境搭建**
- [ ] 同 Yin Fengling 的环境搭建步骤
- [ ] 等待 Yin Fengling 完成数据处理

**1.2 测试检索组件**
- [ ] 测试分块策略 `chunking.py`
  ```python
  from retrieval.chunking import FixedLengthChunker, ClauseLevelChunker

  # 测试固定长度分块
  chunker = FixedLengthChunker(chunk_size=512, overlap=128)
  chunks = chunker.chunk(contract_text)

  # 测试条款级分块
  clause_chunker = ClauseLevelChunker()
  clause_chunks = clause_chunker.chunk(contract_text)
  ```

- [ ] 测试BM25检索器
  ```python
  from retrieval.retrievers.retriever import BM25Retriever

  retriever = BM25Retriever()
  retriever.index(chunks)
  results = retriever.retrieve(query, top_k=5)
  ```

- [ ] 测试Dense检索器
  ```python
  from retrieval.retrievers.retriever import DenseRetriever

  dense_retriever = DenseRetriever()
  dense_retriever.index(chunks)
  results = dense_retriever.retrieve(query, top_k=5)
  ```

#### 优先级 2 (第2天完成 - 11月19日)

**2.1 实现并测试直接提示基线**
- [ ] 确保 `baseline_direct.py` 可以运行
- [ ] 在小样本上测试（5-10个例子）
  ```bash
  python retrieval/baseline_direct.py --model llama3 --dataset contractnli --sample_size 10
  ```
- [ ] 检查输出格式是否正确
- [ ] 验证LLM接口工作正常

**2.2 实现并测试标准RAG基线**
- [ ] 确保 `baseline_rag.py` 可以运行
- [ ] 测试BM25检索器 + 生成
  ```bash
  python retrieval/baseline_rag.py --model llama3 --retriever bm25 --top_k 5 --sample_size 10
  ```
- [ ] 测试不同的top-k值（3, 5, 10）
- [ ] 测试不同的分块策略

#### 优先级 3 (第3天完成 - 11月20日)

**3.1 完整基线测试**
- [ ] 在完整测试集上运行直接提示基线
  ```bash
  python retrieval/baseline_direct.py --model llama3 --dataset contractnli
  ```
- [ ] 在完整测试集上运行标准RAG基线
  ```bash
  python retrieval/baseline_rag.py --model llama3 --retriever bm25 --top_k 5 --dataset contractnli
  ```
- [ ] 保存中间结果和日志

**3.2 消融实验准备**
- [ ] 测试不同检索方法（BM25 vs Dense vs Hybrid）
- [ ] 测试不同分块策略（固定长度 vs 条款级）
- [ ] 测试不同top-k值（3, 5, 10）

#### 优先级 4 (第4-5天 - 11月21-22日)

**4.1 运行所有基线实验**
- [ ] 运行所有配置的实验
- [ ] 确保结果保存到 `experiments/results/` 目录

#### 优先级 5 (第6-7天 - 11月23-25日)

**5.1 结果整理**
- [ ] 整理检索效果分析（Evidence Recall@k）
- [ ] 对比不同检索策略的效果
- [ ] 协助撰写报告的"检索系统"部分

### ⚠️ 注意事项
- 如果内存不足，减小batch_size或chunk_size
- 如果检索速度慢，优先使用BM25而非Dense retrieval
- 确保检索结果包含原文引用和位置信息
- 保存所有中间输出用于调试

### 🔗 依赖关系
- **前置依赖** - 需要 Yin Fengling 完成数据处理
- **被依赖** - Wang Zhengyao 需要你的检索器接口
- **协作** - 与 Wang Zhengyao 协调检索器API接口

---

## 👤 Wang Zhengyao (3036795675)

### 负责模块
- `agent/` - 代理组件（Query Analyzer, Verifier, Reviser）

### 📋 待办事项

#### 优先级 1 (第1天完成 - 11月18日)

**1.1 环境搭建**
- [ ] 同 Li Xinru 的环境搭建步骤
- [ ] 等待 Li Xinru 完成数据处理

**1.2 测试查询分析器**
- [ ] 测试 `query_analyzer/analyzer.py`
  ```python
  from agent.query_analyzer.analyzer import QueryAnalyzer

  analyzer = QueryAnalyzer()
  result = analyzer.analyze("Can this contract be terminated early?")
  print(result)  # 应该返回查询类型、关键实体等
  ```
- [ ] 验证查询分类功能
- [ ] 验证查询细化功能

#### 优先级 2 (第2天完成 - 11月19日)

**2.1 测试答案生成器**
- [ ] 测试 `generator.py`
  ```python
  from agent.generator import AnswerGenerator
  from utils.llm_interface import LLMInterface

  llm = LLMInterface(model_name="llama3")
  generator = AnswerGenerator(llm)

  answer = generator.generate(query, evidence_chunks)
  print(answer)
  ```
- [ ] 确保生成的答案包含证据引用

**2.2 测试证据验证器**
- [ ] 测试 `verifier/verifier.py`
  ```python
  from agent.verifier.verifier import EvidenceVerifier

  verifier = EvidenceVerifier(llm)
  verification = verifier.verify(answer, evidence_chunks)
  print(verification)  # 应该返回支持/矛盾/未提及的声明
  ```
- [ ] 验证声明提取功能
- [ ] 验证声明验证逻辑

**2.3 测试答案修订器**
- [ ] 测试 `reviser/reviser.py`
  ```python
  from agent.reviser.reviser import AnswerReviser

  reviser = AnswerReviser(llm)
  revised = reviser.revise(answer, verification_results, retriever)
  print(revised)
  ```
- [ ] 验证重新检索逻辑
- [ ] 验证答案修订质量

#### 优先级 3 (第3天完成 - 11月20日)

**3.1 集成Agentic RAG工作流**
- [ ] 测试完整的 `agentic_rag.py` 流程
  ```python
  from agent.agentic_rag import AgenticRAG

  agent = AgenticRAG(
      llm=llm,
      enable_verification=True,
      enable_revision=True
  )

  result = agent.run(query, contract_text)
  print(result)
  ```
- [ ] 验证完整流程：分析 → 检索 → 生成 → 验证 → 修订
- [ ] 确保所有中间步骤都被保存

**3.2 小规模测试**
- [ ] 在5-10个样本上测试完整工作流
  ```bash
  python agent/agentic_rag.py --model llama3 --dataset contractnli --sample_size 10
  ```
- [ ] 检查输出格式
- [ ] 验证幻觉检测是否工作

#### 优先级 4 (第4-5天 - 11月21-22日)

**4.1 完整测试集运行**
- [ ] 在完整测试集上运行Agentic RAG
  ```bash
  python agent/agentic_rag.py --model llama3 --dataset contractnli
  ```
- [ ] 保存所有中间输出（验证结果、修订历史等）

**4.2 消融实验**
- [ ] 运行不启用验证的版本
  ```bash
  python agent/agentic_rag.py --model llama3 --no-verification
  ```
- [ ] 运行不启用修订的版本
  ```bash
  python agent/agentic_rag.py --model llama3 --no-revision
  ```

#### 优先级 5 (第6-7天 - 11月23-25日)

**5.1 结果分析**
- [ ] 分析验证器捕获的幻觉案例
- [ ] 分析修订器的改进效果
- [ ] 统计每个组件的贡献
- [ ] 协助撰写报告的"Agentic RAG"部分

### ⚠️ 注意事项
- 确保每个组件的输出格式一致
- 保存所有中间步骤用于分析
- 验证器的阈值可能需要调整
- 注意LLM调用次数和成本

### 🔗 依赖关系
- **前置依赖** - 需要 Yin Fengling 的数据和 Qin Mengmeng 的检索器
- **被依赖** -Li Xinru 需要你的实验结果
- **协作** - 与 Qin Mengmeng 协调检索器接口

---

## 👤 Li Xinru (3036807959)

### 负责模块
- `experiments/` - 实验编排
- `ui/` - 可选UI界面
- 最终报告撰写

### 📋 待办事项

#### 优先级 1 (第1-2天 - 11月18-19日)

**1.1 环境搭建**
- [ ] 同 Li Xinru 的环境搭建步骤

**1.2 测试实验框架**
- [ ] 确保 `experiments/run_experiments.py` 可以运行
- [ ] 测试小规模实验
  ```bash
  python experiments/run_experiments.py --mode all --sample_size 5
  ```
- [ ] 验证结果保存路径和格式

**1.3 准备UI界面（可选）**
- [ ] 测试 `ui/app.py`
  ```bash
  python ui/app.py --model llama3 --port 7860
  ```
- [ ] 确保UI可以加载数据和运行推理

#### 优先级 2 (第3天 - 11月20日)

**2.1 验证集成**
- [ ] 确认所有模块可以通过 `run_experiments.py` 调用
- [ ] 测试端到端流程
- [ ] 解决任何集成问题

**2.2 准备实验配置**
- [ ] 检查 `configs/experiment_config.yaml`
- [ ] 确认所有实验模式都已配置
- [ ] 设置合理的随机种子以保证可复现性

#### 优先级 3 (第4-5天 - 11月21-22日)

**3.1 运行完整实验套件**
- [ ] 运行所有实验
  ```bash
  python experiments/run_experiments.py --mode all --models llama3,mistral
  ```
- [ ] 监控实验进度
- [ ] 确保所有结果正确保存到 `experiments/results/`

**3.2 运行消融实验**
- [ ] 不同检索方法对比
- [ ] 不同模型对比（Llama3 vs Mistral）
- [ ] 不同top-k值对比
- [ ] Agent组件消融（有/无验证、有/无修订）

**3.3 收集实验数据**
- [ ] 整理所有实验结果JSON文件
- [ ] 生成汇总报告
- [ ] 计算所有评估指标

#### 优先级 4 (第6天 - 11月23-24日)

**4.1 结果分析**
- [ ] 使用 `evaluation/error_analysis.py` 进行错误分析
  ```python
  from evaluation.error_analysis import compare_systems

  comparison = compare_systems(
      direct_predictions,
      agentic_predictions,
      ground_truth,
      "Direct Prompting",
      "Agentic RAG"
  )
  ```

**4.2 回答研究问题**
- [ ] **RQ1**: RAG是否减少幻觉？
  - 对比 Direct Prompting vs Standard RAG
  - 计算幻觉率、准确率、F1
  - 准备对比表格和图表

- [ ] **RQ2**: 验证+修订是否改进RAG？
  - 对比 Standard RAG vs Agentic RAG
  - 分析幻觉率、证据召回、弃权精确度
  - 展示具体改进案例

- [ ] **RQ3**: 什么导致幻觉？
  - 按合同条款类型分析（终止、责任等）
  - 按推理类型分析（推断、时间、实体混淆）
  - 使用 `hallucination_detector.py` 分类失败模式

**4.3 可视化结果**
- [ ] 创建对比表格
- [ ] 生成性能图表
- [ ] 准备案例研究示例

#### 优先级 5 (第7天 - 11月25日)

**5.1 撰写最终报告**
- [ ] **引言部分**
  - 背景：LLM幻觉问题
  - 动机：法律领域的重要性
  - 研究目标和问题

- [ ] **相关工作**
  - RAG综述
  - 法律NLP
  - 幻觉检测方法

- [ ] **方法论**
  - 系统架构
  - 数据集描述
  - 基线系统
  - Agentic RAG工作流
  - 评估指标

- [ ] **实验设置**
  - 模型选择
  - 超参数
  - 实验配置

- [ ] **结果与分析**
  - RQ1回答（表格+分析）
  - RQ2回答（表格+分析）
  - RQ3回答（失败模式分析）
  - 消融实验结果

- [ ] **讨论**
  - 主要发现
  - 局限性
  - 未来工作

- [ ] **结论**
  - 总结贡献
  - 实践意义

**5.2 最终检查**
- [ ] 代码清理和文档
- [ ] 更新README.md
- [ ] 验证可复现性
- [ ] 检查所有引用和格式
- [ ] 准备演示材料（如需要）

### ⚠️ 注意事项
- 预留足够时间运行实验（可能需要数小时）
- 提前备份所有实验结果
- 确保报告格式符合课程要求
- 准备应急计划（如果实验失败）

### 🔗 依赖关系
- **前置依赖** - 需要所有团队成员完成各自模块
- **无被依赖** - 最后集成和报告环节
- **协作** - 协调整个团队的进度

---

## 🔄 团队协作检查点

### 每日同步（建议时间：每天晚上9点）
- [ ] 今天完成了什么？
- [ ] 遇到什么问题？
- [ ] 明天计划做什么？
- [ ] 对其他成员有什么依赖？

### 关键里程碑

#### ✅ 检查点1 (11月19日晚)
- [ ] Yin Fengling: 数据处理完成
- [ ] Qin Mengmeng: 检索组件测试通过
- [ ] Wang Zhengyao: Agent组件独立测试通过
- [ ] Li Xinru: 实验框架就绪

#### ✅ 检查点2 (11月20日晚)
- [ ] Qin Mengmeng: 基线系统运行成功
- [ ] Wang Zhengyao: Agentic RAG集成完成
- [ ] Li Xinru: 端到端测试通过

#### ✅ 检查点3 (11月22日晚)
- [ ] 所有实验运行完成
- [ ] 所有结果已保存
- [ ] 初步分析完成

#### ✅ 检查点4 (11月24日晚)
- [ ] 报告初稿完成
- [ ] 代码清理完成
- [ ] 可复现性验证通过

---

## 🆘 常见问题与解决方案

### 问题1: 内存不足
**解决方案**:
- 减小batch_size（在 `configs/model_config.yaml`）
- 减小chunk_size
- 使用较小的模型

### 问题2: 检索速度慢
**解决方案**:
- 使用BM25而非Dense retrieval
- 减少top_k值
- 优化分块策略

### 问题3: 模型下载失败
**解决方案**:
```bash
export HF_HOME=/path/to/large/drive
```
或使用镜像站点

### 问题4: 数据集下载失败
**解决方案**:
手动从官网下载：
- ContractNLI: https://stanfordnlp.github.io/contract-nli/
- CUAD: https://www.atticusprojectai.org/cuad

### 问题5: 实验时间过长
**解决方案**:
- 先在小样本上测试（--sample_size 50）
- 并行运行不同实验
- 使用GPU加速

---

## 📊 预期成果

### 代码交付物
- [ ] 完整的代码库，所有模块可运行
- [ ] 配置文件和文档
- [ ] README和实现指南

### 实验结果
- [ ] 处理后的数据集（`data/processed/`）
- [ ] 实验结果JSON文件（`experiments/results/`）
- [ ] 评估指标报告

### 最终报告
- [ ] 完整的学术报告（PDF格式）
- [ ] 结果分析和可视化
- [ ] 代码使用说明

---

## 🎯 成功标准

1. **功能完整性**: 所有系统都能成功运行
2. **实验完整性**: 所有RQ都有数据支持
3. **可复现性**: 其他人可以重现结果
4. **报告质量**: 清晰的方法论和分析
5. **按时完成**: 11月25日前提交

---

## 💡 最佳实践

1. **频繁提交代码**: 每天至少提交一次
2. **保存中间结果**: 避免重复运行实验
3. **写清晰的日志**: 便于调试
4. **及时沟通**: 遇到问题立即告知团队
5. **备份数据**: 防止意外丢失

---

**祝大家顺利完成项目！加油！🚀**
