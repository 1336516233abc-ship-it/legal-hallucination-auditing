# Part 4 实验交接

负责人：Li Xinru。交付范围为实验编排、真实实验与结果汇总；课程报告暂未撰写。

结果入口：[本次结果表](validation/part4_2026-11-22/RESULTS.md)。原始测试数据、既有全量结果和组员旧文件保留原样。

本文的“全量”指仓库已交付的处理后测试划分；CUAD 覆盖已交付的 10 个类别，并非官方 41 类完整任务集合。[划分与标签说明](../data/processed/README.md)

## 交付范围

|部分|范围|记录|
|---|---|---|
|新跑检索|ContractNLI 2,091 条、CUAD 1,020 条；BM25/Dense/Hybrid × fixed/clause × k=3/5/10，共 36 组|`validation/part4_2026-11-22/*full_rerun.json`|
|新跑 Agent 消融|固定试跑集 ContractNLI 6 条、CUAD 4 条；两个正式模型 × Full/No verification/No revision，共 60 条输出|`validation/part4_2026-11-22/agent_ablation/`|
|归档主实验复核|Mistral/Llama 3 × 两数据集 × Direct/RAG/Agent，共 18,666 条归档输出的汇总；核对数据哈希及混淆矩阵|`validation/part4_2026-11-22/archived_result_audit/`|
|独立答案支持性|方案与盲评导出入口已准备，尚无人工评审结果|[评审方案](ANSWER_SUPPORT_PROTOCOL.md)|

新 Agent 消融是流程检查用的固定小样本，不能当作全测试集消融结果。其样本在旧基线试跑阶段已选定，每类 2 例、seed 42，三个变体使用相同样本。关闭验证同时关闭依赖验证触发的修订；关闭修订保留验证。没有独立评审时，不将 detector 标志当成幻觉真值，也不将“有修订”当成“修订成功”。

## 编排修复

- `experiments/run_experiments.py` 实际应用 dataset、sample_size、seed 和模式选择；消融开关会执行对应变体。
- 使用现有按任务生成/解析/评分接口，CUAD 使用 present/absent，ContractNLI 使用三分类。非法输出显式计错；独立幻觉率等未测指标为 null。
- 每个模型、数据集、变体独立保存 SQLite，逐条提交；恢复前比对数据哈希、样本 ID、代码与参数，避免混用结果。所有模型调用进入轨迹及 token 计数。
- `retrieval/__init__.py` 延迟导出基线类；5 个 Agent 模块将仅用于类型标注的 LLM 导入延迟，支持轻量 dry run 与现有测试。未重写组员的 Agent 决策流程。
- `summarize_results.py` 核对归档；`summarize_part4.py` 从新预测重新评分并核对组统计、配对样本、开关执行与 token 总数。

## 复现

在仓库根目录、Python 3.12 环境执行。依赖只按运行目标安装；Apple Silicon 真模型试跑使用 MLX，CUDA 通用 runner 使用仓库 `requirements.txt`。

```bash
python -m pip install -r requirements-retrieval.txt
python -m experiments.run_experiments --model mistral --dataset cuad \
  --sample_size 4 --mode agent --ablations agent --seed 42 --dry_run
python -m pytest tests -q
```

检索复跑：

```bash
python -m pip install 'sentence-transformers==6.3.0' 'torch==2.15.0' 'transformers==5.24.0'
python -m retrieval.evaluate_retrieval --datasets contractnli cuad --retrievers bm25 \
  --chunkings fixed_length clause_level --top_ks 3 5 10 --seed 42 --output RUN/bm25_full_rerun.json
python -m retrieval.evaluate_retrieval --datasets contractnli cuad --retrievers dense hybrid \
  --chunkings fixed_length clause_level --top_ks 3 5 10 --seed 42 --output RUN/dense_hybrid_full_rerun.json
```

真实 Agent 小样本：

```bash
python -m pip install 'mlx-lm==0.32.2' 'mlx==0.32.4' 'transformers==5.24.0'
# Download the original pinned Mistral checkpoint; provide its snapshot directory below.
hf download mistralai/Mistral-7B-Instruct-v0.2 --revision 63a8b081895390a26e140280378bc85ec8bce07a
python scripts/fetch_verified_llama3.py MODEL_DIR/llama3-pinned
python -m experiments.run_agent_ablation_pilot --model mistral --snapshot MISTRAL_SNAPSHOT \
  --output RUN/agent_ablation/mistral
python -m experiments.run_agent_ablation_pilot --model llama3 --snapshot MODEL_DIR/llama3-pinned \
  --output RUN/agent_ablation/llama3
```

每次只加载一个原版模型。已有目录会核对 manifest 后继续未完成样本；代码或依赖改变须使用新目录。权重和源码 SHA256、配置、样本 ID、Python/MLX/Transformers 版本记录在 manifest；原版未量化，不能换成量化模型后仍沿用同一对照。不同后端和依赖可能使输出不同。首次组耗时可含权重首次加载，不能据此给出严格的速度提升结论。

本次本机环境为 Apple M5、24 GiB 内存、macOS 26.6.2，使用 Metal GPU 串行推理。[硬件与运行环境](validation/part4_2026-11-22/execution_environment.json)

汇总与核验：

```bash
python -m experiments.summarize_results --output RUN/archived_result_audit --plots
python -m experiments.summarize_part4 --directory RUN
```

通用 CUDA 编排示例（另一份新输出目录）：

```bash
python -m experiments.run_experiments --model llama3 mistral --dataset all \
  --mode agent --ablations agent --sample_size 20 --seed 42 --device cuda --output RUN_CUDA
# Resume only with exactly the same arguments and frozen sources:
python -m experiments.run_experiments --model llama3 mistral --dataset all \
  --mode agent --ablations agent --sample_size 20 --seed 42 --device cuda --output RUN_CUDA --resume
```

通用 runner 的模型加载遵循 `model_config.yaml`，记录实际 resolved revision；需要复现本次固定 checkpoint 的 MLX 试跑，应使用专用入口。检索开关 `--ablations retrieval` 是带真实生成的检索配置对照，范围取自配置（当前 BM25/Dense）；纯检索全量含 Hybrid，使用上方 `evaluate_retrieval`。

## 尚待接续

1. 组员取回 AutoDL 的全量 `predictions.sqlite3` 后，可按同一批 example_id 做全量案例分析与盲评导出：`python -m experiments.prepare_support_review --checkpoint FILE --model mistral --output REVIEW`。
2. 团队确认独立答案支持性方案并完成标注，再计算跨三方法的幻觉率。当前无该结果，也无可靠的“100% 修订成功率”。
3. 若需要全量 Agent 组件消融，在可用 GPU/服务器上扩大样本，保留三个变体的同一数据与模型版本。本次小样本不替代该项。

发布包含源码、汇总、逐条 JSONL 和验证记录，不包含模型权重、虚拟环境、数据库临时文件或报告 PDF。SQLite 检查点保留在运行机器；JSONL 提供完整可移植的预测与调用轨迹。
