# Group 55 交付验收

项目为 **Legal Hallucination Auditing with Agentic Retrieval for Contract Review**。组员为 Li Xinru、Qin Mengmeng、Yin Fengling、Wang Zhengyao。

## 解压后运行

从 ZIP 内的 `legal-hallucination-auditing/` 目录运行。使用 Python 3.11 或 3.12，新建虚拟环境后安装依赖。Apple Silicon 使用 MLX，本地已有支持模型的权重时无需远程模型 API。模型权重需要单独取得，使用 Llama 3 时可能需要模型仓库访问权限。CUDA 模式和显存要求见 [UI 交接](UI_HANDOFF.md)。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-retrieval.txt -r requirements-ui.txt
python -m pip install 'mlx-lm==0.32.2' 'mlx==0.32.4' 'transformers==5.23.0'
python ui/app.py --model mistral --backend mlx --snapshot /path/to/mistral-snapshot --port 7860
```

在浏览器打开本机端口 7860。普通、改条件和缺输入样例见 [示例文件](../examples/final_review_cases.json)。使用页面内置的 30 天通知、60 天通知和责任上限按钮也可复现。文本证据来自当前合同，真实模型可能出现非法标签并被安全拦截。官方 UI 同时支持获得相应快照的 Llama 3。本地命令与独立部署的共享演示服务分开配置。

## 验证评估

```bash
python -m pip install 'scikit-learn>=1.3' 'torch>=2.0.0' 'requests>=2.31.0'
python -m pytest -q tests docs/validation/part4_2026-11-23/test_local_delivery.py
python -m experiments.run_experiments --model llama3 mistral --dataset cuad --sample_size 1 --mode all --dry_run
python scripts/verify_full_trace_export.py ../evaluation_traces
```

全量任务的运行环境记录见两个正式 summary 文件。完整回归中依赖 Torch 和 Transformers 的测试需安装对应模型依赖。故障注入用例覆盖非法标签和越界引用，测试中的模型返回已明确标为合成。真实模型的浏览器验收及服务调用见 [验收记录](validation/ui_2026-11-23/README.md)。最终 ZIP 会附带本次运行证据。

对照评估使用任务合法标签。格式失败进入准确率分母，Macro-F1 只对合法标签平均。证据指标要求原文位置，初答 verifier 诊断保留失败计数。词集合召回是词面代理指标。独立终答幻觉率、语义答案完整性及正式弃权精确度在缺标注时保持未测。

## 文件定位

- [报告 PDF](final_report/COMP7608B-Group55-Final-Report.pdf) 使用官方 NeurIPS 2026 样式。课程 ZIP 根目录附有同一文件，LaTeX 源码保存在 `docs/final_report/`。
- `evaluation_traces/` 保留两个原始压缩 JSONL 和导出清单，共 18,666 条预测。提交包中保留等价的 JSONL 逐条数据。原 SQLite 在独立归档中。
- 实验结果和分析见 [全量轨迹分析](validation/part4_2026-11-24/README.md) 与 [最终复算](validation/final_submission_trace_audit.json)。
- 提交版以 ZIP 根目录 `manifest.json` 的版本指纹及逐文件 SHA-256 为准。`acceptance.json` 记录解压后的回归和真实问答。AI 使用及可提供的提示词见 [AI_USE.md](../AI_USE.md)。
- GitHub 版提供[修复记录](FINAL_EVALUATION_FIXES.md)和[解压验收记录](validation/final_submission_acceptance.json)。
