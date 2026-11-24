# 正式 UI 运行与验收

负责人：Li Xinru。按最新分工确认，UI 纳入演示必做事项。入口为 `ui/app.py`，历史候选和旧验收记录保留在原位置。这里是运行交接，不是最终课程报告。

## 页面能做什么

- 核对命题：对英文陈述返回支持、矛盾或现有证据不足的任务结论。
- 查找条款：对英文问题抽取合同原句；没有检索到相关条款不等于证明整份合同不存在该条款。
- 每次请求重新建立本合同的 Agent 与索引，不复用上一份合同的内容。修改输入立即清空旧结果；运行期间禁用输入与示例按钮。
- 展示最终回答实际使用的全部编号证据，以及初始答案检查和一次修订的状态。最终修订的语义尚未再次验证，页面明确提示。
- 模型输出格式、抽取原句或引用范围检查失败时停止展示答案，保留失败原因和原始记录。没有把失败计为正确或零幻觉。
- 查看全部调用次数、生成 token 总数和本次轨迹；没有用最后一次调用冒充总成本。结果属于当前浏览器请求，不保存共享的“上一条合同”。

## 安装与启动

使用 Python 3.12。模型权重与虚拟环境不放入 GitHub。先安装公共依赖和已验证的 UI 版本：

```bash
python -m pip install -r requirements-retrieval.txt -r requirements-ui.txt
```

Apple Silicon 本地运行：

```bash
python -m pip install 'mlx-lm==0.32.2' 'mlx==0.32.4' 'transformers==5.24.0'
python ui/app.py --model llama3 --backend mlx \
  --snapshot /path/to/local/llama3-model --port 7860
```

Mistral 同样使用 `--model mistral` 并指定对应模型目录。MLX 模型在一个专用工作线程内加载和推理，避免 Gradio 工作线程使用其他线程创建的 MLX stream。模型家族不匹配、缺失目录或非 Apple Silicon 平台会在启动时给出错误。

CUDA 服务器使用仓库的 Transformers 接口：

```bash
python -m pip install 'torch>=2.0.0' 'transformers==5.24.0' 'accelerate>=0.26.0'
python ui/app.py --model llama3 --backend transformers --device cuda \
  --snapshot /path/to/local/llama3-model --port 7860
```

不提供 `--snapshot` 时，Transformers 按 `configs/model_config.yaml` 的模型名加载；这可能需要下载和已有模型访问权限。`--device auto` 依次选择 CUDA、MPS、CPU；明确指定不可用设备时会报错。CPU 推理可能很慢。

浏览器访问 `http://127.0.0.1:7860`。远程服务器可以通过用户自行建立的 SSH 本地端口转发访问。默认只监听本机，不创建公网分享链接。结束时在启动终端按 Ctrl+C。

温度为 0、seed 为 42，输入预算默认 4096、输出最多 512 token；检索为 BM25，512 词分块、128 词重叠。页面支持关闭依据检查或修订，但关闭检查时也关闭依赖它的修订。示例均为手工构造输入，不加入正式测试集。

## 演示操作

1. 点击“30 天通知”，再点“核对当前合同”，查看结论和原文证据。
2. 点击“改为 60 天通知”：旧结果应立即清空。再次运行，检查证据是否确实来自 60 天的新合同。
3. 点击“查找责任上限”，运行并检查抽取原句。
4. 展开“查看检查与运行记录”，说明初始检查、是否修订、失败状态和 token 统计。不要把“格式通过”讲成“没有幻觉”。
5. 可以自行改英文输入。模型可能答错或返回非法格式；页面应如实提示，不能用预填答案替代实际运行。

## 验证与范围

验收入口：`scripts/validate_ui_service.py`。先启动正式页面，再运行：

```bash
python scripts/validate_ui_service.py --url http://127.0.0.1:7860 \
  --output /path/to/new-ui-acceptance.json
python -m pytest -q tests docs/validation/part4_2026-11-23/test_local_delivery.py
```

验收脚本逐条保存输出，拒绝覆盖旧文件。模型案例成功数、运行错误、空输入拒绝与浏览器操作应分别说明；脚本调用成功不等于模型结论正确。

本次记录放在 [ui_2026-11-23](validation/ui_2026-11-23/)。首次真实服务调用发现 MLX 工作线程错误，失败记录仍保留；最终结果见该目录的验收说明。后续 [提交包验收](validation/final_submission_acceptance.json) 已在同一台 Mac 的全新 Python 3.11 环境完成解压安装和 Mistral 服务验收，记录 188 项测试通过。原试点和后续环境版本分别保留。CUDA/CPU/MPS 实机推理及另一台机器尚未验收，不能称跨平台验证完成。

## 集成边界与剩余事项

正式 UI 复用已有 Agent 和任务协议，没有修改 `agent/`、`retrieval/` 或原始数据。入口保留 `ContractReviewUI.analyze_contract` 的原四项返回值；新增逻辑在 `ui/service.py` 和 `ui/backends.py`。Gradio 页面接口为 `/review`。

实验编排、主结果、错误案例和正式 UI 已合入 main；课程提交包与 PDF 入口见 [最终提交说明](FINAL_SUBMISSION.md)。服务器逐条输出已取得，[全量分析与可计算指标](validation/part4_2026-11-24/README.md)已补齐；独立幻觉率仍需 Yin Fengling 与团队确认评估方法并完成标注；这里的 UI 验收不是人工盲评。人工评分保持未完成；报告与 UI 说明已按现有验收记录修正，见 [FINAL_REPORT.md](../FINAL_REPORT.md)。
