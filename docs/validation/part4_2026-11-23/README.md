# Part 4 补充验收和人工复核材料

Li Xinru 于 2026-11-23 整理。这个目录提供独立的界面候选、真实运行记录和待真人填写的盲评表，供组员检查和决定集成。原 ui/app.py 与 Agent 模块保持原样。

## 先看结论

**发现的问题：原页面换了一份合同后，仍可能引用上一份合同回答。** 真实 Mistral 验收中，先输入“提前 30 天通知”，再换成“提前 60 天通知”，仍回答 30 天；新建实例才答回 60 天。原因是旧入口复用首次建立的检索索引。

[local_demo.py](local_demo.py) 为每次请求创建独立 Agent/索引，调用已有任务协议，并给格式或引用错误提供停止提示。这是候选入口，没有替换团队原页面。

|真实验收|普通请求|改为 60 天通知后|
|---|---|---|
|原页面回调 / Mistral|回答 30 天|仍回答 30 天，引用旧合同|
|候选 / Llama 3|正确判断 30 天命题得到支持|正确判断 30 天命题与当前合同矛盾|
|候选 / Mistral|正确判断 30 天命题得到支持|出现非法标签，被拦截并显示停止原因|

这些是手工构造的功能验收输入，共 7 次真实模型请求，不加入原有 60 条正式试点。失败记录保留，不把“已拦截”写成“正确回答”。细节在 [UI_ACCEPTANCE.md](UI_ACCEPTANCE.md)。

## 本目录内容

- [RQ_FINDINGS.md](RQ_FINDINGS.md)：RQ1–RQ3 的可支持结论及旧演示措辞修订建议。独立幻觉率、完整修订成功率仍未测得。
- [local_demo.py](local_demo.py)：本地演示候选；逐请求隔离合同，显示最终输入的全部证据、失败原因和完整调用 token 数。
- [human_review/reviewer/评审说明.md](human_review/reviewer/评审说明.md)：60 条输出的盲评规则；2 模型 × 3 变体 × 同样 10 题，不是 60 个独立问题。
- [human_review/reviewer/review.html](human_review/reviewer/review.html)：下载后用浏览器打开。GitHub 文件页不会执行该表单；下载本分支 ZIP 后打开即可。
- [score_reviews.py](score_reviews.py)：校验真人评分的 ID、精确原文区间和判定完整性，统计单份评分。空白指标保持 null，不生成评分或双人裁决。
- [test_local_delivery.py](test_local_delivery.py)：请求隔离、错误处理及评审边界测试。测试用合成数据，不计入真实模型或人工结果。

盲评身份映射及金标对照保留在协调者本地，没有放进这个目录。评审者只拿 reviewer 子目录，不要先读已有案例分析和源轨迹。公开仓库无法防止主动查找方法身份，因此这是隐藏身份的材料包，不声明具有严格的评审者隔离。

## 运行候选

先配置仓库已有的 Python 和检索环境。此候选真实推理依赖 **Apple Silicon 和 MLX**；CUDA/Linux 后端尚未适配。需要自行准备有权使用的本地原版模型权重，本目录不含权重、凭据或虚拟环境。

在仓库根目录运行：

```bash
python -m pip install -r requirements-retrieval.txt
python -m pip install -r docs/validation/part4_2026-11-23/requirements-ui.txt
python docs/validation/part4_2026-11-23/local_demo.py \
  --snapshot /path/to/original-model-snapshot --port 7862
```

将占位路径换成已有模型目录。打开 http://127.0.0.1:7862 ，用页面中的普通示例和改条件示例测试；Ctrl+C 停止。仅监听本机，不生成公网分享链接。

已测 Llama 3 revision：8afb486c1db24fe5011ec46dfbe5b5dccdb575c2；Mistral revision：63a8b081895390a26e140280378bc85ec8bce07a。温度 0、seed 42、输入预算 4096、输出预算 512；BM25 fixed_length，每块 512 个空白分隔词，重叠 128 词，最多一次修订。没有对修订后的语义再次验证，页面明确说明这一点。

发布时仅调整仓库定位、模型目录参数和缓存路径；推理及判定逻辑不变。原记录和发布文件的来源信息在 [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json)。安装列表是可读的直接依赖，完整运行环境版本另见既有 Part 4 manifest；未声称完成一台新电脑的冷启动安装验收。

## 检查命令

```bash
python -m pytest -q tests docs/validation/part4_2026-11-23/test_local_delivery.py
python docs/validation/part4_2026-11-23/score_reviews.py \
  --packet docs/validation/part4_2026-11-23/human_review/reviewer/packet.json \
  --ratings /path/to/actual-human-ratings.json \
  --output /path/to/new-score-file.json
```

评分文件和输出路径需替换成实际位置，脚本拒绝覆盖既有结果。当前人工评分数量为 0。需要两名真人独立评审、讨论并记录分歧，再确认正式口径。

## 交接边界

请界面/Agent 维护者审查请求状态隔离、任务入口和错误反馈，再决定合入方式。服务器全量逐条输出仍需交付后才能开展 Direct/RAG/Agent 的正式配对支持性评估。现有归档结果不能支持“100% 修订成功”的演示结论。

本次使用 Codex 辅助整理代码、验收、盲评材料和结论说明；真实模型调用提示词与输出保存在 acceptance JSON 中。真人评分和用户测试尚未进行，最终报告尚未编写。本目录不是完整课程提交包或完整 AI 提示词声明。
