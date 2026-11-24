"""Build an evidence-linked case inventory from the frozen Part 4 pilot."""

from collections import Counter
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODELS = ("llama3", "mistral")
VARIANTS = ("full", "no_verification", "no_revision")
LABELS = {"contractnli": {"entailment", "contradiction", "neutral"},
          "cuad": {"present", "absent"}}
SOURCES = {}


def read(path):
    payload = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def coverage(context, spans, chunks):
    """Match the published non-whitespace, full-span coverage definition."""
    if not spans:
        return None
    covered = bytearray(len(context))
    for chunk in chunks:
        start, end = chunk["start"], chunk["end"]
        assert 0 <= start < end <= len(context)
        assert context[start:end] == chunk["text"]
        covered[start:end] = b"\x01" * (end - start)
    return sum(all(covered[i] or context[i].isspace() for i in range(a, b))
               for a, b in spans) / len(spans)


def stage_facts(stage, example, dataset):
    chunks = stage["prompt_evidence"]
    for chunk in chunks:
        assert example["context"][chunk["start"]:chunk["end"]] == chunk["text"]
    passed = stage["parse_error"] is None and stage["label"] in LABELS[dataset]
    invalid = [i for i in stage["citations"] if not 1 <= i <= len(chunks)]
    assert invalid == stage["invalid_citations"]
    return {
        "label": stage["label"], "format_pass": passed,
        "label_correct": passed and stage["label"] == example["label"],
        "parse_error": stage["parse_error"], "grounding_error": stage["grounding_error"],
        "citations": stage["citations"], "out_of_range_citations": invalid,
        "prompt_evidence_count": len(chunks),
        "prompt_evidence_spans": [[c["start"], c["end"]] for c in chunks],
        "gold_span_coverage": coverage(example["context"], example["evidence_spans"], chunks),
        "raw_response": stage["raw_response"],
        "answer_is_extractive": stage.get("answer_is_extractive"),
        "input_budget": stage["input_budget"],
    }


def collect():
    data = {ds: {r["id"]: r for r in read(ROOT / "data/processed" / f"{ds}_test.json")}
            for ds in LABELS}
    records, traces = [], {}
    frozen = None
    for model in MODELS:
        base = HERE / "agent_ablation" / model
        manifest = read(base / "manifest.json")
        if frozen is None:
            frozen = manifest["cohort"]
        assert frozen == manifest["cohort"]
        for ds in LABELS:
            source = f"data/processed/{ds}_test.json"
            assert SOURCES[source] == frozen["datasets"][ds]["source_sha256"]
            ids = {r["id"] for r in frozen["datasets"][ds]["rows"]}
            for variant in VARIANTS:
                path = base / ds / variant / "predictions.jsonl"
                payload = path.read_bytes()
                SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
                rows = [json.loads(line) for line in payload.splitlines()]
                assert len(rows) == len(ids) and {r["example_id"] for r in rows} == ids
                for line, raw in enumerate(rows, 1):
                    example = data[ds][raw["example_id"]]
                    trace, assessment = raw["trace"], raw["assessment"]
                    assert trace["id"] == example["id"] and trace["query"] == example["query"]
                    initial, final = (stage_facts(s, example, ds) for s in (trace["initial"], trace))
                    assert assessment["gold_label"] == example["label"]
                    for k in ("label_correct", "format_pass", "parse_error", "grounding_error"):
                        assert assessment[k] == final[k], (path, line, k)
                    assert assessment["prediction"] == final["label"]
                    assert assessment["gold_evidence_coverage"] == final["gold_span_coverage"]
                    assert trace["final_answer"] == ("" if final["parse_error"] or final["grounding_error"] else trace["answer"])
                    revised = trace["revision"] is not None
                    if variant != "full":
                        assert not revised
                    if variant == "no_verification":
                        assert trace["verification"] is None
                    transition = ("correct_to_correct" if initial["label_correct"] and final["label_correct"]
                                  else "wrong_to_correct" if final["label_correct"]
                                  else "correct_to_wrong" if initial["label_correct"] else "wrong_to_wrong")
                    key = (model, ds, example["id"], variant)
                    traces[key] = trace
                    records.append({
                        "model": model, "dataset": ds, "example_id": example["id"], "variant": variant,
                        "gold_label": example["label"], "query": example["query"],
                        "source_path": str(path.relative_to(HERE)), "source_line": line,
                        "initial": initial, "final": final, "revision_performed": revised,
                        "label_transition": transition,
                        "new_grounding_error_after_revision": revised and not initial["grounding_error"] and bool(final["grounding_error"]),
                        "final_answer": trace["final_answer"],
                        "display_suppressed_by_error": bool(final["parse_error"] or final["grounding_error"]),
                        "verifier_detected": (trace["verification"]["hallucination_detected"]
                                              if trace["verification"] is not None else None),
                        "model_calls": len(trace["generation_trace"]),
                    })
    assert len(records) == 60
    assert sum(r["final"]["label_correct"] for r in records) == 37
    assert sum(r["final"]["format_pass"] for r in records) == 57
    return data, records, traces


SPECS = [
    ("C01", "llama3", "contractnli", "contractnli_test_523_nda-8", "标签修正，但引用仍指错证据",
     "label_correction_with_residual_citation_error",
     "初始 contradiction 与模型自己引用的通知义务相冲突；修订后 entailment 与金标和合同一致。但重检索使通知义务从证据 [2] 移到 [1]，最终仍写 Evidence [2]。只能称标签纠正，不能称完整的证据引用修复。编号保留是观察结果，其内部原因未通过新增实验验证。",
     "检查修订后的引用是否支持对应断言，而不只检查编号是否越界。", []),
    ("C02", "mistral", "contractnli", "contractnli_test_493_nda-2", "解释已经否定命题，标签却仍为 neutral",
     "nli_label_rationale_inconsistency",
     "问题含 only technical information。合同定义明确列出 business plans、customer lists、pricing 等范围，金标为 contradiction。初始解释和修订解释均承认不止技术信息，但标签始终 neutral；这不是缺少金标证据，而是标签与解释不一致。",
     "核对 only、all 等限定词及标签与解释的一致性。", []),
    ("C03", "llama3", "cuad", "HPILHOLDING_01_07_2015", "重检索找到签署日期，答案仍为 absent",
     "missed_answer_after_evidence_recovery",
     "初始提示没有覆盖日期金标；重检索后证据 [3] 覆盖签署日期，标签仍为 absent。该样本同时体现初始检索遗漏和重检索后的生成遗漏，不能只归因于检索。Mistral 的三个变体均抽取出该日期，可作为同一问题的对照。",
     "分别检查初始与修订提示中的目标证据，再评估模型是否使用它。", [(3504, 3605, "签署日期上下文")]),
    ("C04", "mistral", "cuad", "PACIRA PHARMACEUTICALS", "重检索覆盖续期通知条款，仍漏掉 180 天",
     "missed_renewal_notice_after_evidence_recovery",
     "初始提示未覆盖金标，重检索后证据 [2] 已含 at least one hundred eighty (180) days 及 any renewal term。最终仍说明没有终止续期的明确通知期，预测 absent，与金标 present 不符。",
     "结合完整条件与 renewal term 审阅条款，避免把存在的续期条件漏判为不存在。", []),
    ("C05", "mistral", "contractnli", "contractnli_test_523_nda-8", "输出 Label: true，任务协议解析失败",
     "invalid_task_label",
     "初始原响应使用 Label: true，而 ContractNLI 只接受 entailment、contradiction、neutral。三个变体均返回相同非法标签并计错；Full 因解析失败跳过验证和修订，最终展示答案为空。即使说明文字提到通知义务，也不能改记为格式通过或标签正确。",
     "后续单独验证结构化输出或格式恢复策略；不能在结果汇总时事后修改原预测。", []),
    ("C06", "llama3", "cuad", "N2KINC_10_16_1997", "占位符逐字匹配通过，但没有回答目标问题",
     "verbatim_but_task_irrelevant_extraction",
     "金标为 absent。模型却将脱敏占位符 [****] 当作 present 的抽取答案。该字符串确实出现在输入中，因此逐字匹配通过；Verifier 的 supported 理由引用的是广告展示次数（Impressions），并未证明终止续期通知期。Full 因未触发检测而没有修订。这里可确认局部任务相关性检查不足，不能据此估计总体幻觉率。",
     "验证器应检查所摘内容是否回答目标条款问题，而不只检查字符串是否出现。", [(6380, 7203, "续期谈判上下文；非金标通知期")]),
    ("C07", "llama3", "contractnli", "contractnli_test_53_nda-18", "修订引入越界引用，正确标签下的答案被清空",
     "revision_introduced_out_of_range_citation",
     "初始和最终标签均为 neutral，与金标一致。修订响应添加了 [18]，但最终输入只有 3 个证据块。原合同确有第 18 条，内容涉及非合伙和代理关系；它既不是证据编号 [18]，也不是禁止招揽条款。范围检查据此标记 grounding_error，展示答案为空。这是新增引用错误，不能称修订成功。neutral 本身不作为人工弃权标签。",
     "将合同条款号与提示证据编号分开，并检查修订是否引入新错误。", [(13542, 13861, "合同第 18 条；用于解释编号混淆，不是招揽限制证据")]),
    ("C08", "mistral", "cuad", "ACCELERATEDTECHNOLOGIESHOLDINGCORP", "标签保持正确，修订却添加大量不存在的引用",
     "revision_introduced_out_of_range_citation",
     "初始 absent 与金标一致且无引用错误。修订后仍为 absent，但加入 [1,2,3,4,5,11,12,13,14,15,16]；最终只有 3 个证据块，8 个编号越界。absent 的空抽取原本是合法结构，不能仅因答案空白判错；这里的确切新增问题是引用越界。",
     "分别统计标签变化、合法空抽取与新增引用错误，避免把它们混成一个成功率。", [(5991, 6263, "合资期限和延期约定；金标 absent 没有目标证据区间")]),
    ("C09", "llama3", "contractnli", "contractnli_test_4_nda-4", "初始引用越界使验证和修订均未执行",
     "invalid_citation_blocks_verification",
     "合同第 2 条支持用途限制，预测 entailment 也正确。但模型使用 [2]，而输入只有 1 个证据块；grounding_error 导致验证和修订跳过，最终展示为空。与 C07 不同，这个错误在初始阶段已出现，不能归因于修订器。",
     "优先检查证据编号与合同条款号的混用；再单独评估是否需要格式或引用恢复分支。", []),
]


def make_cases(data, records, traces):
    cases = []
    for cid, model, ds, fragment, title, category, analysis, recommendation, extra in SPECS:
        candidates = [r for r in records if r["model"] == model and r["dataset"] == ds
                      and r["variant"] == "full" and fragment in r["example_id"]]
        assert len(candidates) == 1
        record = candidates[0]
        example = data[ds][record["example_id"]]
        context = example["context"]
        evidence = []
        for start, end, purpose in [(a, b, "数据集金标证据") for a, b in example["evidence_spans"] or []] + extra:
            assert 0 <= start < end <= len(context)
            evidence.append({"start": start, "end": end, "text": context[start:end], "purpose": purpose,
                             "gold_annotation": purpose == "数据集金标证据"})
        key = (model, ds, example["id"], "full")
        trace = traces[key]
        cases.append({"case_id": cid, "title": title, "category": category,
                      "record": record, "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                      "evidence": evidence, "analysis": analysis, "next_check": recommendation,
                      "verification": trace["verification"],
                      "paired_variants": [r for r in records if r["model"] == model and r["dataset"] == ds
                                          and r["example_id"] == example["id"]]})
    # Verify the non-structural observations used in the selected case notes.
    first = cases[0]["record"]
    assert first["label_transition"] == "wrong_to_correct"
    assert first["initial"]["citations"] == first["final"]["citations"] == [2]
    gold = data["contractnli"][first["example_id"]]["evidence_spans"]
    assert all(first["initial"]["prompt_evidence_spans"][1][0] <= a < b <= first["initial"]["prompt_evidence_spans"][1][1] for a, b in gold)
    assert all(first["final"]["prompt_evidence_spans"][0][0] <= a < b <= first["final"]["prompt_evidence_spans"][0][1] for a, b in gold)
    assert all(not (first["final"]["prompt_evidence_spans"][1][0] <= a < b <= first["final"]["prompt_evidence_spans"][1][1]) for a, b in gold)
    first_trace = traces[(first["model"], first["dataset"], first["example_id"], "full")]
    assert "immediately notify" in first_trace["prompt_evidence"][0]["text"]
    assert "immediately notify" not in first_trace["prompt_evidence"][1]["text"]
    for case in cases[2:4]:
        assert case["record"]["initial"]["gold_span_coverage"] == 0
        assert case["record"]["final"]["gold_span_coverage"] == 1
        assert case["record"]["final"]["label"] == "absent"
    assert all("Label: true" in r["final"]["raw_response"] for r in cases[4]["paired_variants"])
    assert cases[5]["record"]["final_answer"] == "[****]"
    assert cases[5]["record"]["final"]["answer_is_extractive"] is True
    assert "Impressions" in json.dumps(cases[5]["verification"])
    assert all(cases[i]["record"]["new_grounding_error_after_revision"] for i in (6, 7))
    assert cases[8]["record"]["initial"]["grounding_error"] and cases[8]["verification"] is None
    return cases


def build_summary(records):
    full = [r for r in records if r["variant"] == "full"]
    revised = [r for r in full if r["revision_performed"]]
    groups = []
    for model in MODELS:
        for ds in LABELS:
            for variant in VARIANTS:
                rows = [r for r in records if (r["model"], r["dataset"], r["variant"]) == (model, ds, variant)]
                groups.append({"model": model, "dataset": ds, "variant": variant, "n": len(rows),
                               "label_correct": sum(r["final"]["label_correct"] for r in rows),
                               "format_failures": sum(not r["final"]["format_pass"] for r in rows),
                               "grounding_errors": sum(bool(r["final"]["grounding_error"]) for r in rows)})
    transitions = Counter(r["label_transition"] for r in revised)
    return {"output_count": len(records), "unique_examples": len({r["example_id"] for r in records}),
            "label_correct": sum(r["final"]["label_correct"] for r in records),
            "format_failures": sum(not r["final"]["format_pass"] for r in records),
            "grounding_errors": sum(bool(r["final"]["grounding_error"]) for r in records),
            "full_outputs": len(full), "full_revisions": len(revised),
            "full_revision_label_transitions": {k: transitions[k] for k in (
                "wrong_to_correct", "wrong_to_wrong", "correct_to_correct", "correct_to_wrong")},
            "new_grounding_errors_after_revision": sum(r["new_grounding_error_after_revision"] for r in revised),
            "groups": groups}


def source_link(record):
    return f'[{record["model"]}/{record["dataset"]}/{record["variant"]} 第 {record["source_line"]} 行]({record["source_path"]}#L{record["source_line"]})'


def render(result):
    s = result["summary"]
    lines = ["# Part 4 错误与修订案例清单", "",
             "本文件是已有实验输出的定性案例整理，不是课程报告。未重新运行模型，未修改原预测、金标或已发布指标。", "",
             "## 范围与结论边界", "",
             "- 核对了两个模型 × 三个变体的全部 60 条输出，对应 10 个相同问题（ContractNLI 6、CUAD 4），不是 60 个独立样本。下方 9 个案例按不同问题类型选取，不是随机抽样，不能计算总体错误类型占比。",
             "- 原始标签统计仍为 37/60 正确、57/60 格式通过。逐条另查引用范围、金标证据覆盖和最终展示答案；标签正确不保证引用正确或答案可用。",
             f'- 全部输出中有 {s["format_failures"]} 条格式失败、{s["grounding_errors"]} 条程序记录的 grounding_error。重复样本跨变体保留，不当作独立失败事件。',
             f'- 20 条 Full 输出中实际修订 {s["full_revisions"]} 次：标签错→对 1 次、错→错 4 次、对→对 6 次、对→错 0 次。另有 {s["new_grounding_errors_after_revision"]} 次修订引入新的越界引用。唯一标签纠正案例仍有引用指向错误，不能把 1/11 或 1/5 宣称为独立幻觉修订成功率。',
             "- 定性解释由 AI 辅助核对金标与合同原文，尚未经过双人独立盲评；独立幻觉率、完整性、弃权精确度和完整修订成功率均保持未评估。neutral/absent 不自动等于弃权。",
             "- 字符区间为处理后合同 context 的 Python 字符下标 [start,end)，不是 PDF 页码。JSON 摘录均校验等于原文切片；本页展示仅去除行尾空白。完整逐条索引及输入 SHA256 见 [case_analysis.json](case_analysis.json)。", "",
             "## 案例索引", "", "|编号|模型 / 数据集|主要观察|", "|---|---|---|"]
    for c in result["cases"]:
        r = c["record"]
        lines.append(f'|[{c["case_id"]}](#{c["case_id"].lower()})|{r["model"]} / {r["dataset"]}|{c["title"]}|')
    for c in result["cases"]:
        r = c["record"]
        lines += ["", f'<a id="{c["case_id"].lower()}"></a>', "", f'## {c["case_id"]} · {c["title"]}', "",
                  f'- 样本：`{r["example_id"]}`', f'- 问题：{r["query"]}',
                  f'- 金标：`{r["gold_label"]}`；初始 → 最终：`{r["initial"]["label"]}` → `{r["final"]["label"]}`；执行修订：{"是" if r["revision_performed"] else "否"}。',
                  f'- 原始输出：{source_link(r)}。', "", "**初始模型响应**", "", "```text", r["initial"]["raw_response"], "```", "",
                  "**最终模型响应**", "", "```text", r["final"]["raw_response"], "```", "",
                  f'最终展示：{"因解析或引用检查错误而清空" if r["display_suppressed_by_error"] else "合法 absent 的空抽取" if r["dataset"] == "cuad" and r["final"]["label"] == "absent" else "保留模型答案"}。', "",
                  "|阶段|输入证据块数|金标区间覆盖率|模型引用编号|", "|---|---:|---|---|",
                  *[f'|{name}|{r[key]["prompt_evidence_count"]}|{r[key]["gold_span_coverage"] if r[key]["gold_span_coverage"] is not None else "无金标区间，不计"}|{r[key]["citations"]}|'
                    for name, key in (("初始", "initial"), ("最终", "final"))], "",
                  "**合同原文与核对**", ""]
        if not c["evidence"]:
            lines += ["该样本金标无证据区间；不构造不存在的金标摘录。", ""]
        for e in c["evidence"]:
            lines += [f'{e["purpose"]}，`[{e["start"]},{e["end"]})`：', "", "```text", e["text"], "```", ""]
        lines += [c["analysis"], "", "**同一样本的变体对照**", "",
                  "|变体|最终标签|标签正确|格式通过|引用/抽取检查错误|", "|---|---|---|---|---|"]
        for p in c["paired_variants"]:
            lines.append(f'|{p["variant"]}|{p["final"]["label"]}|{"是" if p["final"]["label_correct"] else "否"}|{"是" if p["final"]["format_pass"] else "否"}|{p["final"]["grounding_error"] or "无程序标记；不代表语义已验证"}|')
        lines += ["", f'后续检查方向：{c["next_check"]}', ""]
    lines += ["## 全部 Full 输出索引", "", "保留正确、错误及格式失败输出，便于核查选例范围。", "",
              "|模型 / 数据集|样本（可打开完整轨迹）|金标|初始 → 最终|修订|grounding_error|",
              "|---|---|---|---|---|---|"]
    for r in result["records"]:
        if r["variant"] != "full":
            continue
        name = r["example_id"].replace("cuad_test_", "").replace("contractnli_test_", "")
        lines.append(f'|{r["model"]} / {r["dataset"]}|[{name}]({r["source_path"]}#L{r["source_line"]})|{r["gold_label"]}|{r["initial"]["label"]} → {r["final"]["label"]}|{"是" if r["revision_performed"] else "否"}|{"有" if r["final"]["grounding_error"] else "无程序标记"}|')
    lines += ["", "## 后续优先级", "",
              "1. 检查引用编号：同时覆盖编号越界和编号合法但指错内容两类；重新检索后的编号需要重新对应。",
              "2. 检查任务相关性：逐字摘录或 verifier 的 supported 标志不能替代是否回答目标问题。",
              "3. 检查标签与说明一致性，以及重检索拿到目标条款后的漏答。",
              "4. 单独验证格式恢复策略，保留原始失败记录，不能事后改写已公布结果。",
              "以上是由案例提出的检查方向，本次未更改 Agent 实现，也未运行修复后的对照实验。", "",
              "## 复现与完整性", "", "在仓库根目录执行（仅标准库，不加载模型）：", "", "```bash",
              "python3 docs/validation/part4_2026-11-22/build_case_analysis.py", "```", "",
              "需要原 processed test 文件与本目录 12 份 JSONL。脚本核对模型间固定样本、数据 SHA256、全部 60 条最终评分、初始/最终证据切片、引用范围及展示抑制条件；同时验证上述案例特有的标签变化与证据位置。",
              "原实验主表：[RESULTS.md](RESULTS.md)。独立评审方案：[ANSWER_SUPPORT_PROTOCOL.md](../../ANSWER_SUPPORT_PROTOCOL.md)。", ""]
    return "\n".join(line.rstrip() for line in "\n".join(lines).split("\n"))


def main():
    data, records, traces = collect()
    cases = make_cases(data, records, traces)
    summary = build_summary(records)
    assert summary["full_revisions"] == 11
    assert summary["new_grounding_errors_after_revision"] == 2
    result = {"kind": "part4_qualitative_case_analysis", "analysis_date": "2026-11-23",
              "source_experiment_commit": "f3e8abc689fb46deaa9d806fd66aebbfe759e75f",
              "review_type": "AI-assisted qualitative review against dataset labels and source contracts; not independent human annotation",
              "selection": "Purposeful selection to illustrate distinct observed failure modes; all 60 outputs inventoried.",
              "summary": summary, "cases": cases, "records": records,
              "source_sha256": SOURCES,
              "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "independent_hallucination_rate": None, "independent_revision_success_rate": None}
    (HERE / "case_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "CASE_ANALYSIS.md").write_text(render(result), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
