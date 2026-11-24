"""Supplement verified metrics and label internal/proxy measures explicitly."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import html
import json
import sqlite3
import sys

BASE = Path(__file__).resolve().parent
REPO = BASE.parent / "legal-hallucination-ui-review"
OUT = BASE / "metrics-followup"
sys.path.insert(0, str(REPO))
from retrieval.common import LABELS, document_id
from retrieval.evidence import score_evidence
from evaluation.metrics import MetricsCalculator


def ratio(n, d):
    return n / d if d else None


def mean(values):
    return sum(values) / len(values) if values else None


def words(text):
    return set(text.lower().split()) if isinstance(text, str) else set()


def main():
    OUT.mkdir(exist_ok=False)
    source = json.loads((BASE/"analysis/RESULTS.json").read_text())
    examples = {ds: {r["id"]:r for r in json.loads((REPO/"data/processed"/f"{ds}_test.json").read_text())}
                for ds in sorted(LABELS)}
    calc = MetricsCalculator()
    metrics, checks = [], []
    for model in ("llama3", "mistral"):
        print("Calculating", model, flush=True)
        with sqlite3.connect((BASE/"source"/f"{model}-predictions.sqlite3").as_uri()+"?mode=ro&immutable=1", uri=True) as db:
            db.execute("PRAGMA trusted_schema=OFF")
            groups = defaultdict(list)
            for ds, method, eid, a_json, t_json in db.execute("SELECT dataset,method,example_id,assessment,trace FROM predictions"):
                a, t = json.loads(a_json), json.loads(t_json)
                gt = examples[ds][eid]
                claims = (t.get("verification") or {}).get("claims", [])
                if claims:
                    assert all(c["status"] in {"supported","contradicted","not_enough_info"} for c in claims)
                valid_claims = [c for c in claims if not c.get("parse_error") and not c.get("invalid_evidence_ids")]
                unsupported = lambda c: c["status"] in {"contradicted","not_enough_info"}
                reference = words(gt.get("answer"))
                parsed_overlap = (len(words(t.get("answer")) & reference) / len(reference)) if reference else None
                display = t.get("final_answer", t.get("answer", "")) if a["format_pass"] else ""
                display_overlap = (len(words(display) & reference) / len(reference)) if reference else None
                retrieved_recall = None
                if "retrieved_chunks" in t:
                    expected = document_id(gt["context"], gt["metadata"], ds)
                    chunks = [{**c, "doc_id":expected} for c in t["retrieved_chunks"]]
                    scored = score_evidence(chunks, gt, k=5)
                    retrieved_recall = None if scored is None else scored["evidence_recall_at_k"]
                flagged_abstention = bool(t.get("abstained",False) or t["label"] == "not_enough_info")
                groups[ds,method].append(dict(id=eid, label=t["label"], gold_label=gt["label"],
                    label_correct=a["label_correct"], format_pass=a["format_pass"],
                    legacy_verification_key="verification_results" in t,
                    verified_row=bool(claims), total_claims=len(claims),
                    raw_unsupported=sum(unsupported(c) for c in claims),
                    parse_failed_claims=sum(bool(c.get("parse_error")) for c in claims),
                    failed_claims_excluded=len(claims)-len(valid_claims), valid_claims=len(valid_claims),
                    valid_unsupported=sum(unsupported(c) for c in valid_claims),
                    raw_claim_status=dict(Counter(c["status"] for c in claims)),
                    parsed_word_recall=parsed_overlap, display_word_recall=display_overlap,
                    initial_retrieval_recall_at_5=retrieved_recall, final_prompt_gold_recall=a["gold_evidence_coverage"],
                    flagged_abstention=flagged_abstention,
                    correct_under_legacy_abstention_rule=flagged_abstention and (gt["label"]=="neutral" or not gt.get("evidence_spans")),
                    neutral_as_abstention=flagged_abstention and t["label"]=="neutral",
                    latency=a["latency_seconds"], input_tokens=sum(c["input_tokens"] for c in t["generation_trace"]),
                    output_tokens=sum(c["tokens"] for c in t["generation_trace"]), model_calls=len(t["generation_trace"])))
            for (ds,method), rows in sorted(groups.items()):
                previous = next(g for g in source["groups"] if (g["model"],g["dataset"],g["method"])==(model,ds,method))
                assert len(rows)==previous["completed"] and sum(r["label_correct"] for r in rows)==previous["label_correct"]
                total = sum(r["total_claims"] for r in rows)
                valid = sum(r["valid_claims"] for r in rows)
                raw_bad = sum(r["raw_unsupported"] for r in rows)
                valid_bad = sum(r["valid_unsupported"] for r in rows)
                flags = sum(r["flagged_abstention"] for r in rows)
                pred = [{"label":r["label"]} for r in rows]
                gold = [{"label":r["gold_label"]} for r in rows]
                statuses = Counter()
                for r in rows: statuses.update(r["raw_claim_status"])
                g = dict(model=model,dataset=ds,method=method,n=len(rows),
                    label_accuracy=previous["label_accuracy"],macro_f1=previous["macro_f1"],
                    format_failure_count=sum(not r["format_pass"] for r in rows),
                    initial_retrieval_recall_at_5=mean([r["initial_retrieval_recall_at_5"] for r in rows if r["initial_retrieval_recall_at_5"] is not None]),
                    initial_retrieval_evaluable_rows=sum(r["initial_retrieval_recall_at_5"] is not None for r in rows),
                    final_prompt_gold_span_recall=mean([r["final_prompt_gold_recall"] for r in rows if r["final_prompt_gold_recall"] is not None]),
                    gold_evidence_evaluable_rows=sum(r["final_prompt_gold_recall"] is not None for r in rows),
                    avg_latency_seconds=mean([r["latency"] for r in rows]),
                    total_input_tokens=sum(r["input_tokens"] for r in rows),
                    total_generated_tokens=sum(r["output_tokens"] for r in rows),
                    total_model_calls=sum(r["model_calls"] for r in rows),
                    independent_final_answer_hallucination_rate=None,
                    internal_initial_verifier=dict(verified_rows=sum(r["verified_row"] for r in rows),
                        total_rows=len(rows),claim_status_counts=dict(statuses),total_claims=total,
                        raw_unsupported_claims=raw_bad,raw_unsupported_rate_including_verifier_errors=ratio(raw_bad,total),
                        parse_failed_claims=sum(r["parse_failed_claims"] for r in rows),
                        excluded_failed_claims=sum(r["failed_claims_excluded"] for r in rows),
                        valid_claims=valid,valid_unsupported_claims=valid_bad,
                        unsupported_rate_among_valid_verifier_records=ratio(valid_bad,valid),
                        applies_to="Agent initial answers only; incomplete coverage; same-model judgments"),
                    lexical_reference_overlap_proxy=dict(reference_rows=sum(r["parsed_word_recall"] is not None for r in rows),
                        parsed_answer_word_recall=mean([r["parsed_word_recall"] for r in rows if r["parsed_word_recall"] is not None]),
                        displayable_answer_word_recall=mean([r["display_word_recall"] for r in rows if r["display_word_recall"] is not None]),
                        semantic_completeness=None),
                    legacy_abstention_rule_proxy=dict(flagged_rows=flags,
                        correct_under_rule=sum(r["correct_under_legacy_abstention_rule"] for r in rows),
                        precision=ratio(sum(r["correct_under_legacy_abstention_rule"] for r in rows),flags),
                        neutral_classification_flagged_as_abstention=sum(r["neutral_as_abstention"] for r in rows),
                        independent_should_abstain_ground_truth_available=False),
                    legacy_calculator_diagnostic_only=dict(label_accuracy=float(calc.compute_label_accuracy(pred,gold)),
                        macro_f1=float(calc.compute_macro_f1(pred,gold)),
                        predictions_with_expected_verification_results_key=sum(r["legacy_verification_key"] for r in rows)))
                assert g["total_model_calls"]==previous["model_calls"]
                assert g["total_generated_tokens"]==previous["total_generated_tokens"]
                assert abs(g["avg_latency_seconds"]-previous["mean_latency_seconds"])<1e-10
                assert total==sum(statuses.values()) and total==valid+g["internal_initial_verifier"]["excluded_failed_claims"]
                assert flags==g["legacy_abstention_rule_proxy"]["neutral_classification_flagged_as_abstention"], "Inspect any non-neutral abstentions separately"
                assert total>0 if method=="verified_agentic_rag" else total==0
                metrics.append(g)
    # Concrete checks of the old implementation, kept separate from real metrics.
    diagnostic = dict(invalid_prediction_on_neutral_gold_accuracy=float(calc.compute_label_accuracy([{"label":"invalid"}],[{"label":"neutral"}])),
                      valid_current_trace_read_by_old_hallucination_function=calc.compute_hallucination_rate([{"verification":{"claims":[{"status":"contradicted"}]}}]),
                      current_trace_recognized_by_compute_all_metrics=calc._has_verification_info([{"verification":{"claims":[{"status":"contradicted"}]}}]))
    try:
        calc.compute_answer_completeness([{"answer":"a"},{"answer":""}],[{"answer":"a"},{"answer":None}])
    except AttributeError as exc:
        diagnostic["null_reference_failure"] = str(exc)
    assert diagnostic["invalid_prediction_on_neutral_gold_accuracy"]==1.
    assert diagnostic["valid_current_trace_read_by_old_hallucination_function"]==0.
    assert diagnostic["current_trace_recognized_by_compute_all_metrics"] is False
    assert "null_reference_failure" in diagnostic
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),groups=metrics,
                  supplied_formula="unsupported claims / all claims, where unsupported is contradicted or not_enough_info",
                  verified_source_rows=18666,independent_final_answer_evaluation_completed=False,
                  human_labels_created=0,legacy_implementation_diagnostics=diagnostic,
                  source_archive_sha256=source["source_archive_sha256"],
                  analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  legacy_metrics_source_sha256=hashlib.sha256((REPO/"evaluation/metrics.py").read_bytes()).hexdigest(),
                  live_main_metrics_blob_sha="4f0a612fe5fe483266c7f289955a9169a753c78d",
                  limitations=["Raw verifier rates include parser failures mapped to not_enough_info; valid-only rates condition on a selected subset.",
                               "Neither internal initial-answer rate estimates hallucination in final answers or in Direct/RAG.",
                               "Word-set recall uses only CUAD's 599 nonempty reference strings; all ContractNLI answers are null.",
                               "Recorded abstained flags coincide with neutral classifications; these are task-label proxy results, not independent abstention judgments.",
                               "Retrieval@5 uses initial retrieved chunks; final-prompt gold recall uses the actual final generator input and can differ.",
                               "Input and generated token totals are distinct; monetary cost is not available."])
    (OUT/"METRICS.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    render(result)
    print(json.dumps({"groups":len(metrics),"diagnostics":diagnostic,"agent_initial_rates":[dict(model=g["model"],dataset=g["dataset"],**g["internal_initial_verifier"]) for g in metrics if g["method"]=="verified_agentic_rag"]},ensure_ascii=False,indent=2))


def render(result):
    esc=lambda x:html.escape(str(x))
    pct=lambda x: "N/A" if x is None else f"{x:.2%}"
    names={"direct_prompting":"Direct","standard_rag":"RAG","verified_agentic_rag":"Agent"}
    def table(headers,rows):
        return "<table><thead><tr>"+"".join("<th>"+esc(v)+"</th>" for v in headers)+"</tr></thead><tbody>"+"".join("<tr>"+"".join("<td>"+esc(v)+"</td>" for v in row)+"</tr>" for row in rows)+"</tbody></table>"
    groups=result["groups"]
    content=["<h1>评估口径核对与可计算指标</h1><p>Group 55 · 2026-11-24 · 18,666 条既有输出，无新增推理或人工评分。</p>",
             "<p><strong>结论：</strong>提供的幻觉率公式可用于已有声明判断，但当前代码与数据字段不一致，且记录仅覆盖 Agent 初答。以下明确区分正式任务指标、内部检查统计与替代指标。</p>",
             "<h2>可直接使用的任务指标与运行成本</h2>",
             table(["模型","数据集","方法","准确率","Macro-F1","初检 Recall@5","最终提示金标召回","平均秒/条","生成 token"],
                   [[g["model"],g["dataset"],names[g["method"]],pct(g["label_accuracy"]),f'{g["macro_f1"]:.4f}',pct(g["initial_retrieval_recall_at_5"]),pct(g["final_prompt_gold_span_recall"]),f'{g["avg_latency_seconds"]:.2f}',g["total_generated_tokens"]] for g in groups]),
             "<p>格式失败保留在准确率分母，Macro-F1 对任务合法类别平均。证据指标仅计有金标证据的样本（ContractNLI 1,188；CUAD 599），按同合同字符区间并集覆盖完整片段；Direct 没有检索，Recall@5 为 N/A。初次检索与实际最终生成输入分开统计。生成 token 合计所有调用；输入 token 见 JSON，不当作货币成本。延迟来自原服务器 8 并行运行，不等于串行吞吐。</p>",
             "<h2>按所给声明公式统计 Agent 内部初答检查</h2>",
             table(["模型","数据集","被检查回答/总回答","总声明","原始不支持声明率（含错误）","验证失败声明","有效检查中的不支持率"],
                   [[g["model"],g["dataset"],f'{g["internal_initial_verifier"]["verified_rows"]}/{g["n"]}',g["internal_initial_verifier"]["total_claims"],pct(g["internal_initial_verifier"]["raw_unsupported_rate_including_verifier_errors"]),g["internal_initial_verifier"]["excluded_failed_claims"],pct(g["internal_initial_verifier"]["unsupported_rate_among_valid_verifier_records"])] for g in groups if g["method"]=="verified_agentic_rag"]),
             "<p><strong>这张表不能标成三种方法的最终幻觉率。</strong>原始比例按 contradicted + not_enough_info / 总声明计算；验证器格式失败会被程序转成 not_enough_info，原始比例混入了执行失败。最后一列排除失败记录，只反映成功解析的选择性子集。两列都来自同模型对初答的内部检查，既未覆盖跳过验证的样本，也未重新检查修订后答案。Direct/RAG 没有这些检查，不能填 0。</p>",
             "<h2>可算但需要注明含义的替代指标</h2>",
             table(["模型","数据集","方法","有参考答案的条数","解析答案词集合召回","可展示答案词集合召回","旧弃权规则精确度","旧弃权分母"],
                   [[g["model"],g["dataset"],names[g["method"]],g["lexical_reference_overlap_proxy"]["reference_rows"],pct(g["lexical_reference_overlap_proxy"]["parsed_answer_word_recall"]),pct(g["lexical_reference_overlap_proxy"]["displayable_answer_word_recall"]),pct(g["legacy_abstention_rule_proxy"]["precision"]),g["legacy_abstention_rule_proxy"]["flagged_rows"]] for g in groups]),
             "<p>词集合召回严格使用小写、按空白分词后的集合交集，过滤空参考，不做语义判断。ContractNLI 的 2,091 个参考答案均为 null，不能据此算答案完整性；CUAD 只评 599 个非空参考答案，421 个无答案样本不进入分母。可展示列将格式失败置零，并对 Agent 使用 final_answer，解析答案列保留原解析字段。词重合不保证事实正确或完整。</p>",
             "<p>旧弃权规则按 abstained 标志 / neutral 或无金标证据统计。实际记录中的弃权标志全部对应 neutral 分类，因此这里实质上是 neutral 判断精确度，不是独立的应否拒答评估。CUAD 没有被标为弃权的记录，分母为零，结果 N/A。</p>",
             "<h2>旧 MetricsCalculator 的实测问题</h2><ul><li>读取 verification_results，而全量记录使用 verification.claims。直接调用旧幻觉率函数会在无计数时返回 0；compute_all_metrics 则跳过这一项。</li><li>非法标签 invalid 被默认映射为 neutral；invalid 对 neutral 金标的最小例子会被记为 100% 正确。</li><li>类别映射混合两数据集；CUAD 的非法预测还会成为额外类别进入默认 Macro-F1。正式值继续使用已核验的任务专用评分。</li><li>CUAD 混有 answer=null，直接词重叠函数会在 .lower() 时报错；已在本次只读补算中处理空参考。</li></ul>",
             "<h2>还需团队决定与完成的部分</h2><p>若研究结论要比较三种方法的最终幻觉率，应使用同一判定规则评价三种方法的最终输出，并进行人工核查。内部初答检查可作为系统行为指标另列。此次没有修改 GitHub 或生成虚假的人工判定。</p>",
             '<p>源码依据：<a href="https://github.com/1336516233abc-ship-it/legal-hallucination-auditing/blob/main/evaluation/metrics.py#L237-L260">旧幻觉率实现</a> · <a href="https://github.com/1336516233abc-ship-it/legal-hallucination-auditing/blob/main/agent/agentic_rag.py#L191-L210">Agent 初答验证与返回字段</a>。全部分子、分母、源码哈希和限制见 METRICS.json。</p>']
    style="body{max-width:1250px;margin:40px auto;padding:0 24px;font:15px/1.75 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;color:#203043}h1{font-size:30px}h2{font-size:22px;margin-top:32px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{border-bottom:1px solid #d9e1e8;padding:8px;text-align:left}th{background:#edf3f7}strong{color:#173d66}"
    (OUT/"METRICS_REVIEW.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>评估口径与指标补算</title><style>'+style+'</style><body>'+"\n".join(content)+"</body></html>")


if __name__=="__main__":
    main()
