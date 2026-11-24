"""Audit supplied full traces without inference, source edits, or human labels."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import html
import json
import math
import random
import sqlite3
import sys

BASE = Path(__file__).resolve().parent
REPO = BASE.parent / "legal-hallucination-ui-review"
SOURCE = BASE / "source"
OUT = BASE / "analysis"
sys.path.insert(0, str(REPO))
from retrieval.common import LABELS, document_id, check_cuad_extraction
from retrieval.evidence import score_evidence

METHODS = ("direct_prompting", "standard_rag", "verified_agentic_rag")
NAMES = dict(zip(METHODS, ("Direct", "RAG", "Agent")))


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def equal(actual, expected, name):
    if isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (name, actual, expected)
    else:
        assert actual == expected, (name, actual, expected)


def passed(trace, dataset):
    return trace.get("parse_error") is None and trace.get("label") in LABELS[dataset]


def transition(before, after):
    return ("both_correct" if before and after else "correct_to_wrong" if before
            else "wrong_to_correct" if after else "both_wrong")


def summarize(rows, dataset):
    n = len(rows)
    confusion = Counter((r["gold_label"], r["prediction"]) for r in rows)
    f1 = []
    for label in sorted(LABELS[dataset]):
        actual = sum(v for (g, _), v in confusion.items() if g == label)
        predicted = sum(v for (_, p), v in confusion.items() if p == label)
        f1.append(2 * confusion[label, label] / (actual + predicted) if actual + predicted else 0.)
    correct = sum(r["label_correct"] for r in rows)
    formats = sum(r["format_pass"] for r in rows)
    gold = [r for r in rows if r["gold_evidence_coverage"] is not None]
    covered = [r for r in gold if r["gold_evidence_coverage"] == 1.]
    positives = [r for r in rows if r["gold_label"] == "present"]
    return dict(planned=n, completed=n, format_pass=formats, label_correct=correct,
                format_pass_rate=formats/n, label_accuracy=correct/n, macro_f1=sum(f1)/len(f1),
                confusion={g: {p: v for (y,p),v in sorted(confusion.items()) if y == g}
                           for g in sorted(LABELS[dataset])},
                mean_latency_seconds=sum(r["latency_seconds"] for r in rows)/n,
                model_calls=sum(r["model_calls"] for r in rows),
                format_or_execution_errors=dict(Counter(r["parse_error"] for r in rows if r["parse_error"])),
                grounding_errors=sum(bool(r["grounding_error"]) for r in rows),
                rows_with_out_of_range_citations=sum(bool(r["invalid_citations"]) for r in rows),
                revisions=sum(bool(r["revision_type"]) for r in rows),
                gold_evidence_rows=len(gold), fully_covered_gold_rows=len(covered),
                label_accuracy_when_gold_fully_covered=(sum(r["label_correct"] for r in covered)/len(covered) if covered else None),
                gold_positive_count=len(positives),
                verbatim_positive_correct=sum(r["label_correct"] and r["answer_is_extractive"] is True
                                               and not r["invalid_citations"] and not r["grounding_error"] for r in positives))


def assert_chunks(chunks, example):
    allowed = {document_id(example["context"], example.get("metadata"), example["metadata"]["dataset"]),
               document_id(example["context"], dataset=example["metadata"]["dataset"]),
               document_id(example["context"])}
    for chunk in chunks:
        start, end = chunk["start"], chunk["end"]
        assert type(start) is int and type(end) is int and 0 <= start < end <= len(example["context"])
        assert chunk["doc_id"] in allowed, "Cross-document evidence"
        assert example["context"][start:end] == chunk["text"], "Evidence offset mismatch"


def coverage(trace, example):
    expected = document_id(example["context"], example["metadata"], example["metadata"]["dataset"])
    if "prompt_evidence" in trace:
        chunks = trace["prompt_evidence"]
        assert_chunks(chunks, example)
        chunks = [{**chunk, "doc_id": expected} for chunk in chunks]
    else:
        end = trace["input_budget"]["context_end"]
        assert 0 <= end <= len(example["context"])
        chunks = [dict(text=example["context"][:end], start=0, end=end, doc_id=expected)] if end else []
    score = score_evidence(chunks, example, k=len(chunks))
    return None if score is None else score["evidence_recall_at_k"]


def case_record(model, ds, method, example, a, t, tags):
    return dict(model=model, dataset=ds, method=method, example_id=example["id"],
                automatically_selected_tags=tags, independently_reviewed=False,
                query=example["query"], gold_label=example["label"], prediction=t["label"],
                initial_label=t.get("initial", {}).get("label"),
                initial_answer=t.get("initial", {}).get("answer"), answer=t.get("answer"),
                displayed_answer=t.get("final_answer", t.get("answer")),
                assessment=a, source_contract=example["context"],
                gold_evidence=[dict(start=s, end=e, text=example["context"][s:e]) for s,e in example.get("evidence_spans") or []],
                final_prompt_evidence=t.get("prompt_evidence", []),
                verification=t.get("verification"), revision_type=a.get("revision_type"))


def main():
    OUT.mkdir(exist_ok=False)
    data = {ds: {r["id"]: r for r in json.loads((REPO/"data/processed"/f"{ds}_test.json").read_text())}
            for ds in sorted(LABELS)}
    selected = {ds: set(random.Random(42).sample(sorted(rows), min(50, len(rows)))) for ds,rows in data.items()}
    source_hashes = {ds: digest(REPO/"data/processed"/f"{ds}_test.json") for ds in data}
    manifest = json.loads((SOURCE/"export-manifest.json").read_text())
    checks, all_groups, pairings, revisions, all_cases, clause_rows = [], [], [], [], [], []
    audits_by_key, review_rows, blind_keys = {}, [], {}
    for alias in ("llama3", "mistral"):
        print(f"Auditing {alias} database...", flush=True)
        archive = json.loads((REPO/"docs/validation"/f"{alias}_autodl_full_summary.json").read_text())
        archived_groups = {(g["dataset"],g["method"]):g for g in archive["groups"]}
        exported = next(m for m in manifest["models"] if m["model"] == archive["settings"]["checkpoint"])
        exported_groups = {(g["dataset"],g["method"]):g for g in exported["groups"]}
        db = sqlite3.connect((SOURCE/f"{alias}-predictions.sqlite3").as_uri()+"?mode=ro&immutable=1", uri=True)
        db.execute("PRAGMA trusted_schema=OFF")
        equal(db.execute("PRAGMA integrity_check").fetchall(), [("ok",)], "SQLite integrity")
        frozen = json.loads(db.execute("SELECT settings FROM run WHERE id=1").fetchone()[0])
        equal(frozen, archive["settings"], "Frozen settings vs archived summary")
        for ds in data:
            equal(frozen["datasets"][ds], {"count":len(data[ds]), "sha256":source_hashes[ds]}, "Frozen data")
        groups, row_hashes, seen, features = defaultdict(list), {}, set(), {}
        cases_taken = set()
        for index,(ds, method, eid, raw_a, raw_t) in enumerate(db.execute(
                "SELECT dataset,method,example_id,assessment,trace FROM predictions ORDER BY dataset,method,example_id")):
            key = (ds, method, eid)
            assert key not in seen and ds in data and method in METHODS and eid in data[ds]
            seen.add(key)
            a,t = json.loads(raw_a),json.loads(raw_t)
            example = data[ds][eid]
            equal(t["id"], eid, "Trace ID"); equal(t["dataset"], ds, "Trace dataset")
            equal(t["query"], example["query"], "Query")
            equal(a["gold_label"], example["label"], "Gold label")
            equal(a["prediction"], t["label"], "Prediction")
            equal(a["format_pass"], passed(t, ds), "Format pass")
            equal(a["label_correct"], passed(t, ds) and t["label"] == example["label"], "Correctness")
            for field in ("parse_error", "answer_is_extractive", "invalid_citations", "grounding_error"):
                equal(a.get(field), t.get(field), field)
            equal(a["latency_seconds"], t["latency"], "Latency")
            equal(a["model_calls"], len(t["generation_trace"]), "Model call count")
            equal(a["gold_evidence_coverage"], coverage(t,example), "Gold coverage")
            assert_chunks(t.get("retrieved_chunks", []), example)
            if "prompt_evidence" in t:
                equal(t.get("invalid_citations", []), [i for i in t["citations"] if i > len(t["prompt_evidence"])], "Citation range")
            if ds == "cuad":
                evidence = ([c["text"] for c in t["prompt_evidence"]] if "prompt_evidence" in t
                            else [example["context"][:t["input_budget"]["context_end"]]])
                extraction = check_cuad_extraction(t,evidence)
                equal(t["answer_is_extractive"], extraction["answer_is_extractive"], "Extractiveness")
            groups[ds,method].append(a)
            row_hashes[key] = canonical({"assessment":a,"trace":t})
            f = dict(model=alias,dataset=ds,method=method,example_id=eid,clause_type=example["metadata"]["clause_type"],**a)
            f["total_generated_tokens"] = sum(c["tokens"] for c in t["generation_trace"])
            f["generation_length_limit"] = any(c.get("finish_reason") == "length" and c["stage"] in ("generation","initial_generation","revision") for c in t["generation_trace"])
            f["verifier_parse_error"] = any(c.get("parse_error") for c in (t.get("verification") or {}).get("claims", []))
            f["final_answer_withheld"] = method == "verified_agentic_rag" and bool(t.get("parse_error") or t.get("grounding_error"))
            f["valid_absent_empty_extraction"] = (ds == "cuad" and method == "verified_agentic_rag"
                                                  and passed(t, ds) and t["label"] == "absent"
                                                  and not f["final_answer_withheld"] and not t.get("final_answer"))
            f["revision_transition"] = None
            f["initial_correct"] = None
            if method == "verified_agentic_rag":
                initial = t["initial"]
                assert_chunks(initial["prompt_evidence"], example)
                equal(a["revision_type"], (t.get("revision") or {}).get("revision_type"), "Revision type")
                equal(t["final_answer"], "" if t["parse_error"] or t["grounding_error"] else t["answer"], "Displayed answer")
                f["initial_correct"] = passed(initial,ds) and initial["label"] == example["label"]
                if t["revision"] is not None:
                    f["revision_transition"] = transition(f["initial_correct"], a["label_correct"])
                else:
                    equal(initial["label"], t["label"], "Unrevised label")
            features[key] = f
            tags = []
            if f["revision_transition"] in ("wrong_to_correct","correct_to_wrong"):
                tags.append("revision_"+f["revision_transition"])
            if a["gold_evidence_coverage"] == 1. and not a["label_correct"] and a["format_pass"]:
                tags.append("gold_covered_but_label_wrong")
            if a["invalid_citations"]:
                tags.append("out_of_range_citation")
            if not a["format_pass"]:
                tags.append("format_failure")
            fresh = [tag for tag in tags if (ds,tag) not in cases_taken]
            if fresh:
                all_cases.append(case_record(alias, ds, method, example, a, t, fresh))
                cases_taken.update((ds,tag) for tag in fresh)
            if eid in selected[ds]:
                blind_id = hashlib.sha256(f"{alias}/{ds}/{method}/{eid}".encode()).hexdigest()[:20]
                blind_keys[blind_id] = dict(model=alias,dataset=ds,method=method,example_id=eid)
                review_rows.append(dict(blind_id=blind_id,dataset=ds,query=example["query"],source_contract=example["context"],
                                        predicted_label=t["label"],displayed_answer=t.get("final_answer",t.get("answer","")),
                                        raw_response=t["raw_response"],format_pass=a["format_pass"],parse_error=a["parse_error"],
                                        review=dict(annotator=None,response_type=None,claims=[],all_claims_reviewed=False,needs_adjudication=None,notes="")))
            if (index+1) % 3000 == 0:
                print(f"  {alias}: {index+1} rows checked", flush=True)
        expected = {(ds,m,eid) for ds, rows in data.items() for m in METHODS for eid in rows}
        equal(seen,expected,"Full cohort membership")
        equal(len(seen), exported["rows"], "Export count")
        print(f"Comparing {alias} JSONL export to SQLite...", flush=True)
        gz_seen = set()
        with gzip.open(SOURCE/f"{alias}-predictions.jsonl.gz", "rt") as f:
            for line in f:
                r = json.loads(line); k = r["dataset"],r["method"],r["example_id"]
                assert k not in gz_seen; gz_seen.add(k)
                equal(r["model"],frozen["checkpoint"],"Export model")
                equal(canonical({"assessment":r["assessment"],"trace":r["trace"]}),row_hashes[k],"JSONL vs SQLite")
        equal(gz_seen,seen,"Export cohort")
        for (ds,method),rows in groups.items():
            computed = summarize(rows, ds)
            for field,value in computed.items():
                equal(value, archived_groups[ds,method][field], f"Archive {alias}/{ds}/{method}/{field}")
                equal(value, exported_groups[ds,method][field], f"Manifest {field}")
            fs = [features[ds,method,eid] for eid in sorted(data[ds])]
            extra = dict(total_generated_tokens=sum(r["total_generated_tokens"] for r in fs),
                         verifier_parse_error_rows=sum(r["verifier_parse_error"] for r in fs),
                         final_answer_withheld_rows=sum(r["final_answer_withheld"] for r in fs),
                         valid_absent_empty_extraction_rows=sum(r["valid_absent_empty_extraction"] for r in fs),
                         correct_label_but_withheld_rows=sum(r["label_correct"] and r["final_answer_withheld"] for r in fs),
                         generation_length_limit_rows=sum(r["generation_length_limit"] for r in fs),
                         gold_covered_but_wrong_rows=sum(r["gold_evidence_coverage"] == 1. and not r["label_correct"] for r in fs),
                         independent_hallucination_rate=None)
            all_groups.append(dict(model=alias,dataset=ds,method=method,**computed,**extra))
            if method == "verified_agentic_rag":
                changed = Counter(r["revision_transition"] for r in fs if r["revision_transition"] is not None)
                equal(sum(changed.values()), computed["revisions"], "Revision denominator")
                revisions.append(dict(model=alias,dataset=ds,total=len(fs),revised=sum(changed.values()),
                                      initial_correct=sum(r["initial_correct"] for r in fs),final_correct=computed["label_correct"],
                                      **{k:changed[k] for k in ("both_correct","wrong_to_correct","correct_to_wrong","both_wrong")}))
            for clause in sorted({r["clause_type"] for r in fs}):
                subset = [r for r in fs if r["clause_type"] == clause]
                clause_rows.append(dict(model=alias,dataset=ds,method=method,clause_type=clause,n=len(subset),
                                        label_correct=sum(r["label_correct"] for r in subset),format_failed=sum(not r["format_pass"] for r in subset)))
        for ds in data:
            for left,right in ((METHODS[0],METHODS[1]),(METHODS[1],METHODS[2])):
                counts = Counter(transition(features[ds,left,eid]["label_correct"], features[ds,right,eid]["label_correct"]) for eid in data[ds])
                pairings.append(dict(model=alias,dataset=ds,before=left,after=right,n=len(data[ds]),
                                     **{k:counts[k] for k in ("both_correct","wrong_to_correct","correct_to_wrong","both_wrong")}))
        checks.append(dict(model=alias,rows=len(seen),sqlite_integrity="ok",jsonl_matches_sqlite=True,
                           settings_match_archived_summary=True,all_six_groups_match_archive=True,
                           dataset_hashes_and_ids_match=True,evidence_offsets_and_scope_match=True))
        audits_by_key.update({(alias,*k):v for k,v in features.items()})
        db.close()
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),kind="full_trace_audit_and_descriptive_analysis",
                  total_rows=sum(c["rows"] for c in checks),checks=checks,groups=all_groups,
                  paired_label_transitions=pairings,within_agent_revision_label_transitions=revisions,
                  per_clause_type=clause_rows,independent_hallucination_rate=None,
                  human_review_completed=0,source_dataset_sha256=source_hashes,
                  source_archive_sha256=json.loads((BASE/"file_integrity.json").read_text())["archive_sha256"],
                  analysis_script_sha256=digest(Path(__file__)),evidence_scorer_sha256=digest(REPO/"retrieval/evidence.py"),
                  limitations=["Classification correctness and evidence coverage do not measure hallucination rate.",
                               "Within-Agent initial/final changes are descriptive and do not replace controlled component ablation.",
                               "Cases are selected by deterministic flags, not a random prevalence sample or human judgment.",
                               "The 600 unreviewed forms follow the draft protocol; sample size and annotation ownership remain for team confirmation.",
                               "CUAD covers the supplied 10 categories, not all official 41 categories."])
    (OUT/"RESULTS.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    with gzip.open(OUT/"row_metrics.jsonl.gz","wt") as f:
        for key,r in sorted(audits_by_key.items()): f.write(json.dumps(r,ensure_ascii=False)+"\n")
    with (OUT/"CASES.jsonl").open("w") as f:
        for r in all_cases: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    review_dir = OUT/"unreviewed_materials"; review_dir.mkdir()
    random.Random(42).shuffle(review_rows)
    with (review_dir/"blinded_review.jsonl").open("w") as f:
        for r in review_rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    (review_dir/"private_key.json").write_text(json.dumps(blind_keys,ensure_ascii=False,indent=2))
    (review_dir/"selection.json").write_text(json.dumps(dict(seed=42,sampling="paired simple random sample within each dataset, same questions across models and methods",
                                                             example_ids={ds:sorted(ids) for ds,ids in selected.items()},rows=len(review_rows),reviewed=0,
                                                             independent_hallucination_rate=None,team_protocol_confirmed=False),ensure_ascii=False,indent=2))
    render(result, all_cases)
    print(json.dumps(dict(rows=result["total_rows"],checks=checks,revision_transitions=revisions,cases=len(all_cases),unreviewed_forms=len(review_rows)),ensure_ascii=False,indent=2))


def render(result, cases):
    esc = lambda v: html.escape(str(v))
    def table(head, rows):
        return "<table><thead><tr>"+"".join("<th>"+esc(c)+"</th>" for c in head)+"</tr></thead><tbody>"+"".join("<tr>"+"".join("<td>"+esc(c)+"</td>" for c in row)+"</tr>" for row in rows)+"</tbody></table>"
    blocks = ["<h1>全量逐条记录核验与分析</h1><p>Group 55 · 2026-11-24 · Part 4 接续材料，非最终课程报告。</p>",
              "<p><strong>18,666 条记录核验通过。</strong>两个模型各 9,333 条；各含 ContractNLI 2,091 题和 CUAD 1,020 题 × Direct / RAG / Agent。这里的全量指仓库处理后的测试划分。</p>",
              "<h2>资料完整性</h2><p>ZIP 中四个数据文件的 SHA-256 与清单一致；两个 SQLite 完整性检查通过。逐条核对 JSONL 与 SQLite、模型与参数、原始测试文件哈希、样本 ID、问题、标签评分、证据原文与字符位置。重新汇总的全部 12 组指标与既有仓库汇总一致。此次没有重新运行模型。</p>",
              "<h2>主实验复核</h2>",
              table(["模型","数据集","方法","条数","标签准确率","Macro-F1","格式失败","引用越界"],
                    [[g["model"],g["dataset"],NAMES[g["method"]],g["completed"],f'{g["label_accuracy"]:.2%}',f'{g["macro_f1"]:.4f}',g["completed"]-g["format_pass"],g["rows_with_out_of_range_citations"]] for g in result["groups"]]),
              "<p>格式失败保留在准确率分母；Macro-F1 对任务合法类别平均。引用越界与格式失败可能重叠，不能相加为总错误数。</p>",
              "<h2>修订前后发生了什么</h2><p>只对确实触发修订的样本比较 Agent 初答和最终标签；正确性均按同一金标与格式规则计算。</p>",
              table(["模型","数据集","修订条数","错→对","对→错","一直对","仍然错"],
                    [[r["model"],r["dataset"],r["revised"],r["wrong_to_correct"],r["correct_to_wrong"],r["both_correct"],r["both_wrong"]] for r in result["within_agent_revision_label_transitions"]]),
              "<p><strong>这些是标签变化，不是独立幻觉修复成功率。</strong>修订后的自由文本还未按统一标准人工核对；初答与终答比较也不等于受控的无修订消融实验。</p>",
              "<h2>同题方法对比</h2>",
              table(["模型","数据集","方法变化","前错后对","前对后错","净增正确条数"],
                    [[r["model"],r["dataset"],NAMES[r["before"]]+" → "+NAMES[r["after"]],r["wrong_to_correct"],r["correct_to_wrong"],r["wrong_to_correct"]-r["correct_to_wrong"]] for r in result["paired_label_transitions"]]),
              "<h2>显示答案与标签指标的区别</h2>",
              table(["模型","数据集","Agent 总条数","因格式/依据错误拦截","标签正确但被拦截","合法 absent 空抽取","存在 verifier 格式错误"],
                    [[g["model"],g["dataset"],g["completed"],g["final_answer_withheld_rows"],g["correct_label_but_withheld_rows"],g["valid_absent_empty_extraction_rows"],g["verifier_parse_error_rows"]] for g in result["groups"] if g["method"]==METHODS[2]]),
              "<p>被拦截仅指显式格式错误或依据检查失败，标签正确仍可能被拦截。CUAD 的合法 absent 表示未找到条款，正常返回空抽取，不算拦截；该列并不保证 absent 标签与金标一致。verifier 格式错误描述系统运行失败，不是人工认定的幻觉。</p>",
              "<h2>错误案例与后续评审</h2><p>CASES.jsonl 提供自动筛出的可追溯案例，包含问题、金标、初答/终答、原合同和证据。样例按稳定顺序选择，不代表各错误类型的总体占比。分条款类型计数和全量逐条统计在 RESULTS.json 与 row_metrics.jsonl.gz。</p>",
              "<p>已按待确认方案准备 600 条未评分材料：每个数据集随机选 50 个共同问题 × 两模型 × 三方法。只是备选评审材料，<strong>没有产生人工评分，也没有据此新增个人任务</strong>。评审者不要接触 private_key.json；团队确认规则、人数和样本量后再使用。</p>",
              "<p><strong>仍未完成：</strong>独立幻觉率、答案完整性、弃权精确度、语义层面的修订成功率。Yin 与团队需要确认评估标准和人工检查安排。全量 Agent 组件消融仍不能由本次三方法记录替代。</p>",
              "<h2>复现与来源</h2><p>使用提供的原始文件，只读查询 SQLite，无模型/API 调用，不更改 GitHub、原压缩包或既有实验结果。运行同目录上级 analyze_traces.py 可复现（需先将原 analysis 目录另行归档，脚本拒绝覆盖）。file_integrity.json 保存原始文件校验；RESULTS.json 保存脚本、数据和原压缩包的 SHA-256。</p>"]
    style = "body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.7 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;color:#203043}h1{font-size:30px}h2{font-size:22px;margin-top:32px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px 10px;text-align:left;border-bottom:1px solid #d9e1e8}th{background:#edf3f7}strong{color:#173d66}p{max-width:1000px}"
    (OUT/"SUMMARY.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Group 55 全量记录核验</title><style>'+style+'</style><body>'+"\n".join(blocks)+"</body></html>")


if __name__ == "__main__":
    main()
