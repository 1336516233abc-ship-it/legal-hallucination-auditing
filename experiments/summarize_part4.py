"""Validate new Part 4 runs against portable traces and publish result tables."""

import argparse
import json
from pathlib import Path

from experiments.run_baseline_pilot import cohort
from experiments.run_full_mistral import assess_prediction, group_summary
from experiments.summarize_results import audit, digest, read, validate_group
from retrieval.common import LABELS, ROOT, write_json


def collect(directory):
    """Reject incomplete runs, mismatched cohorts, and altered assessments."""
    archived = audit()
    sources = {}
    environment = directory / "retrieval_environment.json"
    sources[environment.name] = digest(environment)
    retrieval = []
    for name in ("bm25_full_rerun.json", "dense_hybrid_full_rerun.json"):
        path = directory / name
        result = read(path)
        sources[name] = digest(path)
        for dataset, recorded in result["datasets"].items():
            expected = archived["datasets"][dataset]
            if recorded["sha256"] != expected["sha256"] or recorded["examples"] != expected["count"]:
                raise ValueError("Retrieval dataset differs from archived main experiment")
        retrieval.extend(result["experiments"])
    key = lambda r: (r["dataset"], r["retriever"], r["chunking"], r["top_k"])
    old = {key(r): r for r in archived["retrieval"]}
    if len(retrieval) != 36 or {key(r) for r in retrieval} != set(old):
        raise ValueError("Missing or duplicate retrieval configurations")
    maximum_delta = max(abs(value - old[key(row)]["metrics"][metric])
                        for row in retrieval for metric, value in row["metrics"].items())
    frozen_cohort, examples = cohort()
    groups, paired = [], []
    for alias in ("mistral", "llama3"):
        base = directory / "agent_ablation" / alias
        summary = read(base / "summary.json")
        manifest = read(base / "manifest.json")
        if (summary["status"] != "completed" or summary["completed"] != 30
                or summary["settings"] != manifest or manifest["cohort"] != frozen_cohort
                or len(summary["groups"]) != 6):
            raise ValueError(f"Incomplete or inconsistent {alias} pilot")
        expected_keys = {(d, v) for d in LABELS for v in ("full", "no_verification", "no_revision")}
        if {(g["dataset"], g["variant"]["name"]) for g in summary["groups"]} != expected_keys:
            raise ValueError("Pilot variants are missing or duplicated")
        sources[str((base / "manifest.json").relative_to(directory))] = digest(base / "manifest.json")
        sources[str((base / "summary.json").relative_to(directory))] = digest(base / "summary.json")
        by_dataset = {}
        for group in summary["groups"]:
            dataset, variant = group["dataset"], group["variant"]["name"]
            validate_group(group)
            path = base / dataset / variant / "predictions.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            population = {r.id: r for r in examples[dataset]}
            if len(records) != len(population) or {r["example_id"] for r in records} != set(population):
                raise ValueError("Raw predictions do not match the frozen paired cohort")
            for record in records:
                trace = record["trace"]
                if trace["id"] != record["example_id"] or trace["dataset"] != dataset:
                    raise ValueError("Trace identity mismatch")
                if assess_prediction(population[record["example_id"]], dataset, trace) != record["assessment"]:
                    raise ValueError("Assessment differs from raw model prediction")
                if not group["variant"]["verification"] and trace["verification"] is not None:
                    raise ValueError("Verification unexpectedly executed")
                if not group["variant"]["revision"] and trace["revision"] is not None:
                    raise ValueError("Revision unexpectedly executed")
            recomputed = group_summary([r["assessment"] for r in records], len(population), LABELS[dataset])
            for metric, value in recomputed.items():
                if isinstance(value, float):
                    if abs(value - group[metric]) > 1e-10:
                        raise ValueError(f"Group aggregate mismatch: {metric}")
                elif value != group[metric]:
                    raise ValueError(f"Group aggregate mismatch: {metric}")
            tokens = sum(call["tokens"] for r in records for call in r["trace"]["generation_trace"])
            if tokens != group["total_generated_tokens"]:
                raise ValueError("Generated token accounting mismatch")
            sources[str(path.relative_to(directory))] = digest(path)
            groups.append({"model": alias, **group})
            by_dataset.setdefault(dataset, {})[variant] = {r["example_id"]: r["assessment"] for r in records}
        for dataset, variants in by_dataset.items():
            for variant in ("no_verification", "no_revision"):
                transitions = {"both_correct": 0, "full_only_correct": 0,
                               "ablation_only_correct": 0, "both_wrong": 0}
                for example_id, assessment in variants["full"].items():
                    full = assessment["label_correct"]
                    ablated = variants[variant][example_id]["label_correct"]
                    category = ("both_correct" if full and ablated else "full_only_correct" if full
                                else "ablation_only_correct" if ablated else "both_wrong")
                    transitions[category] += 1
                paired.append({"model": alias, "dataset": dataset, "comparison": variant, **transitions})
    return {"kind": "part4_experiment_summary", "source_sha256": sources,
            "archived_main": archived["main"], "new_retrieval": retrieval,
            "retrieval_max_absolute_difference_from_archive": maximum_delta,
            "new_agent_pilot": groups, "paired_correctness": paired,
            "independent_hallucination_rate": None,
            "limitations": ["Agent component ablation is a locked 10-example pilot, not full-test inference.",
                            "MLX pilot latency is not comparable to archived vLLM server latency.",
                            "Independent support judgments and full-server traces are not available.",
                            "Verifier flags and revisions are workflow events, not measured hallucination or correction rates."]}


def tables(result):
    lines = ["# Part 4 实验结果汇总", "", "本页为实验交付清单与数值表，未撰写课程报告。", "",
             "- 新跑：36 组完整测试集检索评测；两个原版模型各 30 条 Agent 消融输出，共 60 条。",
             "- 归档复核：两模型 × 两数据集 × 三方法的 18,666 条全量结果；核对了数据哈希和汇总混淆矩阵，未取回服务器逐条轨迹。",
             "- Agent 消融使用既有的固定 10 例（ContractNLI 6、CUAD 4），每个变体使用同样的样本。小样本结果只支持流程检查，不能外推为全测试集性能。",
             "- 独立幻觉率、答案完整性和拒答精确率仍未评估，不将缺失值写成 0。", "",
             "## 新跑 Agent 组件消融", "",
             "两个模型均为固定 revision、未量化的原版权重，MLX 本机推理，温度 0、seed 42、输入 4,096、单次输出 512 token。Full 开启验证和修订；No verification 同时关闭验证与依赖验证的修订；No revision 保留验证。", "",
             "|模型|数据集|变体|标签正确|格式通过|Macro-F1|平均秒/条|生成 token 总数|修订次数|",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for g in result["new_agent_pilot"]:
        n = g["completed"]
        lines.append(f'|{g["model"]}|{g["dataset"]}|{g["variant"]["name"]}|{g["label_correct"]}/{n}|{g["format_pass"]}/{n}|{g["macro_f1"]:.4f}|{g["mean_latency_seconds"]:.2f}|{g["total_generated_tokens"]}|{g["revisions"]}|')
    lines += ["", "格式失败保留在分母。token 数合计所有模型调用；修订次数只说明修订发生，不代表修订成功。首次组的耗时可能包含权重首次加载，不能用于严谨的跨变体速度结论，也不能与服务器 vLLM 耗时直接比较。", "",
              "## 新跑检索评测", "", f'36 组与归档指标的最大绝对差异：`{result["retrieval_max_absolute_difference_from_archive"]:.3g}`。运行 JSON 没有固定编码器 revision；离线运行所用的唯一缓存版本已另行核对并记录在 `retrieval_environment.json`，依赖版本在运行 JSON 中。', "",
              "|数据集|检索器|分块|k|证据召回率|有效样本|", "|---|---|---|---:|---:|---:|"]
    for r in sorted(result["new_retrieval"], key=lambda r: (r["dataset"], r["retriever"], r["chunking"], r["top_k"])):
        lines.append(f'|{r["dataset"]}|{r["retriever"]}|{r["chunking"]}|{r["top_k"]}|{r["metrics"]["evidence_recall_at_k"]:.2%}|{r["metrics"]["evaluated_examples"]}|')
    lines += ["", "召回率要求完整金标区间被检索块并集覆盖；无金标证据的 903 条 ContractNLI、421 条 CUAD 不进入该指标分母。召回率不等于答案支持率。", "",
              "## 归档主实验", "", "详见 [12 组主实验与旧消融审计表](archived_result_audit/results_tables.md)。原服务器已完成全量生成，本次没有重新运行这 18,666 条预测。旧 `priority4` 消融有样本数据集错配与 CUAD 标签协议不符，不能用于受控对照；本次新跑记录单独保存。", "",
              "## 可回查文件", "", "- `summary.json`：机器可读的汇总、配对正确性变化及输入文件 SHA256。",
              "- `agent_ablation/<model>/manifest.json`：权重哈希、代码哈希、环境、样本 ID。",
              "- `agent_ablation/<model>/<dataset>/<variant>/predictions.jsonl`：逐条预测、评估和完整模型调用轨迹。",
              "- `bm25_full_rerun.json`、`dense_hybrid_full_rerun.json`：36 组全量检索结果和失败样例。",
              "- `tests.xml`：自动化测试记录，与真实模型输出分开。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.directory)
    write_json(args.directory / "summary.json", result)
    (args.directory / "RESULTS.md").write_text(tables(result), encoding="utf-8")
    print("Validated 36 new retrieval groups and 60 new Agent predictions; preserved archived main results.")


if __name__ == "__main__":
    main()
