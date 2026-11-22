"""Audit checked-in results without running or judging a generation model."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"contractnli": ("contradiction", "entailment", "neutral"), "cuad": ("absent", "present")}
METHODS = {"direct_prompting": "Direct", "standard_rag": "RAG", "verified_agentic_rag": "Agent"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_group(group):
    """Recompute scores with invalid outputs retained in the denominator."""
    labels = LABELS[group["dataset"]]
    confusion = group["confusion"]
    n = sum(sum(row.values()) for row in confusion.values())
    if n != group["completed"] or n != group["planned"] or n <= 0:
        raise ValueError("Incomplete or inconsistent group counts")
    correct = sum(confusion.get(label, {}).get(label, 0) for label in labels)
    f1 = []
    for label in labels:
        actual = sum(confusion.get(label, {}).values())
        predicted = sum(row.get(label, 0) for row in confusion.values())
        f1.append(2 * confusion.get(label, {}).get(label, 0) / (actual + predicted) if actual + predicted else 0)
    passed = n - sum(row.get("invalid", 0) for row in confusion.values())
    expected = {"label_correct": correct, "format_pass": passed, "label_accuracy": correct / n,
                "format_pass_rate": passed / n, "macro_f1": sum(f1) / len(f1)}
    if any(abs(group[key] - value) > 1e-10 for key, value in expected.items()):
        raise ValueError("Published score does not match the confusion matrix")
    if set(confusion) - set(labels) or any(set(row) - set(labels) - {"invalid"} for row in confusion.values()):
        raise ValueError("Unexpected task labels in confusion matrix")
    return expected


def audit(root=ROOT):
    """Validate formal aggregates and flag inconsistent legacy Agent artifacts."""
    sources, main, datasets, settings = {}, [], {}, {}
    for dataset in LABELS:
        path = root / "data/processed" / f"{dataset}_test.json"
        rows = read(path)
        datasets[dataset] = {"count": len(rows), "sha256": digest(path),
                             "label_counts": dict(Counter(row["label"] for row in rows)),
                             "gold_evidence_rows": sum(bool(row.get("evidence_spans")) for row in rows)}
    for model in ("mistral", "llama3"):
        path = root / "docs/validation" / f"{model}_autodl_full_summary.json"
        result = read(path)
        sources[str(path.relative_to(root))] = digest(path)
        if result["status"] != "completed" or result["completed"] != result["planned"]:
            raise ValueError(f"Incomplete full experiment: {path}")
        settings[model] = result["settings"]
        for dataset, expected in result["settings"]["datasets"].items():
            if any(datasets[dataset][key] != expected[key] for key in ("count", "sha256")):
                raise ValueError(f"Dataset changed: {dataset}")
        if len(result["groups"]) != 6 or sum(g["completed"] for g in result["groups"]) != result["completed"]:
            raise ValueError("Invalid full experiment group count")
        for group in result["groups"]:
            validate_group(group)
            main.append({"model": model, **group, "independent_hallucination_rate": None})
    keys = {(g["model"], g["dataset"], g["method"]) for g in main}
    expected_keys = {(m, d, method) for m in settings for d in LABELS for method in METHODS}
    if keys != expected_keys:
        raise ValueError("Missing or duplicate main experiment groups")
    retrieval_path = root / "docs/validation/retrieval_ablation_2026-11-20.json"
    retrieval = read(retrieval_path)
    sources[str(retrieval_path.relative_to(root))] = digest(retrieval_path)
    for dataset, recorded in retrieval["datasets"].items():
        if recorded["sha256"] != datasets[dataset]["sha256"] or recorded["examples"] != datasets[dataset]["count"]:
            raise ValueError("Retrieval dataset hash or example count mismatch")
        if len(recorded["selected_ids"]) != len(set(recorded["selected_ids"])) or len(recorded["selected_ids"]) != recorded["examples"]:
            raise ValueError("Invalid retrieval cohort IDs")
    retrieval_rows = [{k: v for k, v in row.items() if k != "failure_examples"} for row in retrieval["experiments"]]
    retrieval_keys = {(r["dataset"], r["retriever"], r["chunking"], r["top_k"]) for r in retrieval_rows}
    expected_retrieval = {(d, r, c, k) for d in LABELS for r in ("bm25", "dense", "hybrid")
                          for c in ("fixed_length", "clause_level") for k in (3, 5, 10)}
    if retrieval_keys != expected_retrieval or len(retrieval_rows) != 36:
        raise ValueError("Incomplete retrieval ablation matrix")
    for row in retrieval_rows:
        metrics, ds = row["metrics"], datasets[row["dataset"]]
        if metrics["evaluated_examples"] != ds["gold_evidence_rows"] or metrics["evaluated_examples"] + metrics["excluded_no_evidence"] != ds["count"]:
            raise ValueError("Retrieval denominator mismatch")
    legacy_dir = root / "experiments/results/priority4"
    published = read(legacy_dir / "both_datasets_summary.json")
    sources[str((legacy_dir / "both_datasets_summary.json").relative_to(root))] = digest(legacy_dir / "both_datasets_summary.json")
    legacy = []
    for path in sorted(legacy_dir.glob("*_predictions.json")):
        rows = read(path)
        declared = "cuad" if path.name.startswith("cuad_") else "contractnli"
        variant = path.stem.removeprefix("cuad_").removeprefix("agentic_rag_").removesuffix("_predictions")
        actual = dict(Counter(row["id"].split("_test_")[0] for row in rows))
        labels = dict(Counter(row.get("label") for row in rows))
        measured = {"total_examples": len(rows), "total_tokens": sum(row.get("tokens", 0) for row in rows),
                    "avg_latency": sum(row.get("latency", 0) for row in rows) / len(rows),
                    "hallucinations_detected": sum(bool(row.get("hallucination_detected")) for row in rows),
                    "revisions_made": sum(bool(row.get("revision")) for row in rows)}
        issues = []
        if set(actual) != {declared}:
            issues.append("dataset_id_mismatch")
        if set(labels) - set(LABELS[declared]):
            issues.append("invalid_task_label_schema")
        for key, value in measured.items():
            if abs(published[declared][variant][key] - value) > 1e-8:
                issues.append(f"published_summary_mismatch:{key}")
        issues.append("legacy_trace_not_independently_evaluated")
        legacy.append({"file": str(path.relative_to(root)), "declared_dataset": declared, "variant": variant,
                       "actual_dataset_counts": actual, "label_counts": labels, "measured": measured,
                       "issues": issues, "usable_for_controlled_performance_comparison": False})
        sources[str(path.relative_to(root))] = digest(path)
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sources, "datasets": datasets, "settings": settings, "main": main,
            "retrieval": retrieval_rows, "legacy_agent_audit": legacy,
            "limitations": ["Main results checked against aggregate confusion matrices, not missing server traces.",
                            "Retrieval recall is not answer support or hallucination rate.",
                            "Legacy detector flags and revision counts do not measure successful correction."]}


def markdown(result):
    lines = ["# 实验结果核对表", "", "主实验：12 组，18,666 条系统输出。数据哈希与混淆矩阵核对通过；服务器逐条轨迹尚未复核。", "",
             "|数据集|模型|方法|n|准确率|Macro-F1|格式通过率|平均秒/条|", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in result["main"]:
        lines.append(f'|{r["dataset"]}|{r["model"]}|{METHODS[r["method"]]}|{r["completed"]}|{r["label_accuracy"]:.2%}|{r["macro_f1"]:.4f}|{r["format_pass_rate"]:.2%}|{r["mean_latency_seconds"]:.2f}|')
    lines += ["", "准确率分母保留格式失败；Macro-F1 对任务合法类别取平均。所有独立幻觉率均为待评估，不填写为 0。", "", "## 检索消融", "",
              "|数据集|检索器|切块|k|证据召回率|有效样本|", "|---|---|---|---:|---:|---:|"]
    for r in result["retrieval"]:
        lines.append(f'|{r["dataset"]}|{r["retriever"]}|{r["chunking"]}|{r["top_k"]}|{r["metrics"]["evidence_recall_at_k"]:.2%}|{r["metrics"]["evaluated_examples"]}|')
    lines += ["", "## 旧 Agent 消融审计", "", "这 6 个文件不作为完整、受控的 2 数据集 × 3 变体结论。", ""]
    for r in result["legacy_agent_audit"]:
        lines += [f'- `{r["file"]}`：实际数据集 {r["actual_dataset_counts"]}；问题：`{"; ".join(r["issues"])}`。']
    return "\n".join(lines) + "\n"


def plots(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#9aa8b7", "#3178a8", "#db8054"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, dataset in zip(axes, LABELS):
        for i, method in enumerate(METHODS):
            values = [next(r["label_accuracy"] * 100 for r in result["main"] if r["dataset"] == dataset and r["model"] == model and r["method"] == method) for model in ("mistral", "llama3")]
            bars = ax.bar(np.arange(2) + (i - 1) * .24, values, width=.23, label=METHODS[method], color=colors[i])
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        ax.set(xticks=[0, 1], xticklabels=["Mistral 7B", "Llama 3 8B"], ylim=(0, 100), title=dataset.upper(), ylabel="Label accuracy (%)")
        ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.savefig(output / "main_accuracy.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, dataset in zip(axes, LABELS):
        for i, retriever in enumerate(("bm25", "dense", "hybrid")):
            for chunking, style in (("fixed_length", "--"), ("clause_level", "-")):
                rows = sorted([r for r in result["retrieval"] if (r["dataset"], r["retriever"], r["chunking"]) == (dataset, retriever, chunking)], key=lambda r: r["top_k"])
                ax.plot([r["top_k"] for r in rows], [r["metrics"]["evidence_recall_at_k"] * 100 for r in rows], style, marker="o", color=colors[i], label=f'{retriever.upper()} / {"fixed" if chunking == "fixed_length" else "clause"}')
        ax.set(title=dataset.upper(), xlabel="Retrieved chunks (k)", ylabel="Gold-span recall (%)", xticks=[3, 5, 10], ylim=(35, 102))
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.savefig(output / "retrieval_recall.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    result = audit()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "verified_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output / "results_tables.md").write_text(markdown(result))
    if args.plots:
        plots(result, args.output)
    print(f'Validated {len(result["main"])} main groups and {len(result["retrieval"])} retrieval groups; audited {len(result["legacy_agent_audit"])} legacy files.')


if __name__ == "__main__":
    main()
