"""Write portable validation summaries from JUnit and saved inference artifacts."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval.common import parse_response, write_json


def junit_summary(path):
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    stats = {key: sum(int(suite.get(key, 0)) for suite in suites)
             for key in ["tests", "failures", "errors", "skipped"]}
    stats["passed"] = stats["tests"] - stats["failures"] - stats["errors"] - stats["skipped"]
    stats["seconds"] = sum(float(suite.get("time", 0)) for suite in suites)
    return stats


def compare_smoke_runs(destination):
    """Replay archived responses and compare the matching fresh inference runs."""
    comparisons = []
    for path in sorted((destination / "qwen_smoke").glob("*.meta.json")):
        metadata = json.loads(path.read_text())
        prediction_name = path.name.replace(".meta.json", ".json")
        original = json.loads(path.with_name(prediction_name).read_text())
        rerun_meta_path = destination / "qwen_format_retest" / path.name
        rerun = []
        if rerun_meta_path.exists():
            rerun_meta = json.loads(rerun_meta_path.read_text())
            for key in ("model", "seed", "device", "split", "dataset_sha256", "selected_ids",
                        "requested_max_input_tokens", "retrieval"):
                if metadata[key] != rerun_meta[key]:
                    raise ValueError(f"Unmatched smoke-test setting {key}: {prediction_name}")
            rerun = json.loads(rerun_meta_path.with_name(prediction_name).read_text())
            if [row["id"] for row in original] != [row["id"] for row in rerun]:
                raise ValueError(f"Unmatched prediction IDs: {prediction_name}")

        replay_errors = [parse_response(row["raw_response"], row["dataset"])["parse_error"]
                         for row in original]
        cases = []
        for index, row in enumerate(original):
            case = {"id": row["id"], "original_parse_error": row["parse_error"],
                    "parser_only_error": replay_errors[index], "original_tokens": row["tokens"]}
            if rerun:
                new = rerun[index]
                case.update(rerun_parse_error=new["parse_error"], rerun_label=new["label"],
                            rerun_tokens=new["tokens"], same_prompt=row["prompt"] == new["prompt"],
                            same_raw_response=row["raw_response"] == new["raw_response"])
            cases.append(case)
        comparisons.append({"mode": metadata["mode"], "dataset": metadata["dataset"],
                            "model": metadata["model"]["name"], "examples": len(original),
                            "original_failures": sum(bool(row["parse_error"]) for row in original),
                            "parser_only_failures": sum(bool(error) for error in replay_errors),
                            "rerun_failures": sum(bool(row["parse_error"]) for row in rerun) if rerun else None,
                            "cases": cases})
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full_junit", type=Path, required=True)
    parser.add_argument("--minimal_junit", type=Path, required=True)
    args = parser.parse_args()
    destination = ROOT / "docs/validation"
    environment = {"python": platform.python_version(), "system": platform.system(),
                   "machine": platform.machine(), "packages": {}}
    for name in ["pytest", "numpy", "rank-bm25", "pyyaml", "torch", "transformers", "sentence-transformers", "accelerate"]:
        try:
            environment["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(destination / "environment.json", environment)
    summary = {"date": "2026-11-20", "full_environment": junit_summary(args.full_junit),
               "minimal_environment": junit_summary(args.minimal_junit), "real_model_smoke": []}
    for path in sorted((destination / "qwen_smoke").glob("*.meta.json")):
        metadata = json.loads(path.read_text())
        predictions = json.loads(path.with_name(path.name.replace(".meta.json", ".json")).read_text())
        errors = sum(parse_response(p["raw_response"], p["dataset"])["parse_error"] is not None for p in predictions)
        summary["real_model_smoke"].append({"mode": metadata["mode"], "dataset": metadata["dataset"],
            "model": metadata["model"]["name"], "model_revision": metadata.get("resolved_model_revision"),
            "examples": len(predictions), "execution_status": metadata["status"],
            "stored_parse_errors": metadata["parse_error_count"], "current_parser_errors": errors,
            "format_validation": "failed" if errors else "passed",
            "research_performance_claim": False})
    summary["response_format_comparison"] = compare_smoke_runs(destination)
    matrix = json.loads((destination / "retrieval_ablation_2026-11-20.json").read_text())
    summary["retrieval_matrix"] = {"configurations": len(matrix["experiments"]),
        "examples": {name: values["examples"] for name, values in matrix["datasets"].items()},
        "kind": matrix["kind"], "definition": matrix["metric_definition"]}
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    summary["source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in sorted(set(paths))
        if Path(name).suffix in {".py", ".yaml"} or name in {"requirements.txt", "requirements-retrieval.txt"}}
    write_json(destination / "test_summary.json", summary)
    runs = {(run["dataset"], run["retriever"], run["chunking"]): run
            for run in matrix["experiments"] if run["top_k"] == 5}
    lines = ["# 检索消融结果", "", "本轮固定配置在两个完整测试集上运行，没有调用生成模型。"
             "结果文件为 retrieval_ablation_2026-11-20.json，包含 36 个配置、数据校验值、样本 ID、参数与失败案例。", "",
             "## 指标口径", "",
             "Evidence Recall@k 在单份合同内按 gold span 统计：只有检索区间的并集覆盖该 span 的全部非空白字符才算命中。"
             "跨 chunk 的连续证据可合并；空白间隔不构成缺失。样本内取 span 召回比例，再在有证据样本间求平均。", "",
             "Hit Rate 表示至少完整覆盖一个 gold span 的样本比例。Character Recall 表示 gold 非空白字符的覆盖比例，"
             "先合并重叠区间以免重复计数。缺少 context、文档不符或切片内容不一致会报错，无证据样本不参与这些指标。", "",
             "ContractNLI：2,091 条中 1,188 条参与证据指标，903 条排除。CUAD：1,020 条中 599 条参与，421 条排除。", "",
             "## Recall@5 对照", "", "| 数据集 | 检索方法 | 固定长度 | 条款边界 |", "|---|---|---:|---:|"]
    for dataset in ["contractnli", "cuad"]:
        for retriever in ["bm25", "dense", "hybrid"]:
            fixed = runs[dataset, retriever, "fixed_length"]["metrics"]["evidence_recall_at_k"]
            clause = runs[dataset, retriever, "clause_level"]["metrics"]["evidence_recall_at_k"]
            lines.append(f"| {dataset} | {retriever} | {fixed:.2%} | {clause:.2%} |")
    lines += ["", "## 配置与解释边界", "",
              "两种策略的最大分块预算均为 512 词；固定长度 overlap=128，条款策略 min_chunk_size=50。"
              "Dense 使用 all-MiniLM-L6-v2，编码上限为 256 个 subword token，长块会在编码时截断；"
              "Hybrid 权重各 0.5。JSON 同时记录 k=3/5/10。", "",
              "表格描述当前固定配置，未据测试集重新选择参数。固定长度与条款策略的边界、重叠方式不同，"
              "不能把差异归因于单一因素。较大的 k 对短合同可能接近全文覆盖；高证据召回不等于答案准确或幻觉更少。", "",
              "当前独立指标的定义不同于旧 evaluation/metrics.py 的字符串包含判断，不能将两者数值直接混合汇总。"
              "最终报告需统一使用这一明确口径，或由评估负责人提供统一的新版本后重新计算。", ""]
    (destination / "RETRIEVAL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
