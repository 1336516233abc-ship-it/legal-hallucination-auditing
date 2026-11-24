"""Run hand-constructed acceptance requests through a running Gradio UI service."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ui.app import EXAMPLES


def main():
    from gradio_client import Client
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file; previous acceptance records must be retained")
    report = {"scope": "Hand-constructed UI service acceptance, not benchmark or independent human evaluation",
              "started_at": datetime.now(timezone.utc).isoformat(), "cases": [],
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("ui/app.py", "ui/service.py", "ui/backends.py")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    save()
    client = Client(args.url, verbose=False)
    requests = [(name, source, query, task, expected) for (name, source, query, task), expected
                in zip(EXAMPLES, ("entailment", "contradiction", "present"))]
    requests.append(("empty_input", " ", "Notice is required.", "contractnli", None))
    for name, source, query, task, expected in requests:
        print(f"Running {name}", flush=True)
        item = {"name": name, "expected_label": expected}
        try:
            outputs = client.predict(source, query, task, True, True, api_name="/review")
            answer, evidence, status, verification, metadata, record = outputs[:6]
            trace = record.get("trace") if record else None
            item.update(answer=answer, evidence=evidence, status=status, verification=verification,
                        metadata=metadata, record=record, observed_label=trace.get("label") if trace else None)
            item["source_is_current"] = (record["contract"] == source and all(
                chunk["text"] == source[chunk["start"]:chunk["end"]] for chunk in trace["prompt_evidence"])) if trace else None
            item["model_case_pass"] = (trace.get("label") == expected and not trace.get("parse_error")
                                       and not trace.get("grounding_error")) if trace else None
            item["empty_input_rejected"] = (not answer and record is None and "请同时填写" in status) if expected is None else None
        except Exception as exc:
            item["transport_error"] = f"{type(exc).__name__}: {exc}"
        report["cases"].append(item)
        save()
        print(f"Finished {name}: label={item.get('observed_label')}, valid={item.get('model_case_pass')}", flush=True)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["model_cases_passed"] = sum(row.get("model_case_pass") is True for row in report["cases"])
    report["model_cases_total"] = 3
    report["transport_errors"] = sum("transport_error" in row for row in report["cases"])
    save()
    print(json.dumps({k: report[k] for k in ("model_cases_passed", "model_cases_total", "transport_errors")}), flush=True)


if __name__ == "__main__":
    main()
