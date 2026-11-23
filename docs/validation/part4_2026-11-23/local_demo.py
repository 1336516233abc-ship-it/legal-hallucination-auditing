"""Local integration candidate using the existing scored Agent path and local weights."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
os.environ.update(HF_HUB_OFFLINE="1", GRADIO_ANALYTICS_ENABLED="False",
                  TOKENIZERS_PARALLELISM="false")
sys.path.insert(0, str(REPO))
from agent.agentic_rag import AgenticRAG


class LocalReviewDemo:
    """Keep request state separate while sharing only the read-only model weights."""

    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()
        self.last_record = None

    def analyze(self, contract, query, dataset):
        if not isinstance(contract, str) or not contract.strip() or not isinstance(query, str) or not query.strip():
            return "请同时填写合同与问题。", "", "未运行", ""
        if dataset not in {"contractnli", "cuad"}:
            return "请选择命题判断或条款抽取任务。", "", "未运行", ""
        # Each request gets a new retriever, index and Agent state.
        with self.lock:
            start = time.perf_counter()
            before = len(self.model.calls)
            record = {"recorded_at": datetime.now(timezone.utc).isoformat(), "dataset": dataset,
                      "query": query, "contract": contract,
                      "contract_sha256": hashlib.sha256(contract.encode()).hexdigest(),
                      "backend": "injected local backend", "trace": None}
            try:
                agent = AgenticRAG(self.model, retriever_type="bm25", chunking_strategy="fixed_length",
                                  chunker_kwargs={"chunk_size": 512, "overlap": 128},
                                  enable_verification=True, enable_revision=True)
                trace = agent.answer(query, context=contract, dataset=dataset, max_input_tokens=4096)
                record["trace"] = trace
                evidence = "\n\n".join(f"[{i}] 字符 {chunk['start']}–{chunk['end']}\n{chunk['text']}"
                                         for i, chunk in enumerate(trace["prompt_evidence"], 1))
                error = trace.get("parse_error") or trace.get("grounding_error")
                if error:
                    answer = f"本次答案未通过检查，已停止展示。原因：{error}。请核对原文或重新提问。"
                    status = "需要人工检查；不能把本次输出视为正确答案。"
                else:
                    label = trace["label"]
                    labels = {"entailment": "命题得到支持", "contradiction": "命题与合同矛盾",
                              "neutral": "合同信息不足以判定命题", "present": "找到相关条款", "absent": "未找到相关条款"}
                    answer = labels[label] + "\n" + (trace["final_answer"] or "（无抽取文本）")
                    status = "格式与引用范围检查通过；仍需人工核对语义和证据是否充分。"
                if trace.get("revision"):
                    status += "\n已执行一次修订；修订后的语义尚未再次验证。"
                elif trace.get("verification"):
                    status += "\n初始答案验证器状态：" + trace["verification"]["overall_status"]
                outputs = [answer, evidence, status, ""]
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                outputs = ["本次处理失败，未生成可用结论。请检查模型服务或缩短输入后重试。", "", "已停止；未自动重试。", ""]
            record["elapsed_seconds"] = time.perf_counter() - start
            record["model_calls"] = self.model.calls[before:]
            # Retain only the latest request record instead of growing a prompt history.
            del self.model.calls[before:]
            metadata = {"dataset": dataset, "contract_sha256": record["contract_sha256"],
                        "elapsed_seconds": round(record["elapsed_seconds"], 3),
                        "model_calls": len(record["model_calls"]),
                        "generated_tokens_all_calls": sum(c.get("tokens", 0) for c in record["model_calls"]),
                        "revision_limit": 1, "final_semantic_reverification": False}
            outputs[3] = json.dumps(metadata, ensure_ascii=False, indent=2)
            record["outputs"] = outputs
            self.last_record = record
            return tuple(outputs)

    def interface(self):
        import gradio as gr
        with gr.Blocks(title="合同证据核对 本地候选") as demo:
            gr.Markdown("# 合同证据核对\n粘贴当前合同，选择任务并核对证据。每次输入作为独立请求处理。此本地候选尚未合入团队项目。")
            dataset = gr.Radio(choices=[("命题判断", "contractnli"), ("条款抽取", "cuad")], value="contractnli", label="任务")
            contract = gr.Textbox(label="合同原文", lines=7)
            query = gr.Textbox(label="待判断命题或条款问题", lines=2)
            go = gr.Button("核对当前合同", variant="primary")
            answer = gr.Textbox(label="结果", lines=5)
            status = gr.Textbox(label="检查状态与下一步", lines=3)
            evidence = gr.Textbox(label="最终答案对应的全部输入证据", lines=8)
            with gr.Accordion("运行记录", open=False):
                metadata = gr.Textbox(label="当前请求", lines=7)
            gr.Examples(examples=[
                ["This Agreement may be terminated by either party upon thirty (30) days written notice.", "Either party may terminate this Agreement with thirty days written notice.", "contractnli"],
                ["This Agreement may be terminated by either party upon sixty (60) days written notice.", "Either party may terminate this Agreement with thirty days written notice.", "contractnli"],
                ["The Contractor's total liability shall not exceed $100,000.", "What is the liability cap?", "cuad"],
            ], inputs=[contract, query, dataset])
            go.click(self.analyze, inputs=[contract, query, dataset], outputs=[answer, evidence, status, metadata],
                     concurrency_limit=1, api_name="analyze")
        return demo


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--snapshot", type=Path, required=True, help="Local original model snapshot directory")
    args = parser.parse_args()
    from experiments.run_baseline_pilot import MLXPilotModel
    snapshot = args.snapshot.resolve()
    if not snapshot.is_dir():
        parser.error("--snapshot must point to a downloaded local model directory")
    model = MLXPilotModel(snapshot, record_calls=True)
    LocalReviewDemo(model).interface().launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
