"""Request-scoped contract review and truthful user-facing result formatting."""

from datetime import datetime, timezone
import hashlib
import threading
import time

from agent.agentic_rag import AgenticRAG
from experiments.run_experiments import TracedModel

LABELS = {"entailment": "命题得到支持", "contradiction": "命题与合同矛盾",
          "neutral": "现有证据不足以判断命题", "present": "找到相关条款", "absent": "检索到的证据中未找到相关条款"}
VERDICTS = {"supported": "原文支持", "contradicted": "与原文矛盾",
            "not_enough_info": "证据不足", "partially_supported": "部分内容缺少依据"}


def empty_result(message):
    return {"answer": "", "evidence": "", "status": message, "verification": "", "metadata": None, "record": None}


def explain_error(trace):
    if trace.get("parse_error"):
        return "模型没有按所选任务返回有效结果。可以缩短问题后重试。"
    if trace.get("invalid_citations"):
        return "回答引用了本次输入中不存在的证据编号。请核对原文。"
    return "抽取内容未通过原文或引用检查。请核对原文。"


class ReviewService:
    def __init__(self, model, *, model_info=None, max_input_tokens=4096):
        self.model = model
        self.model_info = dict(model_info or {})
        self.max_input_tokens = max_input_tokens
        self.lock = threading.Lock()

    def review(self, contract, query, dataset="contractnli", verify=True, revise=True):
        if not isinstance(contract, str) or not contract.strip() or not isinstance(query, str) or not query.strip():
            return empty_result("请同时填写合同原文和问题。")
        if dataset not in {"contractnli", "cuad"}:
            return empty_result("请选择“核对命题”或“查找条款”。")
        if len(contract) > 200_000 or len(query) > 2_000:
            return empty_result("输入过长：合同最多 200,000 个字符，问题最多 2,000 个字符。请缩短后重试。")
        if type(verify) is not bool or type(revise) is not bool:
            return empty_result("检查选项无效，请重新选择。")

        # Serialize the shared model, but never share an Agent, index, or result between requests.
        with self.lock:
            start = time.perf_counter()
            model = TracedModel(self.model)
            record = {"recorded_at": datetime.now(timezone.utc).isoformat(), "dataset": dataset,
                      "query": query, "contract": contract,
                      "contract_sha256": hashlib.sha256(contract.encode()).hexdigest(),
                      "verification_enabled": verify, "revision_enabled": verify and revise,
                      "model": self.model_info, "trace": None}
            result = empty_result("")
            try:
                agent = AgenticRAG(model, retriever_type="bm25", chunking_strategy="fixed_length",
                                  chunker_kwargs={"chunk_size": 512, "overlap": 128},
                                  enable_verification=verify, enable_revision=verify and revise)
                trace = agent.answer(query, context=contract, dataset=dataset,
                                     max_input_tokens=self.max_input_tokens)
                record["trace"] = trace
                result["evidence"] = "\n\n".join(
                    f"[{i}] 原文字符 {chunk['start']}–{chunk['end']}\n{chunk['text']}"
                    for i, chunk in enumerate(trace["prompt_evidence"], 1)) or "本次没有可用的输入证据。"
                if trace.get("parse_error") or trace.get("grounding_error"):
                    result["status"] = "答案未通过检查，已停止展示。" + explain_error(trace)
                else:
                    result["answer"] = LABELS[trace["label"]] + "\n\n" + (trace["final_answer"] or "（无抽取原句）")
                    result["status"] = "格式与引用检查通过。请结合原文核对结论；这不代表已经排除幻觉。"

                verification, revision = trace.get("verification"), trace.get("revision")
                lines = []
                if not verify:
                    lines.append("本次未启用依据检查，也未启用修订。")
                elif verification is None:
                    lines.append("初始输出未通过格式或引用检查，未进入依据检查与修订。")
                elif not verification.get("claims"):
                    lines.append("验证器未返回可评审的断言，不能据此认定答案得到支持。")
                else:
                    lines.append("初始回答检查：" + VERDICTS.get(verification["overall_status"], verification["overall_status"]))
                    for i, claim in enumerate(verification["claims"], 1):
                        lines.append(f"{i}. {claim['claim']}\n   {VERDICTS.get(claim['status'], claim['status'])}")
                        if claim.get("parse_error"):
                            lines.append("   此条检查结果格式无效，不能视为验证成功。")
                    if verification.get("hallucination_detected"):
                        result["status"] += "\n初始回答存在矛盾或依据不足。"
                if revision is not None:
                    lines.append("已尝试修订一次；最终回答的语义尚未再次验证。")
                    result["status"] += "\n已尝试修订；请人工核对最终回答。"
                elif verify and not revise:
                    lines.append("本次未启用修订。")
                elif verification is not None:
                    lines.append("本次未触发修订。")
                result["verification"] = "\n".join(lines)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                result = empty_result("本次处理失败，未生成可用结论。请缩短输入或检查模型服务后重试；没有自动重试。")
            record["elapsed_seconds"] = time.perf_counter() - start
            record["model_calls"] = model.calls
            result["metadata"] = {**self.model_info, "dataset": dataset,
                                  "contract_sha256": record["contract_sha256"],
                                  "elapsed_seconds": round(record["elapsed_seconds"], 3),
                                  "model_calls": len(model.calls),
                                  "generated_tokens_all_calls": sum(call.get("tokens", 0) for call in model.calls),
                                  "verification_enabled": verify, "revision_enabled": verify and revise,
                                  "revision_limit": 1, "final_semantic_reverification": False}
            result["record"] = record
            return result
