"""Interactive contract review using the same task-aware Agent path as experiments."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from ui.service import ReviewService

EXAMPLES = [
    ("30 天通知", "This Agreement may be terminated by either party upon thirty (30) days written notice.",
     "Either party may terminate this Agreement with thirty days written notice.", "contractnli"),
    ("改为 60 天通知", "This Agreement may be terminated by either party upon sixty (60) days written notice.",
     "Either party may terminate this Agreement with thirty days written notice.", "contractnli"),
    ("查找责任上限", "The Contractor's total liability shall not exceed $100,000.",
     "What is the liability cap in this contract?", "cuad"),
]


class ContractReviewUI:
    """Share model weights while keeping contracts and traces local to each request."""

    def __init__(self, model_name="llama3", *, model=None, backend="transformers", snapshot=None,
                 device="auto", max_input_tokens=4096):
        self.model_name = model_name
        if model is None:
            from ui.backends import load_backend
            model, model_info = load_backend(model_name, backend, snapshot, device, max_input_tokens)
        else:
            model_info = {"model": model_name, "backend": backend}
        self.service = ReviewService(model, model_info=model_info, max_input_tokens=max_input_tokens)

    def analyze_contract(self, contract_text, question, enable_verification=True,
                         enable_revision=True, dataset="contractnli"):
        """Keep the original four-output Python callback available to existing callers."""
        result = self.service.review(contract_text, question, dataset, enable_verification, enable_revision)
        return (result["answer"], result["evidence"], result["status"] + "\n" + result["verification"],
                json.dumps(result["metadata"], ensure_ascii=False, indent=2))

    def create_interface(self):
        import gradio as gr

        with gr.Blocks(title="合同证据核对 · Contract Review", analytics_enabled=False) as demo:
            gr.Markdown("# 合同证据核对\n粘贴合同，核对一个命题或查找相关条款。答案旁保留原文证据，便于逐句核实。")
            gr.Markdown(f"**当前模型：{self.model_name}** · 英文合同与英文问题 · 检索 → 回答 → 检查 → 按需修订")
            with gr.Row():
                with gr.Column(scale=5):
                    task = gr.Radio([("核对命题", "contractnli"), ("查找条款", "cuad")],
                                    value="contractnli", label="你要做什么？")
                    contract = gr.Textbox(label="合同原文", lines=10,
                                          placeholder="在这里粘贴当前合同的英文原文")
                    query = gr.Textbox(label="待核对的命题 / 要查找的条款", lines=2,
                                       placeholder="例如：Either party may terminate with thirty days written notice.")
                    gr.Markdown("命题判断：填写一句需要核对的陈述。条款查找：填写问题，返回合同里的相关原句。")
                    with gr.Accordion("检查选项", open=False):
                        verify = gr.Checkbox(label="检查答案依据", value=True)
                        revise = gr.Checkbox(label="检查发现问题后，尝试修订一次", value=True)
                        gr.Markdown("关闭依据检查时，修订也会关闭。自动检查不能代替人工核对。")
                    with gr.Row():
                        run = gr.Button("核对当前合同", variant="primary")
                        clear = gr.Button("清空")
                    gr.Markdown("**试一试** · 以下是演示输入，不计入数据集实验")
                    with gr.Row():
                        example_buttons = [gr.Button(row[0], size="sm") for row in EXAMPLES]
                with gr.Column(scale=6):
                    status = gr.Textbox(label="本次检查状态", value="等待输入合同和问题。", lines=3, interactive=False)
                    answer = gr.Textbox(label="回答", lines=5, interactive=False)
                    evidence = gr.Textbox(label="本次回答使用的原文证据", lines=10, interactive=False)
                    gr.Markdown("证据编号对应本次最终回答的输入。自动检查通过不代表回答一定正确；未找到条款也不等于整份合同没有该条款。")
            with gr.Accordion("查看检查与运行记录", open=False):
                verification = gr.Textbox(label="初始回答的检查与修订情况", lines=6, interactive=False)
                metadata = gr.JSON(label="本次运行信息")
                trace = gr.JSON(label="原始运行记录（含失败输出，用于排查）")

            outputs = [answer, evidence, status, verification, metadata, trace]
            controls = [contract, query, task, verify, revise, run, clear, *example_buttons]

            def clear_outputs():
                return "", "", "输入已更新，请点击“核对当前合同”。", "", None, None

            def analyze(current_contract, current_query, current_task, check, repair):
                # Disable editing while inference runs, so an old request cannot appear under a new contract.
                # A queued request must also restore the exact inputs captured when it was submitted.
                submitted_values = [current_contract, current_query, current_task, check, repair]
                yield ("", "", "正在核对当前合同，请稍候……", "", None, None,
                       *[gr.update(value=value, interactive=False) for value in submitted_values],
                       *[gr.update(interactive=False) for _ in controls[5:]])
                result = self.service.review(current_contract, current_query, current_task, check, repair)
                yield (result["answer"], result["evidence"], result["status"], result["verification"],
                       result["metadata"], result["record"], *[gr.update(interactive=True) for _ in controls])

            run.click(analyze, [contract, query, task, verify, revise], outputs + controls,
                      concurrency_limit=1, api_name="review", show_progress="minimal")
            for component in [contract, query, task, verify, revise]:
                component.input(clear_outputs, outputs=outputs, queue=False, api_visibility="private")
            for button, (_, source, question, dataset) in zip(example_buttons, EXAMPLES):
                def use_example(source=source, question=question, dataset=dataset):
                    return source, question, dataset, *clear_outputs()
                button.click(use_example, outputs=[contract, query, task, *outputs],
                             queue=False, api_visibility="private")
            clear.click(lambda: ("", "", *clear_outputs()), outputs=[contract, query, *outputs],
                        queue=False, api_visibility="private")
        return demo


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["llama3", "mistral"], default="llama3")
    parser.add_argument("--backend", choices=["transformers", "mlx"], default="transformers")
    parser.add_argument("--snapshot", type=Path, help="Existing local model directory; required for MLX")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--port", type=int, default=7860)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not 1024 <= args.max_input_tokens <= 4096:
        parser.error("--max-input-tokens must be between 1024 and 4096")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.backend == "mlx" and args.snapshot is None:
        parser.error("--backend mlx requires --snapshot pointing to local model weights")
    if args.snapshot is not None and not args.snapshot.is_dir():
        parser.error("--snapshot must be an existing directory")
    print(f"Loading {args.model} ({args.backend})...", flush=True)
    try:
        ui = ContractReviewUI(args.model, backend=args.backend, snapshot=args.snapshot,
                              device=args.device, max_input_tokens=args.max_input_tokens)
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Model startup failed: {exc}\nSee docs/UI_HANDOFF.md for installation and model setup.\n")
    ui.create_interface().queue(max_size=8).launch(
        server_name="127.0.0.1", server_port=args.port, share=False, show_error=False)


if __name__ == "__main__":
    main()
