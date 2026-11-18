"""
Optional Gradio UI for interactive contract analysis.

Author: Wang Zhengyao (3036795675)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Install with: pip install gradio")
    sys.exit(1)

from utils.llm_interface import LLMInterface
from utils.helpers import load_yaml
from agent.agentic_rag import AgenticRAG
import json


class ContractReviewUI:
    """Interactive UI for contract review with hallucination auditing."""

    def __init__(self, model_name: str = "llama3"):
        """Initialize UI with model."""
        # Load config
        config = load_yaml("configs/model_config.yaml")
        model_config = config['model'][model_name]

        # Initialize LLM
        self.llm = LLMInterface(
            model_name=model_config['name'],
            max_tokens=model_config['max_tokens'],
            temperature=model_config['temperature'],
        )

        # Initialize Agentic RAG
        self.agent = AgenticRAG(
            llm=self.llm,
            retriever_type="bm25",
            chunking_strategy="clause_level",
            enable_verification=True,
            enable_revision=True,
        )

    def analyze_contract(
        self,
        contract_text: str,
        question: str,
        enable_verification: bool,
        enable_revision: bool
    ):
        """Analyze contract with question."""
        if not contract_text or not question:
            return "Please provide both contract text and a question.", "", "", ""

        # Update agent settings
        self.agent.enable_verification = enable_verification
        self.agent.enable_revision = enable_revision

        # Run analysis
        try:
            result = self.agent.answer(question, context=contract_text)

            # Format output
            answer = result['final_answer']

            # Evidence
            evidence_text = "\n\n".join([
                f"**Evidence {i+1}:**\n{ev}"
                for i, ev in enumerate(result['retrieved_evidence'][:3])
            ])

            # Verification details
            verification_text = ""
            if result.get('verification'):
                verif = result['verification']
                verification_text = f"**Overall Status:** {verif['overall_status']}\n\n"
                verification_text += f"**Hallucination Detected:** {verif['hallucination_detected']}\n\n"

                if verif.get('claims'):
                    verification_text += "**Claims Verification:**\n"
                    for i, claim in enumerate(verif['claims'][:5]):
                        verification_text += f"\n{i+1}. {claim['claim']}\n"
                        verification_text += f"   Status: {claim['status']}\n"

            # Metadata
            metadata = {
                'query_type': result.get('query_analysis', {}).get('query_type', 'N/A'),
                'initial_answer': result.get('initial_answer', 'N/A'),
                'revision_made': result.get('revision', {}).get('changes_made', False) if result.get('revision') else False,
                'latency': f"{result.get('latency', 0):.2f}s",
                'tokens': result.get('tokens', 0),
            }
            metadata_text = json.dumps(metadata, indent=2)

            return answer, evidence_text, verification_text, metadata_text

        except Exception as e:
            return f"Error: {str(e)}", "", "", ""

    def create_interface(self):
        """Create Gradio interface."""
        with gr.Blocks(title="Legal Contract Hallucination Auditing") as demo:
            gr.Markdown(
                """
                # 🔍 Legal Contract Hallucination Auditing Agent

                An Agentic RAG system that audits and reduces hallucinations in legal contract analysis.

                **Features:**
                - Query analysis and intelligent retrieval
                - Evidence-based answer generation
                - Automatic hallucination detection
                - Self-revision for accuracy
                """
            )

            with gr.Row():
                with gr.Column(scale=2):
                    contract_input = gr.Textbox(
                        label="Contract Text",
                        placeholder="Paste your contract text here...",
                        lines=10,
                    )

                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a question about the contract...",
                        lines=2,
                    )

                    with gr.Row():
                        verify_checkbox = gr.Checkbox(
                            label="Enable Verification",
                            value=True
                        )
                        revise_checkbox = gr.Checkbox(
                            label="Enable Revision",
                            value=True
                        )

                    analyze_btn = gr.Button("Analyze Contract", variant="primary")

                with gr.Column(scale=2):
                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=6,
                    )

                    evidence_output = gr.Textbox(
                        label="Retrieved Evidence",
                        lines=6,
                    )

            with gr.Row():
                with gr.Column():
                    verification_output = gr.Textbox(
                        label="Verification Details",
                        lines=8,
                    )

                with gr.Column():
                    metadata_output = gr.Textbox(
                        label="Metadata",
                        lines=8,
                    )

            # Examples
            gr.Examples(
                examples=[
                    [
                        "This Agreement may be terminated by either party upon thirty (30) days written notice. Upon termination, all rights and obligations shall cease except for those provisions that by their nature should survive.",
                        "Can the contract be terminated with 30 days notice?",
                        True,
                        True
                    ],
                    [
                        "The Contractor shall be liable for any damages arising from breach of this agreement, provided that the total liability shall not exceed $100,000.",
                        "What is the liability cap in this contract?",
                        True,
                        True
                    ],
                ],
                inputs=[contract_input, question_input, verify_checkbox, revise_checkbox],
            )

            # Connect button
            analyze_btn.click(
                fn=self.analyze_contract,
                inputs=[contract_input, question_input, verify_checkbox, revise_checkbox],
                outputs=[answer_output, evidence_output, verification_output, metadata_output],
            )

        return demo


def main():
    """Launch UI."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch Contract Review UI")
    parser.add_argument("--model", type=str, default="llama3", help="Model to use")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")

    args = parser.parse_args()

    print(f"Initializing UI with model: {args.model}")
    ui = ContractReviewUI(model_name=args.model)

    print("Creating interface...")
    demo = ui.create_interface()

    print(f"Launching on port {args.port}...")
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
