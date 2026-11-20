"""
Priority 4: Full Dataset Experiments and Ablation Studies
Wang Zhengyao - Day 4-5 (Nov 21-22)

This script runs:
1. Full Agentic RAG on a reasonable sample size
2. Ablation experiments (no-verification, no-revision)
3. Saves all intermediate outputs for analysis
"""

import sys
import os

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
from datetime import datetime
from agent.agentic_rag import AgenticRAG
from utils.openrouter_llm import OpenRouterLLM
from data.contractnli_processor import ContractNLIProcessor
from utils.helpers import save_json
from dotenv import load_dotenv

load_dotenv()

def run_experiment(
    agent,
    examples,
    experiment_name,
    output_dir="experiments/results"
):
    """Run experiment and save results."""
    print(f"\n{'='*70}")
    print(f"Running Experiment: {experiment_name}")
    print(f"{'='*70}")
    print(f"Sample Size: {len(examples)}")
    print(f"Enable Verification: {agent.enable_verification}")
    print(f"Enable Revision: {agent.enable_revision}")

    # Convert to dict format
    example_dicts = [
        {
            'id': ex.id,
            'query': ex.query,
            'context': ex.context,
            'ground_truth_label': ex.label
        }
        for ex in examples
    ]

    # Run predictions
    start_time = datetime.now()
    predictions = agent.batch_answer(example_dicts)
    end_time = datetime.now()

    duration = (end_time - start_time).total_seconds()

    # Calculate statistics
    stats = {
        'experiment_name': experiment_name,
        'total_examples': len(predictions),
        'enable_verification': agent.enable_verification,
        'enable_revision': agent.enable_revision,
        'duration_seconds': duration,
        'avg_latency': sum(p['latency'] for p in predictions) / len(predictions),
        'total_tokens': sum(p['tokens'] for p in predictions),
        'hallucinations_detected': sum(1 for p in predictions if p['hallucination_detected']),
        'revisions_made': sum(1 for p in predictions if p['revision'] and p['revision']['changes_made']),
        'abstentions': sum(1 for p in predictions if p['abstained']),
        'timestamp': datetime.now().isoformat()
    }

    print(f"\n{'='*70}")
    print(f"Experiment Complete: {experiment_name}")
    print(f"{'='*70}")
    print(f"Duration: {duration:.2f}s ({duration/60:.2f} minutes)")
    print(f"Avg Latency: {stats['avg_latency']:.2f}s per query")
    print(f"Total Tokens: {stats['total_tokens']}")
    print(f"Hallucinations Detected: {stats['hallucinations_detected']} ({stats['hallucinations_detected']/len(predictions)*100:.1f}%)")
    print(f"Revisions Made: {stats['revisions_made']} ({stats['revisions_made']/len(predictions)*100:.1f}%)")
    print(f"Abstentions: {stats['abstentions']} ({stats['abstentions']/len(predictions)*100:.1f}%)")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save predictions with all intermediate steps
    predictions_file = os.path.join(output_dir, f"{experiment_name}_predictions.json")
    save_json(predictions, predictions_file)
    print(f"\n✓ Predictions saved to: {predictions_file}")

    # Save statistics
    stats_file = os.path.join(output_dir, f"{experiment_name}_stats.json")
    save_json(stats, stats_file)
    print(f"✓ Statistics saved to: {stats_file}")

    return predictions, stats


def main():
    parser = argparse.ArgumentParser(description="Priority 4: Full Dataset Experiments")
    parser.add_argument("--sample_size", type=int, default=50,
                        help="Number of samples to test (default: 50, use -1 for full dataset)")
    parser.add_argument("--dataset", type=str, default="contractnli",
                        help="Dataset name")
    parser.add_argument("--model", type=str, default="meta-llama/llama-3.1-8b-instruct",
                        help="Model name")
    parser.add_argument("--skip_full", action="store_true",
                        help="Skip full Agentic RAG experiment")
    parser.add_argument("--skip_no_verification", action="store_true",
                        help="Skip no-verification ablation")
    parser.add_argument("--skip_no_revision", action="store_true",
                        help="Skip no-revision ablation")

    args = parser.parse_args()

    print("="*70)
    print("Priority 4: Full Dataset Experiments and Ablation Studies")
    print("="*70)

    # Load dataset
    print(f"\nLoading {args.dataset} test dataset...")
    if args.dataset == "contractnli":
        from data.contractnli_processor import ContractNLIProcessor
        processor = ContractNLIProcessor()
        all_examples = processor.load(split="test")
    elif args.dataset == "cuad":
        from data.cuad_processor import CUADProcessor
        processor = CUADProcessor()
        all_examples = processor.load(split="test")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(f"✓ Loaded {len(all_examples)} total examples")

    # Select sample
    if args.sample_size == -1:
        examples = all_examples
        print(f"✓ Using FULL dataset ({len(examples)} examples)")
        print("⚠ WARNING: This will take a LONG time and consume many API tokens!")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    else:
        examples = all_examples[:args.sample_size]
        print(f"✓ Using sample of {len(examples)} examples")

    # Initialize LLM
    print(f"\nInitializing LLM ({args.model})...")
    llm = OpenRouterLLM(
        model_name=args.model,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_tokens=512,
        temperature=0.3
    )
    print("✓ LLM initialized")

    all_results = {}

    # ═══════════════════════════════════════════════════════════
    # Experiment 4.1: Full Agentic RAG (Verification + Revision)
    # ═══════════════════════════════════════════════════════════
    if not args.skip_full:
        print(f"\n{'#'*70}")
        print("# Experiment 4.1: Full Agentic RAG")
        print(f"{'#'*70}")

        agent_full = AgenticRAG(
            llm=llm,
            retriever_type="bm25",
            chunking_strategy="fixed_length",
            enable_verification=True,
            enable_revision=True
        )

        predictions_full, stats_full = run_experiment(
            agent_full,
            examples,
            experiment_name="agentic_rag_full",
            output_dir="experiments/results/priority4"
        )

        all_results['full'] = stats_full

    # ═══════════════════════════════════════════════════════════
    # Experiment 4.2a: Ablation - No Verification
    # ═══════════════════════════════════════════════════════════
    if not args.skip_no_verification:
        print(f"\n{'#'*70}")
        print("# Experiment 4.2a: Ablation - No Verification")
        print(f"{'#'*70}")

        agent_no_ver = AgenticRAG(
            llm=llm,
            retriever_type="bm25",
            chunking_strategy="fixed_length",
            enable_verification=False,
            enable_revision=True  # Revision needs verification, so this won't actually revise
        )

        predictions_no_ver, stats_no_ver = run_experiment(
            agent_no_ver,
            examples,
            experiment_name="agentic_rag_no_verification",
            output_dir="experiments/results/priority4"
        )

        all_results['no_verification'] = stats_no_ver

    # ═══════════════════════════════════════════════════════════
    # Experiment 4.2b: Ablation - No Revision
    # ═══════════════════════════════════════════════════════════
    if not args.skip_no_revision:
        print(f"\n{'#'*70}")
        print("# Experiment 4.2b: Ablation - No Revision")
        print(f"{'#'*70}")

        agent_no_rev = AgenticRAG(
            llm=llm,
            retriever_type="bm25",
            chunking_strategy="fixed_length",
            enable_verification=True,
            enable_revision=False
        )

        predictions_no_rev, stats_no_rev = run_experiment(
            agent_no_rev,
            examples,
            experiment_name="agentic_rag_no_revision",
            output_dir="experiments/results/priority4"
        )

        all_results['no_revision'] = stats_no_rev

    # ═══════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PRIORITY 4 EXPERIMENTS COMPLETE")
    print(f"{'='*70}")

    print("\nComparative Summary:")
    print("-" * 70)
    print(f"{'Experiment':<30} {'Tokens':<10} {'Halluc.':<10} {'Revised':<10}")
    print("-" * 70)

    for exp_name, stats in all_results.items():
        halluc_pct = f"{stats['hallucinations_detected']/stats['total_examples']*100:.1f}%"
        revised_pct = f"{stats['revisions_made']/stats['total_examples']*100:.1f}%"
        print(f"{exp_name:<30} {stats['total_tokens']:<10} {halluc_pct:<10} {revised_pct:<10}")

    # Save comparative summary
    summary_file = "experiments/results/priority4/comparative_summary.json"
    save_json(all_results, summary_file)
    print(f"\n✓ Comparative summary saved to: {summary_file}")

    print(f"\n{'='*70}")
    print("All results saved to: experiments/results/priority4/")
    print(f"{'='*70}")

    print("\nNext Steps:")
    print("  - Review prediction files for detailed analysis")
    print("  - Calculate evaluation metrics (accuracy, F1, etc.)")
    print("  - Analyze hallucination patterns")
    print("  - Prepare visualizations for report")


if __name__ == "__main__":
    main()
