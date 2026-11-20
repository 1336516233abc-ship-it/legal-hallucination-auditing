"""
Run Priority 4 experiments on CUAD dataset with retry logic
"""
import sys
import os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import time
from datetime import datetime
from agent.agentic_rag import AgenticRAG
from utils.openrouter_llm import OpenRouterLLM
from data.cuad_processor import CUADProcessor
from utils.helpers import save_json
from dotenv import load_dotenv

load_dotenv()

def run_cuad_experiment(exp_name, enable_ver, enable_rev, sample_size=20):
    """Run single CUAD experiment with retry logic"""

    print(f"\n{'='*70}")
    print(f"Running: {exp_name} on CUAD dataset")
    print(f"{'='*70}")
    print(f"Samples: {sample_size}")
    print(f"Verification: {enable_ver}, Revision: {enable_rev}")

    # Load data
    processor = CUADProcessor()
    examples = processor.load(split="test")[:sample_size]

    # Initialize LLM
    llm = OpenRouterLLM(
        model_name="meta-llama/llama-3.1-8b-instruct",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_tokens=512,
        temperature=0.3
    )

    # Initialize agent
    agent = AgenticRAG(
        llm=llm,
        retriever_type="bm25",
        chunking_strategy="fixed_length",
        enable_verification=enable_ver,
        enable_revision=enable_rev
    )

    # Convert to dict format
    example_dicts = [
        {
            'id': ex.id,
            'query': ex.query,
            'context': ex.context,
            'ground_truth_label': getattr(ex, 'label', None)
        }
        for ex in examples
    ]

    # Run with retry on API errors
    predictions = []
    start_time = time.time()

    # Index all documents first
    print("Indexing documents...")
    contexts = [ex['context'] for ex in example_dicts]
    agent.index_documents(contexts)

    # Process each example with retry
    for i, ex_dict in enumerate(example_dicts):
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                print(f"Processing {i+1}/{sample_size}...", end=' ', flush=True)
                result = agent.answer(ex_dict['query'])
                result['id'] = ex_dict['id']
                result['ground_truth_label'] = ex_dict['ground_truth_label']
                predictions.append(result)
                print(f"✓ ({result['latency']:.1f}s)")
                break

            except Exception as e:
                retry_count += 1
                error_msg = str(e)

                if '402' in error_msg or 'credits' in error_msg.lower():
                    # API credit issue - wait longer
                    wait_time = 120  # 2 minutes
                    print(f"\n⚠ API limit reached. Waiting {wait_time}s... (retry {retry_count}/{max_retries})")
                    time.sleep(wait_time)

                elif 'rate' in error_msg.lower() or 'limit' in error_msg.lower():
                    # Rate limit - exponential backoff
                    wait_time = 30 * (2 ** retry_count)
                    print(f"\n⚠ Rate limit. Waiting {wait_time}s... (retry {retry_count}/{max_retries})")
                    time.sleep(wait_time)

                else:
                    # Other error
                    print(f"\n✗ Error: {error_msg}")
                    if retry_count >= max_retries:
                        print(f"Skipping sample {i+1} after {max_retries} retries")
                        break
                    time.sleep(10)

    duration = time.time() - start_time

    # Calculate stats
    if len(predictions) > 0:
        stats = {
            'experiment_name': exp_name,
            'dataset': 'cuad',
            'total_examples': len(predictions),
            'enable_verification': enable_ver,
            'enable_revision': enable_rev,
            'duration_seconds': duration,
            'avg_latency': sum(p['latency'] for p in predictions) / len(predictions),
            'total_tokens': sum(p['tokens'] for p in predictions),
            'hallucinations_detected': sum(1 for p in predictions if p['hallucination_detected']),
            'revisions_made': sum(1 for p in predictions if p['revision'] and p['revision']['changes_made']),
            'abstentions': sum(1 for p in predictions if p['abstained']),
            'timestamp': datetime.now().isoformat()
        }
    else:
        stats = {'error': 'No predictions completed'}

    # Save results
    output_dir = "experiments/results/priority4"
    os.makedirs(output_dir, exist_ok=True)

    pred_file = f"{output_dir}/cuad_{exp_name}_predictions.json"
    stats_file = f"{output_dir}/cuad_{exp_name}_stats.json"

    save_json(predictions, pred_file)
    save_json(stats, stats_file)

    print(f"\n✓ Results saved:")
    print(f"  - {pred_file}")
    print(f"  - {stats_file}")

    if len(predictions) > 0:
        print(f"\nStats:")
        print(f"  Duration: {duration/60:.2f} minutes")
        print(f"  Avg Latency: {stats['avg_latency']:.2f}s")
        print(f"  Tokens: {stats['total_tokens']}")
        print(f"  Hallucinations: {stats['hallucinations_detected']}/{stats['total_examples']}")
        print(f"  Revisions: {stats['revisions_made']}/{stats['total_examples']}")

    return predictions, stats


def main():
    print("="*70)
    print("CUAD Dataset - Priority 4 Experiments")
    print("="*70)

    sample_size = 20

    # Run three experiments
    experiments = [
        ("agentic_rag_full", True, True),
        ("agentic_rag_no_verification", False, True),
        ("agentic_rag_no_revision", True, False)
    ]

    all_results = {}

    for exp_name, enable_ver, enable_rev in experiments:
        # Add delay between experiments to avoid rate limits
        if len(all_results) > 0:
            print(f"\nWaiting 60s before next experiment...")
            time.sleep(60)

        predictions, stats = run_cuad_experiment(
            exp_name, enable_ver, enable_rev, sample_size
        )
        all_results[exp_name] = stats

    # Save summary
    summary_file = "experiments/results/priority4/cuad_summary.json"
    save_json(all_results, summary_file)

    print("\n" + "="*70)
    print("CUAD Experiments Complete!")
    print("="*70)
    print(f"\nSummary saved to: {summary_file}")

    # Print comparison
    print("\nComparison:")
    print("-"*70)
    for name, stats in all_results.items():
        if 'total_examples' in stats:
            print(f"{name}: {stats['hallucinations_detected']}/{stats['total_examples']} hallucinations, "
                  f"{stats['total_tokens']} tokens")

if __name__ == "__main__":
    main()
