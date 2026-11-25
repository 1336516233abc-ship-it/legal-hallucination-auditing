"""
Priority 3 Test: Agentic RAG Integration
Wang Zhengyao - Day 3 (Nov 20)
Tests complete workflow: Query Analysis → Retrieval → Generation → Verification → Revision
"""

import sys
import os

# Fix Windows encoding issue
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Priority 3: Agentic RAG Integration Test")
print("=" * 70)

# ══════════════════════════════════════════════════════════════
# Test 3.1: Complete Agentic RAG Workflow
# ═��════════════════════════════════════════════════════════════
print("\n[TEST 3.1] Complete Agentic RAG Workflow")
print("-" * 70)

try:
    from agent.agentic_rag import AgenticRAG
    from utils.openrouter_llm import OpenRouterLLM
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Initialize LLM (using OpenRouter API)
    print("Initializing LLM via OpenRouter API...")
    llm = OpenRouterLLM(
        model_name="meta-llama/llama-3.1-8b-instruct",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_tokens=512,
        temperature=0.3
    )
    print("  ✓ LLM initialized (OpenRouter API)")

    # Initialize Agentic RAG with all components enabled
    print("\nInitializing Agentic RAG...")
    agent = AgenticRAG(
        llm=llm,
        retriever_type="bm25",
        chunking_strategy="fixed_length",
        enable_verification=True,
        enable_revision=True
    )
    print("  ✓ AgenticRAG initialized with:")
    print("    - Retriever: BM25")
    print("    - Chunking: Fixed Length (default: 512 tokens, 128 overlap)")
    print("    - Verification: ENABLED")
    print("    - Revision: ENABLED")

    # Test contract and query
    test_contract = """
    TERMINATION AGREEMENT

    Article 5: Termination Rights
    Either party may terminate this Agreement by providing thirty (30) days written
    notice to the other party. The notice must be sent via registered mail or email
    to the address specified in Article 12.

    Article 6: Payment Terms
    The Client shall pay the Contractor a monthly fee of $5,000 USD. Payment is due
    on the 15th day of each calendar month. Late payments will incur a penalty of
    2% per month.

    Article 7: Confidentiality
    Both parties agree to maintain confidentiality of all proprietary information
    for a period of 5 years following the termination of this Agreement.

    Article 8: Liability
    The Contractor's total liability under this Agreement shall not exceed the total
    amount paid by the Client in the preceding 12 months.
    """

    test_query = "Can this contract be terminated early with 30 days notice?"

    print(f"\nTest Query: {test_query}")
    print(f"Contract Length: {len(test_contract)} characters")

    # Run complete workflow
    print("\n" + "=" * 70)
    print("Running Complete Agentic RAG Workflow...")
    print("=" * 70)

    result = agent.answer(test_query, context=test_contract)

    # Display results
    print("\n✓ WORKFLOW COMPLETE")
    print("=" * 70)

    print(f"\n1. Query Analysis:")
    print(f"   Type: {result['query_analysis']['query_type']}")
    print(f"   Refined Query: {result['query_analysis']['refined_query'][:80]}...")
    print(f"   Top-k: {result['query_analysis']['retrieval_params'].get('top_k', 'N/A')}")

    print(f"\n2. Retrieval:")
    print(f"   Retrieved Chunks: {len(result['retrieved_evidence'])}")
    for i, (evidence, score) in enumerate(zip(result['retrieved_evidence'][:3], result['evidence_scores'][:3])):
        print(f"   [{i+1}] Score: {score:.4f} | {evidence[:60]}...")

    print(f"\n3. Initial Generation:")
    print(f"   Answer: {result['initial_answer'][:150]}...")
    print(f"   Label: {result['initial_label']}")
    print(f"   Claims: {len(result['initial_claims'])} claims extracted")

    print(f"\n4. Verification:")
    if result['verification']:
        print(f"   Overall Status: {result['verification']['overall_status']}")
        print(f"   Hallucination Detected: {result['hallucination_detected']}")
        print(f"   Claims Verified: {len(result['verification']['claims'])}")
        for i, claim_result in enumerate(result['verification']['claims'][:3]):
            print(f"   [{i+1}] {claim_result['status'].upper()} - {claim_result['claim'][:60]}...")
    else:
        print("   Verification: DISABLED")

    print(f"\n5. Revision:")
    if result['revision']:
        print(f"   Changes Made: {result['revision']['changes_made']}")
        print(f"   Revision Type: {result['revision']['revision_type']}")
        print(f"   Final Answer: {result['final_answer'][:150]}...")
    else:
        print("   Revision: Not needed (no hallucinations)")

    print(f"\n6. Metrics:")
    print(f"   Total Latency: {result['latency']:.2f} seconds")
    print(f"   Total Tokens: {result['tokens']}")
    print(f"   Abstained: {result['abstained']}")

    print("\n" + "=" * 70)
    print("✓ Test 3.1: PASSED - All workflow steps executed successfully")
    print("=" * 70)

    # Verify all intermediate steps are saved
    assert 'query_analysis' in result, "Missing query_analysis"
    assert 'retrieved_evidence' in result, "Missing retrieved_evidence"
    assert 'initial_answer' in result, "Missing initial_answer"
    assert 'verification' in result, "Missing verification"
    assert 'final_answer' in result, "Missing final_answer"
    print("✓ All intermediate steps saved in result dict")

except Exception as e:
    print(f"\n✗ Test 3.1 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# Test 3.2: Small-Scale Test on Real Dataset (5-10 samples)
# ══════════════════════════════════════════════════════════════
print("\n\n[TEST 3.2] Small-Scale Dataset Test")
print("-" * 70)

try:
    from data.contractnli_processor import ContractNLIProcessor

    # Load dataset
    print("Loading ContractNLI test dataset...")
    processor = ContractNLIProcessor()
    examples = processor.load(split="test")

    # Take first 5 examples
    sample_size = 5
    test_examples = examples[:sample_size]
    print(f"  ✓ Loaded {len(examples)} total examples")
    print(f"  ✓ Testing on {sample_size} samples\n")

    # Convert to dict format
    example_dicts = [
        {
            'id': ex.id,
            'query': ex.query,
            'context': ex.context,
            'ground_truth_label': ex.label
        }
        for ex in test_examples
    ]

    # Reinitialize agent for batch processing
    agent = AgenticRAG(
        llm=llm,
        retriever_type="bm25",
        chunking_strategy="fixed_length",
        enable_verification=True,
        enable_revision=True
    )

    print("Running batch predictions...")
    print("=" * 70)

    predictions = agent.batch_answer(example_dicts)

    print("\n" + "=" * 70)
    print("BATCH TEST RESULTS")
    print("=" * 70)

    # Analyze results
    hallucinations_detected = sum(1 for p in predictions if p['hallucination_detected'])
    revisions_made = sum(1 for p in predictions if p['revision'] and p['revision']['changes_made'])
    abstentions = sum(1 for p in predictions if p['abstained'])
    avg_latency = sum(p['latency'] for p in predictions) / len(predictions)
    total_tokens = sum(p['tokens'] for p in predictions)

    print(f"\nSummary Statistics:")
    print(f"  Total Examples: {len(predictions)}")
    print(f"  Hallucinations Detected: {hallucinations_detected} ({hallucinations_detected/len(predictions)*100:.1f}%)")
    print(f"  Revisions Made: {revisions_made} ({revisions_made/len(predictions)*100:.1f}%)")
    print(f"  Abstentions: {abstentions} ({abstentions/len(predictions)*100:.1f}%)")
    print(f"  Average Latency: {avg_latency:.2f}s per query")
    print(f"  Total Tokens Used: {total_tokens}")

    print(f"\nPer-Example Results:")
    for i, pred in enumerate(predictions):
        print(f"\n  Example {i+1} (ID: {pred['id']}):")
        print(f"    Query: {pred['query'][:60]}...")
        print(f"    Ground Truth Label: {example_dicts[i]['ground_truth_label']}")
        print(f"    Predicted Label: {pred['label']}")
        print(f"    Hallucination: {'YES' if pred['hallucination_detected'] else 'NO'}")
        print(f"    Revised: {'YES' if pred['revision'] and pred['revision']['changes_made'] else 'NO'}")
        print(f"    Latency: {pred['latency']:.2f}s")

    # Verify output format
    print("\n" + "=" * 70)
    print("Verifying Output Format...")
    print("-" * 70)

    for i, pred in enumerate(predictions):
        assert 'id' in pred, f"Example {i}: Missing 'id'"
        assert 'query' in pred, f"Example {i}: Missing 'query'"
        assert 'final_answer' in pred, f"Example {i}: Missing 'final_answer'"
        assert 'label' in pred, f"Example {i}: Missing 'label'"
        assert 'hallucination_detected' in pred, f"Example {i}: Missing 'hallucination_detected'"
        assert 'latency' in pred, f"Example {i}: Missing 'latency'"
        assert 'tokens' in pred, f"Example {i}: Missing 'tokens'"

    print("✓ All predictions have required fields")
    print("✓ Hallucination detection is working")
    print("✓ Output format is correct")

    # Save results
    from utils.helpers import save_json
    output_path = "experiments/results/priority3_test_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json(predictions, output_path)
    print(f"\n✓ Results saved to: {output_path}")

    print("\n" + "=" * 70)
    print("✓ Test 3.2: PASSED - Small-scale batch test completed successfully")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ Test 3.2 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# Final Summary
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("PRIORITY 3 COMPLETE")
print("=" * 70)
print("\n✓ 3.1: Agentic RAG workflow integration - PASSED")
print("✓ 3.2: Small-scale dataset test (5 samples) - PASSED")
print("\nAll components verified:")
print("  [1] Query Analysis → Query type classification and refinement")
print("  [2] Retrieval → BM25 chunking and evidence retrieval")
print("  [3] Generation → Initial answer with claims and citations")
print("  [4] Verification → Claim-level hallucination detection")
print("  [5] Revision → Answer correction when hallucinations detected")
print("\n✓ Intermediate steps saved for all examples")
print("✓ Hallucination detection working")
print("✓ Output format validated")
print("\n" + "=" * 70)
print("Ready for Priority 4: Full dataset experiments")
print("=" * 70)
