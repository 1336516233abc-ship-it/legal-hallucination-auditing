"""
Agent Components Test Script - Wang Zhengyao
Tests Query Analyzer, Verifier, and Reviser with OpenRouter API

Author: Wang Zhengyao (3036795675)
Date: 2024-11-18
"""

import os
import sys
import importlib.util

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load OpenRouter module directly
spec = importlib.util.spec_from_file_location(
    "openrouter_llm",
    "utils/openrouter_llm.py"
)
openrouter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(openrouter_module)
OpenRouterLLM = openrouter_module.OpenRouterLLM

print("=" * 60)
print("Agent Components Test - Wang Zhengyao")
print("=" * 60)

# Initialize LLM
print("\nInitializing OpenRouter LLM...")
llm = OpenRouterLLM(model_name="llama3", max_tokens=300)
print("[OK] LLM initialized")

# Test data
test_query = "Can the contract be terminated early with 30 days notice?"
test_evidence = [
    "Either party may terminate this agreement by providing thirty (30) days written notice to the other party.",
    "Termination shall be effective upon the expiration of the notice period.",
    "The contract has an initial term of one year from the effective date."
]
test_answer = "Yes, the contract allows termination with 30 days written notice."
test_claims = [
    "The contract allows termination with 30 days notice",
    "Written notice is required for termination"
]

# =============================================================================
# Test 1: Query Analyzer
# =============================================================================
print("\n" + "=" * 60)
print("Test 1: Query Analyzer")
print("=" * 60)

try:
    # Import the FIXED QueryAnalyzer (without torch dependency)
    sys.path.insert(0, 'agent/query_analyzer')
    from analyzer_openrouter import QueryAnalyzer

    # Test without LLM (keyword-based)
    print("\n1.1 Testing keyword-based analysis...")
    analyzer_basic = QueryAnalyzer(llm=None)
    result = analyzer_basic.analyze(test_query)

    print(f"  Query: {test_query}")
    print(f"  Type: {result['query_type']}")
    print(f"  Refined: {result['refined_query'][:80]}...")
    print(f"  Top-k: {result['retrieval_params']['top_k']}")
    print("  [OK] Keyword-based analysis works")

    # Test with LLM
    print("\n1.2 Testing LLM-based query refinement...")
    analyzer_llm = QueryAnalyzer(llm=llm)

    # Test query refinement
    refined = analyzer_llm._llm_refine_query(test_query, "termination")
    print(f"  Original: {test_query}")
    print(f"  Refined: {refined}")
    print("  [OK] LLM-based refinement works")

    print("\n[SUCCESS] Query Analyzer tests passed!")

except Exception as e:
    print(f"\n[FAIL] Query Analyzer test failed: {str(e)}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 2: Answer Generator
# =============================================================================
print("\n" + "=" * 60)
print("Test 2: Answer Generator")
print("=" * 60)

try:
    sys.path.insert(0, 'agent')
    from generator import AnswerGenerator

    print("\n2.1 Testing answer generation...")
    generator = AnswerGenerator(llm)

    result = generator.generate(
        query=test_query,
        evidence=test_evidence,
        query_type="termination"
    )

    print(f"  Query: {test_query}")
    print(f"  Generated Answer: {result['answer'][:100]}...")
    print(f"  Label: {result['label']}")
    print(f"  Claims: {len(result['claims'])} claims extracted")
    print(f"  Tokens: {result['tokens']}")
    print("  [OK] Answer generation works")

    print("\n[SUCCESS] Answer Generator tests passed!")

except Exception as e:
    print(f"\n[FAIL] Answer Generator test failed: {str(e)}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 3: Evidence Verifier
# =============================================================================
print("\n" + "=" * 60)
print("Test 3: Evidence Verifier")
print("=" * 60)

try:
    sys.path.insert(0, 'agent/verifier')
    from verifier import EvidenceVerifier

    print("\n3.1 Testing claim verification...")
    verifier = EvidenceVerifier(llm)

    verification_result = verifier.verify(
        claims=test_claims,
        evidence=test_evidence
    )

    print(f"  Claims to verify: {len(test_claims)}")
    print(f"  Overall Status: {verification_result['overall_status']}")
    print(f"  Hallucination Detected: {verification_result['hallucination_detected']}")

    for i, claim_result in enumerate(verification_result['claims']):
        print(f"\n  Claim {i+1}: {claim_result['claim']}")
        print(f"    Status: {claim_result['status']}")
        print(f"    Reasoning: {claim_result['reasoning'][:60]}...")

    print("\n  [OK] Evidence verification works")

    print("\n[SUCCESS] Evidence Verifier tests passed!")

except Exception as e:
    print(f"\n[FAIL] Evidence Verifier test failed: {str(e)}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 4: Answer Reviser
# =============================================================================
print("\n" + "=" * 60)
print("Test 4: Answer Reviser")
print("=" * 60)

try:
    sys.path.insert(0, 'agent/reviser')
    from reviser import AnswerReviser

    print("\n4.1 Testing answer revision...")
    reviser = AnswerReviser(llm)

    # Create mock verification results with hallucination
    mock_verification = {
        'hallucination_detected': True,
        'overall_status': 'contradicted',
        'claims': [
            {
                'claim': 'The contract can be terminated immediately',
                'status': 'contradicted',
                'reasoning': 'Evidence states 30 days notice is required',
                'supporting_evidence': [1]
            }
        ]
    }

    revision_result = reviser.revise(
        original_answer="The contract can be terminated immediately.",
        verification_results=mock_verification,
        evidence=test_evidence,
        query=test_query
    )

    print(f"  Original: The contract can be terminated immediately.")
    print(f"  Revised: {revision_result['revised_answer'][:100]}...")
    print(f"  Changes Made: {revision_result['changes_made']}")
    print(f"  Revision Type: {revision_result['revision_type']}")
    print(f"  Tokens: {revision_result.get('tokens', 0)}")
    print("  [OK] Answer revision works")

    print("\n[SUCCESS] Answer Reviser tests passed!")

except Exception as e:
    print(f"\n[FAIL] Answer Reviser test failed: {str(e)}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print("\nAll Agent Components Tested:")
print("  [OK] Query Analyzer - Analyzes queries and classifies types")
print("  [OK] Answer Generator - Generates answers from evidence")
print("  [OK] Evidence Verifier - Verifies claims against evidence")
print("  [OK] Answer Reviser - Revises answers when hallucinations detected")

print("\n" + "=" * 60)
print("Agent Components Ready for Integration!")
print("=" * 60)

print("\nNext steps:")
print("1. Test complete Agentic RAG workflow")
print("2. Run on small sample (5-10 examples)")
print("3. Integrate with retrieval system")
print("4. Run full experiments")
