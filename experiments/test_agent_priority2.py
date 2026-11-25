"""
Priority 2 Test: Generator, Verifier, Reviser
Wang Zhengyao - Day 2 (Nov 19)
Tests agent components with MockLLM (no API key needed)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Prevent agent/__init__.py from triggering retrieval chain imports
import unittest.mock as _mock
sys.modules.setdefault('agent.agentic_rag', _mock.MagicMock())
sys.modules.setdefault('retrieval', _mock.MagicMock())
sys.modules.setdefault('retrieval.chunking', _mock.MagicMock())
sys.modules.setdefault('retrieval.retrievers', _mock.MagicMock())
sys.modules.setdefault('retrieval.retrievers.retriever', _mock.MagicMock())
sys.modules.setdefault('retrieval.baseline_rag', _mock.MagicMock())
sys.modules.setdefault('sentence_transformers', _mock.MagicMock())
sys.modules.setdefault('rank_bm25', _mock.MagicMock())

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=" * 60)
print("Priority 2: Testing Generator, Verifier, Reviser")
print("=" * 60)

# ─── Mock LLM ────────────────────────────────────────────────
class MockLLM:
    """Mock LLM that returns realistic responses for testing."""

    def generate(self, prompt, max_tokens=512, temperature=0.7, **kwargs):
        # Verifier-style prompt
        if "Verify if the following claim" in prompt or "fact-checking" in prompt.lower():
            if "terminated" in prompt.lower() or "terminate" in prompt.lower():
                return {
                    'text': "Status: SUPPORTED\nReasoning: Evidence [1] clearly states termination terms.\nEvidence: [1]",
                    'tokens': 25
                }
            elif "payment" in prompt.lower():
                return {
                    'text': "Status: SUPPORTED\nReasoning: Evidence [2] confirms payment schedule.\nEvidence: [2]",
                    'tokens': 25
                }
            else:
                return {
                    'text': "Status: NOT_ENOUGH_INFO\nReasoning: The evidence does not address this claim.\nEvidence: []",
                    'tokens': 20
                }
        # Generator-style prompt
        elif "Answer the following question" in prompt or "legal contract analysis" in prompt.lower():
            return {
                'text': (
                    "Answer: Based on the evidence, the contract can be terminated with 30 days written notice [1]. "
                    "Payment is due on the 15th of each month [2].\n\n"
                    "Claims:\n"
                    "- The contract can be terminated with 30 days written notice [1]\n"
                    "- Payment is due on the 15th of each month [2]\n\n"
                    "Label: Entailment"
                ),
                'tokens': 60
            }
        # Reviser-style prompt
        elif "Revise" in prompt or "revision" in prompt.lower():
            return {
                'text': "The contract specifies a 30-day termination notice period [1]. "
                        "The contract does not provide clear information on the payment amount.",
                'tokens': 40
            }
        # Chain-of-thought
        elif "step-by-step" in prompt.lower() or "Think through" in prompt:
            return {
                'text': (
                    "Reasoning: First, I look for termination clauses in evidence [1]. "
                    "It clearly states 30 days notice is required.\n\n"
                    "Answer: The contract requires 30 days written notice for termination [1].\n\n"
                    "Label: Entailment"
                ),
                'tokens': 50
            }
        else:
            return {'text': "Generic mock response.", 'tokens': 10}


llm = MockLLM()

# ─── Sample data ──────────────────────────────────────────────
query = "Can this contract be terminated early?"
evidence = [
    "Either party may terminate this agreement by providing 30 days written notice to the other party.",
    "Payment shall be made on the 15th day of each calendar month.",
    "This agreement shall commence on January 1, 2024 and continue for a period of one year."
]

# ══════════════════════════════════════════════════════════════
# TEST 1: AnswerGenerator
# ══════════════════════════════════════════════════════════════
print("\n[TEST 1] AnswerGenerator")
print("-" * 40)

try:
    from agent.generator import AnswerGenerator

    generator = AnswerGenerator(llm)

    # 1a. Basic generation
    result = generator.generate(query, evidence, query_type="termination")
    assert isinstance(result, dict), "Result should be a dict"
    assert 'answer' in result, "Missing 'answer' key"
    assert 'label' in result, "Missing 'label' key"
    assert 'claims' in result, "Missing 'claims' key"
    assert 'citations' in result, "Missing 'citations' key"
    assert len(result['answer']) > 0, "Answer should not be empty"
    assert result['label'] in ('entailment', 'contradiction', 'neutral'), f"Invalid label: {result['label']}"
    print(f"  ✓ generate() - answer: {result['answer'][:60]}...")
    print(f"  ✓ label: {result['label']}")
    print(f"  ✓ claims count: {len(result['claims'])}")
    print(f"  ✓ citations: {result['citations']}")

    # 1b. Chain-of-thought generation
    cot_result = generator.generate_with_chain_of_thought(query, evidence)
    assert 'answer' in cot_result, "CoT missing 'answer'"
    assert 'reasoning' in cot_result, "CoT missing 'reasoning'"
    assert 'label' in cot_result, "CoT missing 'label'"
    print(f"  ✓ generate_with_chain_of_thought() - reasoning: {cot_result['reasoning'][:50]}...")

    print("  → AnswerGenerator: ALL PASSED ✓")

except Exception as e:
    print(f"  ✗ AnswerGenerator FAILED: {e}")
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════════
# TEST 2: EvidenceVerifier
# ══════════════════════════════════════════════════════════════
print("\n[TEST 2] EvidenceVerifier")
print("-" * 40)

try:
    from agent.verifier.verifier import EvidenceVerifier

    verifier = EvidenceVerifier(llm)

    # 2a. Verify claims
    claims = [
        "The contract can be terminated with 30 days written notice.",
        "Payment is due on the 15th of each month."
    ]
    result = verifier.verify(claims, evidence)
    assert isinstance(result, dict), "Result should be dict"
    assert 'overall_status' in result, "Missing 'overall_status'"
    assert 'claims' in result, "Missing 'claims'"
    assert 'hallucination_detected' in result, "Missing 'hallucination_detected'"
    assert len(result['claims']) == 2, f"Expected 2 claim results, got {len(result['claims'])}"
    for cr in result['claims']:
        assert cr['status'] in ('supported', 'contradicted', 'not_enough_info'), f"Invalid status: {cr['status']}"
    print(f"  ✓ verify(claims) - overall: {result['overall_status']}")
    print(f"  ✓ hallucination_detected: {result['hallucination_detected']}")

    # 2b. Verify from answer text (auto-extract claims)
    answer_text = "The contract can be terminated with 30 days notice. Payment is due on the 15th."
    result2 = verifier.verify([], evidence, answer=answer_text)
    assert 'overall_status' in result2
    print(f"  ✓ verify(answer=text) - overall: {result2['overall_status']}")

    # 2c. Confidence score
    conf = verifier.get_verification_confidence(result)
    assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"
    print(f"  ✓ get_verification_confidence(): {conf:.2f}")

    # 2d. Batch verify
    batch_result = verifier.verify_batch([claims], [evidence])
    assert len(batch_result) == 1
    print(f"  ✓ verify_batch(): {len(batch_result)} result(s)")

    print("  → EvidenceVerifier: ALL PASSED ✓")

except Exception as e:
    print(f"  ✗ EvidenceVerifier FAILED: {e}")
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════════
# TEST 3: AnswerReviser
# ══════════════════════════════════════════════════════════════
print("\n[TEST 3] AnswerReviser")
print("-" * 40)

try:
    from agent.reviser.reviser import AnswerReviser
    from agent.verifier.verifier import EvidenceVerifier

    reviser = AnswerReviser(llm)
    verifier = EvidenceVerifier(llm)

    original_answer = "The contract can be terminated with 30 days notice. The penalty fee is $500."
    claims = [
        "The contract can be terminated with 30 days notice.",
        "The penalty fee is $500."  # not in evidence → not_enough_info
    ]

    # Build a fake verification result with hallucination
    fake_verification = {
        'overall_status': 'partially_supported',
        'hallucination_detected': True,
        'claims': [
            {'claim_id': 0, 'claim': claims[0], 'status': 'supported',
             'reasoning': 'Evidence supports this.', 'supporting_evidence': [1]},
            {'claim_id': 1, 'claim': claims[1], 'status': 'not_enough_info',
             'reasoning': 'No evidence for penalty fee.', 'supporting_evidence': []},
        ]
    }

    # 3a. Revise with hallucination detected
    result = reviser.revise(original_answer, fake_verification, evidence, query)
    assert isinstance(result, dict), "Result should be dict"
    assert 'revised_answer' in result, "Missing 'revised_answer'"
    assert 'changes_made' in result, "Missing 'changes_made'"
    assert 'revision_type' in result, "Missing 'revision_type'"
    assert result['changes_made'] == True, "Should have made changes"
    print(f"  ✓ revise() with hallucination - type: {result['revision_type']}")
    print(f"  ✓ revised: {result['revised_answer'][:60]}...")

    # 3b. Revise with no hallucination (should return original)
    clean_verification = {
        'overall_status': 'supported',
        'hallucination_detected': False,
        'claims': [
            {'claim_id': 0, 'claim': claims[0], 'status': 'supported',
             'reasoning': 'Supported.', 'supporting_evidence': [1]},
        ]
    }
    result2 = reviser.revise(original_answer, clean_verification, evidence, query)
    assert result2['changes_made'] == False, "No changes expected when no hallucination"
    assert result2['revision_type'] == 'none'
    print(f"  ✓ revise() without hallucination - no changes made ✓")

    # 3c. Iterative revision
    iter_result = reviser.iterative_revise(
        original_answer, evidence, query, max_iterations=2
    )
    assert 'final_answer' in iter_result, "Missing 'final_answer'"
    assert 'iterations' in iter_result, "Missing 'iterations'"
    assert 'converged' in iter_result, "Missing 'converged'"
    print(f"  ✓ iterative_revise() - iterations: {iter_result['iterations']}, converged: {iter_result['converged']}")

    print("  → AnswerReviser: ALL PASSED ✓")

except Exception as e:
    print(f"  ✗ AnswerReviser FAILED: {e}")
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════════
# TEST 4: Full pipeline (Generator → Verifier → Reviser)
# ══════════════════════════════════════════════════════════════
print("\n[TEST 4] Full Pipeline: Generator → Verifier → Reviser")
print("-" * 40)

try:
    from agent.generator import AnswerGenerator
    from agent.verifier.verifier import EvidenceVerifier
    from agent.reviser.reviser import AnswerReviser

    gen = AnswerGenerator(llm)
    ver = EvidenceVerifier(llm)
    rev = AnswerReviser(llm)

    # Step 1: Generate
    gen_result = gen.generate(query, evidence)
    print(f"  Step 1 Generate - label: {gen_result['label']}, claims: {len(gen_result['claims'])}")

    # Step 2: Verify
    ver_result = ver.verify(gen_result['claims'], evidence)
    print(f"  Step 2 Verify - status: {ver_result['overall_status']}, hallucination: {ver_result['hallucination_detected']}")

    # Step 3: Revise
    rev_result = rev.revise(gen_result['answer'], ver_result, evidence, query)
    print(f"  Step 3 Revise - changes_made: {rev_result['changes_made']}, type: {rev_result['revision_type']}")

    print("  → Full pipeline: PASSED ✓")

except Exception as e:
    print(f"  ✗ Full pipeline FAILED: {e}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 60)
print("Priority 2 Tests Complete")
print("=" * 60)
