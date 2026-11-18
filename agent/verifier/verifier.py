"""
Evidence Verifier module.
Verifies claims against evidence to detect hallucinations.

Author: Yin Fengling (3036807375)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List
from utils.llm_interface import LLMInterface
import logging

logger = logging.getLogger(__name__)


class EvidenceVerifier:
    """Verifies claims against evidence to detect hallucinations."""

    def __init__(self, llm: LLMInterface):
        """
        Initialize evidence verifier.

        Args:
            llm: LLM interface for verification
        """
        self.llm = llm

    def verify(
        self,
        claims: List[str],
        evidence: List[str],
        answer: str = None
    ) -> Dict:
        """
        Verify claims against evidence.

        Args:
            claims: List of claims to verify
            evidence: List of evidence strings
            answer: Optional full answer text

        Returns:
            Dict with verification results
        """
        # If answer provided but no claims, extract claims
        if answer and not claims:
            claims = self._extract_claims(answer)

        # Verify each claim
        verification_results = []

        for i, claim in enumerate(claims):
            result = self._verify_claim(claim, evidence)
            result['claim_id'] = i
            verification_results.append(result)

        # Overall verdict
        has_contradictions = any(r['status'] == 'contradicted' for r in verification_results)
        has_unsupported = any(r['status'] == 'not_enough_info' for r in verification_results)

        if has_contradictions:
            overall_status = "contradicted"
        elif has_unsupported:
            overall_status = "partially_supported"
        else:
            overall_status = "supported"

        return {
            'overall_status': overall_status,
            'claims': verification_results,
            'hallucination_detected': has_contradictions or has_unsupported,
        }

    def _verify_claim(self, claim: str, evidence: List[str]) -> Dict:
        """
        Verify a single claim against evidence.

        Args:
            claim: Claim to verify
            evidence: List of evidence strings

        Returns:
            Dict with claim, status, supporting_evidence
        """
        # Format evidence
        evidence_text = "\n\n".join([
            f"[{i+1}] {ev}"
            for i, ev in enumerate(evidence)
        ])

        # Create verification prompt
        prompt = f"""You are a fact-checking assistant. Verify if the following claim is supported by the evidence.

Evidence:
{evidence_text}

Claim: {claim}

Determine if the claim is:
- SUPPORTED: The evidence clearly supports this claim
- CONTRADICTED: The evidence contradicts this claim
- NOT_ENOUGH_INFO: The evidence does not provide enough information

Format your response as:
Status: [SUPPORTED/CONTRADICTED/NOT_ENOUGH_INFO]
Reasoning: [Brief explanation]
Evidence: [List evidence numbers that are relevant]
"""

        result = self.llm.generate(prompt, max_tokens=200, temperature=0.3)

        # Parse verification result
        parsed = self._parse_verification(result['text'])

        return {
            'claim': claim,
            'status': parsed['status'],
            'reasoning': parsed['reasoning'],
            'supporting_evidence': parsed['evidence_ids'],
        }

    def _parse_verification(self, response: str) -> Dict:
        """Parse verification response."""
        lines = response.strip().split('\n')

        status = "not_enough_info"  # Default
        reasoning = ""
        evidence_ids = []

        for line in lines:
            line = line.strip()

            if line.lower().startswith("status:"):
                status_text = line.split(":", 1)[1].strip().lower()
                if "supported" in status_text and "not" not in status_text:
                    status = "supported"
                elif "contradicted" in status_text:
                    status = "contradicted"
                else:
                    status = "not_enough_info"

            elif line.lower().startswith("reasoning:"):
                reasoning = line.split(":", 1)[1].strip()

            elif line.lower().startswith("evidence:"):
                evidence_text = line.split(":", 1)[1].strip()
                # Extract numbers
                import re
                evidence_ids = [int(m) for m in re.findall(r'\d+', evidence_text)]

        return {
            'status': status,
            'reasoning': reasoning,
            'evidence_ids': evidence_ids,
        }

    def _extract_claims(self, answer: str) -> List[str]:
        """Extract individual claims from answer text."""
        # Simple sentence-based extraction
        import re

        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', answer)

        # Filter out very short sentences
        claims = [s.strip() for s in sentences if len(s.split()) > 3]

        return claims if claims else [answer]

    def verify_batch(
        self,
        claims_list: List[List[str]],
        evidence_list: List[List[str]]
    ) -> List[Dict]:
        """
        Verify multiple sets of claims in batch.

        Args:
            claims_list: List of claim lists
            evidence_list: List of evidence lists

        Returns:
            List of verification result dicts
        """
        results = []

        for claims, evidence in zip(claims_list, evidence_list):
            result = self.verify(claims, evidence)
            results.append(result)

        return results

    def get_verification_confidence(self, verification_result: Dict) -> float:
        """
        Compute confidence score for verification.

        Args:
            verification_result: Verification result dict

        Returns:
            Confidence score between 0 and 1
        """
        # Simple heuristic: based on number of supporting evidence
        claims = verification_result.get('claims', [])

        if not claims:
            return 0.0

        supported_claims = sum(
            1 for c in claims
            if c['status'] == 'supported' and c['supporting_evidence']
        )

        return supported_claims / len(claims)


def main():
    """Test evidence verifier."""
    # Mock LLM for testing
    class MockLLM:
        def generate(self, prompt, **kwargs):
            return {
                'text': "Status: SUPPORTED\nReasoning: The evidence clearly states this.\nEvidence: [1, 2]",
                'tokens': 20
            }

    llm = MockLLM()
    verifier = EvidenceVerifier(llm)

    # Test verification
    claims = [
        "The contract can be terminated with 30 days notice.",
        "The payment is due on the 15th of each month."
    ]

    evidence = [
        "Either party may terminate this agreement by providing 30 days written notice.",
        "Payment shall be made on the 15th day of each calendar month."
    ]

    result = verifier.verify(claims, evidence)

    print("Verification Result:")
    print(f"Overall Status: {result['overall_status']}")
    print(f"Hallucination Detected: {result['hallucination_detected']}")
    print("\nClaims:")
    for claim_result in result['claims']:
        print(f"  - {claim_result['claim']}")
        print(f"    Status: {claim_result['status']}")


if __name__ == "__main__":
    main()
