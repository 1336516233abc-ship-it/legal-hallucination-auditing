"""
Answer Reviser module.
Revises answers when hallucinations are detected.

Author: Yin Fengling (3036807375)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional
from utils.llm_interface import LLMInterface
import logging

logger = logging.getLogger(__name__)


class AnswerReviser:
    """Revises answers to fix hallucinations."""

    def __init__(self, llm: LLMInterface, retriever=None):
        """
        Initialize answer reviser.

        Args:
            llm: LLM interface for revision
            retriever: Optional retriever for re-retrieval
        """
        self.llm = llm
        self.retriever = retriever

    def revise(
        self,
        original_answer: str,
        verification_results: Dict,
        evidence: List[str],
        query: str = None
    ) -> Dict:
        """
        Revise answer based on verification results.

        Args:
            original_answer: Original answer to revise
            verification_results: Verification results from EvidenceVerifier
            evidence: Original evidence used
            query: Optional original query

        Returns:
            Dict with revised_answer, changes_made, confidence
        """
        if not verification_results.get('hallucination_detected', False):
            # No hallucination, return original
            return {
                'revised_answer': original_answer,
                'changes_made': False,
                'revision_type': 'none',
                'confidence': 1.0,
            }

        # Get problematic claims
        problematic_claims = [
            c for c in verification_results['claims']
            if c['status'] in ['contradicted', 'not_enough_info']
        ]

        # Decide revision strategy
        if any(c['status'] == 'contradicted' for c in problematic_claims):
            # Contradicted claims: need correction
            return self._revise_with_correction(
                original_answer,
                problematic_claims,
                evidence,
                query
            )
        else:
            # Not enough info: need additional retrieval or abstention
            return self._revise_with_abstention(
                original_answer,
                problematic_claims,
                evidence,
                query
            )

    def _revise_with_correction(
        self,
        original_answer: str,
        problematic_claims: List[Dict],
        evidence: List[str],
        query: Optional[str]
    ) -> Dict:
        """Revise answer by correcting contradicted claims."""
        # Format evidence
        evidence_text = "\n\n".join([
            f"[{i+1}] {ev}"
            for i, ev in enumerate(evidence)
        ])

        # Format problematic claims
        claims_text = "\n".join([
            f"- {c['claim']} (Status: {c['status']}, Reasoning: {c['reasoning']})"
            for c in problematic_claims
        ])

        prompt = f"""You are a legal contract analysis assistant. The following answer contains claims that are contradicted by the evidence. Revise the answer to be factually accurate.

Evidence:
{evidence_text}

Original Answer:
{original_answer}

Problematic Claims:
{claims_text}

Instructions:
1. Correct or remove claims that are contradicted by the evidence
2. Ensure all statements are supported by the evidence
3. Maintain the structure and clarity of the answer
4. If information is truly absent, state that explicitly

Revised Answer:
"""

        result = self.llm.generate(prompt, max_tokens=400)

        return {
            'revised_answer': result['text'].strip(),
            'changes_made': True,
            'revision_type': 'correction',
            'problematic_claims': [c['claim'] for c in problematic_claims],
            'confidence': 0.7,
            'tokens': result.get('tokens', 0),
        }

    def _revise_with_abstention(
        self,
        original_answer: str,
        problematic_claims: List[Dict],
        evidence: List[str],
        query: Optional[str]
    ) -> Dict:
        """Revise answer by noting insufficient evidence."""
        # If we have a retriever, try re-retrieval first
        if self.retriever and query:
            return self._revise_with_reretrieval(
                original_answer,
                problematic_claims,
                query
            )

        # Otherwise, revise to include abstention
        claims_text = "\n".join([c['claim'] for c in problematic_claims])

        prompt = f"""You are a legal contract analysis assistant. The following answer contains claims that lack sufficient evidence. Revise the answer to be more cautious.

Original Answer:
{original_answer}

Claims lacking evidence:
{claims_text}

Instructions:
1. For claims lacking evidence, either remove them or explicitly state the evidence is insufficient
2. Keep claims that are well-supported
3. Use phrases like "The contract does not provide clear information on..." where appropriate

Revised Answer:
"""

        result = self.llm.generate(prompt, max_tokens=400)

        return {
            'revised_answer': result['text'].strip(),
            'changes_made': True,
            'revision_type': 'abstention',
            'unsupported_claims': [c['claim'] for c in problematic_claims],
            'confidence': 0.5,
            'tokens': result.get('tokens', 0),
        }

    def _revise_with_reretrieval(
        self,
        original_answer: str,
        problematic_claims: List[Dict],
        query: str
    ) -> Dict:
        """Revise by retrieving additional evidence."""
        logger.info("Attempting re-retrieval for unsupported claims...")

        # Generate refined query focusing on problematic claims
        refined_query = self._refine_query_for_reretrieval(query, problematic_claims)

        # Retrieve additional evidence
        additional_evidence = self.retriever.retrieve(refined_query, top_k=3)
        new_evidence = [doc['text'] for doc, score in additional_evidence]

        # Format all evidence
        evidence_text = "\n\n".join([
            f"[{i+1}] {ev}"
            for i, ev in enumerate(new_evidence)
        ])

        # Create revision prompt
        prompt = f"""You are a legal contract analysis assistant. The following answer had claims lacking evidence. Additional evidence has been retrieved. Revise the answer using this new evidence.

Original Answer:
{original_answer}

New Evidence:
{evidence_text}

Instructions:
1. Use the new evidence to support or refute the previously unsupported claims
2. If still insufficient, state that explicitly
3. Cite evidence using [number] notation

Revised Answer:
"""

        result = self.llm.generate(prompt, max_tokens=400)

        return {
            'revised_answer': result['text'].strip(),
            'changes_made': True,
            'revision_type': 'reretrieval',
            'additional_evidence': new_evidence,
            'confidence': 0.8,
            'tokens': result.get('tokens', 0),
        }

    def _refine_query_for_reretrieval(
        self,
        original_query: str,
        problematic_claims: List[Dict]
    ) -> str:
        """Create refined query focusing on problematic claims."""
        # Extract key terms from problematic claims
        claims_text = " ".join([c['claim'] for c in problematic_claims])

        # Combine with original query
        refined = f"{original_query} {claims_text}"

        return refined

    def iterative_revise(
        self,
        answer: str,
        evidence: List[str],
        query: str,
        max_iterations: int = 2
    ) -> Dict:
        """
        Perform iterative revision until no hallucinations remain.

        Args:
            answer: Initial answer
            evidence: Evidence
            query: Query
            max_iterations: Maximum revision iterations

        Returns:
            Final revision result
        """
        from agent.verifier.verifier import EvidenceVerifier

        verifier = EvidenceVerifier(self.llm)

        current_answer = answer
        iteration = 0
        revision_history = []

        while iteration < max_iterations:
            # Verify current answer
            verification = verifier.verify([], evidence, answer=current_answer)

            # If no hallucinations, done
            if not verification['hallucination_detected']:
                logger.info(f"Revision converged after {iteration} iterations")
                break

            # Revise
            revision_result = self.revise(
                current_answer,
                verification,
                evidence,
                query
            )

            revision_history.append({
                'iteration': iteration,
                'verification': verification,
                'revision': revision_result,
            })

            current_answer = revision_result['revised_answer']
            iteration += 1

        return {
            'final_answer': current_answer,
            'iterations': iteration,
            'revision_history': revision_history,
            'converged': iteration < max_iterations,
        }
