"""
Hallucination detection and error pattern analysis.
Identifies and categorizes hallucination failure modes.

Author: Li Xinru (3036807959)
"""

from typing import List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HallucinationCase:
    """A single hallucination instance."""
    example_id: str
    claim: str
    verification_status: str  # "contradicted" or "not_enough_info"
    evidence_used: List[str]
    failure_mode: str
    metadata: Dict


class HallucinationDetector:
    """Detect and analyze hallucinations in model outputs."""

    FAILURE_MODES = [
        "fabrication",          # Completely made up information
        "misinterpretation",   # Misread or misunderstood the evidence
        "overgeneralization",  # Generalized beyond what evidence supports
        "contradictory_evidence",  # Evidence contradicts the claim
        "insufficient_evidence",   # Not enough evidence to support
        "temporal_confusion",      # Confused dates or timelines
        "entity_confusion",        # Mixed up entities or parties
        "negation_error",          # Missed or mishandled negation
        "scope_error",             # Wrong scope (e.g., "all" vs "some")
        "inference_leap",          # Unjustified inference beyond evidence
    ]

    def __init__(self):
        self.hallucination_cases = []

    def detect_hallucinations(
        self,
        predictions: List[Dict],
        verification_results: List[Dict],
        ground_truth: List[Dict]
    ) -> List[HallucinationCase]:
        """
        Detect hallucinations from verification results.

        Args:
            predictions: Model predictions
            verification_results: Verification results for each claim
            ground_truth: Ground truth data

        Returns:
            List of HallucinationCase objects
        """
        cases = []

        for pred, verif, gt in zip(predictions, verification_results, ground_truth):
            example_id = gt.get("id", "unknown")

            # Check each claim in verification results
            for claim_verif in verif.get("claims", []):
                status = claim_verif.get("status", "supported")

                if status in ["contradicted", "not_enough_info"]:
                    # Identify failure mode
                    failure_mode = self._identify_failure_mode(
                        claim_verif, pred, gt
                    )

                    case = HallucinationCase(
                        example_id=example_id,
                        claim=claim_verif.get("claim", ""),
                        verification_status=status,
                        evidence_used=claim_verif.get("evidence", []),
                        failure_mode=failure_mode,
                        metadata={
                            "query": gt.get("query", ""),
                            "ground_truth_label": gt.get("label", ""),
                            "predicted_label": pred.get("label", ""),
                        }
                    )
                    cases.append(case)

        self.hallucination_cases = cases
        return cases

    def _identify_failure_mode(
        self,
        claim_verification: Dict,
        prediction: Dict,
        ground_truth: Dict
    ) -> str:
        """
        Identify the type of hallucination failure mode.

        This is a heuristic-based classification that can be improved
        with more sophisticated NLP analysis.

        Args:
            claim_verification: Verification result for the claim
            prediction: Full prediction
            ground_truth: Ground truth

        Returns:
            Failure mode string
        """
        claim = claim_verification.get("claim", "").lower()
        status = claim_verification.get("status", "")
        evidence = claim_verification.get("evidence", [])

        # Heuristic rules for failure mode classification
        if status == "contradicted":
            # Check for negation errors
            if "not" in claim or "no" in claim:
                return "negation_error"
            # Check for entity confusion
            elif any(word in claim for word in ["party", "parties", "entity"]):
                return "entity_confusion"
            else:
                return "contradictory_evidence"

        elif status == "not_enough_info":
            # Check if claim makes specific assertions without evidence
            if any(word in claim for word in ["must", "shall", "always", "never"]):
                return "overgeneralization"
            # Check for date/time references
            elif any(word in claim for word in ["date", "year", "month", "day", "when"]):
                return "temporal_confusion"
            # Check if making inferences
            elif any(word in claim for word in ["therefore", "thus", "implies", "means"]):
                return "inference_leap"
            # Check if no evidence at all was found
            elif not evidence:
                return "fabrication"
            else:
                return "insufficient_evidence"

        return "misinterpretation"  # Default

    def analyze_failure_patterns(self) -> Dict:
        """
        Analyze patterns in hallucination failures.

        Returns:
            Dictionary with failure mode statistics and examples
        """
        if not self.hallucination_cases:
            logger.warning("No hallucination cases to analyze")
            return {}

        # Count failure modes
        failure_counts = defaultdict(int)
        failure_examples = defaultdict(list)

        for case in self.hallucination_cases:
            failure_counts[case.failure_mode] += 1
            if len(failure_examples[case.failure_mode]) < 3:  # Keep top 3 examples
                failure_examples[case.failure_mode].append({
                    "claim": case.claim,
                    "query": case.metadata.get("query", ""),
                    "status": case.verification_status,
                })

        # Calculate percentages
        total = len(self.hallucination_cases)
        failure_percentages = {
            mode: count / total for mode, count in failure_counts.items()
        }

        # Sort by frequency
        sorted_failures = sorted(
            failure_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        analysis = {
            "total_hallucinations": total,
            "failure_mode_counts": dict(failure_counts),
            "failure_mode_percentages": failure_percentages,
            "sorted_failures": sorted_failures,
            "examples_by_mode": dict(failure_examples),
        }

        return analysis

    def get_statistics_by_query_type(self, query_types: Dict[str, str]) -> Dict:
        """
        Get hallucination statistics broken down by query type.

        Args:
            query_types: Mapping of example_id to query type

        Returns:
            Statistics by query type
        """
        stats_by_type = defaultdict(lambda: defaultdict(int))

        for case in self.hallucination_cases:
            query_type = query_types.get(case.example_id, "unknown")
            stats_by_type[query_type]["total"] += 1
            stats_by_type[query_type][case.failure_mode] += 1

        return dict(stats_by_type)

    def format_analysis_report(self, analysis: Dict) -> str:
        """Format analysis results as a readable report."""
        lines = [
            "=" * 70,
            "HALLUCINATION FAILURE MODE ANALYSIS",
            "=" * 70,
            "",
            f"Total Hallucinations: {analysis['total_hallucinations']}",
            "",
            "Failure Modes (sorted by frequency):",
            ""
        ]

        for mode, count in analysis["sorted_failures"]:
            percentage = analysis["failure_mode_percentages"][mode] * 100
            lines.append(f"  {mode:30s}: {count:4d} ({percentage:5.1f}%)")

        lines.append("")
        lines.append("Examples by Failure Mode:")
        lines.append("")

        for mode, examples in analysis["examples_by_mode"].items():
            lines.append(f"  {mode}:")
            for i, ex in enumerate(examples, 1):
                lines.append(f"    {i}. Query: {ex['query'][:60]}...")
                lines.append(f"       Claim: {ex['claim'][:60]}...")
                lines.append(f"       Status: {ex['status']}")
                lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


def categorize_by_contract_clause_type(
    hallucination_cases: List[HallucinationCase],
    metadata_key: str = "clause_type"
) -> Dict[str, List[HallucinationCase]]:
    """
    Categorize hallucinations by contract clause type.

    Args:
        hallucination_cases: List of hallucination cases
        metadata_key: Key in metadata that contains clause type

    Returns:
        Dictionary mapping clause types to hallucination cases
    """
    by_clause_type = defaultdict(list)

    for case in hallucination_cases:
        clause_type = case.metadata.get(metadata_key, "unknown")
        by_clause_type[clause_type].append(case)

    return dict(by_clause_type)
