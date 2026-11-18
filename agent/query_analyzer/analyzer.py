"""
Query Analyzer module.
Analyzes and classifies user queries for better retrieval.

Author: Yin Fengling (3036807375)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional
from utils.llm_interface import LLMInterface
import logging

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyzes queries to understand intent and guide retrieval."""

    QUERY_TYPES = [
        "rights_obligations",   # Rights and obligations of parties
        "termination",          # Termination conditions
        "duration_timeline",    # Dates, durations, timelines
        "liability",            # Liability and damages
        "exceptions",           # Exceptions and special cases
        "general",              # General queries
    ]

    def __init__(self, llm: Optional[LLMInterface] = None):
        """
        Initialize query analyzer.

        Args:
            llm: Optional LLM for query expansion/refinement
        """
        self.llm = llm

    def analyze(self, query: str) -> Dict:
        """
        Analyze query to understand intent.

        Args:
            query: User query

        Returns:
            Dict with query_type, refined_query, retrieval_params
        """
        # Classify query type
        query_type = self._classify_query(query)

        # Refine query for better retrieval
        refined_query = self._refine_query(query, query_type)

        # Determine retrieval parameters
        retrieval_params = self._get_retrieval_params(query_type)

        result = {
            'original_query': query,
            'query_type': query_type,
            'refined_query': refined_query,
            'retrieval_params': retrieval_params,
        }

        logger.info(f"Analyzed query - Type: {query_type}")

        return result

    def _classify_query(self, query: str) -> str:
        """
        Classify query into predefined types.

        Args:
            query: User query

        Returns:
            Query type string
        """
        query_lower = query.lower()

        # Keyword-based classification
        if any(keyword in query_lower for keyword in ["right", "obligation", "must", "shall", "duty"]):
            return "rights_obligations"

        elif any(keyword in query_lower for keyword in ["terminate", "termination", "cancel", "end"]):
            return "termination"

        elif any(keyword in query_lower for keyword in ["date", "when", "duration", "period", "timeline"]):
            return "duration_timeline"

        elif any(keyword in query_lower for keyword in ["liability", "damage", "liable", "responsible"]):
            return "liability"

        elif any(keyword in query_lower for keyword in ["except", "exception", "unless", "exclude"]):
            return "exceptions"

        else:
            return "general"

    def _refine_query(self, query: str, query_type: str) -> str:
        """
        Refine query for better retrieval.

        Args:
            query: Original query
            query_type: Classified query type

        Returns:
            Refined query
        """
        if self.llm:
            # Use LLM to expand query
            return self._llm_refine_query(query, query_type)
        else:
            # Simple keyword expansion
            return self._keyword_refine_query(query, query_type)

    def _keyword_refine_query(self, query: str, query_type: str) -> str:
        """Simple keyword-based query refinement."""
        # Add relevant keywords based on query type
        expansions = {
            "rights_obligations": " rights obligations duties responsibilities",
            "termination": " termination cancellation expiration end",
            "duration_timeline": " date time period duration deadline",
            "liability": " liability damages compensation indemnification",
            "exceptions": " exception exclusion unless provided",
            "general": "",
        }

        expansion = expansions.get(query_type, "")
        return query + expansion

    def _llm_refine_query(self, query: str, query_type: str) -> str:
        """LLM-based query refinement."""
        prompt = f"""Given the following legal contract query, generate a refined version that would be better for retrieving relevant information.

Query Type: {query_type}
Original Query: {query}

Refined Query:"""

        result = self.llm.generate(prompt, max_tokens=100, temperature=0.3)
        refined = result['text'].strip()

        return refined if refined else query

    def _get_retrieval_params(self, query_type: str) -> Dict:
        """
        Get retrieval parameters based on query type.

        Args:
            query_type: Query type

        Returns:
            Dict with retrieval parameters
        """
        # Default parameters
        params = {
            'top_k': 5,
            'chunking_strategy': 'fixed_length',
            'min_score_threshold': 0.0,
        }

        # Adjust based on query type
        if query_type in ["rights_obligations", "liability"]:
            # These often require more context
            params['top_k'] = 7
            params['chunking_strategy'] = 'clause_level'

        elif query_type == "duration_timeline":
            # Dates are usually specific
            params['top_k'] = 3
            params['min_score_threshold'] = 0.1

        elif query_type == "exceptions":
            # Exceptions might be scattered
            params['top_k'] = 10
            params['chunking_strategy'] = 'clause_level'

        return params

    def generate_sub_queries(self, query: str) -> List[str]:
        """
        Generate sub-queries for complex queries.

        Args:
            query: Original query

        Returns:
            List of sub-queries
        """
        # Check if query is complex (contains "and", "or", multiple questions)
        if not any(word in query.lower() for word in ["and", "or", "also", "?"]):
            return [query]

        if self.llm:
            return self._llm_generate_sub_queries(query)
        else:
            return self._simple_split_query(query)

    def _simple_split_query(self, query: str) -> List[str]:
        """Simple query splitting."""
        # Split by "and", "or", or multiple question marks
        import re
        parts = re.split(r'\s+and\s+|\s+or\s+|\?+', query)
        sub_queries = [part.strip() for part in parts if part.strip()]

        return sub_queries if len(sub_queries) > 1 else [query]

    def _llm_generate_sub_queries(self, query: str) -> List[str]:
        """LLM-based sub-query generation."""
        prompt = f"""Break down the following complex legal query into simpler sub-queries:

Original Query: {query}

Sub-queries (one per line):"""

        result = self.llm.generate(prompt, max_tokens=200, temperature=0.3)
        lines = result['text'].strip().split('\n')

        sub_queries = [line.strip() for line in lines if line.strip()]

        return sub_queries if sub_queries else [query]


def main():
    """Test query analyzer."""
    analyzer = QueryAnalyzer()

    test_queries = [
        "Can the contract be terminated with 30 days notice?",
        "What are the payment obligations of the parties?",
        "When does the agreement expire?",
        "Who is liable for damages in case of breach?",
        "Are there any exceptions to the confidentiality clause?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = analyzer.analyze(query)
        print(f"Type: {result['query_type']}")
        print(f"Refined: {result['refined_query']}")
        print(f"Params: {result['retrieval_params']}")


if __name__ == "__main__":
    main()
