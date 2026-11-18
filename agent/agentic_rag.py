"""
Agentic RAG system - complete workflow.
Integrates Query Analyzer, Retriever, Generator, Verifier, and Reviser.

Author: Yin Fengling (3036807375)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
import time
from utils.llm_interface import LLMInterface
from agent.query_analyzer.analyzer import QueryAnalyzer
from agent.generator import AnswerGenerator
from agent.verifier.verifier import EvidenceVerifier
from agent.reviser.reviser import AnswerReviser
from retrieval.chunking import get_chunker
from retrieval.retrievers.retriever import get_retriever
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AgenticRAG:
    """Complete Agentic RAG system with verification and revision."""

    def __init__(
        self,
        llm: LLMInterface,
        retriever_type: str = "bm25",
        chunking_strategy: str = "fixed_length",
        enable_verification: bool = True,
        enable_revision: bool = True,
        **kwargs
    ):
        """
        Initialize Agentic RAG system.

        Args:
            llm: LLM interface
            retriever_type: Type of retriever
            chunking_strategy: Chunking strategy
            enable_verification: Whether to verify answers
            enable_revision: Whether to revise answers
            **kwargs: Additional arguments
        """
        self.llm = llm
        self.enable_verification = enable_verification
        self.enable_revision = enable_revision

        # Initialize components
        self.query_analyzer = QueryAnalyzer(llm=llm)
        self.chunker = get_chunker(chunking_strategy, **kwargs)
        self.retriever = get_retriever(retriever_type, **kwargs)
        self.generator = AnswerGenerator(llm)
        self.verifier = EvidenceVerifier(llm)
        self.reviser = AnswerReviser(llm, retriever=self.retriever)

        self.indexed = False

        logger.info("Initialized Agentic RAG system")

    def index_documents(self, contexts: List[str]):
        """Index documents for retrieval."""
        logger.info(f"Indexing {len(contexts)} documents...")

        all_chunks = []
        for doc_idx, context in enumerate(contexts):
            chunks = self.chunker.chunk(context, metadata={'doc_id': doc_idx})
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks")
        self.retriever.index(all_chunks)
        self.indexed = True

    def answer(self, query: str, context: str = None) -> Dict:
        """
        Answer query using full agentic workflow.

        Args:
            query: User query
            context: Optional context (will be indexed if not done)

        Returns:
            Complete result dict with all intermediate steps
        """
        start_time = time.time()

        # Step 1: Query Analysis
        logger.info("Step 1: Analyzing query...")
        query_analysis = self.query_analyzer.analyze(query)

        # Step 2: Retrieval
        logger.info("Step 2: Retrieving evidence...")
        if not self.indexed and context:
            self.index_documents([context])

        refined_query = query_analysis['refined_query']
        retrieval_params = query_analysis['retrieval_params']

        retrieved = self.retriever.retrieve(
            refined_query,
            top_k=retrieval_params.get('top_k', 5)
        )
        evidence_chunks = [doc for doc, score in retrieved]
        evidence_texts = [chunk['text'] for chunk in evidence_chunks]

        # Step 3: Initial Answer Generation
        logger.info("Step 3: Generating initial answer...")
        generation_result = self.generator.generate(
            query,
            evidence_texts,
            query_type=query_analysis['query_type']
        )

        initial_answer = generation_result['answer']
        claims = generation_result['claims']

        # Step 4: Verification (if enabled)
        verification_result = None
        if self.enable_verification:
            logger.info("Step 4: Verifying claims...")
            verification_result = self.verifier.verify(
                claims,
                evidence_texts,
                answer=initial_answer
            )
        else:
            logger.info("Step 4: Verification disabled")

        # Step 5: Revision (if needed and enabled)
        final_answer = initial_answer
        revision_result = None

        if self.enable_revision and verification_result:
            if verification_result.get('hallucination_detected', False):
                logger.info("Step 5: Hallucination detected, revising answer...")
                revision_result = self.reviser.revise(
                    initial_answer,
                    verification_result,
                    evidence_texts,
                    query
                )
                final_answer = revision_result['revised_answer']
            else:
                logger.info("Step 5: No hallucinations detected, no revision needed")
        else:
            logger.info("Step 5: Revision disabled or not needed")

        latency = time.time() - start_time

        # Compile complete result
        result = {
            'query': query,
            'query_analysis': query_analysis,
            'retrieved_evidence': evidence_texts,
            'evidence_scores': [score for _, score in retrieved],
            'initial_answer': initial_answer,
            'initial_claims': claims,
            'initial_label': generation_result['label'],
            'verification': verification_result,
            'revision': revision_result,
            'final_answer': final_answer,
            'label': generation_result['label'],  # May need updating based on revision
            'hallucination_detected': verification_result['hallucination_detected'] if verification_result else False,
            'abstained': revision_result and revision_result.get('revision_type') == 'abstention',
            'latency': latency,
            'tokens': self._count_total_tokens(generation_result, verification_result, revision_result),
        }

        logger.info(f"Completed in {latency:.2f}s")

        return result

    def _count_total_tokens(self, *results) -> int:
        """Count total tokens used across all steps."""
        total = 0
        for result in results:
            if result and isinstance(result, dict):
                total += result.get('tokens', 0)
        return total

    def batch_answer(self, examples: List[Dict]) -> List[Dict]:
        """
        Answer multiple queries in batch.

        Args:
            examples: List of example dicts with 'query' and 'context'

        Returns:
            List of prediction dicts
        """
        # Index all documents
        contexts = [ex['context'] for ex in examples]
        self.index_documents(contexts)

        predictions = []

        for i, example in enumerate(examples):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing example {i+1}/{len(examples)}")
            logger.info(f"{'='*60}")

            result = self.answer(example['query'])
            result['id'] = example.get('id', f'example_{i}')

            predictions.append(result)

        return predictions


def main():
    """Example usage of Agentic RAG."""
    import argparse
    from utils.helpers import load_yaml, save_json

    parser = argparse.ArgumentParser(description="Agentic RAG System")
    parser.add_argument("--model", type=str, default="llama3", help="Model name")
    parser.add_argument("--retriever", type=str, default="bm25", help="Retriever type")
    parser.add_argument("--chunking", type=str, default="fixed_length", help="Chunking strategy")
    parser.add_argument("--dataset", type=str, default="contractnli", help="Dataset name")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Config file")
    parser.add_argument("--output", type=str, default="experiments/results/agentic_rag.json", help="Output file")
    parser.add_argument("--no-verification", action="store_true", help="Disable verification")
    parser.add_argument("--no-revision", action="store_true", help="Disable revision")

    args = parser.parse_args()

    # Load config
    config = load_yaml(args.config)
    model_config = config['model'][args.model]

    # Initialize LLM
    logger.info(f"Initializing {args.model}...")
    llm = LLMInterface(
        model_name=model_config['name'],
        max_tokens=model_config['max_tokens'],
        temperature=model_config['temperature'],
    )

    # Initialize Agentic RAG
    agent = AgenticRAG(
        llm=llm,
        retriever_type=args.retriever,
        chunking_strategy=args.chunking,
        enable_verification=not args.no_verification,
        enable_revision=not args.no_revision,
    )

    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    if args.dataset == "contractnli":
        from data.contractnli_processor import ContractNLIProcessor
        processor = ContractNLIProcessor()
        examples = processor.load(split="test")
    elif args.dataset == "cuad":
        from data.cuad_processor import CUADProcessor
        processor = CUADProcessor()
        examples = processor.load(split="test")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Convert to dict format
    example_dicts = [
        {'id': ex.id, 'query': ex.query, 'context': ex.context}
        for ex in examples[:5]  # Test on first 5
    ]

    # Run Agentic RAG
    logger.info("Running Agentic RAG...")
    predictions = agent.batch_answer(example_dicts)

    # Save results
    save_json(predictions, args.output)
    logger.info(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
