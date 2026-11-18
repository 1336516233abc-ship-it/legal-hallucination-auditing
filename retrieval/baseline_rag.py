"""
Standard RAG baseline.
Retrieval + Generation without verification.

Author: Qin Mengmeng (3036843769)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
import time
from utils.llm_interface import LLMInterface
from retrieval.chunking import get_chunker
from retrieval.retrievers.retriever import get_retriever
from utils.logger import setup_logger
import logging

logger = setup_logger(__name__)


class StandardRAGBaseline:
    """Standard RAG baseline: Retrieve + Generate."""

    def __init__(
        self,
        llm: LLMInterface,
        retriever_type: str = "bm25",
        chunking_strategy: str = "fixed_length",
        top_k: int = 5,
        **kwargs
    ):
        """
        Initialize standard RAG baseline.

        Args:
            llm: LLM interface
            retriever_type: Type of retriever ("bm25", "dense", "hybrid")
            chunking_strategy: Chunking strategy ("fixed_length", "clause_level")
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments for retriever and chunker
        """
        self.llm = llm
        self.top_k = top_k

        # Initialize chunker
        self.chunker = get_chunker(chunking_strategy, **kwargs)

        # Initialize retriever
        self.retriever = get_retriever(retriever_type, **kwargs)

        # Document index
        self.indexed = False

    def index_documents(self, contexts: List[str]):
        """
        Index documents for retrieval.

        Args:
            contexts: List of document contexts
        """
        logger.info(f"Chunking {len(contexts)} documents...")

        all_chunks = []
        for doc_idx, context in enumerate(contexts):
            chunks = self.chunker.chunk(context, metadata={'doc_id': doc_idx})
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks total")

        # Index chunks
        logger.info("Indexing chunks...")
        self.retriever.index(all_chunks)

        self.indexed = True
        logger.info("Indexing complete")

    def answer(self, query: str, context: str = None) -> Dict:
        """
        Answer query using RAG.

        Args:
            query: User query
            context: Optional full context (will be chunked if not indexed)

        Returns:
            Dict with answer, label, evidence, latency, tokens
        """
        start_time = time.time()

        # If not indexed, index the single context
        if not self.indexed and context:
            self.index_documents([context])

        # Retrieve relevant chunks
        retrieved = self.retriever.retrieve(query, top_k=self.top_k)
        evidence_chunks = [doc for doc, score in retrieved]

        # Create prompt with retrieved evidence
        prompt = self._create_rag_prompt(query, evidence_chunks)

        # Generate answer
        result = self.llm.generate(prompt)

        # Parse response
        answer_text = result['text']
        parsed = self._parse_response(answer_text)

        latency = time.time() - start_time

        return {
            'answer': parsed['answer'],
            'label': parsed['label'],
            'retrieved_evidence': [chunk['text'] for chunk in evidence_chunks],
            'evidence_scores': [score for _, score in retrieved],
            'raw_response': answer_text,
            'latency': latency,
            'tokens': result.get('tokens', 0),
        }

    def _create_rag_prompt(self, query: str, evidence_chunks: List[Dict]) -> str:
        """Create RAG prompt with retrieved evidence."""
        # Format evidence
        evidence_text = "\n\n".join([
            f"Evidence {i+1}:\n{chunk['text']}"
            for i, chunk in enumerate(evidence_chunks)
        ])

        prompt = f"""You are a legal contract analysis assistant. Answer the following question based on the retrieved evidence from the contract.

Retrieved Evidence:
{evidence_text}

Question: {query}

Based on the evidence above, provide your answer and classify the relationship as:
- Entailment: The evidence clearly supports the statement
- Contradiction: The evidence contradicts the statement
- Neutral: The evidence does not provide enough information

Format your response as:
Answer: [Your answer here]
Label: [Entailment/Contradiction/Neutral]
"""
        return prompt

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response to extract answer and label."""
        lines = response.strip().split('\n')

        answer = ""
        label = "neutral"

        for line in lines:
            line = line.strip()
            if line.lower().startswith("answer:"):
                answer = line.split(":", 1)[1].strip()
            elif line.lower().startswith("label:"):
                label_text = line.split(":", 1)[1].strip().lower()
                if "entailment" in label_text:
                    label = "entailment"
                elif "contradiction" in label_text:
                    label = "contradiction"
                else:
                    label = "neutral"

        # Fallback: use full response as answer
        if not answer:
            answer = response

        return {
            'answer': answer,
            'label': label
        }

    def batch_answer(self, examples: List[Dict]) -> List[Dict]:
        """
        Answer multiple queries in batch.

        Args:
            examples: List of example dicts with 'query' and 'context'

        Returns:
            List of prediction dicts
        """
        # Index all documents first
        contexts = [ex['context'] for ex in examples]
        self.index_documents(contexts)

        predictions = []

        for i, example in enumerate(examples):
            logger.info(f"Processing example {i+1}/{len(examples)}")

            result = self.answer(example['query'])
            result['id'] = example.get('id', f'example_{i}')

            predictions.append(result)

        return predictions


def main():
    """Example usage."""
    import argparse
    from utils.helpers import load_yaml

    parser = argparse.ArgumentParser(description="Standard RAG Baseline")
    parser.add_argument("--model", type=str, default="llama3", help="Model name")
    parser.add_argument("--retriever", type=str, default="bm25", help="Retriever type")
    parser.add_argument("--chunking", type=str, default="fixed_length", help="Chunking strategy")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--dataset", type=str, default="contractnli", help="Dataset name")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Model config")
    parser.add_argument("--output", type=str, default="experiments/results/standard_rag.json", help="Output file")

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

    # Initialize baseline
    baseline = StandardRAGBaseline(
        llm=llm,
        retriever_type=args.retriever,
        chunking_strategy=args.chunking,
        top_k=args.top_k
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
        for ex in examples[:10]  # Test on first 10
    ]

    # Run baseline
    logger.info("Running standard RAG baseline...")
    predictions = baseline.batch_answer(example_dicts)

    # Save results
    from utils.helpers import save_json
    save_json(predictions, args.output)
    logger.info(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
