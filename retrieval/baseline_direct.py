"""
Direct Prompting baseline (no RAG).
Simply prompts the LLM without retrieval.

Author: Qin Mengmeng (3036843769)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
import time
from utils.llm_interface import LLMInterface
from utils.logger import setup_logger
import logging

logger = setup_logger(__name__)


class DirectPromptingBaseline:
    """Direct prompting baseline without retrieval."""

    def __init__(self, llm: LLMInterface):
        """
        Initialize direct prompting baseline.

        Args:
            llm: LLM interface
        """
        self.llm = llm

    def answer(self, query: str, context: str) -> Dict:
        """
        Answer query directly without retrieval.

        Args:
            query: User query
            context: Full document context

        Returns:
            Dict with answer, label, latency, tokens
        """
        start_time = time.time()

        # Create prompt
        prompt = self._create_prompt(query, context)

        # Generate answer
        result = self.llm.generate(prompt)

        # Parse response
        answer_text = result['text']
        parsed = self._parse_response(answer_text)

        latency = time.time() - start_time

        return {
            'answer': parsed['answer'],
            'label': parsed['label'],
            'raw_response': answer_text,
            'latency': latency,
            'tokens': result.get('tokens', 0),
        }

    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for direct answering."""
        prompt = f"""You are a legal contract analysis assistant. Answer the following question based on the contract provided.

Contract:
{context[:2000]}  # Limit context length

Question: {query}

Provide your answer and classify the relationship as one of:
- Entailment: The contract clearly supports the statement
- Contradiction: The contract contradicts the statement
- Neutral: The contract does not provide enough information

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
        predictions = []

        for i, example in enumerate(examples):
            logger.info(f"Processing example {i+1}/{len(examples)}")

            result = self.answer(example['query'], example['context'])
            result['id'] = example.get('id', f'example_{i}')

            predictions.append(result)

        return predictions


def main():
    """Example usage."""
    import argparse
    from utils.helpers import load_yaml
    from data.dataset_loader import LegalExample

    parser = argparse.ArgumentParser(description="Direct Prompting Baseline")
    parser.add_argument("--model", type=str, default="llama3", help="Model name")
    parser.add_argument("--dataset", type=str, default="contractnli", help="Dataset name")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Model config")
    parser.add_argument("--output", type=str, default="experiments/results/direct_prompting.json", help="Output file")

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
    baseline = DirectPromptingBaseline(llm)

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
    logger.info("Running direct prompting baseline...")
    predictions = baseline.batch_answer(example_dicts)

    # Save results
    from utils.helpers import save_json
    save_json(predictions, args.output)
    logger.info(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
