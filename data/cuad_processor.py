"""
CUAD (Contract Understanding Atticus Dataset) processor.
Extracts relevant clause types and converts to QA format.

Dataset: https://www.atticusprojectai.org/cuad
Task: Clause extraction and retrieval from contracts

Author: Li Xinru (3036807959)
"""

import json
import os
import argparse
from typing import List, Dict, Optional
import logging

if __package__:
    from .dataset_loader import DatasetLoader, LegalExample, get_dataset_statistics
else:
    from dataset_loader import DatasetLoader, LegalExample, get_dataset_statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CUADProcessor(DatasetLoader):
    """Processor for CUAD dataset."""

    # Representative clause types to evaluate
    CLAUSE_TYPES = [
        "Document Name",
        "Parties",
        "Agreement Date",
        "Effective Date",
        "Expiration Date",
        "Termination For Convenience",
        "Notice Period To Terminate Renewal",
        "Governing Law",
        "Most Favored Nation",
        "Non-Compete",
        "Exclusivity",
        "Rofr/Rofo/Rofn",
        "Liquidated Damages",
        "Liability Cap",
        "Warranty Duration",
        "Insurance",
        "Covenant Not To Sue",
        "Third Party Beneficiary",
        "Ip Ownership Assignment",
        "Joint Ip Ownership",
        "Post-Termination Services",
        "Audit Rights",
        "Unlimited/All-You-Can-Eat-License",
        "Irrevocable Or Perpetual License",
        "Source Code Escrow",
        "License Grant",
        "Non-Transferable License",
        "Cap On Liability",
        "Competitive Restriction Exception",
        "Anti-Assignment",
        "Change Of Control",
        "Revenue/Profit Sharing",
        "Price Restrictions",
        "Minimum Commitment",
        "Volume Restriction",
    ]

    def __init__(self, data_dir: str = "data", clause_types: Optional[List[str]] = None):
        super().__init__(os.path.join(data_dir, "processed"))
        self.raw_dir = os.path.join(data_dir, "raw", "cuad")
        os.makedirs(self.raw_dir, exist_ok=True)

        # Use default or custom clause types
        self.clause_types = clause_types if clause_types else self.CLAUSE_TYPES[:10]  # Use top 10 by default

    def download(self):
        """Download CUAD dataset from HuggingFace."""
        logger.info("Downloading CUAD dataset...")
        from datasets import load_dataset

        try:
            dataset = load_dataset("cuad", trust_remote_code=True)

            # Save raw data
            for split in ["train", "test"]:
                if split in dataset:
                    output_path = os.path.join(self.raw_dir, f"{split}.json")
                    data = [dict(item) for item in dataset[split]]
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved {len(data)} {split} examples to {output_path}")

            logger.info("CUAD download complete!")
            return dataset

        except Exception as e:
            logger.error(f"Error downloading CUAD: {e}")
            logger.info("You may need to manually download from https://www.atticusprojectai.org/cuad")
            raise

    def process(self, split: str = "test") -> List[LegalExample]:
        """
        Process CUAD data into standard format.

        Args:
            split: Dataset split (train/test)

        Returns:
            List of LegalExample objects
        """
        logger.info(f"Processing CUAD {split} split for clause types: {self.clause_types}")

        raw_path = os.path.join(self.raw_dir, f"{split}.json")
        if not os.path.exists(raw_path):
            logger.warning(f"Raw file not found: {raw_path}. Attempting download...")
            self.download()

        with open(raw_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        examples = []
        for doc_idx, item in enumerate(raw_data):
            context = item.get("context", "")
            title = item.get("title", f"Contract_{doc_idx}")

            # Process each clause type
            for clause_type in self.clause_types:
                # Create query for this clause type
                query = self._create_clause_query(clause_type)

                # Get answers for this clause type
                answers = item.get("answers", {}).get(clause_type, [])

                # Get answer spans
                answer_text = None
                evidence_spans = []

                if answers:
                    answer_text = "; ".join(answers)  # Combine multiple answers
                    # Find spans in context
                    for ans in answers:
                        start_idx = context.find(ans)
                        if start_idx != -1:
                            evidence_spans.append((start_idx, start_idx + len(ans)))

                # Determine label based on presence of answer
                label = "present" if answers else "absent"

                example = LegalExample(
                    id=f"cuad_{split}_{doc_idx}_{clause_type.replace(' ', '_')}",
                    query=query,
                    context=context,
                    answer=answer_text,
                    label=label,
                    evidence_spans=evidence_spans if evidence_spans else None,
                    metadata={
                        "dataset": "cuad",
                        "split": split,
                        "document_title": title,
                        "clause_type": clause_type,
                    }
                )
                examples.append(example)

        logger.info(f"Processed {len(examples)} CUAD examples")

        # Print statistics
        stats = get_dataset_statistics(examples)
        logger.info(f"Dataset statistics: {stats}")

        # Save processed data
        self.save_processed(examples, f"cuad_{split}.json")

        return examples

    def _create_clause_query(self, clause_type: str) -> str:
        """
        Create a natural language query for a clause type.

        Args:
            clause_type: CUAD clause type

        Returns:
            Natural language query
        """
        # Map clause types to natural questions
        query_templates = {
            "Document Name": "What is the name of this document?",
            "Parties": "Who are the parties to this agreement?",
            "Agreement Date": "What is the date of this agreement?",
            "Effective Date": "What is the effective date of this agreement?",
            "Expiration Date": "When does this agreement expire?",
            "Termination For Convenience": "Does the agreement allow termination for convenience?",
            "Notice Period To Terminate Renewal": "What is the notice period required to terminate renewal?",
            "Governing Law": "Which law governs this agreement?",
            "Most Favored Nation": "Does this agreement contain a most favored nation clause?",
            "Non-Compete": "Is there a non-compete clause in this agreement?",
            "Exclusivity": "Does this agreement grant exclusivity?",
            "Rofr/Rofo/Rofn": "Is there a right of first refusal, offer, or negotiation?",
            "Liquidated Damages": "Are there liquidated damages provisions?",
            "Liability Cap": "Is there a cap on liability?",
            "Warranty Duration": "What is the warranty duration?",
            "Insurance": "What are the insurance requirements?",
            "Covenant Not To Sue": "Is there a covenant not to sue?",
            "Third Party Beneficiary": "Are there third party beneficiaries?",
            "Ip Ownership Assignment": "How is IP ownership assigned?",
            "Joint Ip Ownership": "Is there joint IP ownership?",
        }

        # Return specific template or generate generic one
        if clause_type in query_templates:
            return query_templates[clause_type]
        else:
            return f"What are the terms regarding {clause_type.lower()}?"

    def load(self, split: str = "test") -> List[LegalExample]:
        """Load processed CUAD dataset."""
        try:
            return self.load_processed(f"cuad_{split}.json")
        except FileNotFoundError:
            logger.info(f"Processed file not found. Processing {split} split...")
            return self.process(split)


def main():
    parser = argparse.ArgumentParser(description="Process CUAD dataset")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--process", action="store_true", help="Process dataset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--clause_types", type=str, nargs="+", help="Specific clause types to process")

    args = parser.parse_args()

    processor = CUADProcessor(data_dir=args.data_dir, clause_types=args.clause_types)

    if args.download:
        processor.download()

    if args.process:
        processor.process(split=args.split)

    if not args.download and not args.process:
        # Default: load existing data
        examples = processor.load(split=args.split)
        logger.info(f"Loaded {len(examples)} examples")
        logger.info(f"First example: {examples[0]}")


if __name__ == "__main__":
    main()
