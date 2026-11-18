"""
PrivacyQA dataset processor (Optional - for cross-domain evaluation).
Privacy policy question answering dataset.

Dataset: https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP
Task: Question answering on privacy policies

Author: Li Xinru (3036807959)
"""

import json
import os
import argparse
from typing import List
import logging

from dataset_loader import DatasetLoader, LegalExample, get_dataset_statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrivacyQAProcessor(DatasetLoader):
    """Processor for PrivacyQA dataset (optional, for cross-domain evaluation)."""

    def __init__(self, data_dir: str = "data"):
        super().__init__(os.path.join(data_dir, "processed"))
        self.raw_dir = os.path.join(data_dir, "raw", "privacyqa")
        os.makedirs(self.raw_dir, exist_ok=True)

    def download(self):
        """
        Download PrivacyQA dataset.
        Note: May require manual download from GitHub repository.
        """
        logger.info("PrivacyQA requires manual download from:")
        logger.info("https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP")
        logger.info(f"Please place the dataset files in: {self.raw_dir}")
        raise NotImplementedError("Manual download required")

    def process(self, split: str = "test", sample_size: int = 100) -> List[LegalExample]:
        """
        Process PrivacyQA data into standard format.

        Args:
            split: Dataset split
            sample_size: Number of samples to use (for quick evaluation)

        Returns:
            List of LegalExample objects
        """
        logger.info(f"Processing PrivacyQA {split} split (sample_size={sample_size})...")

        # Placeholder implementation - adjust based on actual PrivacyQA format
        raw_path = os.path.join(self.raw_dir, f"{split}.json")
        if not os.path.exists(raw_path):
            logger.warning(f"Raw file not found: {raw_path}")
            logger.info("PrivacyQA is optional. Skipping...")
            return []

        with open(raw_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        examples = []
        for idx, item in enumerate(raw_data[:sample_size]):
            example = LegalExample(
                id=f"privacyqa_{split}_{idx}",
                query=item.get("question", ""),
                context=item.get("context", ""),
                answer=item.get("answer", None),
                label=None,  # PrivacyQA may not have NLI labels
                evidence_spans=None,
                metadata={
                    "dataset": "privacyqa",
                    "split": split,
                }
            )
            examples.append(example)

        logger.info(f"Processed {len(examples)} PrivacyQA examples")

        # Print statistics
        stats = get_dataset_statistics(examples)
        logger.info(f"Dataset statistics: {stats}")

        # Save processed data
        self.save_processed(examples, f"privacyqa_{split}.json")

        return examples

    def load(self, split: str = "test") -> List[LegalExample]:
        """Load processed PrivacyQA dataset."""
        try:
            return self.load_processed(f"privacyqa_{split}.json")
        except FileNotFoundError:
            logger.info("PrivacyQA is optional and not processed yet.")
            return []


def main():
    parser = argparse.ArgumentParser(description="Process PrivacyQA dataset (optional)")
    parser.add_argument("--download", action="store_true", help="Show download instructions")
    parser.add_argument("--process", action="store_true", help="Process dataset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of samples")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")

    args = parser.parse_args()

    processor = PrivacyQAProcessor(data_dir=args.data_dir)

    if args.download:
        processor.download()

    if args.process:
        processor.process(split=args.split, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
