"""
Main experiment runner.
Orchestrates all experiments: baselines, agentic RAG, ablations.

Author: Wang Zhengyao (3036795675)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import List, Dict
import json
from datetime import datetime
from utils.helpers import load_yaml, save_json, set_random_seed, ensure_dir
from utils.logger import get_experiment_logger
from utils.llm_interface import LLMInterface
from evaluation.metrics import MetricsCalculator, EvaluationResult
from evaluation.hallucination_detector import HallucinationDetector
from evaluation.error_analysis import ErrorAnalyzer, compare_systems

logger = None  # Will be initialized in main


class ExperimentRunner:
    """Orchestrates all experiments."""

    def __init__(self, config: Dict):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.metrics_calculator = MetricsCalculator()
        self.hallucination_detector = HallucinationDetector()
        self.error_analyzer = ErrorAnalyzer()

        # Set random seed for reproducibility
        if config['reproducibility']['set_seed']:
            set_random_seed(config['reproducibility']['seed'])

    def run_all_experiments(self):
        """Run complete experiment suite."""
        logger.info("="*70)
        logger.info("STARTING COMPLETE EXPERIMENT SUITE")
        logger.info("="*70)

        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'experiments': {}
        }

        # Load datasets
        datasets = self._load_datasets()

        # Run experiments for each model
        for model_name in self.config['models']:
            logger.info(f"\n{'='*70}")
            logger.info(f"EXPERIMENTS WITH MODEL: {model_name}")
            logger.info(f"{'='*70}\n")

            model_results = self._run_model_experiments(model_name, datasets)
            results['experiments'][model_name] = model_results

        # Save complete results
        output_path = os.path.join(
            self.config['output']['save_path'],
            f"complete_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        save_json(results, output_path)
        logger.info(f"\nSaved complete results to {output_path}")

        # Generate summary report
        self._generate_summary_report(results)

        return results

    def _load_datasets(self) -> Dict:
        """Load all enabled datasets."""
        logger.info("Loading datasets...")
        datasets = {}

        if self.config['datasets']['contractnli']['enabled']:
            from data.contractnli_processor import ContractNLIProcessor
            processor = ContractNLIProcessor()
            datasets['contractnli'] = processor.load(split='test')
            logger.info(f"Loaded ContractNLI: {len(datasets['contractnli'])} examples")

        if self.config['datasets']['cuad']['enabled']:
            from data.cuad_processor import CUADProcessor
            processor = CUADProcessor()
            datasets['cuad'] = processor.load(split='test')
            logger.info(f"Loaded CUAD: {len(datasets['cuad'])} examples")

        if self.config['datasets'].get('privacyqa', {}).get('enabled', False):
            from data.privacyqa_processor import PrivacyQAProcessor
            processor = PrivacyQAProcessor()
            datasets['privacyqa'] = processor.load(split='test')
            logger.info(f"Loaded PrivacyQA: {len(datasets['privacyqa'])} examples")

        return datasets

    def _run_model_experiments(self, model_name: str, datasets: Dict) -> Dict:
        """Run all experiments for a specific model."""
        # Load model config
        model_config_path = "configs/model_config.yaml"
        model_config = load_yaml(model_config_path)['model'][model_name]

        # Initialize LLM
        logger.info(f"Initializing model: {model_name}")
        llm = LLMInterface(
            model_name=model_config['name'],
            max_tokens=model_config['max_tokens'],
            temperature=model_config['temperature'],
        )

        model_results = {}

        # Run for each dataset
        for dataset_name, examples in datasets.items():
            logger.info(f"\n--- Dataset: {dataset_name} ---")

            dataset_results = {}

            # 1. Direct Prompting Baseline
            if 'direct_prompting' in self.config['modes']:
                logger.info("Running Direct Prompting baseline...")
                direct_results = self._run_direct_prompting(llm, examples)
                dataset_results['direct_prompting'] = direct_results

            # 2. Standard RAG Baseline
            if 'standard_rag' in self.config['modes']:
                logger.info("Running Standard RAG baseline...")
                rag_results = self._run_standard_rag(llm, examples)
                dataset_results['standard_rag'] = rag_results

            # 3. Agentic RAG
            if 'agentic_rag' in self.config['modes']:
                logger.info("Running Agentic RAG...")
                agentic_results = self._run_agentic_rag(llm, examples)
                dataset_results['agentic_rag'] = agentic_results

            # 4. Ablation Studies
            if self.config.get('ablations'):
                logger.info("Running ablation studies...")
                ablation_results = self._run_ablations(llm, examples)
                dataset_results['ablations'] = ablation_results

            model_results[dataset_name] = dataset_results

        return model_results

    def _run_direct_prompting(self, llm: LLMInterface, examples: List) -> Dict:
        """Run direct prompting baseline."""
        from retrieval.baseline_direct import DirectPromptingBaseline

        baseline = DirectPromptingBaseline(llm)

        example_dicts = [
            {'id': ex.id, 'query': ex.query, 'context': ex.context}
            for ex in examples
        ]

        predictions = baseline.batch_answer(example_dicts)

        # Evaluate
        ground_truth = [
            {'id': ex.id, 'label': ex.label, 'answer': ex.answer, 'evidence_spans': ex.evidence_spans}
            for ex in examples
        ]

        metrics = self._evaluate_predictions(predictions, ground_truth)

        return {
            'predictions': predictions,
            'metrics': metrics,
        }

    def _run_standard_rag(self, llm: LLMInterface, examples: List) -> Dict:
        """Run standard RAG baseline."""
        from retrieval.baseline_rag import StandardRAGBaseline

        baseline = StandardRAGBaseline(
            llm=llm,
            retriever_type="bm25",
            chunking_strategy="fixed_length",
            top_k=5
        )

        example_dicts = [
            {'id': ex.id, 'query': ex.query, 'context': ex.context}
            for ex in examples
        ]

        predictions = baseline.batch_answer(example_dicts)

        # Evaluate
        ground_truth = [
            {'id': ex.id, 'label': ex.label, 'answer': ex.answer, 'evidence_spans': ex.evidence_spans}
            for ex in examples
        ]

        metrics = self._evaluate_predictions(
            predictions,
            ground_truth,
            retrieved_evidence=[p.get('retrieved_evidence', []) for p in predictions]
        )

        return {
            'predictions': predictions,
            'metrics': metrics,
        }

    def _run_agentic_rag(self, llm: LLMInterface, examples: List) -> Dict:
        """Run agentic RAG system."""
        from agent.agentic_rag import AgenticRAG

        agent = AgenticRAG(
            llm=llm,
            retriever_type="bm25",
            chunking_strategy="fixed_length",
            enable_verification=True,
            enable_revision=True,
        )

        example_dicts = [
            {'id': ex.id, 'query': ex.query, 'context': ex.context}
            for ex in examples
        ]

        predictions = agent.batch_answer(example_dicts)

        # Evaluate
        ground_truth = [
            {'id': ex.id, 'label': ex.label, 'answer': ex.answer, 'evidence_spans': ex.evidence_spans}
            for ex in examples
        ]

        # Analyze hallucinations
        verification_results = [p.get('verification', {}) for p in predictions]
        hallucination_cases = self.hallucination_detector.detect_hallucinations(
            predictions,
            verification_results,
            ground_truth
        )

        hallucination_analysis = self.hallucination_detector.analyze_failure_patterns()

        metrics = self._evaluate_predictions(
            predictions,
            ground_truth,
            retrieved_evidence=[p.get('retrieved_evidence', []) for p in predictions],
            verification_results=verification_results
        )

        return {
            'predictions': predictions,
            'metrics': metrics,
            'hallucination_analysis': hallucination_analysis,
        }

    def _run_ablations(self, llm: LLMInterface, examples: List) -> Dict:
        """Run ablation studies."""
        ablation_results = {}

        # Retriever ablation
        for retriever_type in self.config['ablations']['retriever']:
            logger.info(f"  Ablation: retriever={retriever_type}")
            # Run with different retriever
            # ... similar to standard RAG but with different config

        # Top-k ablation
        for top_k in self.config['ablations']['top_k']:
            logger.info(f"  Ablation: top_k={top_k}")
            # Run with different top_k
            # ...

        # Chunking ablation
        for chunking in self.config['ablations']['chunking']:
            logger.info(f"  Ablation: chunking={chunking}")
            # Run with different chunking
            # ...

        return ablation_results

    def _evaluate_predictions(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        retrieved_evidence: List[List[str]] = None,
        verification_results: List[Dict] = None
    ) -> Dict:
        """Evaluate predictions against ground truth."""
        # Extract latencies and token counts
        latencies = [p.get('latency', 0) for p in predictions]
        token_counts = [p.get('tokens', 0) for p in predictions]

        # Compute metrics
        eval_result = self.metrics_calculator.compute_all_metrics(
            predictions=predictions,
            ground_truth=ground_truth,
            retrieved_evidence=retrieved_evidence,
            latencies=latencies,
            token_counts=token_counts
        )

        # Format as dict
        metrics_dict = {
            'label_accuracy': eval_result.label_accuracy,
            'macro_f1': eval_result.macro_f1,
            'evidence_recall': eval_result.evidence_recall_at_k,
            'hallucination_rate': eval_result.hallucination_rate,
            'avg_latency': eval_result.avg_latency,
            'total_tokens': eval_result.total_tokens,
        }

        logger.info(f"Metrics: {metrics_dict}")

        return metrics_dict

    def _generate_summary_report(self, results: Dict):
        """Generate summary report of all experiments."""
        report_path = os.path.join(
            self.config['output']['save_path'],
            f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        lines = [
            "="*70,
            "EXPERIMENT SUMMARY REPORT",
            "="*70,
            f"\nTimestamp: {results['timestamp']}",
            f"\nModels tested: {', '.join(self.config['models'])}",
            f"Datasets: {', '.join([d for d in self.config['datasets'] if self.config['datasets'][d].get('enabled', False)])}",
            "\n" + "="*70,
            "RESULTS BY MODEL AND DATASET",
            "="*70,
        ]

        for model_name, model_results in results['experiments'].items():
            lines.append(f"\n### {model_name.upper()} ###\n")

            for dataset_name, dataset_results in model_results.items():
                lines.append(f"\n  Dataset: {dataset_name}")

                for mode, mode_results in dataset_results.items():
                    if mode == 'ablations':
                        continue

                    lines.append(f"\n    {mode}:")
                    metrics = mode_results.get('metrics', {})

                    for metric_name, value in metrics.items():
                        if isinstance(value, float):
                            lines.append(f"      {metric_name}: {value:.4f}")
                        else:
                            lines.append(f"      {metric_name}: {value}")

        lines.append("\n" + "="*70)

        report_text = "\n".join(lines)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        logger.info(f"\nGenerated summary report: {report_path}")
        print(report_text)


def main():
    global logger

    parser = argparse.ArgumentParser(description="Run Legal Hallucination Auditing Experiments")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml", help="Experiment config")
    parser.add_argument("--mode", type=str, default="all", choices=['all', 'direct', 'rag', 'agent'], help="Experiment mode")
    parser.add_argument("--models", type=str, help="Comma-separated model names (overrides config)")
    parser.add_argument("--dataset", type=str, help="Single dataset to test")
    parser.add_argument("--sample_size", type=int, help="Limit number of examples per dataset")

    args = parser.parse_args()

    # Load config
    config = load_yaml(args.config)

    # Override config with CLI args
    if args.models:
        config['models'] = args.models.split(',')

    if args.mode != 'all':
        mode_map = {
            'direct': ['direct_prompting'],
            'rag': ['standard_rag'],
            'agent': ['agentic_rag'],
        }
        config['modes'] = mode_map[args.mode]

    # Setup logger
    logger = get_experiment_logger('main_experiments')

    # Ensure output directories exist
    ensure_dir(config['output']['save_path'])
    ensure_dir(config['output']['log_path'])

    # Run experiments
    runner = ExperimentRunner(config)
    results = runner.run_all_experiments()

    logger.info("\n" + "="*70)
    logger.info("ALL EXPERIMENTS COMPLETED!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
