"""Evaluation module."""

from evaluation.metrics import MetricsCalculator, EvaluationResult
from evaluation.hallucination_detector import HallucinationDetector, HallucinationCase
from evaluation.error_analysis import ErrorAnalyzer, compare_systems

__all__ = [
    'MetricsCalculator',
    'EvaluationResult',
    'HallucinationDetector',
    'HallucinationCase',
    'ErrorAnalyzer',
    'compare_systems',
]
