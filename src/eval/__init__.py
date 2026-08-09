"""
Evaluation & Metrics Module exporting metrics functions and CalibrationVisualizer.
"""

from src.eval.metrics import (
    calculate_accuracy,
    calculate_brier_score,
    calculate_ece_mce,
    compute_suite_metrics,
    summarize_benchmark,
)
from src.eval.visualizer import CalibrationVisualizer

__all__ = [
    "calculate_accuracy",
    "calculate_brier_score",
    "calculate_ece_mce",
    "compute_suite_metrics",
    "summarize_benchmark",
    "CalibrationVisualizer",
]
