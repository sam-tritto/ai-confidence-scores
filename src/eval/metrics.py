"""
Evaluation Metrics Suite for LLM Confidence Calibration.
Calculates Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier Score, and Accuracy.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.schema import CalibrationResult, EvaluationMetrics


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier Score: mean squared error between predicted probabilities and binary correctness labels."""
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


def calculate_ece_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, List[Dict[str, float]]]:
    """Calculate Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and bin statistics.
    
    Returns:
        Tuple of (ece, mce, bin_details_list)
    """
    if len(y_true) == 0:
        return 0.0, 0.0, []

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_details = []
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples falling into current bin range
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_size = int(np.sum(in_bin))

        if bin_size > 0:
            bin_acc = float(np.mean(y_true[in_bin]))
            bin_conf = float(np.mean(y_prob[in_bin]))
            abs_diff = abs(bin_acc - bin_conf)

            ece += (bin_size / total_samples) * abs_diff
            mce = max(mce, abs_diff)
        else:
            bin_acc = 0.0
            bin_conf = (bin_lower + bin_upper) / 2.0
            abs_diff = 0.0

        bin_details.append({
            "bin_index": i,
            "bin_lower": bin_lower,
            "bin_upper": bin_upper,
            "bin_size": bin_size,
            "accuracy": bin_acc,
            "confidence": bin_conf,
            "calibration_gap": abs_diff,
        })

    return float(ece), float(mce), bin_details


def compute_suite_metrics(
    method_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    latencies: np.ndarray,
    audit_decisions: List[str],
    n_bins: int = 10,
) -> EvaluationMetrics:
    """Calculate comprehensive evaluation metrics for a single calibration method."""
    accuracy = calculate_accuracy(y_true, y_pred)
    brier_score = calculate_brier_score(y_true, y_prob)
    ece, mce, _ = calculate_ece_mce(y_true, y_prob, n_bins=n_bins)
    mean_confidence = float(np.mean(y_prob)) if len(y_prob) > 0 else 0.0
    mean_latency = float(np.mean(latencies)) if len(latencies) > 0 else 0.0
    automate_ratio = float(np.mean([1 if d == "AUTOMATE" else 0 for d in audit_decisions])) if audit_decisions else 0.0

    return EvaluationMetrics(
        method_name=method_name,
        accuracy=accuracy,
        ece=ece,
        mce=mce,
        brier_score=brier_score,
        mean_confidence=mean_confidence,
        mean_latency_ms=mean_latency,
        automate_ratio=automate_ratio,
    )


def summarize_benchmark(metrics_list: List[EvaluationMetrics]) -> pd.DataFrame:
    """Convert a list of EvaluationMetrics into a formatted pandas DataFrame summary table."""
    records = []
    for m in metrics_list:
        records.append({
            "Framework Method": m.method_name,
            "Accuracy": m.accuracy,
            "ECE": m.ece,
            "MCE": m.mce,
            "Brier Score": m.brier_score,
            "Mean Confidence": m.mean_confidence,
            "Mean Latency (ms)": m.mean_latency_ms,
            "Automation Ratio": m.automate_ratio,
        })
    df = pd.DataFrame(records)
    return df
