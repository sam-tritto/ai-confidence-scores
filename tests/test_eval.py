"""
Unit tests for evaluation metrics and CalibrationVisualizer.
"""

from pathlib import Path
import numpy as np
import pytest

from src.eval.metrics import (
    calculate_accuracy,
    calculate_brier_score,
    calculate_ece_mce,
    compute_suite_metrics,
    summarize_benchmark,
)
from src.eval.visualizer import CalibrationVisualizer


def test_calculate_accuracy():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    assert calculate_accuracy(y_true, y_pred) == 0.5


def test_calculate_brier_score():
    y_true = np.array([1, 0])
    y_prob = np.array([0.9, 0.2])
    # MSE = ((0.9-1)^2 + (0.2-0)^2) / 2 = (0.01 + 0.04)/2 = 0.025
    assert abs(calculate_brier_score(y_true, y_prob) - 0.025) < 1e-5


def test_calculate_ece_mce():
    y_true = np.array([1, 1, 1, 0, 0])
    y_prob = np.array([0.9, 0.85, 0.8, 0.2, 0.1])
    ece, mce, bin_details = calculate_ece_mce(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0
    assert len(bin_details) == 5


def test_compute_suite_metrics_and_summary():
    y_true = np.array([1, 1, 0])
    y_pred = np.array([1, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.1])
    latencies = np.array([120.0, 130.0, 110.0])
    decisions = ["AUTOMATE", "AUTOMATE", "FLAG_FOR_HUMAN_REVIEW"]

    metrics = compute_suite_metrics(
        engine_name="Test Engine",
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        latencies=latencies,
        audit_decisions=decisions,
    )
    assert metrics.accuracy == 1.0
    assert abs(metrics.mean_latency_ms - 120.0) < 1e-3

    df = summarize_benchmark([metrics])
    assert "Framework Engine" in df.columns
    assert df.shape[0] == 1


def test_visualizer_plotting(tmp_path: Path):
    viz = CalibrationVisualizer(output_dir=tmp_path)
    
    # Mock predictions dict for 9 engines
    mock_preds = {}
    engine_names = [
        "Native Token LogProb",
        "LogProb Delta",
        "Temperature Scaling",
        "Platt Scaling Logistic",
        "Self-Consistency & Agreement",
        "Structured Verbalized Confidence",
        "Continuous Numerical Prompting",
        "Grounding & Alignment",
        "LLM-as-a-Judge",
    ]
    for name in engine_names:
        mock_preds[name] = {
            "y_true": np.array([1, 1, 0, 1, 0, 1, 0]),
            "y_prob": np.array([0.9, 0.8, 0.2, 0.85, 0.15, 0.95, 0.05]),
        }

    rel_path = viz.plot_reliability_diagrams(mock_preds, save_name="test_reliability.png")
    assert rel_path.exists()

    metrics_list = []
    for name in engine_names:
        metrics_list.append(
            compute_suite_metrics(
                engine_name=name,
                y_true=np.array([1, 1, 0, 1, 0]),
                y_pred=np.array([1, 1, 0, 1, 0]),
                y_prob=np.array([0.9, 0.8, 0.2, 0.85, 0.15]),
                latencies=np.array([100.0, 110.0]),
                audit_decisions=["AUTOMATE", "AUTOMATE"],
            )
        )
    df_sum = summarize_benchmark(metrics_list)

    bar_path = viz.plot_ece_vs_latency_bars(df_sum, save_name="test_bars.png")
    assert bar_path.exists()

    radar_path = viz.plot_radar_comparison(df_sum, save_name="test_radar.png")
    assert radar_path.exists()
