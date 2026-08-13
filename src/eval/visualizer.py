"""
Visualization Generator for Confidence Calibration Benchmark.
Generates Reliability Diagrams (Calibration Curves), Radar Charts, and ECE vs. Latency Bar Plots.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.eval.metrics import calculate_ece_mce

# Set high quality aesthetic defaults
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


class CalibrationVisualizer:
    """Generates production-grade diagnostic and comparative plots for calibration frameworks."""

    def __init__(self, output_dir: Union[str, Path] = "./notebooks/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colors = sns.color_palette("tab10", 10)

    def plot_reliability_diagrams(
        self,
        method_predictions: Dict[str, Dict[str, np.ndarray]],
        n_bins: int = 10,
        save_name: str = "reliability_diagrams.png",
    ) -> Path:
        """Plot a 2x5 grid of Reliability Diagrams (Calibration Curves) comparing all 10 frameworks.
        
        Args:
            method_predictions: Dict mapping method_name to {"y_true": array, "y_prob": array}
        """
        fig, axes = plt.subplots(2, 5, figsize=(20, 9), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx, (method_name, data) in enumerate(method_predictions.items()):
            if idx >= 10:
                break
            ax = axes[idx]
            y_true = data["y_true"]
            y_prob = data["y_prob"]

            ece, mce, bin_details = calculate_ece_mce(y_true, y_prob, n_bins=n_bins)

            confs = [b["confidence"] for b in bin_details if b["bin_size"] > 0]
            accs = [b["accuracy"] for b in bin_details if b["bin_size"] > 0]
            sizes = [b["bin_size"] for b in bin_details if b["bin_size"] > 0]

            # Reference perfectly calibrated diagonal
            ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect Calibration")

            # Calibration curve
            if confs:
                ax.plot(confs, accs, "o-", color=self.colors[idx], linewidth=2, markersize=6, label=f"ECE: {ece:.3f}")
                ax.bar(confs, accs, width=0.08, alpha=0.2, color=self.colors[idx], align="center")

            ax.set_title(method_name, fontsize=10, fontweight="bold", pad=8)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            ax.set_xlabel("Mean Predicted Confidence", fontsize=9)
            ax.set_ylabel("Empirical Accuracy", fontsize=9)
            ax.legend(loc="upper left", fontsize=8, frameon=True)
            ax.grid(True, linestyle=":", alpha=0.6)

        plt.suptitle("Reliability Diagrams (Calibration Curves) across 10 Frameworks", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    def plot_ece_vs_latency_bars(
        self,
        summary_df: pd.DataFrame,
        save_name: str = "ece_latency_comparison.png",
    ) -> Path:
        """Generate side-by-side bar plots comparing Expected Calibration Error (ECE) and Latency (ms)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        df_sorted_ece = summary_df.sort_values(by="ECE", ascending=True)
        sns.barplot(
            data=df_sorted_ece,
            x="ECE",
            y="Framework Method",
            hue="Framework Method",
            legend=False,
            palette="Blues_r",
            ax=ax1,
        )
        ax1.set_title("Expected Calibration Error (ECE) - Lower is Better", fontsize=12, fontweight="bold")
        ax1.set_xlabel("ECE", fontsize=10)
        ax1.set_ylabel("")
        for p in ax1.patches:
            width = p.get_width()
            ax1.annotate(f"{width:.3f}", (width + 0.005, p.get_y() + p.get_height() / 2.),
                         ha="left", va="center", fontsize=9)

        df_sorted_lat = summary_df.sort_values(by="Mean Latency (ms)", ascending=True)
        sns.barplot(
            data=df_sorted_lat,
            x="Mean Latency (ms)",
            y="Framework Method",
            hue="Framework Method",
            legend=False,
            palette="Oranges_r",
            ax=ax2,
        )
        ax2.set_title("Mean Latency (ms) - Lower is Better", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Latency (ms)", fontsize=10)
        ax2.set_ylabel("")
        for p in ax2.patches:
            width = p.get_width()
            ax2.annotate(f"{width:.1f} ms", (width + 10, p.get_y() + p.get_height() / 2.),
                         ha="left", va="center", fontsize=9)

        plt.suptitle("ECE & Processing Latency Tradeoff across 10 Frameworks", fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    def plot_radar_comparison(
        self,
        summary_df: pd.DataFrame,
        save_name: str = "radar_metrics_comparison.png",
    ) -> Path:
        """Generate a Radar Chart comparing Accuracy, 1-ECE, 1-Brier, and Automation Ratio across frameworks."""
        categories = ["Accuracy", "Calibration (1-ECE)", "Scoring Quality (1-Brier)", "Automation Ratio"]
        num_vars = len(categories)

        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

        for idx, row in summary_df.iterrows():
            method_name = row["Framework Method"]
            values = [
                row["Accuracy"],
                max(0.0, 1.0 - row["ECE"]),
                max(0.0, 1.0 - row["Brier Score"]),
                row["Automation Ratio"],
            ]
            values += values[:1]

            ax.plot(angles, values, linewidth=1.5, linestyle="solid", label=method_name, color=self.colors[idx % 10])
            ax.fill(angles, values, color=self.colors[idx % 10], alpha=0.08)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], categories, size=10, fontweight="bold")
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
        plt.ylim(0, 1.0)

        plt.title("Multi-Metric Radar Comparison across 10 Calibration Frameworks", size=14, fontweight="bold", y=1.08)
        plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path
