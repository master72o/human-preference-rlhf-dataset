"""
Visualization Module for Human Preference RLHF Dataset.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class Visualizer:
    """Generates charts for dataset metrics and inter-annotator agreement."""

    @staticmethod
    def generate_all_figures(metrics: Dict[str, Any], output_dir: str = "reports/figures") -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        generated = {}

        # 1. Preference Distribution
        pref_path = os.path.join(output_dir, "preference_distribution.png")
        Visualizer._plot_preference_distribution(metrics.get("choice_percentages", {}), pref_path)
        generated["preference_distribution"] = pref_path

        # 2. Inter-Annotator Agreement
        agree_path = os.path.join(output_dir, "annotator_agreement.png")
        Visualizer._plot_annotator_agreement(
            metrics.get("fleiss_kappa", 0.0),
            metrics.get("raw_agreement_rate", 0.0),
            agree_path
        )
        generated["annotator_agreement"] = agree_path

        # 3. Error Tag Frequency
        err_path = os.path.join(output_dir, "error_tag_frequency.png")
        Visualizer._plot_error_tag_frequency(
            metrics.get("error_tags_a", {}),
            metrics.get("error_tags_b", {}),
            err_path
        )
        generated["error_tag_frequency"] = err_path

        return generated

    @staticmethod
    def _plot_preference_distribution(choice_pcts: Dict[str, float], output_path: str):
        fig, ax = plt.subplots(figsize=(8, 5))
        choices = ["A", "B", "tie"]
        pcts = [choice_pcts.get(c, 0.0) for c in choices]
        colors = ["#2b5c8f", "#5cb85c", "#f0ad4e"]

        bars = ax.bar(choices, pcts, color=colors, edgecolor="#333333", width=0.5)
        ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_title("Human Preference Label Distribution (A vs B vs Tie)", fontsize=13, fontweight="bold", pad=15)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5, f"{height:.1f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def _plot_annotator_agreement(fleiss_k: float, raw_agree: float, output_path: str):
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics_names = ["Fleiss' Kappa (κ)", "Raw Agreement Rate"]
        values = [fleiss_k, raw_agree]
        colors = ["#2b5c8f", "#5bc0de"]

        bars = ax.bar(metrics_names, values, color=colors, edgecolor="#333333", width=0.4)
        ax.set_ylim(0.0, 1.1)
        ax.axhline(y=0.70, color="#d9534f", linestyle="--", linewidth=1.5, label="Substantial Agreement Benchmark (0.70)")
        ax.set_ylabel("Score / Rate", fontsize=11, fontweight="bold")
        ax.set_title("Inter-Annotator Agreement Metrics", fontsize=13, fontweight="bold", pad=15)
        ax.legend(loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f"{height:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def _plot_error_tag_frequency(tags_a: Dict[str, int], tags_b: Dict[str, int], output_path: str):
        fig, ax = plt.subplots(figsize=(10, 5))
        all_tags = sorted(list(set(tags_a.keys()).union(set(tags_b.keys()))))
        
        if not all_tags:
            ax.text(0.5, 0.5, "No Error Tags Recorded", ha="center", va="center", fontsize=12)
        else:
            counts_a = [tags_a.get(t, 0) for t in all_tags]
            counts_b = [tags_b.get(t, 0) for t in all_tags]
            x = np.arange(len(all_tags))
            width = 0.35

            rects1 = ax.bar(x - width/2, counts_a, width, label="Response A Errors", color="#d9534f")
            rects2 = ax.bar(x + width/2, counts_b, width, label="Response B Errors", color="#f0ad4e")

            ax.set_ylabel("Error Tag Count", fontsize=11, fontweight="bold")
            ax.set_title("Error Tag Breakdown (Response A vs Response B)", fontsize=13, fontweight="bold", pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(all_tags, rotation=30, ha="right")
            ax.legend()
            ax.grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
