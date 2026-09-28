"""
Inter-Annotator Agreement & Statistical Preference Metrics Module.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataset.schema import PreferenceItem, PreferenceChoice, AnnotationRecord


class MetricsCalculator:
    """Calculates statistical metrics for human preference datasets."""

    @staticmethod
    def compute_fleiss_kappa(matrix: np.ndarray) -> float:
        """Compute Fleiss' Kappa for a category assignment matrix N x K.

        Args:
            matrix: N x K numpy array where N is number of subjects, K is categories (A, B, tie).
                    Values in matrix are counts of annotators assigning subject i to category j.
        """
        N, K = matrix.shape
        n = np.sum(matrix[0, :])  # Number of ratings per subject
        if n <= 1:
            return 1.0

        p = np.sum(matrix, axis=0) / (N * n)
        P_i = (np.sum(matrix**2, axis=1) - n) / (n * (n - 1))
        P_bar = np.mean(P_i)
        P_e_bar = np.sum(p**2)

        if P_e_bar == 1.0:
            return 1.0

        kappa = (P_bar - P_e_bar) / (1.0 - P_e_bar)
        return float(kappa)

    @staticmethod
    def compute_cohens_kappa(r1: List[str], r2: List[str]) -> float:
        """Compute Cohen's Kappa between two annotators."""
        if len(r1) != len(r2) or len(r1) == 0:
            return 0.0

        categories = sorted(list(set(r1).union(set(r2))))
        cat_map = {c: i for i, c in enumerate(categories)}
        k = len(categories)

        cm = np.zeros((k, k), dtype=int)
        for a, b in zip(r1, r2):
            cm[cat_map[a], cat_map[b]] += 1

        total = len(r1)
        po = np.trace(cm) / total
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total**2)

        if pe == 1.0:
            return 1.0

        kappa = (po - pe) / (1.0 - pe)
        return float(kappa)

    @classmethod
    def analyze_dataset(cls, items: List[PreferenceItem]) -> Dict[str, Any]:
        if not items:
            return {"total_items": 0, "fleiss_kappa": 0.0, "raw_agreement": 0.0}

        total_items = len(items)
        categories = ["A", "B", "tie"]
        cat_map = {"A": 0, "B": 1, "tie": 2}

        # Preference distribution (consensus or overall)
        all_choices = []
        rating_matrix = []
        raw_agree_count = 0
        total_multi_ann = 0

        annotator_stats: Dict[str, Dict[str, Any]] = {}
        error_tags_a: Dict[str, int] = {}
        error_tags_b: Dict[str, int] = {}
        domain_counts: Dict[str, int] = {}

        for item in items:
            domain_counts[item.domain] = domain_counts.get(item.domain, 0) + 1
            counts = [0, 0, 0]
            item_choices = []

            for ann in item.annotations:
                p_val = ann.preference.value if isinstance(ann.preference, PreferenceChoice) else ann.preference
                all_choices.append(p_val)
                item_choices.append(p_val)
                counts[cat_map[p_val]] += 1

                # Track error tags
                for tag in ann.error_tags_a:
                    error_tags_a[tag] = error_tags_a.get(tag, 0) + 1
                for tag in ann.error_tags_b:
                    error_tags_b[tag] = error_tags_b.get(tag, 0) + 1

                # Track annotator consistency
                if ann.annotator_id not in annotator_stats:
                    annotator_stats[ann.annotator_id] = {"count": 0, "conf_sum": 0, "is_simulated": ann.is_simulated}
                annotator_stats[ann.annotator_id]["count"] += 1
                annotator_stats[ann.annotator_id]["conf_sum"] += ann.confidence

            if len(item.annotations) > 1:
                total_multi_ann += 1
                if len(set(item_choices)) == 1:
                    raw_agree_count += 1
                rating_matrix.append(counts)

        # Compute Fleiss' Kappa
        fleiss_k = 0.0
        if rating_matrix:
            mat = np.array(rating_matrix)
            fleiss_k = cls.compute_fleiss_kappa(mat)

        raw_agree_rate = raw_agree_count / total_multi_ann if total_multi_ann > 0 else 1.0

        # Choice frequencies
        choice_counts = {c: all_choices.count(c) for c in categories}
        choice_percentages = {c: round((count / len(all_choices)) * 100, 2) for c, count in choice_counts.items()} if all_choices else {}

        # Annotator summary
        ann_summary = {}
        for ann_id, stats in annotator_stats.items():
            ann_summary[ann_id] = {
                "annotations_count": stats["count"],
                "avg_confidence": round(stats["conf_sum"] / stats["count"], 2),
                "is_simulated": stats["is_simulated"],
            }

        return {
            "total_items": total_items,
            "total_annotations": len(all_choices),
            "fleiss_kappa": round(fleiss_k, 4),
            "raw_agreement_rate": round(raw_agree_rate, 4),
            "choice_counts": choice_counts,
            "choice_percentages": choice_percentages,
            "domain_distribution": domain_counts,
            "error_tags_a": error_tags_a,
            "error_tags_b": error_tags_b,
            "annotators_summary": ann_summary,
        }
