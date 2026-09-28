"""
Unit tests for agreement metrics and statistical functions.
"""

import pytest
import numpy as np
from dataset.schema import PreferenceItem, PreferenceChoice, AnnotationRecord
from dataset.metrics import MetricsCalculator


def test_cohens_kappa_perfect_agreement():
    r1 = ["A", "B", "A", "tie"]
    r2 = ["A", "B", "A", "tie"]
    kappa = MetricsCalculator.compute_cohens_kappa(r1, r2)
    assert kappa == 1.0


def test_cohens_kappa_chance_agreement():
    r1 = ["A", "A", "B", "B"]
    r2 = ["B", "B", "A", "A"]
    kappa = MetricsCalculator.compute_cohens_kappa(r1, r2)
    assert kappa < 0.0


def test_fleiss_kappa_perfect_agreement():
    # 3 subjects, 3 categories (A, B, tie), 2 annotators per subject
    # All 2 annotators choose category 0 (A) for item 1, category 1 (B) for item 2, category 2 (tie) for item 3
    matrix = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2]
    ])
    kappa = MetricsCalculator.compute_fleiss_kappa(matrix)
    assert kappa == 1.0


def test_dataset_analysis_pipeline():
    item1 = PreferenceItem.from_dict({
        "id": "item_1",
        "prompt": "Prompt 1",
        "response_a": "A1",
        "response_b": "B1",
        "domain": "Code",
        "annotations": [
            {"annotator_id": "ann_1", "preference": "A", "rationale": "Good", "confidence": 5},
            {"annotator_id": "ann_2", "preference": "A", "rationale": "Good", "confidence": 5}
        ]
    })

    metrics = MetricsCalculator.analyze_dataset([item1])
    assert metrics["total_items"] == 1
    assert metrics["fleiss_kappa"] == 1.0
    assert metrics["raw_agreement_rate"] == 1.0
    assert metrics["choice_counts"]["A"] == 2
