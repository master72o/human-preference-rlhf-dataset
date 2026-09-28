"""
Integration test for CLI execution.
"""

import os
import pytest
from dataset.cli import load_preference_dataset
from dataset.metrics import MetricsCalculator
from dataset.report_generator import ReportGenerator


def test_cli_load_dataset(tmp_path):
    jsonl_file = tmp_path / "pref_test.jsonl"
    jsonl_file.write_text(
        '{"id": "p1", "prompt": "Hi", "response_a": "A", "response_b": "B", "annotations": [{"annotator_id": "ann_1", "preference": "A", "rationale": "Better", "confidence": 5}]}\n',
        encoding="utf-8"
    )

    items = load_preference_dataset(str(jsonl_file))
    assert len(items) == 1
    assert items[0].id == "p1"


def test_full_analysis_pipeline_execution(tmp_path):
    dataset_file = tmp_path / "sample.jsonl"
    dataset_file.write_text(
        '{"id": "p1", "prompt": "Hi", "response_a": "A", "response_b": "B", "annotations": [{"annotator_id": "ann_1", "preference": "A", "rationale": "Better", "confidence": 5}, {"annotator_id": "ann_2", "preference": "A", "rationale": "Better", "confidence": 5}]}\n',
        encoding="utf-8"
    )

    items = load_preference_dataset(str(dataset_file))
    metrics = MetricsCalculator.analyze_dataset(items)

    out_dir = tmp_path / "results"
    report_path = tmp_path / "report.md"

    ReportGenerator.export_results(items, str(out_dir))
    ReportGenerator.generate_markdown_report(metrics, str(report_path))

    assert os.path.exists(out_dir / "preference_analysis_results.json")
    assert os.path.exists(out_dir / "preference_annotations.csv")
    assert os.path.exists(report_path)
