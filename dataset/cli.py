"""
Command-Line Interface for Preference Dataset Analyzer.
"""

import os
import json
import argparse
from typing import List
from dataset.schema import PreferenceItem
from dataset.metrics import MetricsCalculator
from dataset.report_generator import ReportGenerator
from dataset.visualizer import Visualizer


def load_preference_dataset(file_path: str) -> List[PreferenceItem]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    items: List[PreferenceItem] = []
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    items.append(PreferenceItem.from_dict(data))
        else:
            data_list = json.load(f)
            for data in data_list:
                items.append(PreferenceItem.from_dict(data))
    return items


def main():
    parser = argparse.ArgumentParser(description="Human Preference Dataset Analyzer CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze preference dataset")
    analyze_parser.add_argument("--input", "-i", required=True, help="Path to preference dataset (.jsonl or .json)")
    analyze_parser.add_argument("--output-dir", "-o", default="results", help="Directory to export JSON/CSV results")
    analyze_parser.add_argument("--report", "-r", default="reports/summary_report.md", help="Path to generate markdown report")
    analyze_parser.add_argument("--figures-dir", "-f", default="reports/figures", help="Directory to save figures")

    args = parser.parse_args()

    if args.command == "analyze":
        print(f"Loading preference dataset from: {args.input}")
        items = load_preference_dataset(args.input)
        print(f"Loaded {len(items)} preference prompt pairs.")

        print("Computing statistical inter-annotator agreement and preference distribution metrics...")
        metrics = MetricsCalculator.analyze_dataset(items)
        print(f" Fleiss' Kappa (κ): {metrics['fleiss_kappa']:.4f}")
        print(f" Raw Agreement Rate: {metrics['raw_agreement_rate']*100:.1f}%")
        print(f" Choice Distribution: {metrics['choice_percentages']}")

        print("\nExporting results to JSON and CSV...")
        ReportGenerator.export_results(items, args.output_dir)

        print("\nGenerating visualization charts...")
        Visualizer.generate_all_figures(metrics, args.figures_dir)

        print(f"\nGenerating Markdown summary report at: {args.report}")
        ReportGenerator.generate_markdown_report(metrics, args.report)

        print("\nDataset Preference Analysis Completed Successfully!")


if __name__ == "__main__":
    main()
