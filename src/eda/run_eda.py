"""
CLI Runner for Exploratory Data Analysis.
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.eda.eda_analyzer import DatasetEDAAnalyzer, generate_eda_report_markdown

def main():
    parser = argparse.ArgumentParser(description="Run Exploratory Data Analysis on Object Detection Dataset")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw",
        help="Path to raw dataset directory"
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/eda/eda_report.md",
        help="Path to save markdown EDA report"
    )

    args = parser.parse_args()
    analyzer = DatasetEDAAnalyzer(args.data_root)
    results = analyzer.analyze()
    generate_eda_report_markdown(results, args.output_report)

if __name__ == "__main__":
    main()
