"""
CLI Runner for Data Splitting.
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.split_data import ObjectDetectionDataSplitter, generate_splitting_report

def main():
    parser = argparse.ArgumentParser(description="Split dataset into Train, Val, Test with stratification.")
    parser.add_argument("--src-root", type=str, default="data/processed", help="Source preprocessed data directory")
    parser.add_argument("--dst-root", type=str, default="data/split", help="Destination split data directory")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (default 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio (default 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (default 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--report", type=str, default="reports/splitting/data_splitting_report.md", help="Report output path")

    args = parser.parse_args()

    splitter = ObjectDetectionDataSplitter(
        src_root=args.src_root,
        dst_root=args.dst_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    summary = splitter.execute_split()
    generate_splitting_report(summary, args.report)

if __name__ == "__main__":
    main()
