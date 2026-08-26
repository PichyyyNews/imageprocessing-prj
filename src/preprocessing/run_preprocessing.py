"""
CLI Runner for Image Preprocessing Pipeline.
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.preprocessing.preprocessor import ImagePreprocessor, generate_preprocessing_report

def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset with letterbox resizing and bbox transformations.")
    parser.add_argument("--src-root", type=str, default="data/raw", help="Source raw data path")
    parser.add_argument("--dst-root", type=str, default="data/processed", help="Destination processed data path")
    parser.add_argument("--width", type=int, default=640, help="Target image width")
    parser.add_argument("--height", type=int, default=640, help="Target image height")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE contrast enhancement")
    parser.add_argument("--report", type=str, default="reports/preprocessing/preprocessing_report.md", help="Output report path")

    args = parser.parse_args()

    preprocessor = ImagePreprocessor(
        target_size=(args.width, args.height),
        apply_clahe=args.clahe
    )
    summary = preprocessor.process_dataset(args.src_root, args.dst_root)
    generate_preprocessing_report(summary, args.report)

if __name__ == "__main__":
    main()
