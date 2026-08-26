"""
Root entrypoint for dataset downloader.
Delegates to src.data.download_data.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.download_data import download_and_setup_dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Kaggle Road Vehicle Dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ashfakyeafi/road-vehicle-images-dataset",
        help="Kaggle dataset handle"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="data/raw",
        help="Destination directory inside project"
    )
    args = parser.parse_args()
    download_and_setup_dataset(args.dataset, args.dest)
