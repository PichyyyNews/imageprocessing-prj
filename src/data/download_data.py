"""
Kaggle Dataset Downloader Module
"""
import os
import sys
import shutil
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Try importing dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import kagglehub


def setup_kaggle_credentials():
    """Setup Kaggle credentials from environment variables if present."""
    token = os.getenv("KAGGLE_API_TOKEN") or os.getenv("KGAT_TOKEN")
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
        try:
            kagglehub.login()
        except Exception:
            pass


def download_and_setup_dataset(
    dataset_handle: str = "ashfakyeafi/road-vehicle-images-dataset",
    destination_dir: str = "data/raw"
):
    """
    Download dataset using kagglehub and copy into the local project data directory.
    """
    print("=" * 60)
    print(f"[+] Downloading dataset: {dataset_handle}")
    print("=" * 60)

    setup_kaggle_credentials()

    # Download latest version of the dataset
    cache_path = kagglehub.dataset_download(dataset_handle)
    print(f"[OK] KaggleHub cache path: {cache_path}")

    # Determine source directory
    src_dir = Path(cache_path)
    if (src_dir / "trafic_data").exists():
        src_data = src_dir / "trafic_data"
    else:
        src_data = src_dir

    # Target directory inside project
    target_path = Path(destination_dir).resolve()
    print(f"[+] Copying dataset to project path: {target_path} ...")
    target_path.mkdir(parents=True, exist_ok=True)

    shutil.copytree(src_data, target_path, dirs_exist_ok=True)
    print(f"[OK] Dataset successfully copied to: {target_path}")

    # Print dataset statistics
    print_dataset_summary(target_path)
    return target_path


def print_dataset_summary(data_path: Path):
    """Print summary of the downloaded dataset."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    yaml_files = list(data_path.glob("*.yaml"))
    if yaml_files:
        print(f"Config file: {yaml_files[0].name}")
        try:
            import yaml
            with open(yaml_files[0], "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                names = cfg.get("names", [])
                print(f"Classes ({len(names)}): {names}")
        except Exception as e:
            print(f"Could not parse YAML: {e}")

    train_img_dir = data_path / "train" / "images"
    valid_img_dir = data_path / "valid" / "images"

    train_count = len(list(train_img_dir.glob("*.*"))) if train_img_dir.exists() else 0
    valid_count = len(list(valid_img_dir.glob("*.*"))) if valid_img_dir.exists() else 0

    print(f"Train Images: {train_count}")
    print(f"Validation Images: {valid_count}")
    print(f"Total Images: {train_count + valid_count}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kaggle Road Vehicle Dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ashfakyeafi/road-vehicle-images-dataset",
        help="Kaggle dataset handle (owner/dataset-name)"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="data/raw",
        help="Destination directory inside project (default: data/raw)"
    )

    args = parser.parse_args()
    download_and_setup_dataset(args.dataset, args.dest)
