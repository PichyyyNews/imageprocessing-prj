"""
End-to-End Master Pipeline Runner.
Executes dataset collection, EDA, preprocessing, and stratified splitting sequentially.
"""
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.download_data import download_and_setup_dataset
from src.eda.eda_analyzer import DatasetEDAAnalyzer, generate_eda_report_markdown
from src.preprocessing.preprocessor import ImagePreprocessor, generate_preprocessing_report
from src.data.split_data import ObjectDetectionDataSplitter, generate_splitting_report


def run_full_pipeline(
    dataset_handle: str = "ashfakyeafi/road-vehicle-images-dataset",
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    split_dir: str = "data/split",
    target_size: tuple = (640, 640),
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    start_time = time.time()
    print("=" * 80)
    print("🚀 STARTING FULL END-TO-END IMAGE PROCESSING PIPELINE 🚀")
    print("=" * 80)

    # 1. Download Dataset
    print("\n[STEP 1/4] 📦 Dataset Acquisition")
    print("-" * 80)
    download_and_setup_dataset(dataset_handle=dataset_handle, destination_dir=raw_dir)

    # 2. Exploratory Data Analysis
    print("\n[STEP 2/4] 📊 Exploratory Data Analysis (EDA)")
    print("-" * 80)
    eda_analyzer = DatasetEDAAnalyzer(data_root=raw_dir)
    eda_results = eda_analyzer.analyze()
    generate_eda_report_markdown(eda_results, output_file="reports/eda/eda_report.md")

    # 3. Image Preprocessing & Bounding Box Recalculation
    print("\n[STEP 3/4] 🖼️ Image Preprocessing & Coordinate Transformation")
    print("-" * 80)
    preprocessor = ImagePreprocessor(target_size=target_size)
    preproc_summary = preprocessor.process_dataset(src_root=raw_dir, dst_root=processed_dir)
    generate_preprocessing_report(preproc_summary, output_path="reports/preprocessing/preprocessing_report.md")

    # 4. Multi-Label Stratified Splitting
    print("\n[STEP 4/4] ✂️ Multi-Label Stratified Splitting (Train/Val/Test)")
    print("-" * 80)
    splitter = ObjectDetectionDataSplitter(
        src_root=processed_dir,
        dst_root=split_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    split_summary = splitter.execute_split()
    generate_splitting_report(split_summary, output_file="reports/splitting/data_splitting_report.md")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s! 🎉")
    print("=" * 80)
    print("📂 Artifacts & Output Summary:")
    print(f"  - Raw Dataset:            {raw_dir}/")
    print(f"  - Preprocessed Dataset:   {processed_dir}/ (640x640 letterbox)")
    print(f"  - Final Splits:           {split_dir}/ (Train 70%, Val 15%, Test 15%)")
    print(f"  - Master Dataset YAML:    {split_dir}/dataset_split.yaml")
    print(f"  - EDA Report:             reports/eda/eda_report.md")
    print(f"  - Preprocessing Report:   reports/preprocessing/preprocessing_report.md")
    print(f"  - Data Splitting Report:  reports/splitting/data_splitting_report.md")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Full End-to-End Image Processing Pipeline")
    parser.add_argument("--dataset", type=str, default="ashfakyeafi/road-vehicle-images-dataset")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--split-dir", type=str, default="data/split")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_full_pipeline(
        dataset_handle=args.dataset,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        split_dir=args.split_dir,
        target_size=(args.width, args.height),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
