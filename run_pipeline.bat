@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ===============================================================================
echo        🚗 ROAD VEHICLE IMAGE PROCESSING END-TO-END PIPELINE 🚗
echo ===============================================================================
echo.

:: 1. Check Python installation
echo [STEP 0/4] Checking Python Environment...
python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.8+!
    pause
    exit /b 1
)
python --version
echo [OK] Python is available.
echo.

:: Install / verify dependencies
echo [*] Checking dependencies from requirements.txt...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [WARNING] Failed to automatically install some requirements. Continuing...
) else (
    echo [OK] Dependencies verified.
)
echo.

:: Step 1: Download dataset
echo ===============================================================================
echo [STEP 1/4] Downloading Road Vehicle Dataset from Kaggle Hub...
echo ===============================================================================
python download_data.py
if errorlevel 1 (
    echo [ERROR] Dataset download failed. Please check network connection and API token.
    pause
    exit /b 1
)
echo.

:: Step 2: Exploratory Data Analysis
echo ===============================================================================
echo [STEP 2/4] Running Comprehensive Exploratory Data Analysis (EDA)...
echo ===============================================================================
python src/eda/run_eda.py --data-root data/raw --output-report reports/eda/eda_report.md
if errorlevel 1 (
    echo [ERROR] EDA analysis failed.
    pause
    exit /b 1
)
echo [OK] EDA Report generated at reports\eda\eda_report.md
echo.

:: Step 3: Image Preprocessing
echo ===============================================================================
echo [STEP 3/4] Running Image Preprocessing & BBox Transformation Pipeline...
echo ===============================================================================
python src/preprocessing/run_preprocessing.py --src-root data/raw --dst-root data/processed --width 640 --height 640 --report reports/preprocessing/preprocessing_report.md
if errorlevel 1 (
    echo [ERROR] Preprocessing failed.
    pause
    exit /b 1
)
echo [OK] Preprocessing Report generated at reports\preprocessing\preprocessing_report.md
echo.

:: Step 4: Stratified Data Splitting
echo ===============================================================================
echo [STEP 4/4] Running Multi-Label Stratified Data Splitting (70/15/15)...
echo ===============================================================================
python src/data/run_split.py --src-root data/processed --dst-root data/split --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42 --report reports/splitting/data_splitting_report.md
if errorlevel 1 (
    echo [ERROR] Data splitting failed.
    pause
    exit /b 1
)
echo [OK] Data Splitting Report generated at reports\splitting\data_splitting_report.md
echo.

echo ===============================================================================
echo 🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉
echo ===============================================================================
echo - Processed Dataset: data\split\ (Train / Val / Test)
echo - Master YAML Config: data\split\dataset_split.yaml
echo - Reports Folder:    reports\
echo.
pause
