@echo off
REM run.bat - Smart MCQ Solver: install -> train -> generate submission.
REM Usage:  run.bat
REM Run this from the repo root (the folder that contains src\ and data\).

setlocal

set DATA_DIR=.\data
set MODEL_DIR=.\models
set SUBMISSION=submission.csv

echo ==================================================
echo  Smart MCQ Solver - full pipeline
echo ==================================================

REM --- 0. Check the data is present ---------------------------------
for %%f in (train.csv test.csv sample_submission.csv) do (
    if not exist "%DATA_DIR%\%%f" (
        echo ERROR: %DATA_DIR%\%%f not found.
        echo Put train.csv, test.csv and sample_submission.csv inside %DATA_DIR%\ first.
        exit /b 1
    )
)

REM --- 1. Install dependencies --------------------------------------
echo.
echo [1/3] Installing dependencies...
pip install -q -r requirements.txt
if errorlevel 1 exit /b 1

REM --- 2. Train -----------------------------------------------------
echo.
echo [2/3] Training BiLSTM...
python src\train.py --data-dir "%DATA_DIR%" --out-dir "%MODEL_DIR%"
if errorlevel 1 exit /b 1

REM --- 3. Inference -------------------------------------------------
echo.
echo [3/3] Generating submission...
python src\inference.py --data-dir "%DATA_DIR%" --model-dir "%MODEL_DIR%" --out "%SUBMISSION%"
if errorlevel 1 exit /b 1

echo.
echo ==================================================
echo  Done. Submission written to %SUBMISSION%
echo ==================================================

endlocal