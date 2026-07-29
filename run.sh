#!/usr/bin/env bash
# run.sh — Smart MCQ Solver: install -> train -> generate submission.
# Usage:  bash run.sh
# Run this from the repo root (the folder that contains src/ and data/).

set -e  # stop immediately if any command fails

DATA_DIR="./data"
MODEL_DIR="./models"
SUBMISSION="submission.csv"

echo "=================================================="
echo " Smart MCQ Solver — full pipeline"
echo "=================================================="

# --- 0. Check the data is present ---------------------------------
for f in train.csv test.csv sample_submission.csv; do
  if [ ! -f "$DATA_DIR/$f" ]; then
    echo "ERROR: $DATA_DIR/$f not found."
    echo "Put train.csv, test.csv and sample_submission.csv inside $DATA_DIR/ first."
    exit 1
  fi
done

# --- 1. Install dependencies --------------------------------------
echo ""
echo "[1/3] Installing dependencies..."
pip install -q -r requirements.txt

# --- 2. Train -----------------------------------------------------
echo ""
echo "[2/3] Training BiLSTM..."
python src/train.py --data-dir "$DATA_DIR" --out-dir "$MODEL_DIR"

# --- 3. Inference -------------------------------------------------
echo ""
echo "[3/3] Generating submission..."
python src/inference.py --data-dir "$DATA_DIR" --model-dir "$MODEL_DIR" --out "$SUBMISSION"

echo ""
echo "=================================================="
echo " Done. Submission written to $SUBMISSION"
echo "=================================================="