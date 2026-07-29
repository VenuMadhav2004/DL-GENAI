"""
inference.py — Load the trained BiLSTM + tokenizer and produce submission.csv.

Reads test.csv, scores the 5 options per question, ranks the top-3, and writes
a Kaggle-format submission (ID, Prediction) validated against sample_submission.csv.

Usage:
    python src/inference.py \
        --data-dir ./data \
        --model-dir ./models \
        --out submission.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from tokenizers import Tokenizer
from utils import OPTIONS, BiLSTM_Classifier, MCQDataset, clean_dataframe

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--model-dir", default="./models")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    # Load artifacts
    ckpt = torch.load(os.path.join(args.model_dir, "bilstm_final.pt"),
                      map_location=DEVICE)
    tokenizer = Tokenizer.from_file(os.path.join(args.model_dir, "bpe_tokenizer.json"))
    model = BiLSTM_Classifier(ckpt["vocab_size"], ckpt["pad_id"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Load + clean test
    test = clean_dataframe(pd.read_csv(os.path.join(args.data_dir, "test.csv")))
    test_ds = MCQDataset(test, tokenizer, is_test=True)

    # Score every question
    test_scores = {}
    with torch.no_grad():
        for batch in DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attn_mask"].to(DEVICE)
            with autocast():
                logits = model(ids, mask)
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()
            for j, qid in enumerate(batch["id"].numpy()):
                test_scores[int(qid)] = probs[j]

    # Build submission
    rows = []
    for _, row in test.iterrows():
        qid = row["id"]
        ranked = [OPTIONS[k] for k in np.argsort(test_scores[qid])[::-1]][:3]
        rows.append({"ID": qid, "Prediction": " ".join(ranked)})
    submission = pd.DataFrame(rows)

    # Validate format against the sample submission if available
    sample_path = os.path.join(args.data_dir, "sample_submission.csv")
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        assert list(submission.columns) == list(sample.columns), "Column mismatch"
        assert len(submission) == len(sample), "Row mismatch"
        assert submission["Prediction"].str.match(r"^[A-E] [A-E] [A-E]$").all(), \
            "Prediction format error"

    submission.to_csv(args.out, index=False)
    print(f"Saved {args.out} — {len(submission)} rows")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
