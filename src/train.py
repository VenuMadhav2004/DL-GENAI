"""
train.py — Train the BiLSTM MCQ solver (the model used for the Kaggle submission).

Pipeline:
  1. load + clean data
  2. leakage-free StratifiedGroupKFold split (duplicates grouped by option-fingerprint)
  3. train BPE tokenizer on the TRAINING SPLIT only
  4. train BiLSTM with early stopping on validation MAP@3
  5. refit on 100% of the data for the best number of epochs
  6. save model checkpoint + tokenizer to models/

Usage:
    python src/train.py --data-dir ./data --out-dir ./models

WandB is optional and off by default. Enable with --wandb (set WANDB_API_KEY
as an env var — never hardcode it).
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold

from utils import (
    OPTIONS, BiLSTM_Classifier, MCQDataset, build_bpe_tokenizer,
    clean_dataframe, evaluate_map3, filter_text,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
CONFIG = {"max_epochs": 25, "patience": 4, "lr": 1e-3, "weight_decay": 1e-2}


def make_split(train):
    """StratifiedGroupKFold: duplicate questions stay on one side, labels balanced."""
    fingerprint = train.apply(
        lambda r: "|||".join(str(r[o]).strip() for o in OPTIONS), axis=1
    )
    groups = fingerprint.factorize()[0]
    y = train["answer"].map({o: i for i, o in enumerate(OPTIONS)}).values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(sgkf.split(train, y, groups))
    return train_idx, val_idx


def run_epoch(model, loader, optimizer, criterion, scaler, train=True):
    model.train() if train else model.eval()
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attn_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            if train:
                optimizer.zero_grad()
            with autocast():
                logits = model(ids, mask)
                loss = criterion(logits, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def validate_map3(model, loader):
    model.eval()
    vp, vt = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attn_mask"].to(DEVICE)
            with autocast():
                logits = model(ids, mask)
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()
            for j, lbl in enumerate(batch["label"].numpy()):
                vp.append([OPTIONS[k] for k in np.argsort(probs[j])[::-1]][:3])
                vt.append(OPTIONS[lbl])
    return evaluate_map3(vt, vp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data",
                    help="folder containing train.csv / test.csv")
    ap.add_argument("--out-dir", default="./models")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="smart-mcq-solver", name="bilstm",
                         config=CONFIG)  # key comes from WANDB_API_KEY env var

    # 1. Load + clean
    train = clean_dataframe(pd.read_csv(os.path.join(args.data_dir, "train.csv")))
    print(f"Train: {train.shape}")

    # 2. Leakage-free split
    train_idx, val_idx = make_split(train)
    print(f"Split -> train {len(train_idx)}, val {len(val_idx)}")

    # 3. Tokenizer on training split only
    corpus = []
    for i in train_idx:
        row = train.iloc[i]
        corpus.append(filter_text(row["prompt"]))
        corpus.extend(filter_text(row[o]) for o in OPTIONS)
    tokenizer = build_bpe_tokenizer(corpus)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<PAD>")
    print(f"BPE vocab: {vocab_size}")

    ds = MCQDataset(train, tokenizer)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=128,
                            shuffle=False, num_workers=2)

    # 4. Train with early stopping on val MAP@3
    model = BiLSTM_Classifier(vocab_size, pad_id).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                                  weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_val, patience_ctr, best_epoch = 0.0, 0, 1
    history = []
    best_path = os.path.join(args.out_dir, "bilstm_best.pt")

    for epoch in range(CONFIG["max_epochs"]):
        loss, acc = run_epoch(model, train_loader, optimizer, criterion,
                              scaler, train=True)
        val_map3 = validate_map3(model, val_loader)
        scheduler.step(val_map3)
        history.append(val_map3)
        print(f"Epoch {epoch+1:2d} | loss {loss:.4f} | acc {acc:.4f} | "
              f"val MAP@3 {val_map3:.4f}")
        if run:
            run.log({"loss": loss, "acc": acc, "val_map3": val_map3})
        if val_map3 > best_val:
            best_val, patience_ctr, best_epoch = val_map3, 0, epoch + 1
            torch.save(model.state_dict(), best_path)
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest val MAP@3 (honest, leakage-free): {best_val:.4f} @ epoch {best_epoch}")

    # 5. Refit on 100% of the data for best_epoch epochs
    print(f"Refitting on full data for {best_epoch} epochs...")
    model = BiLSTM_Classifier(vocab_size, pad_id).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                                  weight_decay=CONFIG["weight_decay"])
    scaler = GradScaler()
    full_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    for epoch in range(best_epoch):
        loss, acc = run_epoch(model, full_loader, optimizer, criterion,
                              scaler, train=True)
        print(f"Refit {epoch+1}/{best_epoch} | loss {loss:.4f} | acc {acc:.4f}")

    # 6. Save final artifacts
    final_path = os.path.join(args.out_dir, "bilstm_final.pt")
    torch.save({"state_dict": model.state_dict(),
                "vocab_size": vocab_size, "pad_id": pad_id}, final_path)
    tokenizer.save(os.path.join(args.out_dir, "bpe_tokenizer.json"))
    print(f"\nSaved -> {final_path} and bpe_tokenizer.json")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
