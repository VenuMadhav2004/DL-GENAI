"""
utils.py — Shared utilities for the Smart MCQ Solver project.

Contains everything used by both train.py and inference.py:
  - data cleaning / text normalization
  - the BPE tokenizer builder + encoding
  - the MCQDataset
  - the BiLSTM model
  - MAP@3 and evaluation metrics

All logic is lifted directly from the project notebook so results reproduce.
"""

import re
import string

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, Lowercase, NFD, StripAccents

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
OPTIONS = ["A", "B", "C", "D", "E"]
NN_MAX_LEN = 200          # subword sequences are longer than word sequences
TEXT_COLS = ["prompt", "A", "B", "C", "D", "E"]


# --------------------------------------------------------------------------- #
# 1. Data cleaning
# --------------------------------------------------------------------------- #
def clean_dataframe(df):
    """Defensive cleaning: fill NaNs, coerce to str, normalize whitespace."""
    df = df.copy()
    for col in TEXT_COLS:
        df[col] = df[col].fillna("").astype(str).str.strip()
        df[col] = df[col].str.replace(r"\s+", " ", regex=True)
        df[col] = df[col].replace(
            {"nan": "", "None": "", "null": "", "N/A": "", "NA": ""}, regex=False
        )
        df.loc[df[col] == "", col] = "[no text]"
    if "answer" in df.columns:
        df["answer"] = df["answer"].str.strip().str.upper()
        assert df["answer"].isin(OPTIONS).all(), "Invalid answer values!"
    return df


def filter_text(text):
    """Consistent text filtering applied identically to train / val / test."""
    text = str(text).strip()
    text = " ".join(text.split())
    return text if text else "[empty]"


def normalize_text(text):
    """Lowercase, strip punctuation, collapse whitespace."""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# 2. MAP@3 + evaluation metrics
# --------------------------------------------------------------------------- #
def map_at_3(truth, preds):
    score, hits = 0.0, 0
    for i, p in enumerate(preds[:3]):
        if p == truth:
            hits += 1
            score += hits / (i + 1)
    return score


def evaluate_map3(truths, all_preds):
    return float(np.mean([map_at_3(t, p) for t, p in zip(truths, all_preds)]))


def probs_to_ranked(prob_matrix):
    """(n, 5) probability matrix -> list of top-3 letter lists."""
    return [[OPTIONS[k] for k in np.argsort(p)[::-1]][:3] for p in prob_matrix]


def scores_to_preds(id_scores_dict, id_list):
    preds = []
    for qid in id_list:
        scores = id_scores_dict[qid]
        preds.append([OPTIONS[i] for i in np.argsort(scores)[::-1]][:3])
    return preds


def evaluate_model(val_truths, val_ranked):
    """Accuracy + macro-F1 (rank-1) + MAP@3."""
    from sklearn.metrics import accuracy_score, f1_score
    top1 = [r[0] for r in val_ranked]
    return {
        "accuracy": accuracy_score(val_truths, top1),
        "macro_f1": f1_score(val_truths, top1, average="macro",
                             labels=OPTIONS, zero_division=0),
        "map3": evaluate_map3(val_truths, val_ranked),
    }


# --------------------------------------------------------------------------- #
# 3. BPE tokenizer
# --------------------------------------------------------------------------- #
def build_bpe_tokenizer(corpus, vocab_size=8000, min_frequency=2):
    """Train a BPE tokenizer on a list of texts (training split only)."""
    tok = Tokenizer(BPE(unk_token="<UNK>"))
    tok.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>", "<SEP>"],
        min_frequency=min_frequency,
    )
    tok.train_from_iterator(corpus, trainer)
    return tok


def encode_pair(tokenizer, prompt, option_text, max_len=NN_MAX_LEN):
    """filter -> BPE encode -> truncate -> pad. Returns (ids, attn_mask)."""
    pad_id = tokenizer.token_to_id("<PAD>")
    text = filter_text(prompt) + " <SEP> " + filter_text(option_text)
    ids = tokenizer.encode(text).ids[:max_len]
    attn = [1] * len(ids) + [0] * (max_len - len(ids))
    ids = ids + [pad_id] * (max_len - len(ids))
    return ids, attn


# --------------------------------------------------------------------------- #
# 4. Dataset
# --------------------------------------------------------------------------- #
class MCQDataset(Dataset):
    """One question per item -> 5 encoded (prompt + option) pairs."""

    def __init__(self, df, tokenizer, max_len=NN_MAX_LEN, is_test=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.items = []
        for _, row in df.iterrows():
            item = {"id": row["id"], "prompt": row["prompt"],
                    "options": {o: row[o] for o in OPTIONS}}
            if not is_test:
                item["answer"] = row["answer"]
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        all_ids, all_masks = [], []
        for o in OPTIONS:
            ids, mask = encode_pair(self.tokenizer, item["prompt"],
                                    item["options"][o], self.max_len)
            all_ids.append(ids)
            all_masks.append(mask)
        result = {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attn_mask": torch.tensor(all_masks, dtype=torch.float),
            "id": item["id"],
        }
        if "answer" in item:
            result["label"] = torch.tensor(ord(item["answer"]) - ord("A"),
                                           dtype=torch.long)
        return result


# --------------------------------------------------------------------------- #
# 5. Model
# --------------------------------------------------------------------------- #
class BiLSTM_Classifier(nn.Module):
    """
    Embedding (64d) -> BiLSTM (1 layer, bidirectional) -> masked mean-pool
    -> Dropout -> Dense(128->64) -> ReLU -> Dropout -> Dense(64->1).
    Each (prompt, option) pair gets one score; 5 scores -> softmax over options.
    """

    def __init__(self, vocab_size, pad_id, embed_dim=64, hidden_dim=64,
                 num_layers=1, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward_one(self, ids, mask):
        emb = self.embedding(ids)
        lstm_out, _ = self.lstm(emb)
        mask_exp = mask.unsqueeze(-1)
        summed = (lstm_out * mask_exp).sum(dim=1)
        counts = mask_exp.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        h = self.dropout(pooled)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h).squeeze(-1)

    def forward(self, input_ids, attn_mask):
        scores = [self.forward_one(input_ids[:, i], attn_mask[:, i])
                  for i in range(5)]
        return torch.stack(scores, dim=1)
