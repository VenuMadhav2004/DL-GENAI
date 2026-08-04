"""
utils.py — Shared utilities for the Smart MCQ Solver project.

Everything used by both train.py and inference.py lives here, so the two scripts
stay short and the logic is defined in exactly one place:
  - data cleaning / text normalization
  - the BPE tokenizer builder + encoding
  - the MCQDataset (turns a dataframe into model-ready tensors)
  - the BiLSTM model
  - MAP@3 and evaluation metrics

All logic is lifted directly from the project notebook so results reproduce exactly.
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
OPTIONS = ["A", "B", "C", "D", "E"]        # the 5 answer choices (labels 0..4)
NN_MAX_LEN = 200                            # pad/truncate length; subword seqs run long
TEXT_COLS = ["prompt", "A", "B", "C", "D", "E"]  # every free-text column to clean


# --------------------------------------------------------------------------- #
# 1. Data cleaning
# --------------------------------------------------------------------------- #
def clean_dataframe(df):
    """Defensive cleaning: fill NaNs, coerce to str, normalize whitespace.

    Runs on train AND test with identical rules, so the model never sees
    inconsistent input between the two (no preprocessing skew).
    """
    df = df.copy()
    for col in TEXT_COLS:
        # fill missing -> string -> trim; collapse any run of spaces into one
        df[col] = df[col].fillna("").astype(str).str.strip()
        df[col] = df[col].str.replace(r"\s+", " ", regex=True)
        # kill "disguised" nulls that arrive as literal text, not real NaN
        df[col] = df[col].replace(
            {"nan": "", "None": "", "null": "", "N/A": "", "NA": ""}, regex=False
        )
        # never hand the model an empty string — give it a placeholder token
        df.loc[df[col] == "", col] = "[no text]"
    if "answer" in df.columns:                      # test set has no 'answer' column
        df["answer"] = df["answer"].str.strip().str.upper()
        # fail loudly if a label isn't one of A-E (catches data corruption early)
        assert df["answer"].isin(OPTIONS).all(), "Invalid answer values!"
    return df


def filter_text(text):
    """Light text filter used at tokenization time (same rule for all splits)."""
    text = str(text).strip()
    text = " ".join(text.split())          # normalize whitespace to single spaces
    return text if text else "[empty]"     # guard against empty strings


def normalize_text(text):
    """Lowercase, strip punctuation, collapse whitespace (for the classical model)."""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# 2. MAP@3 + evaluation metrics
# --------------------------------------------------------------------------- #
def map_at_3(truth, preds):
    """Score ONE question. Reward = 1/rank if the correct answer is in the top-3.

    Correct at rank 1 -> 1.0, rank 2 -> 0.5, rank 3 -> 0.33, otherwise -> 0.
    """
    score, hits = 0.0, 0
    for i, p in enumerate(preds[:3]):      # only the top-3 predictions count
        if p == truth:
            hits += 1
            score += hits / (i + 1)        # i is 0-based, so rank = i + 1
    return score


def evaluate_map3(truths, all_preds):
    """Average MAP@3 across every question = the competition metric."""
    return float(np.mean([map_at_3(t, p) for t, p in zip(truths, all_preds)]))


def probs_to_ranked(prob_matrix):
    """(n, 5) probability matrix -> list of top-3 letter lists (one per question)."""
    # argsort is ascending, so [::-1] flips it to descending (highest prob first)
    return [[OPTIONS[k] for k in np.argsort(p)[::-1]][:3] for p in prob_matrix]


def scores_to_preds(id_scores_dict, id_list):
    """Same as probs_to_ranked but for a {id: scores} dict keyed by question id."""
    preds = []
    for qid in id_list:
        scores = id_scores_dict[qid]
        preds.append([OPTIONS[i] for i in np.argsort(scores)[::-1]][:3])
    return preds


def evaluate_model(val_truths, val_ranked):
    """Return accuracy + macro-F1 (both rank-1) + MAP@3 in one dict.

    Accuracy/F1 only look at the top pick; MAP@3 rewards good ranking of the top-3.
    """
    from sklearn.metrics import accuracy_score, f1_score
    top1 = [r[0] for r in val_ranked]      # rank-1 prediction per question
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
    """Train a Byte-Pair-Encoding tokenizer on a list of texts.

    IMPORTANT: pass the TRAINING SPLIT ONLY (never val/test) to avoid leakage.
    BPE splits rare/unseen words into known sub-units, so out-of-vocab is ~0%,
    unlike a fixed word vocabulary that would emit lots of <UNK>.
    """
    tok = Tokenizer(BPE(unk_token="<UNK>"))
    # normalizer: unicode-normalize, lowercase, strip accents before splitting
    tok.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tok.pre_tokenizer = Whitespace()       # split on whitespace first, then BPE-merge
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>", "<SEP>"],  # PAD=0, then UNK, then SEP
        min_frequency=min_frequency,       # ignore merges seen fewer than this
    )
    tok.train_from_iterator(corpus, trainer)
    return tok


def encode_pair(tokenizer, prompt, option_text, max_len=NN_MAX_LEN):
    """Encode one (prompt, option) pair: filter -> BPE -> truncate -> pad.

    Returns (ids, attn_mask). The attention mask is 1 for real tokens and 0 for
    padding, so downstream pooling can ignore the padded positions.
    """
    pad_id = tokenizer.token_to_id("<PAD>")
    # join prompt and option with a separator token so the model sees the boundary
    text = filter_text(prompt) + " <SEP> " + filter_text(option_text)
    ids = tokenizer.encode(text).ids[:max_len]          # BPE ids, truncated to max_len
    attn = [1] * len(ids) + [0] * (max_len - len(ids))  # 1 = real token, 0 = padding
    ids = ids + [pad_id] * (max_len - len(ids))         # pad the ids to fixed length
    return ids, attn


# --------------------------------------------------------------------------- #
# 4. Dataset
# --------------------------------------------------------------------------- #
class MCQDataset(Dataset):
    """One question per item -> 5 encoded (prompt + option) pairs.

    Each __getitem__ returns a (5, max_len) tensor of token ids + matching masks,
    plus the integer label (0-4) during training. is_test=True omits the label.
    """

    def __init__(self, df, tokenizer, max_len=NN_MAX_LEN, is_test=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.items = []
        # pre-extract rows once (faster than touching the dataframe each __getitem__)
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
        for o in OPTIONS:                              # encode all 5 options
            ids, mask = encode_pair(self.tokenizer, item["prompt"],
                                    item["options"][o], self.max_len)
            all_ids.append(ids)
            all_masks.append(mask)
        result = {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),   # (5, max_len)
            "attn_mask": torch.tensor(all_masks, dtype=torch.float),# (5, max_len)
            "id": item["id"],
        }
        if "answer" in item:
            # convert letter -> class index: 'A'->0, 'B'->1, ... 'E'->4
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

    Kept deliberately small (dropout 0.4) because only ~2000 questions are
    available and a bigger network overfits fast.
    """

    def __init__(self, vocab_size, pad_id, embed_dim=64, hidden_dim=64,
                 num_layers=1, dropout=0.4):
        super().__init__()
        # padding_idx keeps the PAD token's embedding fixed at zero
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        # bidirectional => output width is hidden_dim * 2
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)   # 128 -> 64
        self.fc2 = nn.Linear(64, 1)                # 64 -> 1 score

    def forward_one(self, ids, mask):
        """Score a single (prompt, option) pair -> one scalar per example."""
        emb = self.embedding(ids)                  # (batch, seq, embed)
        lstm_out, _ = self.lstm(emb)               # (batch, seq, hidden*2)
        # --- masked mean pooling: average over REAL tokens only ---
        mask_exp = mask.unsqueeze(-1)              # (batch, seq, 1) for broadcasting
        summed = (lstm_out * mask_exp).sum(dim=1)  # zero-out padding, then sum
        counts = mask_exp.sum(dim=1).clamp(min=1)  # real-token count (avoid /0)
        pooled = summed / counts                   # -> (batch, hidden*2)
        # --- classifier head ---
        h = self.dropout(pooled)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h).squeeze(-1)           # (batch,) returns one score per example

    def forward(self, input_ids, attn_mask):
        """Run all 5 options and stack their scores -> (batch, 5) logits."""
        scores = [self.forward_one(input_ids[:, i], attn_mask[:, i])
                  for i in range(5)]
        return torch.stack(scores, dim=1)          # softmax/argmax applied by caller