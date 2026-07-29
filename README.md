# Smart MCQ Solver — Intro to DL & GenAI Project [BSDA2001P]

Ranking-based multiple-choice question solver for the IIT Madras *Smart MCQ Solver
Challenge* (Kaggle). Given a `prompt` and five options `A–E`, the model ranks the
options and submits the **top-3** (metric: **MAP@3**).

Three models are compared on the same leakage-free validation split:

| Model | Type | Notes |
|---|---|---|
| **BiLSTM** | From scratch | BPE tokenizer + Embedding → BiLSTM → masked mean-pool → classifier. **Submission model.** |
| **TF-IDF + LogReg** | Classical ML | Per-(prompt, option) pair scored, ranked per question. |
| **DeBERTa-v3-small** | Pretrained | Fine-tuned transformer, fp16, early stopping on macro-F1. |

## Repository structure

```
├── notebooks/
│   └── final_notebook.ipynb     # full EDA + all 3 models + comparison
├── src/
│   ├── utils.py                 # cleaning, tokenizer, dataset, model, MAP@3
│   ├── train.py                 # trains the BiLSTM submission model
│   └── inference.py             # loads model → writes submission.csv
├── reports/                     # milestone + final reports (PDF)
├── models/                      # saved weights + tokenizer (git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Place the competition files in a `data/` folder (git-ignored):

```
data/train.csv
data/test.csv
data/sample_submission.csv
```

## Train

```bash
python src/train.py --data-dir ./data --out-dir ./models
```

Writes `models/bilstm_final.pt` and `models/bpe_tokenizer.json`.

Optional experiment tracking (do **not** hardcode your key):

```bash
export WANDB_API_KEY=your_key_here
python src/train.py --data-dir ./data --out-dir ./models --wandb
```

## Inference / Submission

```bash
python src/inference.py --data-dir ./data --model-dir ./models --out submission.csv
```

Produces a validated Kaggle `submission.csv` (`ID`, `Prediction` = `"A B C"`).

## Method notes

- **Leakage-free split** — the dataset has duplicate questions, so a random split
  leaks copies across train/val. `StratifiedGroupKFold` groups duplicates by their
  five-option fingerprint *and* keeps the answer distribution balanced.
- **Honest evaluation** — the reported MAP@3 is the best pre-refit validation score.
  The model is then refit on 100% of the data for the best epoch count before test
  prediction; any post-refit training score is memorization, not evaluation.

## Milestones

| Milestone | Branch | Content |
|---|---|---|
| 1 | `milestone-1` | Data cleaning + EDA |
| 2 | `milestone-2` | Baseline + models |
| 3 | `milestone-3` | Tuning, comparison, submission |

Milestone branches are kept after merging (per course policy).
