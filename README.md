# Smart MCQ Solver — Intro to DL & GenAI Project [BSDA2001P]

Ranking-based multiple-choice question solver for the IIT Madras *Smart MCQ Solver
Challenge* (Kaggle). Given a `prompt` and five options `A–E` (exactly one correct),
the model ranks the options and submits the **top-3**. Metric: **MAP@3** (Mean
Average Precision at 3).

**Roll No:** 22f3000768

## Models

Three models are compared on the **same leakage-free validation split**, covering the
required scratch, classical, and pretrained categories:

| Model | Type | Val MAP@3 | Notes |
|---|---|---|---|
| **BiLSTM** | From scratch | 0.6645 | BPE tokenizer + Embedding → BiLSTM → masked mean-pool → classifier. **Submission model.** |
| **TF-IDF + LogReg** | Classical ML | 0.6769 | Each (prompt, option) pair scored, ranked per question. |
| **DeBERTa-v3-small** | Pretrained | **0.7292** | Fine-tuned transformer, fp16, early stopping on macro-F1. |

## Results & key finding

| Model | Val MAP@3 (generalisation) | Kaggle leaderboard |
|---|---|---|
| BiLSTM | 0.6645 | **0.742** |
| DeBERTa-v3-small (refit) | 0.7292 | 0.736 |

**Why the ranking flips.** On the honest, leakage-free validation set DeBERTa is best
(0.7292). But on the leaderboard the BiLSTM scores *higher* (0.742 vs 0.736). This is
because **91.6% of the test set overlaps the training set** — so the leaderboard largely
rewards *memorisation*, not generalisation. A smaller, high-capacity model recalls seen
questions more sharply, which is exactly why the BiLSTM was submitted. The two numbers
measure different things, and both are reported honestly.

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

Optional experiment tracking (do **not** hardcode your key — read it from an env var
or Kaggle Secrets):

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
  leaks copies across train/val and only measures memorisation. `StratifiedGroupKFold`
  groups duplicates by their five-option fingerprint (all copies of a question stay on
  one side) *and* keeps the answer distribution balanced. Zero group overlap is asserted.
- **Honest evaluation** — the reported MAP@3 is the best *pre-refit* validation score.
  The model is then refit on 100% of the data for the best epoch count before test
  prediction; any post-refit training accuracy is memorisation, not an evaluation metric.
- **Ranking loss** — each (prompt, option) pair gets one score; the 5 scores per question
  go through softmax, and the top-3 letters form the prediction and evaluation.

## Experiment tracking

All runs are logged to Weights & Biases (`smart-mcq-solver` project): training loss,
validation accuracy / macro-F1 / MAP@3 per epoch, and per-model config. The API key is
loaded from Kaggle Secrets (`WANDB_API_KEY`), never hardcoded in the notebook.

## Milestones

| Milestone | Branch | Content |
|---|---|---|
| 1 | `milestone-1` | Data cleaning + EDA |
| 2 | `milestone-2` | Baseline + models |
| 3 | `milestone-3` | Tuning, comparison, submission |

Milestone branches are kept after merging (per course policy).