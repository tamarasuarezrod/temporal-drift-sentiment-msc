"""Test whether the strategy that wins on the practice sets also wins on the
test sets.

Reports the practice winner, the true test winner, their rank correlation and
the regret from deploying the practice winner. Writes
results/metrics/practice_selection.csv."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    METRICS_DIR,
    data_root,
    SEEDS,
    SYSTEMS,
    TABLES_DIR,
    load_checkpoint,
)

PRACTICE_FILES = [
    ("practice_2016", "train_eval/interim_eval_2016.json"),
    ("practice_2018", "train_eval/interim_eval_2018.json"),
]

PAIRINGS = [
    ("practice_2016", "within"),
    ("practice_2018", "short"),
    ("practice_2018", "long"),
]

LABEL2ID = {"negative": 0, "positive": 1}

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@torch.no_grad()
def predict(model, tokenizer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(texts), batch_size):
        chunk = [t.replace("@MENTION", "@user") for t in texts[i:i + batch_size]]
        enc = tokenizer(
            chunk, truncation=True, padding=True, max_length=128, return_tensors="pt"
        ).to(DEVICE)
        logits = model(**enc).logits
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    print(f"Device: {DEVICE}")
    root = data_root()

    # Load the two practice sets (distant labels, available before deployment)
    practice = {}
    for name, rel in PRACTICE_FILES:
        with open(root / rel) as f:
            df = pd.DataFrame(json.load(f))
        df["label_id"] = df["distant_label"].map(LABEL2ID)
        practice[name] = df
        print(f"  {name}: n={len(df)}  positive_rate={df['label_id'].mean():.3f}")

    # Majority-vote predictions per system on each practice set
    prac_f1 = {name: {} for name, _ in PRACTICE_FILES}
    for strategy, ckpt_id in SYSTEMS:
        votes = {name: [] for name, _ in PRACTICE_FILES}
        for seed in SEEDS:
            ckpt = f"{ckpt_id}-seed{seed}"
            print(f"[ {strategy}  seed {seed} ]  {ckpt}")
            tokenizer = load_checkpoint(AutoTokenizer, ckpt)
            model = load_checkpoint(AutoModelForSequenceClassification, ckpt).to(DEVICE)
            for name, df in practice.items():
                votes[name].append(predict(model, tokenizer, df["pp_text"].tolist()))
            del model, tokenizer
        for name, df in practice.items():
            stacked = np.stack(votes[name])  # 3 x n
            majority = (stacked.sum(axis=0) >= 2).astype(int)
            prac_f1[name][strategy] = f1_score(
                df["label_id"], majority, average="macro", zero_division=0
            )

    # Test F1 from the regenerated results table
    t3 = pd.read_csv(TABLES_DIR / "results.csv").set_index("Method")

    # Put each system's practice and test F1 side by side
    rows = []
    for strategy, _ in SYSTEMS:
        rows.append({
            "system": strategy,
            "practice_2016": prac_f1["practice_2016"][strategy],
            "practice_2018": prac_f1["practice_2018"][strategy],
            "test_within": t3.loc[strategy, "within"],
            "test_short": t3.loc[strategy, "short"],
            "test_long": t3.loc[strategy, "long"],
        })
    out = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(METRICS_DIR / "practice_selection.csv", index=False)
    print(f"\nWrote {METRICS_DIR / 'practice_selection.csv'}")
    print()
    print(out.round(4).to_string(index=False))

    print("\n=== Selection protocol: pick the practice winner, deploy on test ===")
    for prac_name, test_split in PAIRINGS:
        p = out.set_index("system")[prac_name]
        t = out.set_index("system")[f"test_{test_split}"]
        pick = p.idxmax()
        best = t.idxmax()
        regret = t[best] - t[pick]
        rho, _ = spearmanr(p, t)
        print(f"{prac_name} -> test_{test_split}:")
        print(f"  practice winner : {pick} ({p[pick]:.4f} on practice, {t[pick]:.4f} on test)")
        print(f"  true test winner: {best} ({t[best]:.4f})")
        print(f"  regret          : {100 * regret:.2f} pp   rank correlation (Spearman): {rho:.2f}")


if __name__ == "__main__":
    main()
