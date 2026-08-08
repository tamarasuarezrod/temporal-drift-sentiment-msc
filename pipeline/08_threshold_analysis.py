"""Measure how much of each strategy's gain over the baseline is just a
decision-threshold shift.

The threshold is tuned on the 2016 practice set and applied to the baseline
probabilities on the test splits. Writes results/metrics/threshold_analysis*.csv."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    METRICS_DIR,
    data_root,
    PARQUET,
    SEEDS,
    SYSTEM_COLUMN_PREFIX,
    load_checkpoint,
)
PRACTICE_FILE = "train_eval/interim_eval_2016.json"

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

LABEL2ID = {"negative": 0, "positive": 1}


@torch.no_grad()
def predict_probs(model, tokenizer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = [t.replace("@MENTION", "@user") for t in texts[i:i + batch_size]]
        enc = tokenizer(
            chunk, truncation=True, padding=True, max_length=128, return_tensors="pt"
        ).to(DEVICE)
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        out.append(probs[:, LABEL2ID["positive"]])
    return np.concatenate(out)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def main() -> None:
    print(f"Device: {DEVICE}")
    root = data_root()
    with open(root / PRACTICE_FILE) as f:
        practice = pd.DataFrame(json.load(f))
    y_prac = (practice["distant_label"] == "positive").astype(int).to_numpy()
    print(f"Practice 2016: n={len(practice)}")

    # Mean positive probability across the three baseline seeds
    probs = []
    for seed in SEEDS:
        ckpt = f"baseline-seed{seed}"
        print(f"  loading {ckpt} ...")
        tok = load_checkpoint(AutoTokenizer, ckpt)
        model = load_checkpoint(AutoModelForSequenceClassification, ckpt).to(DEVICE)
        probs.append(predict_probs(model, tok, practice["pp_text"].tolist()))
        del model, tok
    prob_prac = np.mean(probs, axis=0)

    # Threshold sweep on the practice set
    thresholds = np.arange(0.30, 0.71, 0.01)
    scores = [(t, macro_f1(y_prac, (prob_prac >= t).astype(int))) for t in thresholds]
    best_t, best_prac_f1 = max(scores, key=lambda x: x[1])
    default_prac_f1 = macro_f1(y_prac, (prob_prac >= 0.5).astype(int))
    print(f"\nBest threshold on practice_2016: {best_t:.2f} "
          f"(macro-F1 {best_prac_f1:.4f} vs {default_prac_f1:.4f} at 0.50)")

    # Apply to the test splits
    df = pd.read_parquet(PARQUET)
    rows = []
    for split in ["within", "short", "long"]:
        sub = df[df["split"] == split]
        y = (sub["gold"] == "positive").astype(int).to_numpy()
        prob = sub["baseline_prob_pos"].to_numpy()

        f1_base = macro_f1(y, (prob >= 0.5).astype(int))
        f1_thresh = macro_f1(y, (prob >= best_t).astype(int))

        # Best strategy on this split (excluding baseline)
        best_strat, best_f1 = None, -1.0
        for label, prefix in SYSTEM_COLUMN_PREFIX.items():
            if label == "Baseline":
                continue
            pred = (sub[f"{prefix}_pred"] == "positive").astype(int).to_numpy()
            f1s = macro_f1(y, pred)
            if f1s > best_f1:
                best_strat, best_f1 = label, f1s

        gain_strat = best_f1 - f1_base
        gain_thresh = f1_thresh - f1_base
        recovered = gain_thresh / gain_strat if gain_strat > 0 else float("nan")
        rows.append({
            "split": split,
            "baseline_f1": f1_base,
            "thresholded_f1": f1_thresh,
            "best_strategy": best_strat,
            "best_strategy_f1": best_f1,
            "gain_recovered": recovered,
        })
        print(f"  {split}: base={f1_base:.4f}  thresh={f1_thresh:.4f}  "
              f"best={best_strat} {best_f1:.4f}  recovered={recovered*100:.0f}%")

    # Oracle variant: tune on a held-out half of the manually annotated
    # within test set. This is NOT deployment-realistic (manual labels
    # are not available at development time) but isolates the question
    # "is the strategies' gain reproducible by a threshold move, if one
    # knew where to move it?"
    print("\nOracle variant (tuned on a held-out half of test within):")
    rng = np.random.default_rng(42)
    within = df[df["split"] == "within"]
    idx = rng.permutation(len(within))
    tune_idx, eval_idx = idx[: len(within) // 2], idx[len(within) // 2 :]

    y_tune = (within["gold"] == "positive").astype(int).to_numpy()[tune_idx]
    prob_tune = within["baseline_prob_pos"].to_numpy()[tune_idx]
    scores = [(t, macro_f1(y_tune, (prob_tune >= t).astype(int))) for t in thresholds]
    oracle_t, _ = max(scores, key=lambda x: x[1])
    print(f"  oracle threshold: {oracle_t:.2f}")

    oracle_rows = []
    for split in ["within", "short", "long"]:
        sub = df[df["split"] == split]
        if split == "within":
            sub = sub.iloc[eval_idx]  # evaluate only on the held-out half
        y = (sub["gold"] == "positive").astype(int).to_numpy()
        prob = sub["baseline_prob_pos"].to_numpy()

        f1_base = macro_f1(y, (prob >= 0.5).astype(int))
        f1_thresh = macro_f1(y, (prob >= oracle_t).astype(int))

        best_strat, best_f1 = None, -1.0
        for label, prefix in SYSTEM_COLUMN_PREFIX.items():
            if label == "Baseline":
                continue
            pred = (sub[f"{prefix}_pred"] == "positive").astype(int).to_numpy()
            f1s = macro_f1(y, pred)
            if f1s > best_f1:
                best_strat, best_f1 = label, f1s

        gain_strat = best_f1 - f1_base
        gain_thresh = f1_thresh - f1_base
        recovered = gain_thresh / gain_strat if gain_strat > 0 else float("nan")
        oracle_rows.append({
            "split": split,
            "baseline_f1": f1_base,
            "oracle_thresholded_f1": f1_thresh,
            "best_strategy": best_strat,
            "best_strategy_f1": best_f1,
            "gain_recovered": recovered,
        })
        print(f"  {split}: base={f1_base:.4f}  thresh={f1_thresh:.4f}  "
              f"best={best_strat} {best_f1:.4f}  recovered={recovered*100:.0f}%")

    # In-sample per-period upper bound: for each test period, pick the
    # threshold that maximises macro-F1 on that period itself. This is
    # the number quoted in the paper's threshold section (about half of
    # the top strategy's gain on within and short, 18% on long). It is
    # an upper bound on what boundary placement alone can deliver, not
    # a deployable method
    print("\nIn-sample per-period upper bound:")
    insample_rows = []
    for split in ["within", "short", "long"]:
        sub = df[df["split"] == split]
        y = (sub["gold"] == "positive").astype(int).to_numpy()
        prob = sub["baseline_prob_pos"].to_numpy()

        f1_base = macro_f1(y, (prob >= 0.5).astype(int))
        best_split_t, f1_thresh = max(
            ((t, macro_f1(y, (prob >= t).astype(int))) for t in thresholds),
            key=lambda x: x[1],
        )

        best_strat, best_f1 = None, -1.0
        for label, prefix in SYSTEM_COLUMN_PREFIX.items():
            if label == "Baseline":
                continue
            pred = (sub[f"{prefix}_pred"] == "positive").astype(int).to_numpy()
            f1s = macro_f1(y, pred)
            if f1s > best_f1:
                best_strat, best_f1 = label, f1s

        gain_strat = best_f1 - f1_base
        gain_thresh = f1_thresh - f1_base
        recovered = gain_thresh / gain_strat if gain_strat > 0 else float("nan")
        insample_rows.append({
            "split": split,
            "baseline_f1": f1_base,
            "optimal_threshold": best_split_t,
            "insample_thresholded_f1": f1_thresh,
            "best_strategy": best_strat,
            "best_strategy_f1": best_f1,
            "gain_recovered": recovered,
        })
        print(f"  {split}: base={f1_base:.4f}  opt_t={best_split_t:.2f}  "
              f"thresh={f1_thresh:.4f}  best={best_strat} {best_f1:.4f}  "
              f"recovered={recovered*100:.0f}%")

    out = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(METRICS_DIR / "threshold_analysis.csv", index=False)
    pd.DataFrame(oracle_rows).to_csv(METRICS_DIR / "threshold_analysis_oracle.csv", index=False)
    pd.DataFrame(insample_rows).to_csv(METRICS_DIR / "threshold_analysis_insample.csv", index=False)
    print(f"\nWrote {METRICS_DIR / 'threshold_analysis.csv'}, "
          f"threshold_analysis_oracle.csv and threshold_analysis_insample.csv")


if __name__ == "__main__":
    main()
