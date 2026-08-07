"""Aggregate the per-seed predictions, drift metrics and flags into the
canonical per-tweet table (results/per_tweet_analysis.parquet), one row per
test tweet with majority-vote fields per system."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from config import (
    DRIFT_CSV,
    HF_DATA_REPO,
    PARQUET,
    PRAGMATIC_CSV,
    PREDS_RAW,
    SPLITS,
    SYSTEM_COLUMN_PREFIX,
)


def load_gold() -> pd.DataFrame:
    """Gold labels and text for the three test splits, keyed by (split, idx)."""
    root = Path(snapshot_download(repo_id=HF_DATA_REPO, repo_type="dataset"))
    frames = []
    for name, rel in SPLITS:
        with open(root / rel) as f:
            recs = json.load(f)
        df = pd.DataFrame(recs)
        df["split"] = name
        df["idx"] = np.arange(len(df))
        df = df.rename(columns={"pp_text": "text", "label": "gold"})
        frames.append(df[["split", "idx", "text", "gold"]])
    return pd.concat(frames, ignore_index=True)


def aggregate_predictions(raw: pd.DataFrame, prefix_map: dict[str, str]) -> pd.DataFrame:
    """Turn the long (strategy, seed) frame into wide per-tweet columns."""
    out = None
    for strategy, prefix in prefix_map.items():
        sub = raw[raw["strategy"] == strategy]
        if sub.empty:
            raise RuntimeError(f"No predictions found for {strategy}. Did 04 finish?")

        grouped = sub.groupby(["split", "idx"], as_index=False).agg(
            prob_pos=("prob_pos", "mean"),
            conf=("conf", "mean"),
            n_correct=("correct", "sum"),
        )
        # majority-vote prediction (positive when mean prob >= 0.5)
        grouped[f"{prefix}_pred"] = np.where(
            grouped["prob_pos"] >= 0.5, "positive", "negative"
        )
        grouped[f"{prefix}_prob_pos"] = grouped["prob_pos"]
        grouped[f"{prefix}_conf"] = grouped["conf"]
        grouped[f"{prefix}_n_correct"] = grouped["n_correct"].astype(int)
        grouped = grouped.drop(columns=["prob_pos", "conf", "n_correct"])
        out = grouped if out is None else out.merge(grouped, on=["split", "idx"], how="inner")
    return out


def main() -> None:
    if not PREDS_RAW.exists():
        raise SystemExit(f"Missing {PREDS_RAW}. Run 04_build_predictions.py first.")

    # Raw per-seed predictions (from 04) and the gold labels
    raw = pd.read_parquet(PREDS_RAW)
    print(f"Loaded {len(raw)} prediction rows from {PREDS_RAW}")

    gold = load_gold()
    print(f"Loaded gold labels: {len(gold)} rows")

    # Collapse the three seeds into one majority-vote row per tweet, then
    # join onto the gold labels to give each tweet all per-system columns
    preds = aggregate_predictions(raw, SYSTEM_COLUMN_PREFIX)
    print(f"Aggregated predictions: {len(preds)} rows")

    df = gold.merge(preds, on=["split", "idx"], how="inner")

    # Mark correctness of the majority-vote prediction
    for _, prefix in SYSTEM_COLUMN_PREFIX.items():
        df[f"{prefix}_correct"] = (df[f"{prefix}_pred"] == df["gold"]).astype(int)

    # Merge drift metrics (per-tweet)
    if DRIFT_CSV.exists():
        drift = pd.read_csv(DRIFT_CSV)
        df = df.merge(drift, on=["split", "idx"], how="left")
        print(f"Merged drift metrics ({DRIFT_CSV})")
    else:
        print(f"  (skipping drift merge, {DRIFT_CSV} not found)")

    # Merge pragmatic flag
    if PRAGMATIC_CSV.exists():
        prag = pd.read_csv(PRAGMATIC_CSV)
        df = df.merge(prag, on=["split", "idx"], how="left")
        print(f"Merged pragmatic flags ({PRAGMATIC_CSV})")
    else:
        print(f"  (skipping pragmatic merge, {PRAGMATIC_CSV} not found)")

    # Save the canonical per-tweet table that the rest of the pipeline reads
    PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET, index=False)
    print(f"\nWrote {PARQUET}  ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
