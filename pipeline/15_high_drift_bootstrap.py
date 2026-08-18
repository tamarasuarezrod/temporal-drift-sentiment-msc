"""Paired-bootstrap intervals on the highest-drift quintile of every test split.

For each period, this script compares every mitigation strategy with the
baseline and the descriptively strongest strategy with every other strategy.
It writes results/all_systems/high_drift_bootstrap_by_split.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config import PARQUET


REPO = Path(__file__).resolve().parents[1]
ALL_SYSTEMS_DIR = REPO / "results" / "all_systems"
OUT = ALL_SYSTEMS_DIR / "high_drift_bootstrap_by_split.csv"

SPLITS = ["within", "short", "long"]
CANONICAL_SYSTEMS = [
    "baseline",
    "DatePrefix",
    "MLMPretrain",
    "RecencyWeight",
    "TimeLMs",
    "TEA",
]
ADDITIONAL_SYSTEMS = ["PretrainedTEA", "T5"]
MITIGATION_SYSTEMS = CANONICAL_SYSTEMS[1:] + ADDITIONAL_SYSTEMS
N_BOOT = 5_000


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, average="macro", zero_division=0))


def aggregate_raw(raw: pd.DataFrame, system: str, split: str) -> np.ndarray:
    """Soft-vote raw seed probabilities and threshold their mean at 0.5."""
    block = raw[(raw.system == system) & (raw.split == split)]
    mean_prob = block.groupby("idx").prob_pos.mean().sort_index().to_numpy()
    return (mean_prob >= 0.5).astype(np.int8)


def bootstrap_macro_f1(
    y: np.ndarray, pred: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """Vectorised macro-F1 for a shared matrix of paired bootstrap samples."""
    sampled_y = y[indices]
    sampled_pred = pred[indices]
    tp = ((sampled_y == 1) & (sampled_pred == 1)).sum(axis=1)
    tn = ((sampled_y == 0) & (sampled_pred == 0)).sum(axis=1)
    fp = ((sampled_y == 0) & (sampled_pred == 1)).sum(axis=1)
    fn = ((sampled_y == 1) & (sampled_pred == 0)).sum(axis=1)
    pos_den = 2 * tp + fp + fn
    neg_den = 2 * tn + fp + fn
    pos_f1 = np.divide(
        2 * tp, pos_den, out=np.zeros_like(tp, dtype=float), where=pos_den != 0
    )
    neg_f1 = np.divide(
        2 * tn, neg_den, out=np.zeros_like(tn, dtype=float), where=neg_den != 0
    )
    return (pos_f1 + neg_f1) / 2


def main() -> None:
    canonical = pd.read_parquet(PARQUET)
    proposed_raw = pd.read_parquet(ALL_SYSTEMS_DIR / "proposed_preds_raw.parquet")
    t5_raw = pd.read_parquet(ALL_SYSTEMS_DIR / "t5_preds_raw.parquet")
    raw_by_system = {"PretrainedTEA": proposed_raw, "T5": t5_raw}

    rows: list[dict[str, object]] = []
    for split_index, split in enumerate(SPLITS):
        block = (
            canonical[canonical.split == split]
            .sort_values("idx")
            .reset_index(drop=True)
        )
        y = (block.gold == "positive").astype(np.int8).to_numpy()
        drift_score = block.oov_frac.fillna(0) + block.sem_dist.fillna(0)
        quintile = pd.qcut(drift_score, 5, labels=[1, 2, 3, 4, 5])
        q5 = (quintile == 5).to_numpy()
        y_q5 = y[q5]

        predictions: dict[str, np.ndarray] = {}
        for system in CANONICAL_SYSTEMS:
            predictions[system] = (
                block[f"{system}_pred"] == "positive"
            ).astype(np.int8).to_numpy()[q5]
        for system in ADDITIONAL_SYSTEMS:
            predictions[system] = aggregate_raw(
                raw_by_system[system], system, split
            )[q5]

        rng = np.random.default_rng(42 + split_index)
        indices = rng.integers(0, len(y_q5), size=(N_BOOT, len(y_q5)))
        bootstrap_scores = {
            system: bootstrap_macro_f1(y_q5, pred, indices)
            for system, pred in predictions.items()
        }
        point_scores = {
            system: macro_f1(y_q5, pred)
            for system, pred in predictions.items()
        }

        def add_comparison(kind: str, system_a: str, system_b: str) -> None:
            diffs = 100 * (
                bootstrap_scores[system_a] - bootstrap_scores[system_b]
            )
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            observed = 100 * (point_scores[system_a] - point_scores[system_b])
            rows.append(
                {
                    "split": split,
                    "n": int(q5.sum()),
                    "comparison_type": kind,
                    "system_a": system_a,
                    "system_b": system_b,
                    "diff_pp": round(float(observed), 2),
                    "ci_lo_pp": round(float(lo), 2),
                    "ci_hi_pp": round(float(hi), 2),
                    "significant": bool(lo > 0 or hi < 0),
                }
            )

        for system in MITIGATION_SYSTEMS:
            add_comparison("vs_baseline", system, "baseline")

        winner = max(MITIGATION_SYSTEMS, key=point_scores.get)
        for system in MITIGATION_SYSTEMS:
            if system != winner:
                add_comparison("winner_vs_strategy", winner, system)

    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    for split in SPLITS:
        block = result[
            (result.split == split)
            & (result.comparison_type == "vs_baseline")
        ]
        winner = block.loc[block.diff_pp.idxmax(), "system_a"]
        significant = block.loc[block.significant, "system_a"].tolist()
        print(
            f"  {split}: descriptive winner={winner}; "
            f"significant vs baseline={significant or 'none'}"
        )


if __name__ == "__main__":
    main()
