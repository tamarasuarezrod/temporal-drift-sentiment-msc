"""Paired-bootstrap 95% confidence intervals for macro-F1 differences between
systems (5,000 resamples). Writes results/metrics/paired_bootstrap.csv."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config import METRICS_DIR, PARQUET

N_BOOT = 5000

COMPARISONS = [
    # (split, system A prefix, system B prefix, label)
    ("within", "S7", "baseline", "TEA vs Baseline (within winner)"),
    ("short",  "S2", "baseline", "MLMPretrain vs Baseline (short winner)"),
    ("long",   "S7", "baseline", "TEA vs Baseline (long winner)"),
    ("long",   "S7", "S3",       "TEA vs RecencyWeight (top tier, long)"),
    ("within", "S7", "S3",       "TEA vs RecencyWeight (top tier, within)"),
    ("short",  "S2", "S7",       "MLMPretrain vs TEA (top tier, short)"),
]


def macro_f1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def main() -> None:
    df = pd.read_parquet(PARQUET)
    rng = np.random.default_rng(42)

    rows = []
    for split, a, b, label in COMPARISONS:
        sub = df[df["split"] == split]
        y = (sub["gold"] == "positive").astype(int).to_numpy()
        pa = (sub[f"{a}_pred"] == "positive").astype(int).to_numpy()
        pb = (sub[f"{b}_pred"] == "positive").astype(int).to_numpy()
        n = len(y)

        # Resample the tweets with replacement and take the F1 gap each time
        diffs = np.empty(N_BOOT)
        for k in range(N_BOOT):
            i = rng.integers(0, n, n)
            diffs[k] = macro_f1(y[i], pa[i]) - macro_f1(y[i], pb[i])

        mean = diffs.mean()
        lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
        significant = bool(lo > 0 or hi < 0)
        rows.append({
            "comparison": label, "split": split,
            "mean_diff_pp": mean * 100,
            "ci_lo_pp": lo * 100, "ci_hi_pp": hi * 100,
            "significant": significant,
        })
        tag = "SIG" if significant else "n.s."
        print(f"  {label}: {mean*100:+.2f}pp  [{lo*100:+.2f}, {hi*100:+.2f}]  {tag}")

    out = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(METRICS_DIR / "paired_bootstrap.csv", index=False)
    print(f"\nWrote {METRICS_DIR / 'paired_bootstrap.csv'}")


if __name__ == "__main__":
    main()
