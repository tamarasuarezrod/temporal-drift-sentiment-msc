"""Correction overlap between the two components of PretrainedTEA.

For each test split, this script identifies baseline errors corrected by
MLMPretrain, TEA and PretrainedTEA, then computes the Jaccard overlap between
the MLMPretrain and TEA correction sets. It writes the canonical all-period
result to results/all_systems/correction_overlap_by_split.csv.
"""

from pathlib import Path

import pandas as pd

from config import PARQUET


REPO = Path(__file__).resolve().parents[1]
ALL_SYSTEMS_DIR = REPO / "results" / "all_systems"
OUT = ALL_SYSTEMS_DIR / "correction_overlap_by_split.csv"
SPLITS = ["within", "short", "long"]


def main() -> None:
    tweets = pd.read_parquet(PARQUET)
    raw = pd.read_parquet(ALL_SYSTEMS_DIR / "proposed_preds_raw.parquet")
    proposed = (
        raw.groupby(["split", "idx"], as_index=False)
        .prob_pos.mean()
        .rename(columns={"prob_pos": "mean_prob_pos"})
    )
    proposed["PretrainedTEA_pred"] = proposed.mean_prob_pos.ge(0.5).map(
        {True: "positive", False: "negative"}
    )
    tweets = tweets.merge(
        proposed[["split", "idx", "PretrainedTEA_pred"]],
        on=["split", "idx"],
        validate="one_to_one",
    )

    rows = []
    for split in SPLITS:
        block = tweets[tweets.split == split]
        baseline_wrong = block.baseline_pred.ne(block.gold)
        mlm = baseline_wrong & block.MLMPretrain_pred.eq(block.gold)
        tea = baseline_wrong & block.TEA_pred.eq(block.gold)
        combined = baseline_wrong & block.PretrainedTEA_pred.eq(block.gold)
        intersection = int((mlm & tea).sum())
        union = int((mlm | tea).sum())
        rows.append(
            {
                "split": split,
                "n": len(block),
                "baseline_errors": int(baseline_wrong.sum()),
                "mlmpretrain_corrected": int(mlm.sum()),
                "tea_corrected": int(tea.sum()),
                "pretrainedtea_corrected": int(combined.sum()),
                "shared_corrections": intersection,
                "union_corrections": union,
                "jaccard_overlap": round(intersection / union, 4),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
