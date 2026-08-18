"""Always-fail residual: the tweets every one of the eight evaluated systems
gets wrong under all three seeds, with the matching always-correct set and the
statistics reported in the paper (macro error floor, gold-positive share,
baseline confidence, polarity-mismatch enrichment).

A cross-strategy analysis, so it runs last, after every system is scored, and
reads only the cached per-tweet predictions (no inference). Writes
results/all_systems/residual.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import PARQUET

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "all_systems"

TEST_SPLITS = ["within", "short", "long"]
# The eight systems the paper reports. Baseline and the five literature
# strategies come from the per-tweet parquet; T5 and PretrainedTEA are added
# from their cached raw predictions below.
PRESENTED = ["baseline", "DatePrefix", "MLMPretrain", "RecencyWeight",
             "TimeLMs", "TEA", "T5", "PretrainedTEA"]


def n_correct_from_raw(path: Path, system: str) -> pd.DataFrame:
    """(split, idx) -> n_correct over the three seeds for one system."""
    r = pd.read_parquet(path)
    r = r[(r.system == system) & (r.split.isin(TEST_SPLITS))].copy()
    r["ok"] = (r["pred"] == r["gold"]).astype(int)
    return (r.groupby(["split", "idx"])["ok"].sum()
             .rename(f"{system}_n_correct").reset_index())


def all_n(d: pd.DataFrame, n: int) -> np.ndarray:
    m = np.ones(len(d), dtype=bool)
    for s in PRESENTED:
        m &= (d[f"{s}_n_correct"] == n).to_numpy()
    return m


def main() -> None:
    # Start with the six systems stored in the canonical per-tweet parquet
    pt = pd.read_parquet(PARQUET)
    pt = pt[pt.split.isin(TEST_SPLITS)].copy()

    # Add the per-seed correctness counts for T5 and PretrainedTEA
    for system, path in [("T5", OUT / "t5_preds_raw.parquet"),
                         ("PretrainedTEA", OUT / "proposed_preds_raw.parquet")]:
        pt = pt.merge(n_correct_from_raw(path, system), on=["split", "idx"], how="left")
        assert not pt[f"{system}_n_correct"].isna().any(), f"{system}: rows missing"
        pt[f"{system}_n_correct"] = pt[f"{system}_n_correct"].astype(int)

    # Summarise tweets missed by every seed of every system, by split and pooled
    rows = []
    for sp in TEST_SPLITS + ["pooled"]:
        d = pt if sp == "pooled" else pt[pt.split == sp]
        af, ac = all_n(d, 0), all_n(d, 3)
        afd, acd = d[af], d[ac]
        pm_af = float(afd["polarity_mismatch"].fillna(0).mean()) if len(afd) else 0.0
        pm_all = float(d["polarity_mismatch"].fillna(0).mean())
        rows.append({
            "split": sp, "n": len(d),
            "always_fail_n": int(af.sum()),
            "always_fail_pct": round(100 * af.mean(), 1),
            "always_correct_n": int(ac.sum()),
            "always_correct_pct": round(100 * ac.mean(), 1),
            "af_gold_pos_pct": round(100 * (afd.gold == "positive").mean(), 1) if len(afd) else None,
            "baseline_conf_af": round(float(afd.baseline_conf.mean()), 3) if len(afd) else None,
            "baseline_conf_ac": round(float(acd.baseline_conf.mean()), 3) if len(acd) else None,
            "polarity_mismatch_rate_af": round(100 * pm_af, 1),
            "polarity_mismatch_rate_all": round(100 * pm_all, 1),
            "polarity_mismatch_enrichment": round(pm_af / pm_all, 2) if pm_all else None,
        })
    # Write the summary used by the paper and dashboard
    pd.DataFrame(rows).to_csv(OUT / "residual.csv", index=False)
    for r in rows:
        print(f"  {r['split']:7s} always-fail {r['always_fail_pct']}% (n={r['always_fail_n']})  "
              f"always-correct {r['always_correct_pct']}%  gold+ {r['af_gold_pos_pct']}%  "
              f"conf AF/AC {r['baseline_conf_af']}/{r['baseline_conf_ac']}")
    print(f"wrote {OUT / 'residual.csv'}")


if __name__ == "__main__":
    main()
