"""Evaluate the proposed method (PretrainedTEA).

Scores PretrainedTEA (annual experts on the MLM backbone) and writes the numbers
behind the 2x2 factorial to results/all_systems/: the factorial table (with TEA
as the averaging-only cell), paired bootstrap CIs, the metric panel,
drift-severity, polarity-mismatch and practice-set breakdowns, and the
comparison against the LongEval 2023 teams."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    LABEL2ID, PARQUET, PRACTICE_SPLITS, PROPOSED_SYSTEMS, SEEDS, TEST_SPLITS,
    data_root, get_device, load_checkpoint,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "all_systems"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = get_device()

# Table 7 of the LNCS overview (evaluation phase)
PARTICIPANTS = {
    "Official baseline": (0.7459, 0.6839, 0.6549),
    "Pablojmed": (0.7377, 0.6739, 0.6971),
    "Cordyceps": (0.7246, 0.6771, 0.6751),
    "saroyehun": (0.7203, 0.6674, 0.6874),
    "pakapro": (0.5033, 0.4648, 0.4910),
}


def mf1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


@torch.no_grad()
def predict(model, tokenizer, texts, batch_size=64):
    model.eval()
    probs_all = []
    for i in range(0, len(texts), batch_size):
        chunk = [t.replace("@MENTION", "@user") for t in texts[i:i + batch_size]]
        enc = tokenizer(chunk, truncation=True, padding=True, max_length=128,
                        return_tensors="pt").to(DEVICE)
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        probs_all.append(probs)
    return np.concatenate(probs_all)


def run_inference() -> pd.DataFrame:
    raw_path = OUT / "proposed_preds_raw.parquet"
    if raw_path.exists():
        print(f"reusing {raw_path}")
        return pd.read_parquet(raw_path)

    root = data_root()
    splits = {}
    for name, rel, label_col in TEST_SPLITS + PRACTICE_SPLITS:
        with open(root / rel) as f:
            df = pd.DataFrame(json.load(f))
        df["label_id"] = df[label_col].map(LABEL2ID)
        splits[name] = df

    rows = []
    t0 = time.time()
    for sys_key, prefix in PROPOSED_SYSTEMS.items():
        for seed in SEEDS:
            ckpt = f"{prefix}-seed{seed}"
            print(f"[ {sys_key} seed {seed} ] {ckpt}")
            tok = load_checkpoint(AutoTokenizer, ckpt)
            model = load_checkpoint(AutoModelForSequenceClassification, ckpt).to(DEVICE)
            for split_name, df in splits.items():
                probs = predict(model, tok, df["pp_text"].tolist())
                pred = probs.argmax(axis=1)
                for idx in range(len(df)):
                    rows.append({
                        "system": sys_key, "seed": int(seed), "split": split_name,
                        "idx": idx, "pred": int(pred[idx]),
                        "prob_pos": float(probs[idx, 1]),
                        "gold": int(df["label_id"].iloc[idx]),
                    })
            del model, tok
    out = pd.DataFrame(rows)
    out.to_parquet(raw_path, index=False)
    print(f"inference done in {(time.time()-t0)/60:.1f} min -> {raw_path}")
    return out


def aggregate(raw: pd.DataFrame) -> dict:
    """Soft-vote prediction and mean probability per (system, split, idx)."""
    agg = {}
    for (system, split), g in raw.groupby(["system", "split"]):
        wide = g.pivot_table(index="idx", columns="seed", values="pred")
        prob = g.groupby("idx")["prob_pos"].mean()
        pred = (prob >= 0.5).astype(int)
        gold = g.groupby("idx")["gold"].first()
        ncorr = (wide.eq(gold, axis=0)).sum(axis=1)
        disagree = (wide.nunique(axis=1) > 1).mean()
        agg[(system, split)] = {
            "pred": pred, "prob": prob, "gold": gold,
            "n_correct": ncorr, "seed_disagree": float(disagree),
            "per_seed_pred": wide,
        }
    return agg


def paired_bootstrap(y, pa, pb, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    m = len(y)
    for i in range(n):
        ix = rng.integers(0, m, m)
        diffs[i] = mf1(y[ix], pa[ix]) - mf1(y[ix], pb[ix])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)) * 100, float(lo) * 100, float(hi) * 100


def ece(y, prob, bins=10):
    conf = np.maximum(prob, 1 - prob)
    pred = (prob >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.sum():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def main() -> None:
    raw = run_inference()
    agg = aggregate(raw)

    canon = pd.read_parquet(PARQUET)
    old_systems = ["baseline", "DatePrefix", "MLMPretrain", "RecencyWeight", "TimeLMs", "TEA"]
    splits = ["within", "short", "long"]

    # Convenience accessors over the canonical parquet
    def canon_split(sp):
        d = canon[canon.split == sp].sort_values("idx").reset_index(drop=True)
        y = (d.gold == "positive").astype(int).to_numpy()
        return d, y

    # Factorial table: baseline / MLMPretrain / TEA / PretrainedTEA per split.
    # TEA is the averaging-only cell (annual experts on the base backbone);
    # PretrainedTEA is the same annual averaging on the MLM backbone.
    fact_rows = []
    f1_all: dict[tuple[str, str], float] = {}
    preds_all: dict[tuple[str, str], np.ndarray] = {}
    probs_all: dict[tuple[str, str], np.ndarray] = {}

    for sp in splits:
        d, y = canon_split(sp)
        for s in old_systems:
            preds_all[(s, sp)] = (d[f"{s}_pred"] == "positive").astype(int).to_numpy()
            probs_all[(s, sp)] = d[f"{s}_prob_pos"].to_numpy()
            f1_all[(s, sp)] = mf1(y, preds_all[(s, sp)])
        for s in PROPOSED_SYSTEMS:
            a = agg[(s, sp)]
            preds_all[(s, sp)] = a["pred"].to_numpy()
            probs_all[(s, sp)] = a["prob"].to_numpy()
            f1_all[(s, sp)] = mf1(y, preds_all[(s, sp)])

    for sp in splits:
        b, s2, tea, s10 = (f1_all[(k, sp)] for k in
                          ["baseline", "MLMPretrain", "TEA", "PretrainedTEA"])
        fact_rows.append({
            "split": sp, "baseline": round(b, 4), "backbone": round(s2, 4),
            "mechanism": round(tea, 4), "both": round(s10, 4),
            "backbone_effect_pp": round(100 * (s2 - b), 2),
            "mechanism_effect_pp": round(100 * (tea - b), 2),
            "interaction_pp": round(100 * ((s10 - tea) - (s2 - b)), 2),
        })
    pd.DataFrame(fact_rows).to_csv(OUT / "factorial.csv", index=False)
    print("factorial.csv done")

    # Paired bootstrap CIs for the key comparisons
    comps = []
    for sp in splits:
        _, y = canon_split(sp)
        for a, b, label in [
            ("PretrainedTEA", "baseline", "PretrainedTEA vs baseline"),
            ("PretrainedTEA", "MLMPretrain", "PretrainedTEA vs MLMPretrain (mechanism on top of backbone)"),
            ("PretrainedTEA", "TEA", "PretrainedTEA vs TEA (backbone on top of mechanism)"),
            ("TEA", "baseline", "TEA vs baseline (averaging alone)"),
        ]:
            d, lo, hi = paired_bootstrap(y, preds_all[(a, sp)], preds_all[(b, sp)])
            comps.append({
                "comparison": label, "split": sp, "mean_diff_pp": round(d, 2),
                "ci_lo_pp": round(lo, 1), "ci_hi_pp": round(hi, 1),
                "significant": bool(lo > 0 or hi < 0),
            })
            print(f"  {sp:7s} {label}: {d:+.2f} [{lo:+.1f}, {hi:+.1f}]")
    pd.DataFrame(comps).to_csv(OUT / "paired_bootstrap.csv", index=False)

    # Metric panel: F1 / AUROC / ECE / Brier / seed disagreement
    panel = []
    all_systems = old_systems + list(PROPOSED_SYSTEMS)
    for sp in splits:
        d, y = canon_split(sp)
        for s in all_systems:
            prob = probs_all[(s, sp)]
            if s in PROPOSED_SYSTEMS:
                dis = agg[(s, sp)]["seed_disagree"]
            else:
                dis = float((d[f"{s}_n_correct"] % 3 != 0).mean())
            panel.append({
                "system": s, "split": sp,
                "f1": round(f1_all[(s, sp)], 4),
                "auroc": round(float(roc_auc_score(y, prob)), 4),
                "ece": round(ece(y, prob), 4),
                "brier": round(float(brier_score_loss(y, prob)), 4),
                "seed_disagree": round(dis, 4),
            })
    pd.DataFrame(panel).to_csv(OUT / "metric_panel.csv", index=False)
    print("metric_panel.csv done")

    # Drift-severity quintiles on long, incl. PretrainedTEA
    d, y = canon_split("long")
    score = d["oov_frac"].fillna(0) + d["sem_dist"].fillna(0)
    q = pd.qcut(score, 5, labels=[1, 2, 3, 4, 5])
    sev = []
    for qq in [1, 2, 3, 4, 5]:
        m = (q == qq).to_numpy()
        base = mf1(y[m], preds_all[("baseline", "long")][m])
        row = {"q": qq, "n": int(m.sum())}
        for s in all_systems[1:]:
            row[s] = round(100 * (mf1(y[m], preds_all[(s, "long")][m]) - base), 2)
        sev.append(row)
    pd.DataFrame(sev).to_csv(OUT / "severity.csv", index=False)
    print("severity.csv done")

    # Composite base on long (T5 row added in stage 13): the four robustness
    # measures per system, from the metric panel and the Q5 severity row.
    q5row = sev[-1]  # q == 5
    comp_rows = []
    for s in all_systems:
        r = next(p for p in panel if p["system"] == s and p["split"] == "long")
        comp_rows.append({
            "system": s,
            "f1": r["f1"],
            "auroc": r["auroc"],
            "stability": round(1 - r["seed_disagree"], 4),
            "q5_gain": 0.0 if s == "baseline" else q5row[s],
        })
    pd.DataFrame(comp_rows).to_csv(OUT / "composite.csv", index=False)
    print("composite.csv (base) done")

    # Polarity-mismatch subset per system per split
    pm_rows = []
    for sp in splits:
        d, y = canon_split(sp)
        m = d["polarity_mismatch"].fillna(0).astype(int).to_numpy() == 1
        for s in all_systems:
            v = preds_all[(s, sp)]
            pm_rows.append({
                "system": s, "split": sp,
                "f1_polarity_mismatch": round(mf1(y[m], v[m]), 4),
                "f1_rest": round(mf1(y[~m], v[~m]), 4),
            })
    pd.DataFrame(pm_rows).to_csv(OUT / "polarity_mismatch.csv", index=False)
    print("polarity_mismatch.csv done")

    # Selection trap re-test: PretrainedTEA practice scores
    sel_rows = []
    for s in PROPOSED_SYSTEMS:
        row = {"system": s}
        for prac in ["practice_2016", "practice_2018"]:
            a = agg[(s, prac)]
            row[prac] = round(mf1(a["gold"].to_numpy(), a["pred"].to_numpy()), 4)
        for sp in splits:
            row[f"test_{sp}"] = round(f1_all[(s, sp)], 4)
        sel_rows.append(row)
    pd.DataFrame(sel_rows).to_csv(OUT / "selection.csv", index=False)
    print("selection.csv done")

    # Participant comparison
    part = [{"system": k, "within": v[0], "short": v[1], "long": v[2],
             "source": "Table 7, LNCS overview"} for k, v in PARTICIPANTS.items()]
    part.append({
        "system": "PretrainedTEA (ours)",
        "within": round(f1_all[("PretrainedTEA", "within")], 4),
        "short": round(f1_all[("PretrainedTEA", "short")], 4),
        "long": round(f1_all[("PretrainedTEA", "long")], 4),
        "source": "this work, mean probability over 3 seeds",
    })
    pd.DataFrame(part).to_csv(OUT / "participants.csv", index=False)
    print("participants.csv done")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
