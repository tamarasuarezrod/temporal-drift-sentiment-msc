"""Evaluate the proposed method (PretrainedTEA) and its ablations.

Scores the averaging-only (S8) and full (S10) systems and writes the numbers
behind the 2x2 factorial to results/all_systems/: the factorial table, paired
bootstrap CIs, the metric panel, drift-severity, residual, pragmatic and
practice-set breakdowns, and the comparison against the LongEval 2023 teams."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import PARQUET, SEEDS, data_root, load_checkpoint

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "all_systems"
OUT.mkdir(parents=True, exist_ok=True)

NEW_SYSTEMS = {
    "S8": "expertaveraging",
    "S10": "pretrainedtea",
}
TEST_SPLITS = [
    ("within", "test/interim_test_2016.json", "label"),
    ("short", "test/interim_test_2018.json", "label"),
    ("long", "test/interim_test_2021.json", "label"),
]
PRACTICE_SPLITS = [
    ("practice_2016", "train_eval/interim_eval_2016.json", "distant_label"),
    ("practice_2018", "train_eval/interim_eval_2018.json", "distant_label"),
]
LABEL2ID = {"negative": 0, "positive": 1}

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

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
    for sys_key, prefix in NEW_SYSTEMS.items():
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
    """Majority vote + mean prob per (system, split, idx)."""
    agg = {}
    for (system, split), g in raw.groupby(["system", "split"]):
        wide = g.pivot_table(index="idx", columns="seed", values="pred")
        vote = (wide.sum(axis=1) >= 2).astype(int)
        prob = g.groupby("idx")["prob_pos"].mean()
        gold = g.groupby("idx")["gold"].first()
        ncorr = (wide.eq(gold, axis=0)).sum(axis=1)
        disagree = (wide.nunique(axis=1) > 1).mean()
        agg[(system, split)] = {
            "vote": vote, "prob": prob, "gold": gold,
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
    old_systems = ["baseline", "S1", "S2", "S3", "S6", "S7"]
    splits = ["within", "short", "long"]

    # Convenience accessors over the canonical parquet
    def canon_split(sp):
        d = canon[canon.split == sp].sort_values("idx").reset_index(drop=True)
        y = (d.gold == "positive").astype(int).to_numpy()
        return d, y

    # Factorial table: baseline / S2 / S8 / S10 per split
    fact_rows = []
    f1_all: dict[tuple[str, str], float] = {}
    votes_all: dict[tuple[str, str], np.ndarray] = {}
    probs_all: dict[tuple[str, str], np.ndarray] = {}

    for sp in splits:
        d, y = canon_split(sp)
        for s in old_systems:
            votes_all[(s, sp)] = (d[f"{s}_pred"] == "positive").astype(int).to_numpy()
            probs_all[(s, sp)] = d[f"{s}_prob_pos"].to_numpy()
            f1_all[(s, sp)] = mf1(y, votes_all[(s, sp)])
        for s in NEW_SYSTEMS:
            a = agg[(s, sp)]
            votes_all[(s, sp)] = a["vote"].to_numpy()
            probs_all[(s, sp)] = a["prob"].to_numpy()
            f1_all[(s, sp)] = mf1(y, votes_all[(s, sp)])

    for sp in splits:
        b, s2, s8, s10 = (f1_all[(k, sp)] for k in ["baseline", "S2", "S8", "S10"])
        fact_rows.append({
            "split": sp, "baseline": round(b, 4), "S2_backbone": round(s2, 4),
            "S8_mechanism": round(s8, 4), "S10_both": round(s10, 4),
            "backbone_effect_pp": round(100 * (s2 - b), 2),
            "mechanism_effect_pp": round(100 * (s8 - b), 2),
            "interaction_pp": round(100 * ((s10 - s8) - (s2 - b)), 2),
        })
    pd.DataFrame(fact_rows).to_csv(OUT / "factorial.csv", index=False)
    print("factorial.csv done")

    # Paired bootstrap CIs for the key comparisons
    comps = []
    for sp in splits:
        _, y = canon_split(sp)
        for a, b, label in [
            ("S10", "baseline", "S10 vs baseline"),
            ("S10", "S2", "S10 vs S2 (mechanism on top of backbone)"),
            ("S10", "S8", "S10 vs S8 (backbone on top of mechanism)"),
            ("S10", "S7", "S10 vs S7 (TEA)"),
            ("S8", "baseline", "S8 vs baseline"),
        ]:
            d, lo, hi = paired_bootstrap(y, votes_all[(a, sp)], votes_all[(b, sp)])
            comps.append({
                "comparison": label, "split": sp, "mean_diff_pp": round(d, 2),
                "ci_lo_pp": round(lo, 1), "ci_hi_pp": round(hi, 1),
                "significant": bool(lo > 0 or hi < 0),
            })
            print(f"  {sp:7s} {label}: {d:+.2f} [{lo:+.1f}, {hi:+.1f}]")
    pd.DataFrame(comps).to_csv(OUT / "paired_bootstrap.csv", index=False)

    # Metric panel: F1 / AUROC / ECE / Brier / seed disagreement
    panel = []
    all_systems = old_systems + list(NEW_SYSTEMS)
    for sp in splits:
        d, y = canon_split(sp)
        for s in all_systems:
            prob = probs_all[(s, sp)]
            if s in NEW_SYSTEMS:
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

    # Drift-severity quintiles on long, incl. S8 / S10
    d, y = canon_split("long")
    score = d["oov_frac"].fillna(0) + d["sem_dist"].fillna(0)
    q = pd.qcut(score, 5, labels=[1, 2, 3, 4, 5])
    sev = []
    for qq in [1, 2, 3, 4, 5]:
        m = (q == qq).to_numpy()
        base = mf1(y[m], votes_all[("baseline", "long")][m])
        row = {"q": qq, "n": int(m.sum())}
        for s in all_systems[1:]:
            row[s] = round(100 * (mf1(y[m], votes_all[(s, "long")][m]) - base), 2)
        sev.append(row)
    pd.DataFrame(sev).to_csv(OUT / "severity.csv", index=False)
    print("severity.csv done")

    # Residual: always-fail with 8 systems x 3 seeds (24 predictions)
    res = []
    for sp in splits:
        d, y = canon_split(sp)
        ncorr6 = sum(d[f"{s}_n_correct"] for s in old_systems).to_numpy()
        ncorr8 = ncorr6 + agg[("S8", sp)]["n_correct"].to_numpy() \
                        + agg[("S10", sp)]["n_correct"].to_numpy()
        af6 = ncorr6 == 0
        af8 = ncorr8 == 0
        cracked = int(af6.sum() - af8.sum())
        s10_fixes = int(((agg[("S10", sp)]["n_correct"].to_numpy() > 0) & af6).sum())
        res.append({
            "split": sp,
            "always_fail_6sys_pct": round(100 * af6.mean(), 1),
            "always_fail_8sys_pct": round(100 * af8.mean(), 1),
            "cracked_by_S8_or_S10": cracked,
            "old_af_where_S10_correct_any_seed": s10_fixes,
        })
    pd.DataFrame(res).to_csv(OUT / "residual.csv", index=False)
    print("residual.csv done")

    # Pragmatic subset per system per split
    prag_rows = []
    for sp in splits:
        d, y = canon_split(sp)
        m = d["pragmatic"].fillna(0).astype(int).to_numpy() == 1
        for s in all_systems:
            v = votes_all[(s, sp)]
            prag_rows.append({
                "system": s, "split": sp,
                "f1_pragmatic": round(mf1(y[m], v[m]), 4),
                "f1_rest": round(mf1(y[~m], v[~m]), 4),
            })
    pd.DataFrame(prag_rows).to_csv(OUT / "pragmatic.csv", index=False)
    print("pragmatic.csv done")

    # Selection trap re-test: S8/S10 practice scores
    sel_rows = []
    for s in NEW_SYSTEMS:
        row = {"system": s}
        for prac in ["practice_2016", "practice_2018"]:
            a = agg[(s, prac)]
            row[prac] = round(mf1(a["gold"].to_numpy(), a["vote"].to_numpy()), 4)
        for sp in splits:
            row[f"test_{sp}"] = round(f1_all[(s, sp)], 4)
        sel_rows.append(row)
    pd.DataFrame(sel_rows).to_csv(OUT / "selection.csv", index=False)
    print("selection.csv done")

    # Participant comparison
    part = [{"system": k, "within": v[0], "short": v[1], "long": v[2],
             "source": "Table 7, LNCS overview"} for k, v in PARTICIPANTS.items()]
    part.append({
        "system": "PretrainedTEA (S10, ours)",
        "within": round(f1_all[("S10", "within")], 4),
        "short": round(f1_all[("S10", "short")], 4),
        "long": round(f1_all[("S10", "long")], 4),
        "source": "this work, majority vote over 3 seeds",
    })
    pd.DataFrame(part).to_csv(OUT / "participants.csv", index=False)
    print("participants.csv done")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
