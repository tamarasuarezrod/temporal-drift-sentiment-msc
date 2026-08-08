"""Evaluate the generative T5 strategy.

T5 reframes sentiment as a text-to-text task and is scored by generation
likelihood of "positive" against "negative" rather than by the classification
path the other systems use. Writes its rows into results/all_systems/
(per-split F1, practice scores, long-period metrics, per-tweet predictions)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoTokenizer, T5ForConditionalGeneration

from config import PARQUET, SEEDS, data_root, load_checkpoint

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "all_systems"
OUT.mkdir(parents=True, exist_ok=True)

PREFIX = "sentiment: "
LABELS = ["negative", "positive"]
# baseline + five RoBERTa strategies. T5 is the sixth comparison system
COMPARISON = ["baseline", "S1", "S2", "S3", "S6", "S7"]
SPLITS = {  # name -> (rel path, label column)
    "practice_2016": ("train_eval/interim_eval_2016.json", "distant_label"),
    "practice_2018": ("train_eval/interim_eval_2018.json", "distant_label"),
    "within": ("test/interim_test_2016.json", "label"),
    "short": ("test/interim_test_2018.json", "label"),
    "long": ("test/interim_test_2021.json", "label"),
}
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def mf1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


@torch.no_grad()
def t5_pos_prob(model, tokenizer, texts, batch_size=64):
    """P(positive) via summed token log-likelihood of the label word."""
    model.eval()
    cand = {l: tokenizer(l, return_tensors="pt").input_ids.to(DEVICE) for l in LABELS}
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = [PREFIX + str(t).replace("@MENTION", "@user") for t in texts[i:i + batch_size]]
        enc = tokenizer(chunk, truncation=True, max_length=128, padding=True,
                        return_tensors="pt").to(DEVICE)
        n = enc["input_ids"].size(0)
        logp = {}
        for label, ids in cand.items():
            lab = ids.repeat(n, 1)
            logits = model(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"], labels=lab).logits
            tok_lp = torch.log_softmax(logits, -1).gather(-1, lab.unsqueeze(-1)).squeeze(-1)
            logp[label] = (tok_lp * (lab != tokenizer.pad_token_id).float()).sum(1)
        stacked = torch.stack([logp["negative"], logp["positive"]], 1)
        out.append(torch.softmax(stacked, 1)[:, 1].float().cpu().numpy())
    return np.concatenate(out)


def load_splits():
    root = data_root()
    data = {}
    for name, (rel, label_col) in SPLITS.items():
        with open(root / rel) as f:
            df = pd.DataFrame(json.load(f))
        data[name] = (df["pp_text"].tolist(), (df[label_col] == "positive").astype(int).to_numpy())
    return data


def main() -> None:
    data = load_splits()

    # T5 inference per seed, aligned to the split order above
    preds = {nm: [] for nm in SPLITS}
    probs = {nm: [] for nm in SPLITS}
    for seed in SEEDS:
        ckpt = f"t5-seed{seed}"
        print(f"[ T5 seed {seed} ] {ckpt}")
        tok = load_checkpoint(AutoTokenizer, ckpt)
        model = load_checkpoint(T5ForConditionalGeneration, ckpt).to(DEVICE)
        for nm in SPLITS:
            p = t5_pos_prob(model, tok, data[nm][0])
            preds[nm].append((p >= 0.5).astype(int))
            probs[nm].append(p)
        del model, tok

    # Per-split aggregates (majority vote + mean over the three seeds)
    t5 = {}
    for nm in SPLITS:
        y = data[nm][1]
        sp = np.array(preds[nm])
        mv = (sp.sum(0) >= 2).astype(int)
        t5[nm] = {
            "y": y, "mv": mv,
            "mv_f1": mf1(y, mv),
            "mean_f1": float(np.mean([mf1(y, sp[i]) for i in range(len(SEEDS))])),
            "prob": np.mean(probs[nm], 0),
            "seed_disagree": float(np.mean([len(set(sp[:, j])) > 1 for j in range(sp.shape[1])])),
            "n_correct": (sp == y).sum(0),
        }

    print("\n=== T5 macro-F1 (mean / majority vote) ===")
    for nm in SPLITS:
        print(f"  {nm:14s} mean={t5[nm]['mean_f1']:.4f}  mv={t5[nm]['mv_f1']:.4f}")

    # Long composite metrics, computed against the canonical parquet baseline
    canon = pd.read_parquet(PARQUET)
    cl = canon[canon.split == "long"].sort_values("idx").reset_index(drop=True)
    yL = (cl.gold == "positive").astype(int).to_numpy()
    baseL = (cl.baseline_pred == "positive").astype(int).to_numpy()
    score = cl["oov_frac"].fillna(0).to_numpy() + cl["sem_dist"].fillna(0).to_numpy()
    q5 = (pd.qcut(pd.Series(score), 5, labels=[1, 2, 3, 4, 5]) == 5).to_numpy()
    long_metrics = {
        "f1": round(t5["long"]["mv_f1"], 4),
        "auroc": round(float(roc_auc_score(yL, t5["long"]["prob"])), 4),
        "stability": round(1 - t5["long"]["seed_disagree"], 4),
        "q5_gain": round(100 * (mf1(yL[q5], t5["long"]["mv"][q5]) - mf1(yL[q5], baseL[q5])), 2),
    }
    print(f"\n=== T5 long composite metrics ===\n  {long_metrics}")

    # Always-fail over the seven comparison systems (baseline + 6 strategies
    # incl. T5), aligned to the parquet, plus the long residual stats
    pt = canon.reset_index(drop=True).copy()
    t5nc = np.zeros(len(pt), dtype=int)
    for nm in ["within", "short", "long"]:
        idx = pt.index[pt.split == nm]
        # T5 n_correct aligned by split order (parquet is sorted by idx per split)
        order = pt.loc[idx].sort_values("idx").index
        t5nc[order] = t5[nm]["n_correct"]
    pt["T5_n_correct"] = t5nc

    def always_fail(df, systems):
        fail = np.ones(len(df), dtype=bool)
        for s in systems:
            fail &= (df[f"{s}_n_correct"] == 0).to_numpy()
        return fail

    print("\n=== always-fail rates ===")
    for label, systems in [("6 comparison systems", COMPARISON),
                           ("7 comparison systems (+T5)", COMPARISON + ["T5"])]:
        rates = {sp: 100 * always_fail(pt[pt.split == sp], systems).mean() for sp in ["within", "short", "long"]}
        counts = {sp: int(always_fail(pt[pt.split == sp], systems).sum()) for sp in ["within", "short", "long"]}
        print(f"  {label}: " + "  ".join(f"{sp} {rates[sp]:.1f}% ({counts[sp]})" for sp in rates))

    dl = pt[pt.split == "long"].reset_index(drop=True)
    fail7 = always_fail(dl, COMPARISON + ["T5"])
    afset = dl[fail7]
    ac = np.ones(len(dl), dtype=bool)
    for s in COMPARISON + ["T5"]:
        ac &= (dl[f"{s}_n_correct"] == 3).to_numpy()
    print("\n=== long always-fail set (7 systems) ===")
    print(f"  n={len(afset)}  gold-positive={100*(afset.gold=='positive').mean():.1f}%")
    print(f"  baseline mean conf: always-fail={afset.baseline_conf.mean():.3f}  "
          f"always-correct={dl[ac].baseline_conf.mean():.3f}")

    # Write t5.csv and upsert the T5 row into composite.csv
    t5_summary = {
        "within_mean": round(t5["within"]["mean_f1"], 4),
        "short_mean": round(t5["short"]["mean_f1"], 4),
        "long_mean": round(t5["long"]["mean_f1"], 4),
        "within_mv": round(t5["within"]["mv_f1"], 4),
        "short_mv": round(t5["short"]["mv_f1"], 4),
        "long_mv": round(t5["long"]["mv_f1"], 4),
        "practice_2016": round(t5["practice_2016"]["mv_f1"], 4),
        "practice_2018": round(t5["practice_2018"]["mv_f1"], 4),
        **long_metrics,
    }
    pd.DataFrame([t5_summary]).to_csv(OUT / "t5.csv", index=False)
    print(f"\nt5.csv done -> {t5_summary}")

    # Per-tweet T5 predictions for the three test splits, in the same schema as
    # proposed_preds_raw.parquet, to aggregate T5 like the other systems. The
    # split arrays are in idx order (JSON file order matches the parquet)
    raw_rows = []
    for si, seed in enumerate(SEEDS):
        for nm in ["within", "short", "long"]:
            y = data[nm][1]
            pr = probs[nm][si]
            pd_seed = preds[nm][si]
            for idx in range(len(y)):
                raw_rows.append({
                    "system": "T5",
                    "seed": int(seed),
                    "split": nm,
                    "idx": int(idx),
                    "pred": int(pd_seed[idx]),
                    "prob_pos": float(pr[idx]),
                    "gold": int(y[idx]),
                })
    raw = pd.DataFrame(raw_rows)
    raw.to_parquet(OUT / "t5_preds_raw.parquet", index=False)
    print(f"t5_preds_raw.parquet done -> {len(raw)} rows "
          f"({raw.split.nunique()} splits x {len(SEEDS)} seeds)")

    comp_path = OUT / "composite.csv"
    comp = pd.read_csv(comp_path)
    comp = comp[comp.system != "T5"]
    t5_row = {"system": "T5", **long_metrics}
    comp = pd.concat([comp, pd.DataFrame([t5_row])], ignore_index=True)
    # recompute the composite column over the eight deployment systems (excl. S8 ablation)
    order = ["baseline", "S1", "S2", "S3", "S6", "S7", "S10", "T5"]
    sub = comp.set_index("system").loc[order, ["f1", "auroc", "stability", "q5_gain"]]
    norm = (sub - sub.min()) / (sub.max() - sub.min())
    tot = norm.sum(1)
    comp["composite"] = comp.system.map(tot).fillna(comp.get("composite"))
    comp.to_csv(comp_path, index=False)
    print("composite.csv updated; eight-system ranking:")
    for r, (s, v) in enumerate(tot.sort_values(ascending=False).items(), 1):
        print(f"  {r}. {s:<10} {v:.3f}")
    print("\nALL DONE")


if __name__ == "__main__":
    main()
