"""Evaluate the generative T5 strategy.

T5 reframes sentiment as a text-to-text task and is scored by generation
likelihood of "positive" against "negative" rather than by the classification
path the other systems use. Writes its rows into results/all_systems/
(per-split F1, practice scores, robustness metrics and per-tweet predictions)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoTokenizer, T5ForConditionalGeneration

from config import (
    PARQUET, PRACTICE_SPLITS, SEEDS, TEST_SPLITS,
    aggregate_seed_probabilities, data_root, get_device, load_checkpoint,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "all_systems"
OUT.mkdir(parents=True, exist_ok=True)

PREFIX = "sentiment: "
LABELS = ["negative", "positive"]
# name -> (rel path, label column), practice sets first
SPLITS = {name: (rel, lc) for name, rel, lc in PRACTICE_SPLITS + TEST_SPLITS}
DEVICE = get_device()


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

    # Per-split aggregates: mean probability across seeds, thresholded at 0.5.
    t5 = {}
    for nm in SPLITS:
        y = data[nm][1]
        sp = np.array(preds[nm])
        mean_prob, pred = aggregate_seed_probabilities(probs[nm])
        t5[nm] = {
            "y": y, "pred": pred,
            "f1": mf1(y, pred),
            "mean_seed_f1": float(np.mean([mf1(y, sp[i]) for i in range(len(SEEDS))])),
            "prob": mean_prob,
            "seed_disagree": float(np.mean([len(set(sp[:, j])) > 1 for j in range(sp.shape[1])])),
            "n_correct": (sp == y).sum(0),
        }

    print("\n=== T5 macro-F1 (mean per-seed F1 / soft vote) ===")
    for nm in SPLITS:
        print(f"  {nm:14s} mean_seed={t5[nm]['mean_seed_f1']:.4f}  "
              f"soft={t5[nm]['f1']:.4f}")

    # Composite inputs for all three test periods, computed against the
    # canonical parquet baseline. The highest-drift quintile is defined within
    # each period, matching the long-only calculation used previously.
    canon = pd.read_parquet(PARQUET)
    test_splits = ["within", "short", "long"]
    robustness = {}
    t5_severity = {}
    for nm in test_splits:
        split_df = canon[canon.split == nm].sort_values("idx").reset_index(drop=True)
        y = (split_df.gold == "positive").astype(int).to_numpy()
        baseline_pred = (split_df.baseline_pred == "positive").astype(int).to_numpy()
        score = split_df["oov_frac"].fillna(0) + split_df["sem_dist"].fillna(0)
        quintile = pd.qcut(score, 5, labels=[1, 2, 3, 4, 5])
        q5 = (quintile == 5).to_numpy()
        robustness[nm] = {
            "f1": round(t5[nm]["f1"], 4),
            "auroc": round(float(roc_auc_score(y, t5[nm]["prob"])), 4),
            "stability": round(1 - t5[nm]["seed_disagree"], 4),
            "q5_gain": round(
                100 * (mf1(y[q5], t5[nm]["pred"][q5]) - mf1(y[q5], baseline_pred[q5])),
                2,
            ),
        }
        for qq in [1, 2, 3, 4, 5]:
            mask = (quintile == qq).to_numpy()
            t5_severity[(nm, qq)] = round(
                100 * (
                    mf1(y[mask], t5[nm]["pred"][mask])
                    - mf1(y[mask], baseline_pred[mask])
                ),
                2,
            )
    print("\n=== T5 composite inputs by split ===")
    for nm in test_splits:
        print(f"  {nm:7s} {robustness[nm]}")

    # Write t5.csv. The unprefixed metric names remain the backwards-compatible
    # long-period view; prefixed columns expose every test period.
    t5_summary = {
        "within_mean_seed_f1": round(t5["within"]["mean_seed_f1"], 4),
        "short_mean_seed_f1": round(t5["short"]["mean_seed_f1"], 4),
        "long_mean_seed_f1": round(t5["long"]["mean_seed_f1"], 4),
        "within_f1": round(t5["within"]["f1"], 4),
        "short_f1": round(t5["short"]["f1"], 4),
        "long_f1": round(t5["long"]["f1"], 4),
        "practice_2016": round(t5["practice_2016"]["f1"], 4),
        "practice_2018": round(t5["practice_2018"]["f1"], 4),
        **{
            f"{nm}_{metric}": value
            for nm in test_splits
            for metric, value in robustness[nm].items()
        },
        **robustness["long"],
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

    # Add T5 to the per-period severity table.
    severity_path = OUT / "severity_by_split.csv"
    severity = pd.read_csv(severity_path)
    severity["T5"] = [t5_severity[(r.split, int(r.q))] for r in severity.itertuples()]
    severity.to_csv(severity_path, index=False)

    # Add T5 and recompute the composite independently within every test split.
    comp_path = OUT / "composite_by_split.csv"
    comp = pd.read_csv(comp_path)
    comp = comp[comp.system != "T5"]
    t5_rows = [
        {"split": nm, "system": "T5", **robustness[nm]}
        for nm in test_splits
    ]
    comp = pd.concat([comp, pd.DataFrame(t5_rows)], ignore_index=True)
    order = ["baseline", "DatePrefix", "MLMPretrain", "RecencyWeight", "TimeLMs", "TEA", "PretrainedTEA", "T5"]
    scored = []
    rankings = {}
    for nm in test_splits:
        block = comp[comp.split == nm].set_index("system").loc[
            order, ["f1", "auroc", "stability", "q5_gain"]
        ]
        norm = (block - block.min()) / (block.max() - block.min())
        total = norm.sum(axis=1)
        out = block.reset_index()
        out.insert(0, "split", nm)
        out["composite"] = out.system.map(total)
        scored.append(out)
        rankings[nm] = total.sort_values(ascending=False)
    comp = pd.concat(scored, ignore_index=True)
    comp.to_csv(comp_path, index=False)
    print("composite_by_split.csv updated; eight-system rankings:")
    for nm in test_splits:
        print(f"  [{nm}]")
        for rank, (system, value) in enumerate(rankings[nm].items(), 1):
            print(f"    {rank}. {system:<14} {value:.3f}")
    print("\nALL DONE")


if __name__ == "__main__":
    main()
