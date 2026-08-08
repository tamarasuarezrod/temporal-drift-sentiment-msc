"""Run inference for the six RoBERTa classifiers on the three test splits.

Covers the baseline and the five sequence-classification strategies, which
all share this scoring path. The generative T5 needs a different path and is
handled in 13_evaluate_t5.py, the proposed method is analysed in 12_evaluate_pretrainedtea.py.

Writes results/preds_raw.parquet, one row per (system, seed, split, tweet).
Seeds are aggregated in 05_build_parquet.py."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    ANALYSIS_DIR,
    data_root,
    PREDS_RAW,
    SEEDS,
    SPLITS,
    SYSTEMS,
    load_checkpoint,
)

LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {DEVICE}")


def load_split(root: Path, rel_path: str) -> pd.DataFrame:
    with open(root / rel_path) as f:
        records = json.load(f)
    df = pd.DataFrame(records).reset_index(drop=True)
    df["label_id"] = df["label"].map(LABEL2ID)
    return df


@torch.no_grad()
def predict(model, tokenizer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Softmax class probabilities for a list of tweets, in batches."""
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        chunk = [t.replace("@MENTION", "@user") for t in texts[i:i + batch_size]]
        enc = tokenizer(
            chunk, truncation=True, padding=True, max_length=128, return_tensors="pt"
        ).to(DEVICE)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the three test splits with their gold labels
    root = data_root()
    splits = {name: load_split(root, rel) for name, rel in SPLITS}
    for name, df in splits.items():
        pos_rate = df["label_id"].mean()
        print(f"  {name}: n={len(df)}  positive_rate={pos_rate:.3f}")

    # For each system and seed, download the checkpoint and predict on every split
    rows = []
    total_start = time.time()
    for strategy, ckpt_id in SYSTEMS:
        for seed in SEEDS:
            name = f"{ckpt_id}-seed{seed}"
            print(f"\n[ {strategy}  seed {seed} ]  {name}")
            t0 = time.time()
            try:
                tokenizer = load_checkpoint(AutoTokenizer, name)
                model = load_checkpoint(AutoModelForSequenceClassification, name).to(DEVICE)
            except Exception as e:
                print(f"  FAILED to load {name}: {e}")
                continue
            print(f"  loaded in {time.time() - t0:.1f}s")

            for split_name, df in splits.items():
                probs = predict(model, tokenizer, df["pp_text"].tolist())
                pred = probs.argmax(axis=1)
                prob_pos = probs[:, LABEL2ID["positive"]]
                conf = probs.max(axis=1)
                correct = (pred == df["label_id"].to_numpy()).astype(int)

                for idx, (p, pp, c, cor) in enumerate(zip(pred, prob_pos, conf, correct)):
                    rows.append({
                        "strategy": strategy,
                        "seed": int(seed),
                        "split": split_name,
                        "idx": int(idx),
                        "pred": ID2LABEL[int(p)],
                        "prob_pos": float(pp),
                        "conf": float(c),
                        "correct": int(cor),
                    })
            # Free memory
            del model, tokenizer
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out.to_parquet(PREDS_RAW, index=False)
    total = time.time() - total_start
    print(f"\nWrote {PREDS_RAW}  ({len(out)} rows, {total/60:.1f} min total)")


if __name__ == "__main__":
    main()
