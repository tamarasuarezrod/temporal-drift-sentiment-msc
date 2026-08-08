"""Flag each test tweet whose surface word polarity contradicts its gold label.

Positive and negative words are counted with the Hu & Liu (2004) lexicon, a
tweet is flagged when the words point one way and the label the other. An
irony-model probability is stored alongside for reference. Writes
results/metrics/pragmatic_flags.csv."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    data_root,
    IRONY_MODEL,
    PRAGMATIC_CSV,
    SPLITS,
)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

TOKEN_RE = re.compile(r"[a-z']+")


def load_lexicon() -> tuple[set[str], set[str]]:
    import nltk
    try:
        from nltk.corpus import opinion_lexicon
        pos = set(opinion_lexicon.positive())
    except LookupError:
        nltk.download("opinion_lexicon", quiet=True)
        from nltk.corpus import opinion_lexicon
        pos = set(opinion_lexicon.positive())
    from nltk.corpus import opinion_lexicon
    neg = set(opinion_lexicon.negative())
    return pos, neg


def polarity_counts(text: str, pos: set[str], neg: set[str]) -> tuple[int, int]:
    toks = TOKEN_RE.findall(str(text).lower())
    return sum(1 for t in toks if t in pos), sum(1 for t in toks if t in neg)


def flag_direction(lex_pos: int, lex_neg: int, gold: str) -> str | None:
    """Flag a tweet whose words point strictly one way while the gold label
    points the other: all-positive words with a negative label (sarcasm), or
    all-negative words with a positive label (softened complaint or joke)."""
    if lex_pos > 0 and lex_neg == 0 and gold == "negative":
        return "B_sarcasm"
    if lex_neg > 0 and lex_pos == 0 and gold == "positive":
        return "A_joking_complaint"
    return None


@torch.no_grad()
def irony_scores(model, tokenizer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        chunk = [t.replace("@MENTION", "@user") for t in texts[i:i + batch_size]]
        enc = tokenizer(
            chunk, truncation=True, padding=True, max_length=128, return_tensors="pt"
        ).to(DEVICE)
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        all_probs.append(probs[:, 1])  # class 1 = irony
    return np.concatenate(all_probs)


def main() -> None:
    print(f"Device: {DEVICE}")
    pos, neg = load_lexicon()
    print(f"Hu-Liu opinion lexicon: {len(pos)} positive, {len(neg)} negative words")

    print(f"Loading {IRONY_MODEL} for the corroboration column ...")
    tokenizer = AutoTokenizer.from_pretrained(IRONY_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(IRONY_MODEL).to(DEVICE)

    root = data_root()
    frames = []
    for split_name, rel in SPLITS:
        with open(root / rel) as f:
            records = json.load(f)
        texts = [r["pp_text"] for r in records]
        golds = [r["label"] for r in records]

        rows = []
        for idx, (text, gold) in enumerate(zip(texts, golds)):
            p, n = polarity_counts(text, pos, neg)
            direction = flag_direction(p, n, gold)
            rows.append({
                "split": split_name, "idx": idx,
                "lex_pos": p, "lex_neg": n,
                "pragmatic": int(direction is not None),
                "pragmatic_dir": direction,
            })
        df = pd.DataFrame(rows)
        df["irony_prob"] = irony_scores(model, tokenizer, texts)

        rate = df["pragmatic"].mean() * 100
        irony_in_flagged = df.loc[df["pragmatic"] == 1, "irony_prob"].mean()
        irony_overall = df["irony_prob"].mean()
        print(f"  {split_name}: pragmatic={rate:.1f}%  "
              f"(irony prob: flagged {irony_in_flagged:.3f} vs overall {irony_overall:.3f})")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    PRAGMATIC_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PRAGMATIC_CSV, index=False)
    print(f"\nWrote {PRAGMATIC_CSV}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
