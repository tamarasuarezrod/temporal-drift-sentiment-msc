"""Check whether any test tweet also appears in the MLM pretraining corpus,
by exact match and near-duplicate hash. Writes results/metrics/contamination_check.csv."""

from __future__ import annotations

import hashlib
import json
import re

import pandas as pd

from config import METRICS_DIR, PARQUET, UNLABELED_CORPUS

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
NONALNUM_RE = re.compile(r"[^a-z0-9 ]")
WS_RE = re.compile(r"\s+")


def normalise(text: str) -> str:
    t = str(text).lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = t.replace("@mention", " ").replace("@user", " ")
    t = NONALNUM_RE.sub(" ", t)
    return WS_RE.sub(" ", t).strip()


def token_hash(norm: str) -> str:
    toks = sorted(set(norm.split()))
    return hashlib.md5(" ".join(toks).encode()).hexdigest()


def main() -> None:
    if not UNLABELED_CORPUS.exists():
        raise SystemExit(f"Missing {UNLABELED_CORPUS}. Run 01_fetch_data.py first.")

    # Index every test tweet by its normalised text and its token-set hash
    df = pd.read_parquet(PARQUET)
    test = df[["split", "idx", "text"]].copy()
    test["norm"] = test["text"].map(normalise)
    test["thash"] = test["norm"].map(token_hash)
    norm_index = {n: i for i, n in zip(test.index, test["norm"]) if n}
    thash_index: dict[str, int] = {}
    for i, h in zip(test.index, test["thash"]):
        thash_index.setdefault(h, i)
    print(f"Indexed {len(test)} test tweets "
          f"({len(norm_index)} unique normalised, {len(thash_index)} token hashes)")

    # Stream the pretraining corpus once, flagging any tweet that matches a
    # test tweet exactly or as a near-duplicate
    exact_hits, near_hits = [], []
    n_corpus = 0
    with open(UNLABELED_CORPUS) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_corpus += 1
            raw = rec.get("text") or ""
            norm = normalise(raw)
            if not norm:
                continue
            if norm in norm_index:
                exact_hits.append((norm_index[norm], raw))
            else:
                h = token_hash(norm)
                if h in thash_index:
                    near_hits.append((thash_index[h], raw))
            if n_corpus % 200_000 == 0:
                print(f"  scanned {n_corpus:,} corpus tweets ... "
                      f"exact={len(exact_hits)} near={len(near_hits)}")

    print(f"\nScanned {n_corpus:,} corpus tweets.")
    print(f"Exact-normalised matches: {len(exact_hits)}")
    print(f"Near-duplicate (token-set) matches: {len(near_hits)}")

    rows = []
    for kind, hits in [("exact", exact_hits), ("near", near_hits)]:
        for test_i, corpus_text in hits:
            r = test.loc[test_i]
            rows.append({
                "kind": kind, "split": r["split"], "idx": r["idx"],
                "test_text": r["text"], "corpus_text": corpus_text,
            })
    out = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(METRICS_DIR / "contamination_check.csv", index=False)
    print(f"Wrote {METRICS_DIR / 'contamination_check.csv'} ({len(out)} rows)")

    if not rows:
        print("\nNo overlap found: the MLM pretraining corpus does not "
              "contain any of the 2,724 test tweets.")


if __name__ == "__main__":
    main()
