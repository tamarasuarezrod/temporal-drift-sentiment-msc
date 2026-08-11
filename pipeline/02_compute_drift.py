"""Compute lexical and semantic drift for each test split.

Writes results/metrics/drift_metrics.csv (per-tweet lexical drift) and
results/metrics/drift_summary.csv (per-split semantic shift)."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (
    DRIFT_CSV,
    DRIFT_EMBED_BATCH,
    DRIFT_LABEL_MIN_FREQ,
    DRIFT_MAX_TWEETS_PER_YEAR,
    DRIFT_MIN_TWEETS_PER_WORD,
    DRIFT_YEARS,
    data_root,
    MINILM_MODEL,
    SPLITS,
    UNLABELED_CORPUS,
)

STOPWORDS = {
    "the","a","an","and","or","but","if","of","in","on","at","to","for",
    "with","from","by","as","is","was","are","were","be","been","being",
    "i","you","he","she","we","they","it","this","that","these","those",
    "my","your","his","her","our","their","me","him","us","them","do",
    "does","did","have","has","had","will","would","can","could","should",
    "may","might","must","not","no","so","than","then","there","here",
    "when","where","what","which","who","whom","whose","how","why","just",
    "now","also","too","very","much","more","most","some","any","all",
    "both","each","few","many","other","such","only","own","same","rt",
    "user","amp","http","https","'s","'m","'ll","'re","'ve","'d",
}

TOKEN_RE = re.compile(r"[a-z][a-z']{1,}")


def tokenize(text: str, drop_stop: bool = False) -> list[str]:
    """Tokeniser for lexical drift. The default splits on whitespace. The
    stopword-removing regex path is only for the semantic-drift step."""
    if drop_stop:
        return [t for t in TOKEN_RE.findall(text.lower())
                if t not in STOPWORDS and len(t) >= 3]
    return text.lower().split()


# Lexical drift (against train vocabulary)
def build_train_vocab(root: Path) -> set[str]:
    with open(root / "train_eval/train.json") as f:
        recs = json.load(f)
    vocab: set[str] = set()
    for r in recs:
        vocab.update(tokenize(r["pp_text"]))
    return vocab


def lexical_metrics(root: Path, train_vocab: set[str]) -> pd.DataFrame:
    """Per-tweet OOV counts, aggregated to a token-level rate for Table 2."""
    rows = []
    per_split_new_types = {}
    for name, rel in SPLITS:
        with open(root / rel) as f:
            recs = json.load(f)
        split_types: set[str] = set()
        for idx, r in enumerate(recs):
            toks = tokenize(r["pp_text"])
            split_types.update(toks)
            n_oov = sum(1 for t in toks if t not in train_vocab)
            rows.append({
                "split": name, "idx": idx,
                "n_tokens": int(len(toks)),
                "n_oov": int(n_oov),
                "oov_frac": (n_oov / len(toks)) if toks else 0.0,
            })
        new = split_types - train_vocab
        per_split_new_types[name] = len(new) / len(split_types) if split_types else 0.0
    df = pd.DataFrame(rows)
    df["new_type_rate"] = df["split"].map(per_split_new_types)
    return df


# Semantic drift (contextual per-word prototypes)
def sample_corpus_by_year() -> dict[int, list[str]]:
    if not UNLABELED_CORPUS.exists():
        print(f"  {UNLABELED_CORPUS} not found. Skipping semantic drift.")
        return {}
    by_year: dict[int, list[str]] = {y: [] for y in DRIFT_YEARS}
    full = {y: False for y in DRIFT_YEARS}
    with open(UNLABELED_CORPUS) as f:
        for line in f:
            if all(full.values()):
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = rec.get("created_at") or rec.get("date") or ""
            # Twitter format is "Tue Jan 01 23:02:00 +0000 2013" (year at
            # the end). Fall back to ISO if the format is different
            year = None
            if ts and ts[:4].isdigit():
                year = int(ts[:4])
            elif ts and ts[-4:].isdigit():
                year = int(ts[-4:])
            if year in by_year and not full[year]:
                by_year[year].append(rec.get("text") or rec.get("pp_text") or "")
                if len(by_year[year]) >= DRIFT_MAX_TWEETS_PER_YEAR:
                    full[year] = True
    for y, ts in by_year.items():
        print(f"  {y}: {len(ts)} tweets")
    return by_year


def _word_index(embs: np.ndarray, texts: list[str]) -> dict[str, list[int]]:
    """Return {word -> list of tweet indices that contain it}."""
    idx: dict[str, list[int]] = defaultdict(list)
    for i, text in enumerate(texts):
        for tok in set(tokenize(text, drop_stop=True)):
            idx[tok].append(i)
    return idx


def _prototype(embs: np.ndarray, positions: list[int]) -> np.ndarray:
    vec = embs[positions].mean(axis=0)
    return vec / (np.linalg.norm(vec) + 1e-12)


def encode_by_year(model: SentenceTransformer, by_year: dict[int, list[str]]):
    embeddings: dict[int, np.ndarray] = {}
    indices: dict[int, dict[str, list[int]]] = {}
    for year, tweets in by_year.items():
        if not tweets:
            embeddings[year] = np.zeros((0, 384), dtype="float32")
            indices[year] = {}
            continue
        print(f"  Encoding {len(tweets)} tweets for {year} ...")
        embeddings[year] = model.encode(
            tweets, batch_size=DRIFT_EMBED_BATCH,
            convert_to_numpy=True, show_progress_bar=False,
        )
        indices[year] = _word_index(embeddings[year], tweets)
        n_words = sum(1 for w, p in indices[year].items() if len(p) >= DRIFT_MIN_TWEETS_PER_WORD)
        print(f"    {n_words} words with >= {DRIFT_MIN_TWEETS_PER_WORD} occurrences")
    return embeddings, indices


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def semantic_summary(embeddings: dict[int, np.ndarray],
                     indices: dict[int, dict[str, list[int]]]) -> pd.DataFrame:
    """Per-word contextual shift, with a noise floor from a 2016 self-split
    (share_above_floor is the fraction of shifts exceeding that floor)."""
    if not embeddings.get(2016, np.array([])).size:
        return pd.DataFrame()

    # Full-year prototypes (used for real 2016->2018 and 2016->2021 shifts)
    protos: dict[int, dict[str, np.ndarray]] = {}
    for year in embeddings:
        protos[year] = {
            w: _prototype(embeddings[year], positions)
            for w, positions in indices[year].items()
            if len(positions) >= DRIFT_MIN_TWEETS_PER_WORD
        }

    # Noise floor: 2016 self-split, half prototypes vs half prototypes
    print("  Building 2016 self-split noise floor ...")
    n16 = embeddings[2016].shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(n16)
    halves = (set(perm[: n16 // 2].tolist()),
              set(perm[n16 // 2 : 2 * (n16 // 2)].tolist()))
    half_protos = []
    for half in halves:
        h_protos: dict[str, np.ndarray] = {}
        for w, positions in indices[2016].items():
            local = [i for i in positions if i in half]
            if len(local) >= DRIFT_MIN_TWEETS_PER_WORD // 2:
                h_protos[w] = _prototype(embeddings[2016], local)
        half_protos.append(h_protos)
    common_half = set(half_protos[0]) & set(half_protos[1])
    noise_shifts = [cosine(half_protos[0][w], half_protos[1][w]) for w in common_half]
    p90_noise = float(np.percentile(noise_shifts, 90)) if noise_shifts else 0.0
    print(f"    noise floor n={len(noise_shifts)}  p90={p90_noise:.4f}")

    # Real shifts (restricted to words present in ALL three years, to keep the
    # short and long numbers on the same vocabulary, matching the project's
    # earlier drift notebook)
    common_all = set(protos.get(2016, {})) & set(protos.get(2018, {})) & set(protos.get(2021, {}))
    rows = []
    for pair_name, year in [("short", 2018), ("long", 2021)]:
        if not protos.get(year) or not common_all:
            continue
        shifts = [cosine(protos[2016][w], protos[year][w]) for w in sorted(common_all)]
        rows.append({
            "split": pair_name,
            "mean_shift": float(np.mean(shifts)),
            "share_above_floor": float(np.mean([s > p90_noise for s in shifts])),
            "noise_floor": p90_noise,
            "n_words": len(shifts),
        })
    return pd.DataFrame(rows)


def label_conditional_drift(root: Path) -> pd.DataFrame:
    """Real-drift probe: does P(positive | word), the label distribution
    conditioned on a word, change between periods? For each content word frequent
    enough in both within and a later split, a two-proportion z-test compares
    the positive rate; we report how many shifts are significant at p<0.05 and
    how many survive a Bonferroni correction. Power is limited by the small
    (908-tweet) test splits."""
    from scipy.stats import norm

    def load(rel):
        with open(root / rel) as f:
            recs = json.load(f)
        return [(set(tokenize(r["pp_text"], drop_stop=True)),
                 1 if r["label"] == "positive" else 0) for r in recs]

    def counts(data):
        cnt: dict[str, int] = defaultdict(int)
        pos: dict[str, int] = defaultdict(int)
        for toks, y in data:
            for w in toks:
                cnt[w] += 1
                pos[w] += y
        return {w: (pos[w], cnt[w]) for w in cnt if cnt[w] >= DRIFT_LABEL_MIN_FREQ}

    within = counts(load("test/interim_test_2016.json"))
    rows = []
    for name, rel in [("short", "test/interim_test_2018.json"),
                      ("long", "test/interim_test_2021.json")]:
        later = counts(load(rel))
        common = sorted(set(within) & set(later))
        if not common:
            continue
        diffs, pvals = [], []
        for w in common:
            aw, nw = within[w]
            al, nl = later[w]
            pw, pl = aw / nw, al / nl
            pool = (aw + al) / (nw + nl)
            se = np.sqrt(pool * (1 - pool) * (1 / nw + 1 / nl))
            z = (pw - pl) / se if se > 0 else 0.0
            pvals.append(2 * (1 - norm.cdf(abs(z))))
            diffs.append(abs(pw - pl))
        diffs, pvals = np.array(diffs), np.array(pvals)
        n = len(common)
        rows.append({
            "pair": f"within->{name}",
            "n_words": n,
            "min_freq": DRIFT_LABEL_MIN_FREQ,
            "mean_abs_delta": round(float(diffs.mean()), 3),
            "sig_p05": int((pvals < 0.05).sum()),
            "sig_p05_pct": round(100 * float((pvals < 0.05).mean()), 1),
            "bonferroni_survivors": int((pvals < 0.05 / n).sum()),
        })
    return pd.DataFrame(rows)


def per_tweet_semantic_distance(root: Path, model: SentenceTransformer,
                                train_centroid: np.ndarray) -> pd.DataFrame:
    """Cosine distance of each test tweet to the training centroid (the
    semantic component of the drift score)."""
    rows = []
    for name, rel in SPLITS:
        with open(root / rel) as f:
            recs = json.load(f)
        texts = [r["pp_text"] for r in recs]
        embs = model.encode(texts, batch_size=DRIFT_EMBED_BATCH,
                            convert_to_numpy=True, show_progress_bar=False)
        for idx, e in enumerate(embs):
            rows.append({"split": name, "idx": idx, "sem_dist": cosine(e, train_centroid)})
    return pd.DataFrame(rows)


def train_centroid(root: Path, model: SentenceTransformer, cap: int = 20000) -> np.ndarray:
    with open(root / "train_eval/train.json") as f:
        recs = json.load(f)
    texts = [r["pp_text"] for r in recs][:cap]
    embs = model.encode(texts, batch_size=DRIFT_EMBED_BATCH,
                        convert_to_numpy=True, show_progress_bar=False)
    return embs.mean(axis=0)


def main() -> None:
    root = data_root()

    print("Building train vocabulary ...")
    train_vocab = build_train_vocab(root)
    print(f"  {len(train_vocab)} unique tokens")

    print("\nComputing lexical drift per tweet ...")
    lex = lexical_metrics(root, train_vocab)

    print("\nLoading MiniLM ...")
    model = SentenceTransformer(MINILM_MODEL)
    centroid = train_centroid(root, model)

    print("Per-tweet semantic distance ...")
    sem = per_tweet_semantic_distance(root, model, centroid)
    df = lex.merge(sem, on=["split", "idx"], how="left")
    DRIFT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DRIFT_CSV, index=False)
    print(f"Wrote {DRIFT_CSV}  ({len(df)} rows)")

    print("\nComputing semantic per-word shifts (contextual) ...")
    by_year = sample_corpus_by_year()
    if by_year:
        embeddings, indices = encode_by_year(model, by_year)
        summary = semantic_summary(embeddings, indices)
        if not summary.empty:
            out = DRIFT_CSV.parent / "drift_summary.csv"
            summary.to_csv(out, index=False)
            print(f"Wrote {out}")
            print(summary.round(3).to_string(index=False))
    else:
        print("Skipped semantic per-word shift (no unlabelled corpus).")

    print("\nProbing real drift (label-conditional per word) ...")
    rdc = label_conditional_drift(root)
    if not rdc.empty:
        out = DRIFT_CSV.parent / "real_drift_check.csv"
        rdc.to_csv(out, index=False)
        print(f"Wrote {out}")
        print(rdc.to_string(index=False))


if __name__ == "__main__":
    main()
