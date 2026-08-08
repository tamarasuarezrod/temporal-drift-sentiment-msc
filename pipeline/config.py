"""Shared paths, data location and system list for the pipeline."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
ANALYSIS_DIR = REPO_ROOT / "results"
TABLES_DIR = ANALYSIS_DIR / "tables"
FIGURES_DIR = ANALYSIS_DIR / "figures"
METRICS_DIR = ANALYSIS_DIR / "metrics"
LOCAL_CHECKPOINTS = REPO_ROOT / "checkpoints"

# Canonical per-tweet data stays at the results/ root
PREDS_RAW = ANALYSIS_DIR / "preds_raw.parquet"
PARQUET = ANALYSIS_DIR / "per_tweet_analysis.parquet"

# Six-system analysis outputs live under results/metrics/
DRIFT_CSV = METRICS_DIR / "drift_metrics.csv"
PRAGMATIC_CSV = METRICS_DIR / "pragmatic_flags.csv"

SPLITS = [
    ("within", "test/interim_test_2016.json"),
    ("short",  "test/interim_test_2018.json"),
    ("long",   "test/interim_test_2021.json"),
]

# Splits the pipeline reads from data/. Download them from LongEval 2023 Task 2
# (https://clef-longeval-2023.github.io/data/) and place them under data/.
EXPECTED_DATA = [
    "train_eval/train.json",
    "train_eval/interim_eval_2016.json",
    "train_eval/interim_eval_2018.json",
    "test/interim_test_2016.json",
    "test/interim_test_2018.json",
    "test/interim_test_2021.json",
]


def data_root():
    """Return the local data/ directory, checking the expected splits are there."""
    missing = [f for f in EXPECTED_DATA if not (DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing data files under data/: " + ", ".join(missing) + ". "
            "Download the LongEval 2023 Task 2 data from "
            "https://clef-longeval-2023.github.io/data/ and place the splits "
            "under data/ (see the README)."
        )
    return DATA_DIR


def load_checkpoint(loader, name):
    """Load a model or tokenizer for a checkpoint id such as "tea-seed1" from
    checkpoints/{name}/.

    Download checkpoints.zip from Zenodo and unzip it at the repository root
    (see the README) before running the steps that need the checkpoints.
    """
    local = LOCAL_CHECKPOINTS / name
    if not local.exists():
        raise FileNotFoundError(
            f"No checkpoint at {local}. Download checkpoints.zip from Zenodo "
            f"(https://doi.org/10.5281/zenodo.21850584) and unzip it into "
            f"{LOCAL_CHECKPOINTS}/."
        )
    return loader.from_pretrained(str(local))

# The six comparison systems, as (display name, checkpoint id). T5 and
# PretrainedTEA need different scoring and are handled in 13_evaluate_t5.py and
# 12_evaluate_pretrainedtea.py.
SYSTEMS = [
    ("Baseline",       "baseline"),
    ("DatePrefix",     "dateprefix"),
    ("MLMPretrain",    "mlmpretrain"),
    ("RecencyWeight",  "recencyweight"),
    ("TimeLMs",        "timelms"),
    ("TEA",            "tea"),
]

# Short codes used as column prefixes in the per-tweet parquet
SYSTEM_COLUMN_PREFIX = {
    "Baseline":       "baseline",
    "DatePrefix":     "S1",
    "MLMPretrain":    "S2",
    "RecencyWeight":  "S3",
    "TimeLMs":        "S6",
    "TEA":            "S7",
}

SEEDS = [42, 1, 2]

# A tweet is flagged as a surface-polarity mismatch when the irony detector's
# positive-class probability exceeds IRONY_THRESHOLD
IRONY_MODEL = "cardiffnlp/twitter-roberta-base-irony"
IRONY_THRESHOLD = 0.5

# Semantic drift uses contextual word prototypes (Hamilton et al.)
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
UNLABELED_CORPUS = DATA_DIR / "sample-10k-monthly-120k-yearly.jl"
DRIFT_YEARS = [2016, 2018, 2021]
DRIFT_MAX_TWEETS_PER_YEAR = 60_000
DRIFT_MIN_TWEETS_PER_WORD = 80
DRIFT_EMBED_BATCH = 128
