"""Shared paths, HuggingFace repo names and system list for the pipeline.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
ANALYSIS_DIR = REPO_ROOT / "results"
TABLES_DIR = ANALYSIS_DIR / "tables"
FIGURES_DIR = ANALYSIS_DIR / "figures"
METRICS_DIR = ANALYSIS_DIR / "metrics"

# Canonical per-tweet data stays at the results/ root
PREDS_RAW = ANALYSIS_DIR / "preds_raw.parquet"
PARQUET = ANALYSIS_DIR / "per_tweet_analysis.parquet"

# Six-system analysis outputs live under results/metrics/
DRIFT_CSV = METRICS_DIR / "drift_metrics.csv"
PRAGMATIC_CSV = METRICS_DIR / "pragmatic_flags.csv"

HF_DATA_REPO = "tamarasuarezrod/longeval-data"

SPLITS = [
    ("within", "test/interim_test_2016.json"),
    ("short",  "test/interim_test_2018.json"),
    ("long",   "test/interim_test_2021.json"),
]

HF_USER = "tamarasuarezrod"

# Checkpoints load from HuggingFace by default. If that fails, they are read
# from a local folder instead (see load_checkpoint below and the README).
LOCAL_CHECKPOINTS = REPO_ROOT / "checkpoints"


def load_checkpoint(loader, name):
    """Load a model or tokenizer by checkpoint name, trying HuggingFace first.

    Tries the repo {HF_USER}/{name}; if it cannot be reached, falls back to a
    local folder checkpoints/{name}. See the README for where to download the
    checkpoints if you do not have them locally.
    """
    try:
        return loader.from_pretrained(f"{HF_USER}/{name}")
    except Exception:
        local = LOCAL_CHECKPOINTS / name
        if not local.exists():
            raise FileNotFoundError(
                f"Could not load {HF_USER}/{name} from HuggingFace and no local "
                f"copy at {local}. Download the checkpoints (see README) into "
                f"{LOCAL_CHECKPOINTS}/."
            )
        return loader.from_pretrained(str(local))

# The six comparison systems, as (display name, HuggingFace repo prefix). Each
# seeded checkpoint is f"{HF_USER}/{prefix}-seed{seed}". T5 and PretrainedTEA
# need different scoring and are handled in 13_evaluate_t5.py and 12_evaluate_pretrainedtea.py
SYSTEMS = [
    ("Baseline",       "longeval-baseline-valloss"),
    ("DatePrefix",     "longeval-s1-date-prefix"),
    ("MLMPretrain",    "longeval-s2-continued-pretraining"),
    ("RecencyWeight",  "longeval-s3-temporal-reweighting"),
    ("TimeLMs",        "longeval-s6-timelms"),
    ("TEA",            "longeval-s7-tea"),
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
