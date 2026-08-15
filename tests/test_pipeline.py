"""Lightweight unit tests for the pure helper functions in the pipeline.

These check the small building blocks (tokenisation, the polarity flag, the
metric, the data/checkpoint loaders) without touching the models or the large
data files. Run with:  pytest

They do not re-run the experiments. Reproducing the paper's numbers is done by
running the pipeline itself (see pipeline/README.md).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))

import config  # noqa: E402


def _load(filename):
    """Import a numbered pipeline script by path (names start with a digit)."""
    path = PIPELINE / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- tokenisation (02) -----------------------------------------------------

def test_tokenize_lowercases_on_whitespace():
    drift = _load("02_compute_drift.py")
    # the default path is text.lower().split(): lowercased, split on spaces
    assert drift.tokenize("Hello WORLD Covid") == ["hello", "world", "covid"]


def test_tokenize_dropstop_removes_stopwords_and_short_tokens():
    drift = _load("02_compute_drift.py")
    toks = drift.tokenize("the COVID lockdown is here", drop_stop=True)
    assert "covid" in toks and "lockdown" in toks
    assert "the" not in toks and "is" not in toks  # stopwords gone


# --- polarity flag (03) ----------------------------------------------------

def test_polarity_counts():
    prag = _load("03_compute_pragmatic.py")
    pos, neg = {"good", "great"}, {"bad", "awful"}
    assert prag.polarity_counts("good and great day", pos, neg) == (2, 0)
    assert prag.polarity_counts("an awful bad mess", pos, neg) == (0, 2)


def test_flag_direction():
    prag = _load("03_compute_pragmatic.py")
    # all-positive words but a negative label -> sarcasm
    assert prag.flag_direction(2, 0, "negative") == "B_sarcasm"
    # all-negative words but a positive label -> softened complaint / joke
    assert prag.flag_direction(0, 2, "positive") == "A_joking_complaint"
    # words and label agree, or mixed -> not flagged
    assert prag.flag_direction(2, 0, "positive") is None
    assert prag.flag_direction(1, 1, "negative") is None


# --- config: data and checkpoint loaders -----------------------------------

def test_data_root_raises_when_files_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # empty dir
    with pytest.raises(FileNotFoundError):
        config.data_root()


def test_load_checkpoint_raises_for_missing_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOCAL_CHECKPOINTS", tmp_path)
    with pytest.raises(FileNotFoundError):
        config.load_checkpoint(object, "does-not-exist-seed99")


def test_expected_data_and_splits_are_consistent():
    # the three test splits referenced by SPLITS must be in EXPECTED_DATA
    for _, rel in config.SPLITS:
        assert rel in config.EXPECTED_DATA


def test_seed_aggregation_uses_mean_probability_not_hard_vote():
    import numpy as np

    # Two seeds are just above 0.5, but one confident negative makes the mean
    # fall below 0.5. Hard majority voting would predict positive here.
    probs = np.array([[0.51], [0.51], [0.01]])
    mean_prob, pred = config.aggregate_seed_probabilities(probs)
    assert mean_prob[0] == pytest.approx(0.3433333333)
    assert pred[0] == 0


# --- smoke test: every pipeline script parses --------------------------------

def test_all_pipeline_scripts_compile():
    import py_compile
    scripts = sorted(PIPELINE.glob("*.py"))
    assert len(scripts) >= 13
    for s in scripts:
        py_compile.compile(str(s), doraise=True)
