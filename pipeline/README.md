# Reproduction pipeline

Regenerates every number, table and figure in the paper from scratch, to verify
the results end to end.

## Two paths

### Fast path (no GPU required)

Runs from the six trained checkpoints already published on
[HuggingFace](https://huggingface.co/tamarasuarezrod). Downloads the
data, downloads the checkpoints, runs inference on the three test
splits, rebuilds the parquet, and regenerates every table and figure in
the paper.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash pipeline/run_all.sh
```

Outputs land in:
- `results/per_tweet_analysis.parquet` (canonical per-tweet predictions)
- `results/tables/` (dataset, drift and results tables as CSV plus LaTeX-ready fragments)
- `results/figures/` (the figure used in the paper)
- `results/metrics/` (drift, threshold, bootstrap, selection and contamination outputs)
- `results/all_systems/` (the eight-system battery from steps 12 and 13)

### Checkpoints

The scripts download the trained checkpoints from HuggingFace by default. If a
checkpoint cannot be reached there, they fall back to a local copy under
`checkpoints/<name>/` (for example
`checkpoints/longeval-baseline-valloss-seed42/`). If you need the checkpoints,
download them from [OneDrive](ONEDRIVE-CHECKPOINTS-URL-PLACEHOLDER) and unzip
them into `checkpoints/`.

### Retraining from scratch

Each system has a training notebook under `strategies/` (one per subfolder).
They were run on Kaggle with a T4 GPU and push their checkpoints to
HuggingFace. Two things are wired to the original account and must be changed
to retrain under your own:

- The notebooks authenticate with an `HF_TOKEN` stored as a Kaggle secret
  (`UserSecretsClient().get_secret("HF_TOKEN")`). Add your own token as a
  Kaggle secret of that name before running them.
- The HuggingFace username is hardcoded as `tamarasuarezrod` in each notebook
  (the upload target) and in `pipeline/config.py` (`HF_USER`, the download
  source). Change both to your account.

With those in place, run the notebooks to produce the checkpoints, then run
the fast path above to reproduce the tables and figures.

## Scripts

Called in this order by `run_all.sh`:

| Step | Script | What it does |
|---|---|---|
| 01 | `01_fetch_data.py` | Downloads `data/` (labelled splits + unlabelled corpus) from HF |
| 02 | `02_compute_drift.py` | OOV rate, new-type rate, per-word semantic shift → `drift_metrics.csv` |
| 03 | `03_compute_pragmatic.py` | Runs `cardiffnlp/twitter-roberta-base-irony` on the three test sets → `pragmatic_flags.csv` |
| 04 | `04_build_predictions.py` | For each of 6 systems × 3 seeds: loads HF checkpoint, predicts on `within` / `short` / `long`. Writes `preds_raw.parquet` |
| 05 | `05_build_parquet.py` | Aggregates per-seed predictions by tweet, merges with drift + pragmatic → `per_tweet_analysis.parquet` |
| 06 | `06_regen_tables.py` | Recomputes the dataset, drift and results tables from the parquet |
| 07 | `07_regen_figures.py` | Regenerates the paper figure (threshold decomposition) |
| 08 | `08_threshold_analysis.py` | Threshold decomposition: practice-tuned, oracle-half, and in-sample per-period sweeps |
| 09 | `09_contamination_check.py` | Overlap between the MLM corpus and the test tweets |
| 10 | `10_paired_bootstrap.py` | Paired bootstrap CIs quoted in the Limitations section |
| 11 | `11_practice_selection.py` | Practice-set macro-F1 against test macro-F1 (selection trap) |
| 12 | `12_evaluate_pretrainedtea.py` | Scores PretrainedTEA and the averaging-only ablation, and the 2×2 factorial, into `results/all_systems/` |
| 13 | `13_evaluate_t5.py` | Scores the generative T5 system (its own generation-likelihood path) into `results/all_systems/` |

Each script is independently runnable (assuming its inputs are on disk).

**Aggregation note.** The results table reports the macro-F1 of the
majority-vote prediction across the three seeds (mean positive
probability thresholded at 0.5), computed by `06_regen_tables.py`.
Per-seed F1 scores can be recomputed from `results/preds_raw.parquet`
(written by `04_build_predictions.py`), which keeps each seed's
predictions separate. The mean of per-seed F1 scores is a different,
usually lower, aggregate than the majority-vote F1.
The six systems currently included are:

- **Baseline** (val-loss early stopping): `tamarasuarezrod/longeval-baseline-valloss-seed{1,2,42}`
- **DatePrefix**: `tamarasuarezrod/longeval-s1-date-prefix-seed{1,2,42}`
- **MLMPretrain (continued pretraining)**: `tamarasuarezrod/longeval-s2-continued-pretraining-seed{1,2,42}`
- **RecencyWeight (temporal reweighting)**: `tamarasuarezrod/longeval-s3-temporal-reweighting-seed{1,2,42}`
- **TimeLMs**: `tamarasuarezrod/longeval-s6-timelms-seed{1,2,42}`
- **TEA (temporal experts averaging)**: `tamarasuarezrod/longeval-s7-tea-seed{1,2,42}`

## Environment

Uses the same `requirements.txt` and `.venv` as the rest of the repo.
GPU is optional: PyTorch will fall back to MPS (Apple Silicon) or CPU
automatically. The pragmatic-flag step and semantic-drift step both use
sentence-transformers, which are installed on demand.

A `HF_TOKEN` is not required for the fast path (all repos are public).
