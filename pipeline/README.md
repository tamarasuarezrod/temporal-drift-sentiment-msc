# Reproduction pipeline

Regenerates every number, table and figure in the paper from scratch, to verify
the results end to end.

## Before running

Two inputs are not stored in the repository and must be placed locally first.

- **Data.** Download the LongEval 2023 Task 2 splits from
  https://clef-longeval-2023.github.io/data/ and place them under `data/` so
  that `data/train_eval/` and `data/test/` hold the JSON splits (see the file
  list in `01_check_data.py`). The unlabelled corpus
  `sample-10k-monthly-120k-yearly.jl` goes in `data/` too, and is only needed
  by the semantic-drift step.
- **Checkpoints.** Download `checkpoints.zip` from Zenodo
  (https://doi.org/10.5281/zenodo.21850584) and unzip it at the repository
  root so the models land under `checkpoints/<id>-seed<n>/` (for example
  `checkpoints/tea-seed1/`). Only the inference steps (04, 08, 11, 12, 13) need
  them.

## Running

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash pipeline/run_all.sh
```

This checks the data, runs inference on the three test splits, rebuilds the
parquet, and regenerates every table and figure. Outputs land in:
- `results/per_tweet_analysis.parquet` (canonical per-tweet predictions)
- `results/tables/` (dataset, drift and results tables as CSV plus LaTeX-ready fragments)
- `results/figures/` (the figure used in the paper)
- `results/metrics/` (drift, threshold, bootstrap, selection and contamination outputs)
- `results/all_systems/` (the eight-system battery from steps 12 and 13)

### Retraining from scratch

Each system has a training notebook under `strategies/` (one per subfolder).
They were run on Kaggle with a T4 GPU and push their checkpoints to
HuggingFace. To retrain under your own account:

- The notebooks authenticate with an `HF_TOKEN` stored as a Kaggle secret
  (`UserSecretsClient().get_secret("HF_TOKEN")`). Add your own token as a
  Kaggle secret of that name before running them.
- The HuggingFace username is hardcoded as `tamarasuarezrod` in each notebook
  (the upload target). Change it to your account.

Run the notebooks to produce the checkpoints, place them under `checkpoints/`,
then run the pipeline to reproduce the tables and figures.

## Scripts

Called in this order by `run_all.sh`:

| Step | Script | What it does |
|---|---|---|
| 01 | `01_check_data.py` | Checks that the LongEval splits are present under `data/` |
| 02 | `02_compute_drift.py` | OOV rate, new-type rate, per-word contextual shift, and a word-level real-drift probe → `drift_metrics.csv`, `drift_summary.csv`, `real_drift_check.csv` |
| 03 | `03_compute_polarity_mismatch.py` | Flags tweets whose surface word polarity contradicts the gold label (Hu & Liu lexicon; irony model as a corroboration column) → `polarity_mismatch_flags.csv` |
| 04 | `04_build_predictions.py` | For each of 6 systems × 3 seeds: loads HF checkpoint, predicts on `within` / `short` / `long`. Writes `preds_raw.parquet` |
| 05 | `05_build_parquet.py` | Aggregates per-seed predictions by tweet, merges with drift + polarity-mismatch → `per_tweet_analysis.parquet` |
| 06 | `06_regen_tables.py` | Recomputes the dataset, drift and results tables from the parquet |
| 07 | `07_regen_figures.py` | Regenerates the paper figure (threshold decomposition) |
| 08 | `08_threshold_analysis.py` | Threshold decomposition: practice-tuned, oracle-half, and in-sample per-period sweeps |
| 09 | `09_contamination_check.py` | Overlap between the MLM corpus and the test tweets |
| 10 | `10_paired_bootstrap.py` | Paired bootstrap CIs quoted in the Limitations section |
| 11 | `11_practice_selection.py` | Practice-set macro-F1 against test macro-F1 (selection trap) |
| 12 | `12_evaluate_pretrainedtea.py` | Scores PretrainedTEA and the averaging-only ablation, and the 2×2 factorial, into `results/all_systems/` |
| 13 | `13_evaluate_t5.py` | Scores the generative T5 system (its own generation-likelihood path) into `results/all_systems/` |
| 14 | `14_always_fail_residual.py` | Always-fail / always-correct residual across the eight systems → `residual.csv` |

Each script is independently runnable (assuming its inputs are on disk).

**Aggregation note.** The results table reports the macro-F1 obtained by soft
voting across the three seeds. For each tweet, the positive-class
probabilities are averaged and the resulting score is thresholded at 0.5.
The same rule is used for all eight systems.
Per-seed F1 scores can be recomputed from `results/preds_raw.parquet`
(written by `04_build_predictions.py`), which keeps each seed's
predictions separate. The mean of per-seed F1 scores is a different,
usually lower, aggregate than the soft-vote F1.

## Environment

GPU is optional: PyTorch falls back to MPS (Apple Silicon) or CPU
automatically. The polarity-mismatch step and the semantic-drift step both use
sentence-transformers, installed on demand.
