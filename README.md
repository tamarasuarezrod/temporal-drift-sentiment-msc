# Mitigating Temporal Drift in Social Media Sentiment Classification: A Comparative Study

Supporting material for the MSc dissertation *Mitigating Temporal Drift in
Social Media Sentiment Classification: A Comparative Study* (Queen Mary
University of London, MSc Artificial Intelligence).

The study trains a RoBERTa sentiment classifier on 2014–2016 tweets from the
LongEval 2023 Task 2 benchmark (TM-Senti) and evaluates it on three test
periods (2016, 2018, 2021) to measure temporal drift. It compares six
literature mitigation strategies and a validation-loss baseline, decomposes
each strategy's gain at the per-tweet level, and proposes **PretrainedTEA**, a
single model that combines continued encoder pretraining with recency-weighted
temporal expert averaging.

This repository contains the code, the reproduction pipeline and the analysis
outputs. The trained checkpoints and the evaluation data are hosted separately
(see [Data and checkpoints](#data-and-checkpoints)).

## Repository structure

```
strategies/         Training notebook for each system (one per subfolder):
                    baseline, DatePrefix, MLMPretrain, RecencyWeight,
                    TimeLMs, TEA, T5, PretrainedTEA
pipeline/           Reproduction and evaluation (13 steps) + run_all.sh + config.py
results/            Reproducible outputs
                    per_tweet_analysis.parquet, preds_raw.parquet (canonical data)
  tables/           dataset, drift and results tables as CSV (+ LaTeX fragment)
  figures/          fig_threshold.png
  metrics/          drift, pragmatic, threshold, bootstrap, selection, contamination
  all_systems/      outputs covering all eight systems (incl. PretrainedTEA, T5)
requirements.txt    Python dependencies
```

Each system is trained in its own notebook under `strategies/` (run on Kaggle,
uploading checkpoints to HuggingFace). The `pipeline/` scripts then download
those checkpoints and reproduce every table and figure in the paper.

## Quick start

The fast path runs from the published checkpoints and needs no GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash pipeline/run_all.sh
```

This downloads the data and checkpoints, runs inference on the three test
splits, rebuilds `results/per_tweet_analysis.parquet`, and regenerates every
table and figure in the paper. See
[`pipeline/README.md`](pipeline/README.md) for a step-by-step
description of the pipeline.

The `results/` outputs are committed, which lets the tables and figures be
inspected without running anything.

## Systems

| System | Family | Checkpoint prefix |
|---|---|---|
| Baseline (val-loss) | — | `longeval-baseline-valloss` |
| DatePrefix | input tagging | `longeval-s1-date-prefix` |
| MLMPretrain | encoder pretraining | `longeval-s2-continued-pretraining` |
| RecencyWeight | recency weighting | `longeval-s3-temporal-reweighting` |
| TimeLMs | encoder pretraining | `longeval-s6-timelms` |
| TEA | recency weighting | `longeval-s7-tea` |
| T5 | generative classification | `longeval-t5-generative` |
| PretrainedTEA (proposed) | recency + pretraining | `longeval-s10-s2backbone-s8mechanism` |

Each system has three seeds (42, 1, 2) and is trained in its own notebook under
`strategies/`. Reported results are the majority vote across the three seeds. In
the pipeline, the six literature systems and the baseline are scored together in
step 04, while T5 and PretrainedTEA need their own scoring and are handled in
steps 12 and 13.

## Data and checkpoints

The labelled splits and the trained checkpoints are public on
[HuggingFace](https://huggingface.co/tamarasuarezrod) and are downloaded
automatically by the pipeline (no token required). If a checkpoint cannot be
reached on HuggingFace, the pipeline falls back to a local copy under
`checkpoints/<name>/`. Download the checkpoints from
[OneDrive](ONEDRIVE-CHECKPOINTS-URL-PLACEHOLDER) and unzip them there if you
need them.

The underlying tweets come from TM-Senti and the LongEval 2023 shared task.
Please refer to those sources for their own terms of use.

## Environment

Python 3.11+. PyTorch runs on GPU, Apple MPS or CPU automatically. The
semantic-drift and irony-flag steps use `sentence-transformers` and
`cardiffnlp/twitter-roberta-base-irony`, downloaded on demand.

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). The datasets
and pretrained models retain the licenses of their original authors.

## Citation

```bibtex
@mastersthesis{suarez2026temporaldrift,
  title  = {Mitigating Temporal Drift in Social Media Sentiment Classification: A Comparative Study},
  author = {Su\'arez Rodr\'iguez, Tamara},
  school = {Queen Mary University of London},
  year   = {2026}
}
```
