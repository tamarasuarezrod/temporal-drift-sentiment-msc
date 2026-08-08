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

Each system is trained in its own notebook under `strategies/` (run on Kaggle).
The `pipeline/` scripts then score the trained checkpoints and reproduce every
table and figure in the paper.

## Quick start

The pipeline needs two inputs placed locally first (see
[Data and checkpoints](#data-and-checkpoints)): the LongEval splits under
`data/` and the checkpoints under `checkpoints/`. No GPU is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash pipeline/run_all.sh
```

This runs inference on the three test splits, rebuilds
`results/per_tweet_analysis.parquet`, and regenerates every table and figure in
the paper. See [`pipeline/README.md`](pipeline/README.md) for a step-by-step
description of the pipeline.

The `results/` outputs are committed, which lets the tables and figures be
inspected without running anything.

## Systems

| System | Family | Checkpoint id |
|---|---|---|
| Baseline (val-loss) | — | `baseline` |
| DatePrefix | input tagging | `dateprefix` |
| MLMPretrain | encoder pretraining | `mlmpretrain` |
| RecencyWeight | recency weighting | `recencyweight` |
| TimeLMs | encoder pretraining | `timelms` |
| TEA | recency weighting | `tea` |
| T5 | generative classification | `t5` |
| PretrainedTEA (proposed) | recency + pretraining | `pretrainedtea` |

Each checkpoint folder is `<id>-seed<n>` (for example `tea-seed1`). The
recency-weighted expert-averaging ablation (`expertaveraging`) is used in the
factorial in step 12.

Each system has three seeds (42, 1, 2) and is trained in its own notebook under
`strategies/`. Reported results are the majority vote across the three seeds. In
the pipeline, the six literature systems and the baseline are scored together in
step 04, while T5 and PretrainedTEA need their own scoring and are handled in
steps 12 and 13.

## Data and checkpoints

Neither the data nor the checkpoints are stored in this repository. Place both
locally before running the pipeline.

**Data.** The evaluation data comes from the LongEval 2023 shared task
(Task 2), which builds on TM-Senti. It is not redistributed here. Download it
from the official source and place the splits under `data/` so that
`data/train_eval/` and `data/test/` hold the JSON files (the pipeline lists the
expected files and fails early if any are missing). Refer to the source for its
terms of use: https://clef-longeval-2023.github.io/data/

**Checkpoints.** The trained checkpoints are archived on Zenodo
(https://doi.org/10.5281/zenodo.21850584). Download `checkpoints.zip` and unzip
it at the repository root so the models land under `checkpoints/<id>-seed<n>/`
(for example `checkpoints/tea-seed1/`).

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
