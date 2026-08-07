"""Recompute the dataset-size, drift and results tables from the canonical
parquet, writing CSVs and a .tex fragment to results/tables/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.metrics import f1_score

from config import (
    DRIFT_CSV,
    HF_DATA_REPO,
    PARQUET,
    SPLITS,
    SYSTEM_COLUMN_PREFIX,
    TABLES_DIR,
)


def dataset_table() -> pd.DataFrame:
    root = Path(snapshot_download(repo_id=HF_DATA_REPO, repo_type="dataset"))
    rows = [
        ("Train",           "train_eval/train.json",              "2014-2016", "distant"),
        ("Practice (2016)", "train_eval/interim_eval_2016.json",  "2016",      "distant"),
        ("Practice (2018)", "train_eval/interim_eval_2018.json",  "2018",      "distant"),
        ("Test (within)",   "test/interim_test_2016.json",        "2016",      "manual"),
        ("Test (short)",    "test/interim_test_2018.json",        "2018",      "manual"),
        ("Test (long)",     "test/interim_test_2021.json",        "2021",      "manual"),
    ]
    out = []
    for label, rel, period, lab_type in rows:
        with open(root / rel) as f:
            n = len(json.load(f))
        out.append({"Split": label, "Period": period, "Size": n, "Labels": lab_type})
    return pd.DataFrame(out)


def drift_table() -> pd.DataFrame:
    """Aggregated from drift_metrics.csv (per-tweet) and the per-word
    semantic summary written by 02_compute_drift.py."""
    df = pd.read_csv(DRIFT_CSV)
    summary_path = DRIFT_CSV.parent / "drift_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else None

    rows = []
    for split in ["within", "short", "long"]:
        sub = df[df["split"] == split]
        # OOV rate as total OOV tokens / total tokens, matching the
        # drift-characterisation notebook and the reported figure
        if "n_tokens" in sub.columns and "n_oov" in sub.columns:
            oov_pct = sub["n_oov"].sum() / sub["n_tokens"].sum() * 100
        else:
            oov_pct = sub["oov_frac"].mean() * 100
        row = {
            "split": split,
            "OOV rate (%)": oov_pct,
            "New types (%)": sub["new_type_rate"].iloc[0] * 100
                if "new_type_rate" in sub.columns else np.nan,
        }
        if summary is not None:
            m = summary[summary["split"] == split]
            if not m.empty:
                row["Per-word semantic shift (mean)"] = float(m["mean_shift"].iloc[0])
                row["Share above noise floor (%)"] = float(m["share_above_floor"].iloc[0]) * 100
        rows.append(row)
    return pd.DataFrame(rows).set_index("split").T.reset_index().rename(columns={"index": "Metric"})


def results_table() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    rows = []
    for label, prefix in SYSTEM_COLUMN_PREFIX.items():
        row = {"Method": label}
        for split in ["within", "short", "long"]:
            sub = df[df["split"] == split]
            y_true = (sub["gold"] == "positive").astype(int).to_numpy()
            y_pred = (sub[f"{prefix}_pred"] == "positive").astype(int).to_numpy()
            row[split] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        rows.append(row)
    return pd.DataFrame(rows)


def results_to_latex(df: pd.DataFrame) -> str:
    """Emit the paper-style booktabs table grouped by mechanism family."""
    def cell(x: float, best: bool) -> str:
        s = f"{x:.3f}"
        return f"\\textbf{{{s}}}" if best else s

    best_within = df["within"].max()
    best_short  = df["short"].max()
    best_long   = df["long"].max()

    def row(method: str) -> str:
        r = df[df["Method"] == method].iloc[0]
        return (
            f"{method:<24} & "
            f"{cell(r['within'], r['within'] == best_within)} & "
            f"{cell(r['short'],  r['short']  == best_short)} & "
            f"{cell(r['long'],   r['long']   == best_long)} \\\\"
        )

    lines = [
        r"\begin{table}[t]",
        r"\caption{Macro-F1 by strategy (majority vote across three seeds), grouped by delivered mechanism. The best score in each column is in bold.}",
        r"\label{tab:results}",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & within & short & long \\",
        r"\midrule",
        row("Baseline"),
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Recency weighting}} \\",
        row("RecencyWeight"),
        row("TEA"),
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Vocabulary adaptation}} \\",
        row("MLMPretrain"),
        row("TimeLMs"),
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Structure-agnostic}} \\",
        row("DatePrefix"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    t1 = dataset_table()
    t1.to_csv(TABLES_DIR / "dataset.csv", index=False)
    print("\nDataset table:")
    print(t1.to_string(index=False))

    if DRIFT_CSV.exists():
        t2 = drift_table()
        t2.to_csv(TABLES_DIR / "drift.csv", index=False)
        print("\nDrift table:")
        print(t2.round(3).to_string(index=False))
    else:
        print(f"\nSkipping drift table: {DRIFT_CSV} not found.")

    if PARQUET.exists():
        t3 = results_table()
        t3.round(4).to_csv(TABLES_DIR / "results.csv", index=False)
        (TABLES_DIR / "results.tex").write_text(results_to_latex(t3))
        print("\nResults table:")
        print(t3.round(3).to_string(index=False))
        print(f"\nWrote LaTeX fragment to {TABLES_DIR / 'results.tex'}")
    else:
        print(f"\nSkipping results table: {PARQUET} not found.")


if __name__ == "__main__":
    main()
