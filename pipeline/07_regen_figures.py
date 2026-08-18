"""Regenerate the paper figures into results/figures/."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "temporal-drift-matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "temporal-drift-cache")
)

import matplotlib.pyplot as plt
import pandas as pd

from config import (
    METRICS_DIR,
    FIGURES_DIR,
)

INK = "#1b1b19"
INK_SOFT = "#4b4b47"
INK_FAINT = "#75756e"
HAIRLINE = "#e4e4e1"
GRID = "#ececea"

NEUTRAL = "#8f8f88"
ACCENT = "#2c6fa3"    # blue: the best strategy / proposed method (the target)
ACCENT2 = "#c26a4a"   # rust: the deployable, dev-tuned threshold (can fail)

SPLITS = ["within", "short", "long"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": HAIRLINE,
    "axes.labelcolor": INK_SOFT,
    "xtick.color": INK_FAINT,
    "ytick.color": INK_FAINT,
    "text.color": INK,
    "font.size": 9,
    "axes.linewidth": 0.8,
})


def strip_spines(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


# Threshold decomposition per period (dumbbells)
def fig_threshold() -> None:
    ins_path = METRICS_DIR / "threshold_analysis_insample.csv"
    prac_path = METRICS_DIR / "threshold_analysis.csv"
    if not ins_path.exists() or not prac_path.exists():
        print("  fig_threshold: threshold CSVs missing, run 08 first — skipping.")
        return
    ins = pd.read_csv(ins_path).set_index("split")
    prac = pd.read_csv(prac_path).set_index("split")

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ys = {"within": 2, "short": 1, "long": 0}
    for sp in SPLITS:
        y = ys[sp]
        base = ins.loc[sp, "baseline_f1"]
        opt = ins.loc[sp, "insample_thresholded_f1"]
        best = ins.loc[sp, "best_strategy_f1"]
        dev = prac.loc[sp, "thresholded_f1"]
        rec = round(100 * ins.loc[sp, "gain_recovered"])

        ax.plot([base, best], [y, y], color=HAIRLINE, lw=2.4, zorder=1, solid_capstyle="round")
        ax.scatter([dev], [y], marker="x", s=44, c=ACCENT2, lw=1.8, zorder=3)
        ax.scatter([base], [y], s=48, c=NEUTRAL, zorder=3)
        ax.scatter([opt], [y], s=58, facecolors="white", edgecolors=ACCENT, lw=1.6, zorder=3)
        ax.scatter([best], [y], s=54, c=ACCENT, zorder=3)

        note = f"threshold alone\n{rec}% of the gain"
        if sp == "within":
            ax.annotate(note, (min(base, dev), y), xytext=(-10, -1), textcoords="offset points",
                        ha="right", va="center", fontsize=7.2, color=INK_SOFT, linespacing=1.5)
        else:
            ax.annotate(note, (best, y), xytext=(10, -1), textcoords="offset points",
                        ha="left", va="center", fontsize=7.2, color=INK_SOFT, linespacing=1.5)
        ax.annotate(sp, (0.653, y + 0.18), fontsize=8.5, fontweight="bold", color=INK,
                    annotation_clip=False)

    ax.set_xlim(0.652, 0.744)
    ax.set_ylim(-0.55, 2.75)
    ax.set_yticks([])
    ax.set_xlabel("Macro-F1", fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.7)
    strip_spines(ax, keep=("bottom",))
    ax.tick_params(length=0, labelsize=8)

    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=NEUTRAL, ms=6, label="baseline"),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor=ACCENT, ms=6.5, label="optimal threshold (test labels)"),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=ACCENT, ms=6,
                   label="best non-generative strategy"),
        plt.Line2D([], [], marker="x", color=ACCENT2, ls="none", ms=6,
                   label="practice-tuned threshold"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              ncol=4, fontsize=7, frameon=False, columnspacing=1.0, handletextpad=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_threshold.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


# Robustness composite rankings for each period (horizontal bars)
def fig_composite() -> None:
    from config import ANALYSIS_DIR
    composite_accent = "#2e7d6b"  # presentation green
    composite_other = "#b8bab6"   # light neutral grey
    path = ANALYSIS_DIR / "all_systems" / "composite_by_split.csv"
    if not path.exists():
        print("  fig_composite: composite_by_split.csv missing, run 12/13 first — skipping.")
        return
    all_periods = pd.read_csv(path)

    # Native IEEE two-column width, so labels remain legible without being
    # scaled down from an oversized canvas in LaTeX.
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.85), sharex=True)
    for ax, split in zip(axes, ["within", "short", "long"]):
        df = all_periods[all_periods.split == split].sort_values("composite")
        ys = range(len(df))
        colours = [
            composite_accent if s == "PretrainedTEA" else composite_other
            for s in df["system"]
        ]
        ax.barh(list(ys), df["composite"], color=colours, height=0.66, zorder=2)
        for y, (system, value) in enumerate(zip(df["system"], df["composite"])):
            ax.annotate(f"{value:.2f}", (value, y), xytext=(3, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=6.0,
                        color=composite_accent if system == "PretrainedTEA" else INK_SOFT,
                        fontweight="bold" if system == "PretrainedTEA" else "normal")
        ax.set_yticks(list(ys))
        labels = ["Baseline" if system == "baseline" else system for system in df["system"]]
        ax.set_yticklabels(labels, fontsize=6.2)
        for tick, system in zip(ax.get_yticklabels(), df["system"]):
            if system == "PretrainedTEA":
                tick.set_color(composite_accent)
                tick.set_fontweight("bold")
        ax.set_xlim(0, 4.4)
        ax.set_title(split, fontsize=8, fontweight="normal")
        ax.grid(axis="x", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        strip_spines(ax, keep=("bottom",))
        ax.tick_params(length=0, labelsize=6.2)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_composite.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Rendering figures ...")
    fig_threshold()
    fig_composite()


if __name__ == "__main__":
    main()
