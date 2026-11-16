from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = {
    "AP": "Mask AP@[0.50:0.95]",
    "AP50": "Mask AP50",
    "AP75": "Mask AP75",
    "mIoU": "Mean IoU",
}


def _ranked_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    work = df.copy()
    work["score_percent"] = pd.to_numeric(work[metric], errors="coerce") * 100.0
    work = work.dropna(subset=["score_percent"]).sort_values("score_percent", ascending=True)
    return work


def _bar_chart(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    ranked = _ranked_frame(df, metric)
    if ranked.empty:
        return
    fig_height = max(6.5, 0.38 * len(ranked) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    values = ranked["score_percent"].to_numpy()
    names = ranked["experiment"].astype(str).to_numpy()
    colors = ["#2A9D72"] * len(values)
    colors[-1] = "#E07A32"  # best result is highlighted at the top/right
    bars = ax.barh(names, values, color=colors, height=0.72)
    ax.set_xlim(0, max(5.0, float(values.max()) * 1.18))
    ax.set_xlabel("Score (%)")
    ax.set_title(f"Band-combination ranking — {METRICS[metric]}", loc="left", weight="bold", pad=14)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, value in zip(bars, values):
        ax.text(value + max(0.2, ax.get_xlim()[1] * 0.008), bar.get_y() + bar.get_height()/2,
                f"{value:.2f}", va="center", ha="left", fontsize=8)
    best = ranked.iloc[-1]
    ax.text(
        1.0, 1.015,
        f"Best: {best['experiment']} ({best['name']}) — {best['score_percent']:.2f}%",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#B65E24",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_all(results_csv: str | Path, output_dir: str | Path) -> list[Path]:
    results_csv = Path(results_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv)
    if df.empty:
        raise ValueError(f"No results found in {results_csv}")

    # Save ranked summary: highest AP first, retaining all metrics.
    ranked = df.sort_values("AP", ascending=False).copy()
    ranked.to_csv(output_dir / "ranked_results_by_AP.csv", index=False)

    saved = []
    for metric in ("AP", "AP50", "AP75", "mIoU"):
        out = output_dir / f"ranked_{metric}.png"
        _bar_chart(df, metric, out)
        saved.append(out)

    # A compact 2x2 dashboard, each panel independently ranked high-to-low.
    fig, axes = plt.subplots(2, 2, figsize=(17, 14))
    for ax, metric in zip(axes.flat, ("AP", "AP50", "AP75", "mIoU")):
        work = _ranked_frame(df, metric)
        vals = work["score_percent"].to_numpy()
        labels = work["experiment"].astype(str).to_numpy()
        colors = ["#2A9D72"] * len(vals)
        if len(colors):
            colors[-1] = "#E07A32"
        bars = ax.barh(labels, vals, color=colors, height=0.72)
        ax.set_title(METRICS[metric], loc="left", weight="bold")
        ax.set_xlabel("Score (%)")
        ax.set_xlim(0, max(5.0, float(vals.max()) * 1.20) if len(vals) else 1.0)
        ax.grid(axis="x", alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=8)
        for bar, value in zip(bars, vals):
            ax.text(value + max(0.15, ax.get_xlim()[1] * 0.006),
                    bar.get_y() + bar.get_height()/2, f"{value:.1f}",
                    va="center", fontsize=6.8)
    fig.suptitle(f"COCO-pretrained Mask R-CNN — {len(df)} completed band combinations", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    dashboard = output_dir / "metrics_dashboard.png"
    fig.savefig(dashboard, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(dashboard)

    # Size-specific AP chart, also sorted by headline AP.
    size_cols = ["AP_small", "AP_medium", "AP_large"]
    available = [col for col in size_cols if col in df.columns]
    if available:
        order = df.sort_values("AP", ascending=True)
        y = np.arange(len(order))
        height = 0.23
        fig, ax = plt.subplots(figsize=(12, max(7, 0.4 * len(order) + 2)))
        palette = ["#6A8CAF", "#2A9D72", "#E5A13A"]
        for i, col in enumerate(available):
            vals = pd.to_numeric(order[col], errors="coerce").fillna(0).to_numpy() * 100
            ax.barh(y + (i - (len(available)-1)/2) * height, vals, height=height,
                    label=col.replace("AP_", "").title(), color=palette[i])
        ax.set_yticks(y, order["experiment"].astype(str))
        ax.set_xlabel("Score (%)")
        ax.set_title("Instance-size AP by band combination", loc="left", weight="bold")
        ax.grid(axis="x", alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        size_path = output_dir / "ranked_AP_by_size.png"
        fig.savefig(size_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(size_path)

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    for path in plot_all(args.results, args.out):
        print(path)


if __name__ == "__main__":
    main()
