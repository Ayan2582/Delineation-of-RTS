"""Figures for the results folder: training curves, train/val/test metrics, predicted chips.

    python plots.py --out_dir outputs     # every run in results/history/ + the cross-run summary

    results/figures/<run>/overview.png        loss per epoch, val metrics per epoch, train/val/test bars
    results/figures/<run>/samples_<split>.png chips with GT and predicted outlines (written by eval.py)
    results/figures/summary.png               mean ± std per experiment and split, across folds/seeds
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import resolve_path  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # fixed order, never cycled
SPLIT_COLOR = {"train": SERIES[0], "val": SERIES[1], "test": SERIES[2]}
GT_COLOR, PRED_COLOR = "#1baf7a", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = ["AP", "AP50", "AP75", "mIoU"]
RESULT_FILES = {"train": "results_train.csv", "val": "results.csv", "test": "results_test.csv"}

plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 9,
                     "lines.linewidth": 2, "legend.frameon": False})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- predicted chips


def _rgb(cfg: dict, rec: dict) -> np.ndarray:
    """(H, W, 3) in [0, 1] from the raw tile's red/green/blue bands, 2-98 % stretch per band."""
    from data import band_index, load_raw
    idx = band_index(cfg)
    img = load_raw(rec)[[idx["red"], idx["green"], idx["blue"]]]
    out = np.zeros(img.shape, np.float32)
    for c, band in enumerate(img):
        ok = np.isfinite(band)
        if ok.any():
            lo, hi = np.percentile(band[ok], (2, 98))
            out[c] = np.clip((np.nan_to_num(band, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
    return out.transpose(1, 2, 0)


def plot_samples(cfg: dict, recs: list[dict], preds: dict, out: Path, n: int, split: str,
                 metrics: dict | None = None) -> Path:
    """Up to `n` chips (chips with RTS first, fixed seed): GT outlines vs predictions >= score_thresh."""
    from data import decode_masks
    rng = np.random.default_rng(0)
    pos = [r for r in recs if r["anns"]]
    neg = [r for r in recs if not r["anns"]]
    pick = [pos[i] for i in rng.permutation(len(pos))[:n]]
    pick += [neg[i] for i in rng.permutation(len(neg))[: n - len(pick)]]
    pick.sort(key=lambda r: r["id"])
    thresh = cfg["eval"]["score_thresh"]

    cols = min(4, max(1, len(pick)))
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows + 0.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, rec in zip(axes.flat, pick):
        ax.imshow(_rgb(cfg, rec))
        gt = decode_masks(rec)
        for g in gt:
            ax.contour(g, levels=[0.5], colors=GT_COLOR, linewidths=1.6)
        masks, scores = preds.get(rec["id"], (np.zeros((0, 1, 1)), np.zeros(0)))
        keep = np.where(scores >= thresh)[0]
        for i in keep:
            if masks[i].any():
                ax.contour(masks[i], levels=[0.5], colors=PRED_COLOR, linewidths=1.2, linestyles="--")
                y, x = np.argwhere(masks[i]).min(0)
                ax.text(x, y, f"{scores[i]:.2f}", color="white", fontsize=7, va="bottom",
                        bbox=dict(facecolor=PRED_COLOR, edgecolor="none", pad=1))
        ax.set_title(f"id {rec['id']}  GT {len(gt)}  pred {len(keep)}", fontsize=8, color=MUTED)

    head = f"{split} chips: GT (solid) vs predictions with score >= {thresh} (dashed)"
    if metrics:
        head += "   |   " + "  ".join(f"{k} {metrics[k]:.3f}" for k in METRICS if k in metrics)
    fig.suptitle(head, fontsize=10, x=0.01, ha="left")
    handles = [plt.Line2D([], [], color=GT_COLOR, lw=1.6, label="ground truth"),
               plt.Line2D([], [], color=PRED_COLOR, lw=1.2, ls="--", label="prediction")]
    fig.legend(handles=handles, loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out)


# ----------------------------------------------------------------------------- per run


def _split_rows(results: Path, exp: str, fold: int, seed: int) -> dict[str, dict]:
    """The latest row of this run in each split's results CSV."""
    import pandas as pd
    out = {}
    for split, name in RESULT_FILES.items():
        if (results / name).exists():
            df = pd.read_csv(results / name)
            df = df[(df["experiment"] == exp) & (df["fold"].astype(str) == str(fold)) & (df["seed"] == seed)]
            if len(df):
                out[split] = df.iloc[-1].to_dict()
    return out


def plot_run(results: Path, run: str) -> Path:
    """overview.png for one run (`<exp>_fold<k>_seed<s>`, k may be "all"): losses, val curves, split bars."""
    import pandas as pd
    exp, fold, seed = re.fullmatch(r"(.+)_fold(\d+|all)_seed(\d+)", run).groups()
    h = pd.read_csv(results / "history" / f"{run}.csv")
    rows = _split_rows(results, exp, fold, int(seed))

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 4.4))
    losses = [k for k in h.columns if k.startswith("train_loss_")]
    a.plot(h["epoch"], h["train_loss"], color=INK, label="total")
    for k, col in zip(losses, SERIES):
        a.plot(h["epoch"], h[k], color=col, label=k.removeprefix("train_loss_"))
    a.set(title="Train loss per epoch", xlabel="epoch", ylabel="mean loss")
    a.legend(fontsize=8)

    if "val_AP" in h:
        best = h.loc[h["val_AP"].idxmax()]
        for k, col in zip(METRICS, SERIES):
            b.plot(h["epoch"], h[f"val_{k}"], color=col, label=k)
        b.axvline(best["epoch"], color=MUTED, lw=1, ls=":")
        b.annotate(f"best epoch {int(best['epoch'])}\nAP {best['val_AP']:.3f}", (best["epoch"], best["val_AP"]),
                   xytext=(6, 10), textcoords="offset points", fontsize=8, color=MUTED)
        b.legend(fontsize=8)
    else:                                       # fold "all": no val, best.pth is the last epoch
        b.text(0.5, 0.5, "no val (fold all)\nbest.pth = last epoch", ha="center", va="center",
               transform=b.transAxes, color=MUTED)
    b.set(title="Validation metrics per epoch", xlabel="epoch", ylabel="score", ylim=(0, None))

    splits = [s for s in RESULT_FILES if s in rows]
    x, w = np.arange(len(METRICS)), 0.8 / max(1, len(splits))
    for j, s in enumerate(splits):
        vals = [float(rows[s][k]) for k in METRICS]
        bars = c.bar(x + (j - (len(splits) - 1) / 2) * w, vals, w - 0.03, color=SPLIT_COLOR[s], label=s)
        c.bar_label(bars, fmt="%.2f", fontsize=7, color=MUTED, padding=2)
    c.set_xticks(x, METRICS)
    c.set(title="Best checkpoint: train / val / test", ylabel="score", ylim=(0, 1.05))
    c.grid(axis="x", visible=False)
    if splits:
        c.legend(fontsize=8)
    else:
        c.text(0.5, 0.5, "no results rows yet", ha="center", transform=c.transAxes, color=MUTED)

    for ax in (a, b):
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.suptitle(f"{exp}  fold {fold}  seed {seed}", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, results / "figures" / run / "overview.png")


# ----------------------------------------------------------------------------- across runs


def plot_summary(results: Path) -> Path | None:
    """summary.png: mean ± std over folds/seeds per experiment, one bar per split, AP and AP50."""
    import pandas as pd
    frames = []
    for split, name in RESULT_FILES.items():
        if (results / name).exists():
            df = pd.read_csv(results / name).drop_duplicates(["experiment", "fold", "seed"], keep="last")
            frames.append(df.assign(split=split))
    if not frames:
        return None
    df = pd.concat(frames)
    exps = sorted(df["experiment"].unique())
    splits = [s for s in RESULT_FILES if s in set(df["split"])]

    fig, axes = plt.subplots(1, 2, figsize=(max(8, 2.2 * len(exps) * 2), 4.2))
    x, w = np.arange(len(exps)), 0.8 / len(splits)
    for ax, metric in zip(axes, ["AP", "AP50"]):
        for j, s in enumerate(splits):
            g = df[df["split"] == s].groupby("experiment")[metric]
            mean, std = g.mean().reindex(exps), g.std(ddof=1).reindex(exps).fillna(0)
            bars = ax.bar(x + (j - (len(splits) - 1) / 2) * w, mean.fillna(0), w - 0.03, yerr=std,
                          color=SPLIT_COLOR[s], ecolor=MUTED, capsize=2, label=s)
            ax.bar_label(bars, labels=[f"{v:.2f}" if v == v else "" for v in mean], fontsize=7,
                         color=MUTED, padding=2)
        ax.set_xticks(x, exps)
        ax.set(title=f"{metric} (mean ± std over folds/seeds)", ylabel=metric, ylim=(0, 1.05))
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, results / "figures" / "summary.png")


def plot_all(out_dir: str | Path) -> list[Path]:
    results = resolve_path(out_dir) / "results"
    made = [plot_run(results, p.stem) for p in sorted((results / "history").glob("*.csv"))]
    s = plot_summary(results)
    return made + ([s] if s else [])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    for p in plot_all(ap.parse_args().out_dir):
        print("->", p)
