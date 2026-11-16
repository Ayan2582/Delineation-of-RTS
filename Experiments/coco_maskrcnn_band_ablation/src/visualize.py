from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_prediction_samples(
    dataset,
    records: list[dict],
    per_image: dict,
    output_path: Path,
    band_names: list[str],
    n_samples: int = 8,
    score_threshold: float = 0.5,
    seed: int = 42,
):
    """Save selected-band composite with solid GT and dashed predicted outlines."""
    rng = np.random.default_rng(seed)
    positive = [i for i, rec in enumerate(records) if rec["anns"]]
    negative = [i for i, rec in enumerate(records) if not rec["anns"]]
    chosen = list(rng.permutation(positive)[:n_samples])
    if len(chosen) < n_samples:
        remaining = n_samples - len(chosen)
        chosen.extend(list(rng.permutation(negative)[:remaining]))
    chosen = sorted(chosen, key=lambda i: records[i]["id"])
    if not chosen:
        return None

    cols = min(4, len(chosen))
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.8 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, index in zip(axes.flat, chosen):
        image, target = dataset[index]
        composite = image.detach().cpu().numpy().transpose(1, 2, 0)
        ax.imshow(np.clip(composite, 0.0, 1.0))

        gt_masks = target["masks"].detach().cpu().numpy()
        for gt_mask in gt_masks:
            if gt_mask.any():
                ax.contour(gt_mask, levels=[0.5], colors=["#20a36b"], linewidths=1.5)

        image_id = int(records[index]["id"])
        pred_masks, scores = per_image.get(
            image_id, (np.zeros((0, *composite.shape[:2]), dtype=np.uint8), np.zeros(0))
        )
        kept = np.flatnonzero(scores >= score_threshold)
        for pred_idx in kept:
            pred_mask = pred_masks[pred_idx]
            if pred_mask.any():
                ax.contour(
                    pred_mask, levels=[0.5], colors=["#f07832"],
                    linewidths=1.2, linestyles="--"
                )
                ys, xs = np.where(pred_mask)
                ax.text(
                    float(xs.min()), float(ys.min()), f"{scores[pred_idx]:.2f}",
                    color="white", fontsize=7,
                    bbox={"facecolor": "#f07832", "edgecolor": "none", "pad": 1.2},
                )

        ax.set_title(
            f"ID {image_id} | GT {len(gt_masks)} | Pred {len(kept)}",
            fontsize=9,
        )

    handles = [
        plt.Line2D([], [], color="#20a36b", lw=1.7, label="Ground truth"),
        plt.Line2D([], [], color="#f07832", lw=1.3, ls="--", label="Prediction"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        f"{', '.join(band_names)} | Ground truth vs predictions (score ≥ {score_threshold:.2f})",
        y=0.995, fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
