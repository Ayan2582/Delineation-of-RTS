"""Turn fetched DARTS chips into a GeoAI-format dataset: COCO annotations + previews.

    python DARTS_EDA/finalize_darts_dataset.py --dir DARTS_EDA/dataset

Writes
    <dir>/annotations/instances_darts.json   COCO, RLE masks, file_name = <chip>.npz (like the GeoAI train json)
    <dir>/previews/00_size_comparison.png    chip size, slump size, slumps per chip vs GeoAI
    <dir>/previews/01_gallery.png            RGB with every slump outlined
    <dir>/previews/02_bands.png              RGB / mask / ndvi / relative elevation / shaded relief
    <dir>/previews/03_geoai_vs_darts.png     the two datasets side by side
Chips are read one at a time, so memory stays small. Uses the base env (pycocotools).
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from pycocotools import mask as mask_utils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL = os.path.join(ROOT, "competition_release")
MIN_PX = 10                       # GeoAI has no instance under 10 px
SCORER_BINS = (300, 2000)


def stretch(im):
    rgb = im[..., :3].astype(np.float32)
    lo, hi = np.nanpercentile(rgb, 2, (0, 1)), np.nanpercentile(rgb, 98, (0, 1))
    return np.nan_to_num(np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1))


def grey(a_, x, title=None, cmap="gray"):
    a_.imshow(x, cmap=cmap, vmin=np.nanpercentile(x, 2), vmax=np.nanpercentile(x, 98))
    a_.axis("off")
    if title:
        a_.set_title(title, fontsize=9)


def outline(a_, inst):
    for k in np.unique(inst)[1:]:
        a_.contour(inst == k, [.5], colors=[plt.cm.tab10(int(k) % 10)], linewidths=1)


def write_coco(d, chips):
    images, anns, rows = [], [], []
    for img_id, c in enumerate(chips, 1):
        z = np.load(os.path.join(d, c + ".npz"))
        h, w = z["inst"].shape
        images.append({"id": img_id, "file_name": c + ".npz", "width": int(w), "height": int(h)})
        n = 0
        for k in np.unique(z["inst"])[1:]:
            m = z["inst"] == k
            if m.sum() < MIN_PX:      # slivers where a polygon only clips the chip edge
                continue
            rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": 1, "iscrowd": 0,
                         "area": float(mask_utils.area(rle)), "bbox": [float(v) for v in mask_utils.toBbox(rle)],
                         "segmentation": {"size": rle["size"], "counts": rle["counts"].decode("ascii")}})
            n += 1
        rows.append({"chip": c, "width": w, "height": h, "n_inst": n})
    coco = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "rts"}],
            "info": {"description": "DARTS v1.2 training labels on Sentinel-2 + ArcticDEM, GeoAI band order, 4 m grid"}}
    os.makedirs(os.path.join(d, "annotations"), exist_ok=True)
    json.dump(coco, open(os.path.join(d, "annotations", "instances_darts.json"), "w"))
    return coco, pd.DataFrame(rows)


def size_figure(coco, chips_df, out):
    gm = pd.read_csv(os.path.join(REL, "metadata", "train_manifest.csv"))
    gc = json.load(open(os.path.join(REL, "train", "annotations", "instances_train.json")))
    g_area = np.array([a["area"] for a in gc["annotations"]])
    d_area = np.array([a["area"] for a in coco["annotations"]])
    g_per = pd.Series([a["image_id"] for a in gc["annotations"]]).value_counts().reindex(
        [im["id"] for im in gc["images"]], fill_value=0)
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.5))
    ax[0].scatter(gm.width, gm.height, s=6, alpha=.4, label=f"GeoAI ({len(gm)})")
    ax[0].scatter(chips_df.width, chips_df.height, s=6, alpha=.4, c="C3", label=f"DARTS ({len(chips_df)})")
    ax[0].set(xlabel="width (px)", ylabel="height (px)", title="chip size (4 m px)"); ax[0].legend()
    for a_, col, lab in ((ax[1], "width", "chip width"), (ax[2], "height", "chip height")):
        b = np.linspace(0, max(gm[col].max(), chips_df[col].max()) + 10, 40)
        a_.hist(gm[col], b, density=True, alpha=.55, label="GeoAI"); a_.hist(chips_df[col], b, density=True, alpha=.55, color="C3", label="DARTS")
        a_.set_title(lab); a_.legend()
    b = np.logspace(1, 5.5, 50)
    ax[3].hist(g_area, b, density=True, alpha=.55, label=f"GeoAI, median {np.median(g_area):.0f} px")
    ax[3].hist(d_area, b, density=True, alpha=.55, color="C3", label=f"DARTS, median {np.median(d_area):.0f} px")
    for t in SCORER_BINS:
        ax[3].axvline(t, c="k", ls="--", lw=1)
    ax[3].set(xscale="log", xlabel="slump area (px)", title="slump size (scorer bins 300 / 2000 px)"); ax[3].legend()
    plt.tight_layout(); fig.savefig(os.path.join(out, "00_size_comparison.png"), dpi=110); plt.close(fig)

    def bins(a):
        return [(a < 300).mean(), ((a >= 300) & (a < 2000)).mean(), (a >= 2000).mean()]
    return pd.DataFrame({
        "GeoAI": [len(gm), f"{gm.height.median():.0f} x {gm.width.median():.0f}", len(g_area), np.median(g_area),
                  g_per.mean(), *bins(g_area)],
        "DARTS": [len(chips_df), f"{chips_df.height.median():.0f} x {chips_df.width.median():.0f}", len(d_area),
                  np.median(d_area), chips_df.n_inst.mean(), *bins(d_area)]},
        index=["chips", "median chip (h x w px)", "slumps", "median slump (px)", "slumps per chip",
               "small <300", "medium 300-2000", "large >=2000"])


def previews(d, chips_df, out, rng):
    pick = chips_df[chips_df.n_inst > 0].chip.to_numpy()
    load = lambda c: np.load(os.path.join(d, c + ".npz"))

    sel = rng.choice(pick, min(16, len(pick)), replace=False)
    fig, ax = plt.subplots(4, 4, figsize=(18, 12))
    for a_, c in zip(ax.flat, sel):
        z = load(c)
        a_.imshow(stretch(z["image"])); outline(a_, z["inst"]); a_.axis("off")
        m = json.loads(str(z["meta"]))
        a_.set_title(f"{c} · {m.get('region')} · {m.get('s2_date', '')}", fontsize=8)
    fig.suptitle("DARTS chips: Sentinel-2 RGB (2-98 % stretch), each slump outlined", y=1.0)
    plt.tight_layout(); fig.savefig(os.path.join(out, "01_gallery.png"), dpi=100); plt.close(fig)

    sel = rng.choice(pick, min(6, len(pick)), replace=False)
    fig, ax = plt.subplots(len(sel), 5, figsize=(19, 3.3 * len(sel)))
    cmap = ListedColormap(["black"] + [plt.cm.tab10(i) for i in range(10)])
    for r, c in enumerate(sel):
        z = load(c); im, inst = z["image"], z["inst"]
        ax[r, 0].imshow(stretch(im)); outline(ax[r, 0], inst); ax[r, 0].axis("off")
        ax[r, 1].imshow(np.where(inst > 0, (inst - 1) % 10 + 1, 0), cmap=cmap, vmin=0, vmax=10, interpolation="nearest"); ax[r, 1].axis("off")
        grey(ax[r, 2], im[..., 3], cmap="RdYlGn"); grey(ax[r, 3], im[..., 4]); grey(ax[r, 4], im[..., 5])
        for k in (2, 3, 4):
            outline(ax[r, k], inst)
        ax[r, 0].set_ylabel(c)
    for k, t in enumerate(["RGB (S2)", "instance mask", "ndvi", "relative elevation (100 m)", "shaded relief"]):
        ax[0, k].set_title(t)
    plt.tight_layout(); fig.savefig(os.path.join(out, "02_bands.png"), dpi=95); plt.close(fig)

    gc = json.load(open(os.path.join(REL, "train", "annotations", "instances_train.json")))
    by_img = {}
    for a in gc["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    gimgs = [im for im in gc["images"] if im["id"] in by_img]
    fig, ax = plt.subplots(2, 6, figsize=(22, 6.5))
    for a_, im in zip(ax[0], rng.choice(gimgs, 6, replace=False)):
        x = np.load(os.path.join(REL, "train", "images", os.path.basename(im["file_name"])))["image"]
        inst = np.zeros(x.shape[:2], int)
        for k, a in enumerate(by_img[im["id"]], 1):
            inst[mask_utils.decode(a["segmentation"]).astype(bool)] = k
        a_.imshow(stretch(x)); outline(a_, inst); a_.axis("off"); a_.set_title(f"GeoAI {x.shape[0]}x{x.shape[1]}", fontsize=9)
    for a_, c in zip(ax[1], rng.choice(pick, 6, replace=False)):
        z = load(c)
        a_.imshow(stretch(z["image"])); outline(a_, z["inst"]); a_.axis("off")
        a_.set_title(f"DARTS {z['inst'].shape[0]}x{z['inst'].shape[1]}", fontsize=9)
    fig.suptitle("top: GeoAI (Maxar 4 m)   bottom: DARTS (Sentinel-2 10 m on the 4 m grid)")
    plt.tight_layout(); fig.savefig(os.path.join(out, "03_geoai_vs_darts.png"), dpi=95); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "DARTS_EDA", "dataset"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    man = pd.read_csv(os.path.join(args.dir, "manifest.csv"))
    chips = man.loc[man.status == "ok", "chip"].tolist()
    coco, chips_df = write_coco(args.dir, chips)
    out = os.path.join(args.dir, "previews")
    os.makedirs(out, exist_ok=True)
    table = size_figure(coco, chips_df, out)
    previews(args.dir, chips_df, out, np.random.default_rng(args.seed))
    print(man.status.str.split(":").str[0].value_counts().to_string(), "\n")
    print(table.to_string(float_format=lambda v: f"{v:.3g}"))
    print("\nwritten:", os.path.join(args.dir, "annotations", "instances_darts.json"), "and", out)


if __name__ == "__main__":
    main()
