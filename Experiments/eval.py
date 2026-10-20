"""Scoring: COCO mask AP (official scorer settings and COCO defaults) plus mean IoU.

    python eval.py --config configs/e4.yaml --fold 0 --seed 0   # score best.pth, append a CSV row
    python eval.py --config configs/e4.yaml --fold 0 --split test   # local held-out test -> results_test.csv
    python eval.py --summary [--split test]                     # mean ± std across folds
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import sys
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config import load_config, resolve_path, run_name
from data import build_loaders, build_test_loader, load_coco, release_root

CSV_FIELDS = ["experiment", "fold", "seed", "best_epoch", "AP", "AP50", "AP75",
              "AP_small", "AP_medium", "AP_large", "AP_default", "mIoU"]


def _scorer(cfg: dict):
    """The official scorer module, tools/evaluate_coco.py (settings + summarize_metric)."""
    tools = str(release_root(cfg) / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import evaluate_coco
    return evaluate_coco


@torch.no_grad()
def predict(model, loader, val_recs, device, amp: bool, mask_thresh: float = 0.5):
    """Run the model on the val loader.

    Returns COCO results (RLE masks) and binary masks per image, all cropped to orig_size.
    """
    model.eval()
    results, per_image = [], {}
    recs = iter(val_recs)                                     # val loader is not shuffled
    for images, targets in loader:
        with torch.autocast(device.type, enabled=amp):
            outs = model(list(images.to(device)))
        for out, t in zip(outs, targets):
            rec = next(recs)
            h, w = (int(v) for v in t["orig_size"])
            assert (h, w) == (rec["height"], rec["width"]), "val order / orig_size mismatch"
            masks = (out["masks"][:, 0, :h, :w] > mask_thresh).cpu().numpy().astype(np.uint8)
            scores = out["scores"].float().cpu().numpy()
            per_image[rec["id"]] = (masks, scores)
            for mk, sc in zip(masks, scores):
                rle = mask_utils.encode(np.asfortranarray(mk))
                rle["counts"] = rle["counts"].decode("utf-8")
                results.append({"image_id": rec["id"], "category_id": 1,
                                "segmentation": rle, "score": float(sc)})
    return results, per_image


def coco_gt(cfg: dict, image_ids) -> COCO:
    d = load_coco(cfg)
    keep = set(image_ids)
    sub = {"images": [im for im in d["images"] if im["id"] in keep],
           "annotations": [a for a in d["annotations"] if a["image_id"] in keep],
           "categories": d["categories"],
           "info": d.get("info", {}), "licenses": d.get("licenses", [])}   # some pycocotools copy these
    gt = COCO()
    gt.dataset = sub
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def _cocoeval(gt: COCO, results: list[dict], scorer=None) -> COCOeval:
    """Run COCOeval segm; with `scorer`, use the official maxDets and area bins."""
    with contextlib.redirect_stdout(io.StringIO()):
        # loadRes adds 'area'/'bbox' (numpy) to the dicts it gets; reusing them in a second
        # loadRes breaks on numpy 2 (`bbox == []`), so give each call its own copies.
        dt = gt.loadRes([dict(r) for r in results])
        ev = COCOeval(gt, dt, iouType="segm")
        if scorer is not None:
            ev.params.maxDets = list(scorer.DEFAULT_MAX_DETECTIONS)
            ev.params.areaRng = [list(r) for r in scorer.DEFAULT_AREA_RANGES]
            ev.params.areaRngLbl = list(scorer.DEFAULT_AREA_LABELS)
        ev.evaluate(); ev.accumulate()
    return ev


def official_ap(gt: COCO, results: list[dict], scorer) -> dict:
    """AP as the challenge scorer computes it (maxDets=10). Not COCOeval.summarize(): its
    headline AP is hard-wired to maxDets=100, which is absent here, so it would return -1."""
    if not results:
        return {k: 0.0 for k in ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")}
    ev = _cocoeval(gt, results, scorer)
    top = ev.params.maxDets[-1]

    def m(iou=None, area="all"):
        return max(0.0, scorer.summarize_metric(ev, ap=True, iou_thr=iou, area_label=area, max_det=top))
    return {"AP": m(), "AP50": m(0.5), "AP75": m(0.75),
            "AP_small": m(area="small"), "AP_medium": m(area="medium"), "AP_large": m(area="large")}


def default_ap(gt: COCO, results: list[dict]) -> float:
    """COCO-default mask AP@[.5:.95] (maxDets=100, standard area bins), for reference."""
    if not results:
        return 0.0
    ev = _cocoeval(gt, results)
    with contextlib.redirect_stdout(io.StringIO()):
        ev.summarize()
    return max(0.0, float(ev.stats[0]))


def mean_iou(gt: COCO, per_image: dict, score_thresh: float) -> float:
    """Per image: IoU of the union of predicted masks (score >= thresh) against the union of GT."""
    ious = []
    for img_id, (masks, scores) in per_image.items():
        info = gt.imgs[img_id]
        g = np.zeros((info["height"], info["width"]), bool)
        for a in gt.imgToAnns[img_id]:
            g |= mask_utils.decode(a["segmentation"]).astype(bool)
        keep = scores >= score_thresh
        p = masks[keep].any(0) if keep.any() else np.zeros_like(g)
        union = (g | p).sum()
        ious.append(1.0 if union == 0 else (g & p).sum() / union)
    return float(np.mean(ious)) if ious else 0.0


def evaluate(model, loader, val_recs, cfg, device, amp: bool) -> dict:
    results, per_image = predict(model, loader, val_recs, device, amp, cfg["eval"]["mask_thresh"])
    gt = coco_gt(cfg, [r["id"] for r in val_recs])
    return {**official_ap(gt, results, _scorer(cfg)), "AP_default": default_ap(gt, results),
            "mIoU": mean_iou(gt, per_image, cfg["eval"]["score_thresh"])}


def results_file(split: str) -> str:
    return "results.csv" if split == "val" else f"results_{split}.csv"


def append_row(cfg: dict, row: dict, split: str = "val") -> Path:
    path = resolve_path(cfg["out_dir"]) / "results" / results_file(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    return path


def summarize(out_dir: str | Path, split: str = "val") -> "pd.DataFrame":
    """Mean ± std across folds (and seeds) per experiment; writes summary[_split].csv next to the results."""
    import pandas as pd

    res = resolve_path(out_dir) / "results"
    df = pd.read_csv(res / results_file(split))
    df = df.drop_duplicates(["experiment", "fold", "seed"], keep="last")   # re-runs overwrite
    metrics = ["AP", "AP50", "AP75", "AP_default", "mIoU"]
    g = df.groupby("experiment")
    table = pd.DataFrame({"runs": g.size(), "folds": g["fold"].nunique()})
    for m in metrics:
        table[m] = [f"{a:.4f} ± {s:.4f}" for a, s in zip(g[m].mean(), g[m].std(ddof=1).fillna(0))]
    table.to_csv(res / ("summary.csv" if split == "val" else f"summary_{split}.csv"))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--split", choices=["val", "test"], default="val",
                    help="test = the grouped split's held-out local test chips")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--out_dir", default=None, help="for --summary without --config")
    args = ap.parse_args()

    if args.summary:
        out = args.out_dir or load_config(args.config or "configs/base.yaml", args.override)["out_dir"]
        print(summarize(out, args.split).to_string())
        return

    from models.build import build_model

    cfg = load_config(args.config, args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_path(cfg["out_dir"]) / "runs" / run_name(cfg) / f"fold{args.fold}_seed{args.seed}" / "best.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg["model"]["pretrained"] = False                 # weights come from the checkpoint
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    if args.split == "test":
        dl, recs = build_test_loader(cfg, args.fold)
    else:
        _, dl, recs = build_loaders(cfg, args.fold, args.seed)
    m = evaluate(model, dl, recs, cfg, device, cfg["train"]["amp"] and device.type == "cuda")
    row = {"experiment": run_name(cfg), "fold": args.fold, "seed": args.seed,
           "best_epoch": ckpt["epoch"], **{k: float(v) for k, v in m.items()}}
    print(row)
    print("->", append_row(cfg, row, args.split))


if __name__ == "__main__":
    main()
