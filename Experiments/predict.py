"""Test-set inference: one trained run -> submission.json (COCO results, official test ids).

    python predict.py --run ../results/e3_e4_result/outputs/runs/e4/fold2_seed0 \
        --root ../competition_release --stats ../results/e3_e4_result/cache/stats_fold2.json \
        --out ../submission.json
    add --check_val to first re-score the run's val fold (should reproduce its results.csv AP)

Preprocessing is the training path: neighbour-mean NaN fill (train-mean fallback), the fold's
train mean/std, channel select (swin then terrain), mean padding. Ids come from test_manifest.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from pycocotools import mask as mask_utils

from data import stream_indices
from rts_preprocess import fill_nan_neighbor_mean, normalize, pad_collate


def load_run(run_dir: Path, root: Path, cache_dir: Path):
    cfg = yaml.safe_load((run_dir / "config_resolved.yaml").read_text())
    cfg.pop("run", None)
    cfg["data"].update(root=str(root.resolve()), cache_dir=str(cache_dir.resolve()), num_workers=0)
    cfg["model"]["pretrained"] = False                 # weights come from the checkpoint
    from models.build import build_model
    model = build_model(cfg)
    ckpt = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    return cfg, model.eval(), ckpt


def test_records(root: Path) -> list[dict]:
    with (root / "metadata/test_manifest.csv").open() as f:
        return [{"id": int(r["image_id"]), "path": root / r["file_name"],
                 "height": int(r["height"]), "width": int(r["width"])} for r in csv.DictReader(f)]


@torch.no_grad()
def predict_test(model, recs, mean, std, channels, mask_thresh, top_k, batch_size, device):
    results = []
    for i in range(0, len(recs), batch_size):
        chunk, batch = recs[i:i + batch_size], []
        for r in chunk:
            with np.load(r["path"]) as z:
                raw = np.ascontiguousarray(z["image"].transpose(2, 0, 1), dtype=np.float32)
            assert raw.shape[1:] == (r["height"], r["width"]), f"size mismatch for {r['path']}"
            filled, _ = fill_nan_neighbor_mean(raw, fallback=mean)
            img = torch.from_numpy(normalize(filled, mean, std))[channels]
            batch.append((img, {}))
        images, targets = pad_collate(batch)
        outs = model(list(images.to(device)))
        for r, out, t in zip(chunk, outs, targets):
            h, w = (int(v) for v in t["orig_size"])
            order = out["scores"].argsort(descending=True)[:top_k]   # scorer uses maxDets=10
            masks = (out["masks"][order, 0, :h, :w] > mask_thresh).cpu().numpy().astype(np.uint8)
            for mk, sc in zip(masks, out["scores"][order].float().cpu().numpy()):
                if not mk.any():
                    continue
                rle = mask_utils.encode(np.asfortranarray(mk))
                rle["counts"] = rle["counts"].decode("utf-8")
                results.append({"image_id": r["id"], "category_id": 1,
                                "segmentation": rle, "score": float(sc)})
        print(f"{min(i + batch_size, len(recs))}/{len(recs)}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, help="runs/<exp>/fold<k>_seed<s> dir")
    ap.add_argument("--root", required=True, type=Path, help="competition_release dir")
    ap.add_argument("--stats", required=True, type=Path, help="stats_fold<k>.json of that run")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--check_val", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model, ckpt = load_run(args.run, args.root, args.stats.parent)
    model.to(device)
    fold = ckpt["run"]["fold"]
    assert args.stats.name == f"stats_fold{fold}.json", "stats must belong to the run's fold"
    s = json.loads(args.stats.read_text())
    mean, std = np.array(s["mean"], np.float32), np.array(s["std"], np.float32)
    si = stream_indices(cfg)
    channels = si["swin"] + si["terrain"]
    assert channels == ckpt["run"]["channel_indices"]["swin"] + ckpt["run"]["channel_indices"]["terrain"]

    if args.check_val:
        from data import build_loaders
        from eval import evaluate
        _, val_dl, val_recs = build_loaders(cfg, fold, ckpt["run"]["seed"])
        m = evaluate(model, val_dl, val_recs, cfg, device, amp=False)
        print("val fold", fold, {k: round(float(v), 4) for k, v in m.items()},
              "| recorded AP", round(ckpt["metrics"]["AP"], 4))

    recs = test_records(args.root)
    results = predict_test(model, recs, mean, std, channels, cfg["eval"]["mask_thresh"],
                           args.top_k, args.batch_size, device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results))
    print(f"{len(results)} predictions over {len({r['image_id'] for r in results})}/{len(recs)} images -> {args.out}")


if __name__ == "__main__":
    main()
