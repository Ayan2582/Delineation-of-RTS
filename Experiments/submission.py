"""Make a leaderboard submission.json from one trained checkpoint.

Edit the constants below, then:
    python submission.py
or override any of them from the command line:
    python submission.py --model ../results/e3_e4_result/outputs/runs/e4/fold2_seed0/best.pth --max_dets 5
    add --check_val to first re-score the run's val fold (should reproduce its results.csv AP)

The checkpoint's folder must also hold the run's config_resolved.yaml (train.py writes both).
Preprocessing matches training: neighbour-mean NaN fill, the fold's train mean/std, the run's
band order. Image ids come from test_manifest.csv. The file is checked with the official validator.

Per image, the model's detections are processed highest score first:
    1. drop scores below MIN_SCORE
    2. NO_OVERLAP: a pixel belongs to the first (highest-scoring) mask that claims it. Later masks
       lose already-claimed pixels; a mask left with < MIN_KEEP of its area (or < 10 px) is dropped
    3. keep the first MAX_DETS that remain
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from pycocotools import mask as mask_utils

# ============================== constants: edit these ==============================
MODEL      = "../results/e6_pretrained/outputs/runs/e6_darts_darts_ft/fold1_seed0/best.pth"   # checkpoint to use
MAX_DETS   = 10   # predictions kept per image (the scorer uses at most 10)
NO_OVERLAP = True     # later masks may not reuse pixels of higher-scoring masks
MIN_KEEP   = 1      # with NO_OVERLAP: drop a mask left with less than this share of its pixels
MIN_SCORE  = 0.1  # drop predictions below this score (the model already drops < 0.05)
DATA_ROOT  = "../competition_release"                                       # release folder
STATS      = None     # fold's stats_*.json; None = the run's stats.json next to best.pth, else recomputed
OUT        = "../submissions/e6_pretrained_best_1/submission.json"
BATCH_SIZE = 4
# ===================================================================================

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from data import input_norm, load_raw, split_records, stream_indices, test_records   # noqa: E402
from models.build import build_model                                                 # noqa: E402
from rts_preprocess import compute_channel_stats, fill_nan_neighbor_mean, normalize, pad_collate  # noqa: E402

MIN_PIXELS = 10       # the labels drop components smaller than this, so predictions do too


def resolve(p: str | Path) -> Path:
    """Relative paths are relative to this script's folder, not the cwd."""
    p = Path(p)
    return p if p.is_absolute() else (CODE_DIR / p).resolve()


def load_model(model_path: Path, root: Path):
    cfg_path = model_path.parent / "config_resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found; the checkpoint needs its run's config next to it")
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.pop("run", None)
    cfg["data"].update(root=str(root), limit=None)
    cfg["model"]["pretrained"] = False                  # weights come from the checkpoint
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    return cfg, model.eval(), ckpt


def run_stats(cfg: dict, fold: int, stats_path: Path | None):
    """The normalisation the run was trained with: its stats file, or recomputed from the same chips."""
    if stats_path is not None:
        s = json.loads(stats_path.read_text())
        return np.array(s["mean"], np.float32), np.array(s["std"], np.float32), str(stats_path)
    folds_file = resolve(cfg["data"]["folds_file"])
    if not folds_file.exists():
        raise FileNotFoundError(f"{folds_file} (the run's folds_file) not found; set STATS instead")
    cfg = {**cfg, "data": {**cfg["data"], "folds_file": str(folds_file)}}
    train_recs, _ = split_records(cfg, fold)
    mean, std = compute_channel_stats([load_raw(r) for r in train_recs])
    return mean, std, f"recomputed from {len(train_recs)} fold-{fold} train chips ({folds_file.name})"


def select(masks: np.ndarray, scores: np.ndarray, max_dets: int, no_overlap: bool,
           min_keep: float, min_score: float) -> list[tuple[np.ndarray, float]]:
    """Masks (N, H, W) bool and scores (N,) -> up to max_dets (mask, score), highest score first."""
    kept, claimed = [], np.zeros(masks.shape[1:], bool)
    for i in np.argsort(-scores):
        if scores[i] < min_score or len(kept) == max_dets:
            break
        m = masks[i]
        area = int(m.sum())
        if no_overlap:
            m = m & ~claimed
            if m.sum() < max(MIN_PIXELS, min_keep * area):
                continue
            claimed |= m
        elif area < MIN_PIXELS:
            continue
        kept.append((m, float(scores[i])))
    return kept


@torch.no_grad()
def predict(model, recs, mean, channels, mask_thresh, args, device, norm) -> list[dict]:
    """`mean` is the raw train mean (NaN fallback); `norm` = input_norm(...) is what normalises."""
    results = []
    for i in range(0, len(recs), args.batch_size):
        chunk, batch = recs[i:i + args.batch_size], []
        for r in chunk:
            raw = load_raw(r)
            assert raw.shape[1:] == (r["height"], r["width"]), f"size mismatch for {r['path']}"
            filled = fill_nan_neighbor_mean(raw, fallback=mean)
            batch.append((torch.from_numpy(normalize(filled, *norm))[channels], {}))
        images, targets = pad_collate(batch)
        for r, out, t in zip(chunk, model(list(images.to(device))), targets):
            h, w = (int(v) for v in t["orig_size"])
            masks = (out["masks"][:, 0, :h, :w] > mask_thresh).cpu().numpy()
            scores = out["scores"].float().cpu().numpy()
            for m, s in select(masks, scores, args.max_dets, args.no_overlap, args.min_keep, args.min_score):
                rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")
                results.append({"image_id": r["id"], "category_id": 1, "segmentation": rle, "score": s})
        print(f"{min(i + args.batch_size, len(recs))}/{len(recs)}", end="\r", flush=True)
    print()
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL, help="path to best.pth")
    ap.add_argument("--max_dets", type=int, default=MAX_DETS, help="predictions kept per image")
    ap.add_argument("--no_overlap", action=argparse.BooleanOptionalAction, default=NO_OVERLAP)
    ap.add_argument("--min_keep", type=float, default=MIN_KEEP)
    ap.add_argument("--min_score", type=float, default=MIN_SCORE)
    ap.add_argument("--root", default=DATA_ROOT, help="competition_release folder")
    ap.add_argument("--stats", default=STATS, help="stats json of the run's fold (optional)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--check_val", action="store_true", help="re-score the run's val fold before predicting")
    args = ap.parse_args()
    if args.max_dets < 1:
        ap.error("--max_dets must be at least 1")
    if not 0 <= args.min_keep <= 1:
        ap.error("--min_keep must be in [0, 1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path, root, out = resolve(args.model), resolve(args.root), resolve(args.out)
    cfg, model, ckpt = load_model(model_path, root)
    model.to(device)
    fold = ckpt["run"]["fold"]
    run_stats = model_path.parent / "stats.json"          # train.py copies the fold's stats here
    stats = resolve(args.stats) if args.stats else (run_stats if run_stats.exists() else None)
    mean, std, stats_src = run_stats(cfg, fold, stats)

    si = stream_indices(cfg)
    channels = si["swin"] + si["terrain"]
    trained = ckpt["run"]["channel_indices"]
    assert channels == trained["swin"] + trained["terrain"], "band order differs from the trained run"

    val = f"val AP {ckpt['metrics']['AP']:.4f}" if ckpt.get("metrics") else "no val: fold 'all', last epoch"
    print(f"model   {model_path}  ({ckpt['run']['experiment']}, fold {fold}, epoch {ckpt['epoch']}, {val})")
    print(f"stats   {stats_src}")
    print(f"device  {device} | max_dets {args.max_dets} | no_overlap {args.no_overlap} "
          f"(min_keep {args.min_keep}) | min_score {args.min_score}")

    if args.check_val and not ckpt.get("metrics"):
        print("--check_val skipped: this run (fold 'all') has no val split")
    elif args.check_val:
        from data import build_loaders
        from eval import evaluate
        vcfg = {**cfg, "data": {**cfg["data"], "folds_file": str(resolve(cfg["data"]["folds_file"])),
                                "cache_dir": str(CODE_DIR / "cache"), "num_workers": 0}}
        _, val_dl, val_recs = build_loaders(vcfg, fold, ckpt["run"]["seed"])
        m = evaluate(model, val_dl, val_recs, vcfg, device, amp=False)
        print(f"val     fold {fold}: AP {m['AP']:.4f} (recorded {ckpt['metrics']['AP']:.4f}), "
              f"AP50 {m['AP50']:.4f}, mIoU {m['mIoU']:.4f}")

    recs = test_records(root)
    results = predict(model, recs, mean, channels, cfg["eval"]["mask_thresh"], args, device,
                      input_norm(cfg, mean, std))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results))
    n_img = len({r["image_id"] for r in results})
    print(f"wrote   {out}: {len(results)} predictions over {n_img}/{len(recs)} images")

    check = subprocess.run([sys.executable, "tools/validate_submission.py", "--submission", str(out)],
                           cwd=root, capture_output=True, text=True)
    print((check.stdout + check.stderr).strip())
    if check.returncode:
        sys.exit("validation failed: do not upload this file")


if __name__ == "__main__":
    main()
