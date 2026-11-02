#!/usr/bin/env python3
"""Convert the RTS .npz chips to RGB PNGs and rewrite the COCO labels to match.

Why this exists
---------------
Every chip is an (H, W, 8) float32 array, but bands 0-2 (red, green, blue) hold 8-bit
digital numbers: measured over all 894 chips, every RGB value is an exact integer in
[0, 255]. So writing them as uint8 PNG is *bit-exact lossless*, and it lets detectron2's
own DatasetMapper read the data - no custom mapper, no .npz handling, no manual
category_id remapping.

The labels are pure geometry (RLE masks + pixel-coordinate boxes) and this conversion
never touches H or W, so annotations carry over verbatim. Only `file_name` changes.

Output (ready to zip and upload as a Kaggle Dataset):

    data_png/
      train/*.png                         756 chips
      test/*.png                          138 chips
      annotations/instances_train_fold.json
      annotations/instances_val_fold.json
      test_manifest.csv

Run from the project root:  python prepare_png_dataset.py
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

RELEASE = Path("competition_release")
OUT = Path("data_png")

VAL_FRACTION = 0.15
SEED = 42
STRAT_CLIP = 6  # instance counts >= this are pooled into one stratification bin


def load_rgb_u8(npz_path: Path) -> tuple[np.ndarray, int]:
    """Load one chip's RGB bands as uint8. Returns (array, n_nan_pixels_filled)."""
    with np.load(npz_path) as data:
        rgb = data["image"][..., :3]

    n_nan = int(np.isnan(rgb).sum())
    rgb = np.nan_to_num(rgb, nan=0.0)

    # Guard the lossless claim rather than trusting it.
    if rgb.min() < 0 or rgb.max() > 255:
        raise ValueError(f"{npz_path.name}: RGB outside [0, 255] ({rgb.min()}, {rgb.max()})")
    if not np.array_equal(rgb, np.round(rgb)):
        raise ValueError(f"{npz_path.name}: RGB values are not integral; PNG would lose data")

    return rgb.astype(np.uint8), n_nan


def write_png(dst: Path, rgb_u8: np.ndarray) -> None:
    """Write an RGB uint8 array as PNG and verify the round trip is bit-exact."""
    # cv2 writes BGR to disk, so reverse to get a correct-looking RGB PNG.
    if not cv2.imwrite(str(dst), rgb_u8[:, :, ::-1]):
        raise IOError(f"cv2.imwrite failed for {dst}")

    back = cv2.imread(str(dst), cv2.IMREAD_COLOR)
    if back is None:
        raise IOError(f"could not read back {dst}")
    if not np.array_equal(back[:, :, ::-1], rgb_u8):
        raise ValueError(f"{dst.name}: PNG round trip is not bit-exact - aborting")


def convert_split(split: str) -> tuple[int, int]:
    """Convert every .npz in one split to PNG. Returns (n_chips, n_nan_pixels)."""
    src_dir = RELEASE / split / "images"
    dst_dir = OUT / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"error: no .npz files under {src_dir}")

    total_nan = 0
    for i, npz_path in enumerate(files, 1):
        rgb_u8, n_nan = load_rgb_u8(npz_path)
        total_nan += n_nan
        write_png(dst_dir / f"{npz_path.stem}.png", rgb_u8)
        if i % 100 == 0 or i == len(files):
            print(f"  {split}: {i}/{len(files)}", flush=True)

    return len(files), total_nan


def stratified_split(images: list[dict], counts: dict[int, int]) -> tuple[set[int], set[int]]:
    """Split image ids into train/val, stratified by (clipped) instance count."""
    from sklearn.model_selection import train_test_split

    ids = [im["id"] for im in images]
    bins = [min(counts.get(i, 0), STRAT_CLIP) for i in ids]

    train_ids, val_ids = train_test_split(
        ids, test_size=VAL_FRACTION, random_state=SEED, stratify=bins
    )
    return set(train_ids), set(val_ids)


def write_fold(path: Path, coco: dict, keep: set[int]) -> tuple[int, int]:
    """Write one COCO fold, pointing file_name at the PNGs. Returns (n_images, n_anns)."""
    images = []
    for im in coco["images"]:
        if im["id"] not in keep:
            continue
        rec = dict(im)
        # The only field the conversion changes. Note the release writes file_name as
        # "images/train_000001.npz" - we take the stem and point at a flat PNG directory.
        rec["file_name"] = f"{Path(im['file_name']).stem}.png"
        images.append(rec)

    anns = [a for a in coco["annotations"] if a["image_id"] in keep]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"images": images, "annotations": anns, "categories": coco["categories"]},
            indent=1,
        ),
        encoding="utf-8",
    )
    return len(images), len(anns)


def main() -> None:
    if not RELEASE.is_dir():
        raise SystemExit(f"error: {RELEASE} not found - run this from the project root")

    print("Converting chips to PNG (RGB, bands 0-2)")
    n_train, nan_train = convert_split("train")
    n_test, nan_test = convert_split("test")

    print("\nRewriting COCO labels")
    coco = json.loads((RELEASE / "train/annotations/instances_train.json").read_text())

    counts = Counter(a["image_id"] for a in coco["annotations"])
    train_ids, val_ids = stratified_split(coco["images"], counts)

    ann_dir = OUT / "annotations"
    n_tr_img, n_tr_ann = write_fold(ann_dir / "instances_train_fold.json", coco, train_ids)
    n_va_img, n_va_ann = write_fold(ann_dir / "instances_val_fold.json", coco, val_ids)

    shutil.copy(RELEASE / "metadata/test_manifest.csv", OUT / "test_manifest.csv")

    print("\n" + "=" * 62)
    print("CONVERSION REPORT")
    print("=" * 62)
    print(f"train PNGs written : {n_train}   (NaN pixels filled with 0: {nan_train})")
    print(f"test  PNGs written : {n_test}   (NaN pixels filled with 0: {nan_test})")
    print(f"train fold         : {n_tr_img} images, {n_tr_ann} annotations")
    print(f"val   fold         : {n_va_img} images, {n_va_ann} annotations")
    print(f"folds disjoint     : {not (train_ids & val_ids)}")
    print(f"folds cover all    : {len(train_ids | val_ids) == len(coco['images'])}")

    print("\ninstance-count distribution per fold (share of fold):")
    print(f"  {'instances':<12}{'train':>10}{'val':>10}")
    tr_counts = Counter(min(counts.get(i, 0), STRAT_CLIP) for i in train_ids)
    va_counts = Counter(min(counts.get(i, 0), STRAT_CLIP) for i in val_ids)
    for b in sorted(set(tr_counts) | set(va_counts)):
        label = f"{b}+" if b >= STRAT_CLIP else str(b)
        print(f"  {label:<12}{tr_counts[b] / len(train_ids):>9.1%}{va_counts[b] / len(val_ids):>10.1%}")

    print(f"\nwrote {OUT}/  - zip this directory and upload it as a Kaggle Dataset")


if __name__ == "__main__":
    sys.exit(main())
