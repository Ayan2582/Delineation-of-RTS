"""Copy the DARTS dataset into the competition release's layout, ready to upload as a Kaggle dataset.

    python DARTS_EDA/export_darts_kaggle.py            # -> DARTS_EDA/kaggle_darts/

    kaggle_darts/train/images/darts_XXXXX.npz          `image` only (H, W, 8) float32, GeoAI band order
    kaggle_darts/train/annotations/instances_train.json   = instances_darts.json, file_name "images/<chip>.npz"
    kaggle_darts/metadata/band_names.json              copied from the release (same band order)
    kaggle_darts/tools/{evaluate_coco,coco_utils}.py   copied from the release (eval.py imports the scorer)

Because the layout matches the release, Experiments/ trains on it with `data.root=<kaggle_darts>` and
`data.folds_file=splits/darts_grouped_folds.json`: the NaN fill, fold stats and scoring are unchanged.
The hillshade/slope extras, the `inst` raster and `meta` are left out (training reads the COCO masks).
"""
import argparse
import json
import os
import shutil

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL = os.path.join(ROOT, "competition_release")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "DARTS_EDA", "dataset"))
    ap.add_argument("--out", default=os.path.join(ROOT, "DARTS_EDA", "kaggle_darts"))
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "train", "images")
    for d in (img_dir, os.path.join(args.out, "train", "annotations"),
              os.path.join(args.out, "metadata"), os.path.join(args.out, "tools")):
        os.makedirs(d, exist_ok=True)

    coco = json.load(open(os.path.join(args.src, "annotations", "instances_darts.json")))
    for im in coco["images"]:
        name = os.path.basename(im["file_name"])
        with np.load(os.path.join(args.src, name)) as z:
            image = z["image"].astype(np.float32)
        assert image.shape == (im["height"], im["width"], 8), (name, image.shape)
        np.savez_compressed(os.path.join(img_dir, name), image=image)
        im["file_name"] = "images/" + name
    json.dump(coco, open(os.path.join(args.out, "train", "annotations", "instances_train.json"), "w"))

    shutil.copy(os.path.join(REL, "metadata", "band_names.json"), os.path.join(args.out, "metadata"))
    for f in ("evaluate_coco.py", "coco_utils.py"):
        shutil.copy(os.path.join(REL, "tools", f), os.path.join(args.out, "tools"))
    size = sum(os.path.getsize(os.path.join(img_dir, f)) for f in os.listdir(img_dir))
    print(f"{len(coco['images'])} chips, {len(coco['annotations'])} instances, images {size / 1e9:.2f} GB -> {args.out}")


if __name__ == "__main__":
    main()
