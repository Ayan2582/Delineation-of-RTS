"""Write one fold of the grouped split to disk as train / val / test folders.

    python export_split.py                 # fold 0 -> ../data_split_grouped_fold0
    python export_split.py --fold 2 --out ../my_split

Layout (each split mirrors the release's train/ folder):
    <out>/train/images/*.npz   <out>/train/annotations/instances_train.json
    <out>/val/images/*.npz     <out>/val/annotations/instances_val.json
    <out>/test/images/*.npz    <out>/test/annotations/instances_test.json   (local held-out test)
    <out>/split_manifest.csv   one row per chip: split, ids, size, instances, overlap group
    <out>/split_info.json      counts and the source folds file

Image ids, file names and annotations are the release's own, so any chip can be traced back.
Train/val/test come from data.split_records / test_split_records, the same code training uses.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from config import load_config, resolve_path
from data import load_coco, split_records, test_split_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--fold", type=int, default=0, help="which grouped fold is the val split")
    ap.add_argument("--out", default=None, help="default: ../data_split_grouped_fold<k>")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["limit"] = None
    folds = json.loads(resolve_path(cfg["data"]["folds_file"]).read_text())
    out = resolve_path(args.out or f"../data_split_grouped_fold{args.fold}")
    if out.exists():
        raise SystemExit(f"{out} already exists; delete it or pass --out")

    train, val = split_records(cfg, args.fold)
    splits = {"train": train, "val": val, "test": test_split_records(cfg)}
    group_of = {i: g for g, ids in enumerate(folds["groups"]) for i in ids}
    coco = load_coco(cfg)

    rows, info = [], {"folds_file": cfg["data"]["folds_file"], "val_fold": args.fold, "splits": {}}
    for name, recs in splits.items():
        (out / name / "images").mkdir(parents=True)
        (out / name / "annotations").mkdir()
        ids = {r["id"] for r in recs}
        for r in recs:
            shutil.copy2(r["path"], out / name / "images" / r["path"].name)
            rows.append({"split": name, "public_id": r["path"].stem, "image_id": r["id"],
                         "file_name": f"{name}/images/{r['path'].name}", "width": r["width"],
                         "height": r["height"], "instances": len(r["anns"]), "group": group_of[r["id"]]})
        sub = {**coco, "info": {**coco.get("info", {}), "description": f"RTS {name} split, grouped fold {args.fold}"},
               "images": [im for im in coco["images"] if im["id"] in ids],
               "annotations": [a for a in coco["annotations"] if a["image_id"] in ids]}
        (out / name / "annotations" / f"instances_{name}.json").write_text(json.dumps(sub))
        info["splits"][name] = {"images": len(recs), "instances": len(sub["annotations"]),
                                "groups": len({group_of[i] for i in ids})}

    # no overlap group may appear in two splits
    seen = {}
    for r in rows:
        assert seen.setdefault(r["group"], r["split"]) == r["split"], f"group {r['group']} straddles splits"

    rows.sort(key=lambda r: (["train", "val", "test"].index(r["split"]), r["image_id"]))
    with (out / "split_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "split_info.json").write_text(json.dumps(info, indent=2))
    for name, s in info["splits"].items():
        print(f"{name:5s} {s['images']:4d} images  {s['instances']:4d} instances  {s['groups']:3d} groups")
    print("->", out)


if __name__ == "__main__":
    main()
