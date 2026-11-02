"""Leak-free split of the DARTS chips, same format as Experiments/splits/grouped_folds.json.

DARTS chips have real coordinates, so chips are grouped by ground footprint instead of pixel hashes:

    1. footprint  each chip's box (x0, y1, width, height at 4 m) moved to EPSG:3413, so chips from
                  different UTM zones, PlanetScope images and years are comparable
    2. groups     two chips are linked if their footprints, grown by --buffer metres, touch.
                  buffer 0 catches the same ground (and the same slump seen in another year);
                  a buffer > 0 also keeps neighbouring, spatially correlated terrain together
    3. assign     whole groups, largest first, to the least full bucket (test + N folds),
                  balancing chips and instances, as in make_grouped_split.py

    python DARTS_EDA/make_darts_split.py                 # -> Experiments/splits/darts_grouped_folds.json
    python DARTS_EDA/make_darts_split.py --buffer 1000 --test_frac 0.15 --n_folds 5

Ids are the COCO image ids of annotations/instances_darts.json (chip names are stored too).
"""
import argparse
import json
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Experiments"))
from make_grouped_split import assign, connected_groups  # noqa: E402

GSD = 4.0


def footprints(man: pd.DataFrame) -> gpd.GeoSeries:
    parts = []
    for crs, g in man.groupby("crs"):
        geom = [box(r.x0, r.y1 - r.height * GSD, r.x0 + r.width * GSD, r.y1) for r in g.itertuples()]
        parts.append(gpd.GeoSeries(geom, index=g.index, crs=int(crs)).to_crs(3413))
    return pd.concat(parts).sort_index()


def touching_pairs(fp: gpd.GeoSeries, buffer: float) -> set[tuple[int, int]]:
    grown = gpd.GeoDataFrame(geometry=fp.buffer(buffer / 2) if buffer else fp)   # both grow by half
    j = gpd.sjoin(grown, grown, predicate="intersects")
    return {(min(a, b), max(a, b)) for a, b in zip(j.index, j.index_right) if a != b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "DARTS_EDA", "dataset"))
    ap.add_argument("--buffer", type=float, default=500.0, help="metres between chips of different groups")
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "Experiments", "splits", "darts_grouped_folds.json"))
    args = ap.parse_args()

    coco = json.load(open(os.path.join(args.dir, "annotations", "instances_darts.json")))
    img_id = {im["file_name"][:-4]: im["id"] for im in coco["images"]}
    n_ann = pd.Series([a["image_id"] for a in coco["annotations"]]).value_counts()
    man = pd.read_csv(os.path.join(args.dir, "manifest.csv"))
    man = man[man.chip.isin(img_id)].reset_index(drop=True)
    man["id"] = man.chip.map(img_id)

    fp = footprints(man)
    pairs = touching_pairs(fp, args.buffer)
    groups = connected_groups(len(man), pairs)
    n_inst = [int(n_ann.get(i, 0)) for i in man.id]
    share = (1 - args.test_frac) / args.n_folds
    targets = {"test": args.test_frac, **{f"fold{k}": share for k in range(args.n_folds)}}
    buckets = assign(groups, n_inst, targets, args.seed)

    where = {i: b for b, members in buckets.items() for i in members}
    # leak checks: no shared ground across buckets, and every cross-bucket pair is >= buffer apart
    same_ground = touching_pairs(fp, 0.0)
    assert not [p for p in same_ground if where[p[0]] != where[p[1]]], "overlapping chips straddle two splits"
    assert not [p for p in pairs if where[p[0]] != where[p[1]]], "chips closer than --buffer straddle two splits"

    ids = lambda b: sorted(int(man.id[i]) for i in buckets[b])
    chips = lambda b: sorted(man.chip[i] for i in buckets[b])
    out = {"n_folds": args.n_folds, "seed": args.seed, "test_frac": args.test_frac, "buffer_m": args.buffer,
           "method": f"grouped by ground footprint (EPSG:3413), chips within {args.buffer:g} m share a group",
           "coco": "DARTS_EDA/dataset/annotations/instances_darts.json",
           "test_ids": ids("test"), "val_ids": [ids(f"fold{k}") for k in range(args.n_folds)],
           "test_chips": chips("test"), "val_chips": [chips(f"fold{k}") for k in range(args.n_folds)],
           "groups": sorted((sorted(int(man.id[i]) for i in g) for g in groups), key=lambda g: (-len(g), g[0]))}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"))

    sizes = sorted((len(g) for g in groups), reverse=True)
    print(f"buffer {args.buffer:g} m | same-ground pairs {len(same_ground)} | linked pairs {len(pairs)} | "
          f"groups {len(groups)}, largest {sizes[:5]}, singletons {sizes.count(1)}")
    rows = []
    for b, members in buckets.items():
        m = man.loc[members]
        rows.append({"bucket": b, "chips": len(m), "% chips": 100 * len(m) / len(man),
                     "instances": sum(n_inst[i] for i in members),
                     "groups": sum(1 for g in groups if where[g[0]] == b), "regions": m.region.nunique()})
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.1f"))
    reg = pd.crosstab(man.region, man.index.map(where))[list(buckets)]
    print("\nchips per region and bucket\n" + reg.to_string())
    print("\n0 overlapping or near pairs cross splits ->", args.out)


if __name__ == "__main__":
    main()
