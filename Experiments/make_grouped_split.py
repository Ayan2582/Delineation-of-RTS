"""Leak-free split: overlapping chips are grouped, and a group never straddles two splits.

The release chips are overlapping windows cut from larger scenes (634/756 train chips share
pixels with another chip), so a random split puts the same ground in train and val. Here:

    1. overlap   two chips overlap if any 8x8 patch is pixel-identical in RGB or in
                 shaded_relief (min std filters out flat/no-data patches)
    2. groups    connected components of the overlap graph = one piece of terrain each
    3. assign    whole groups, largest first, to the bucket that is least full:
                 a held-out local TEST bucket plus N grouped CV folds over the rest

    python make_grouped_split.py                       # -> splits/grouped_folds.json
    python make_grouped_split.py --test_frac 0.15 --n_folds 5 --seed 0

The output keeps folds.json's format (`val_ids` per fold) and adds `test_ids`; data.py drops
test ids from every fold's train and val, so the local test chips are never trained on or
used for checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from config import load_config
from data import band_index, load_records

PATCH = 8


def patch_hashes(img: np.ndarray, min_std: float, w: np.ndarray):
    """Hash of every PATCH x PATCH window of a (C, H, W) image -> (hashes, ys, xs)."""
    p = sliding_window_view(img, (PATCH, PATCH), axis=(1, 2))      # C, h, w, P, P
    h = np.einsum("chwij,cij->hw", p, w)
    keep = np.isfinite(h) & (np.nan_to_num(p.std(axis=(0, 3, 4))) > min_std)
    ys, xs = np.nonzero(keep)
    return h[keep], ys, xs


def overlap_pairs(recs: list[dict], bands: list[int], min_std: float, seed: int) -> set[tuple[int, int]]:
    """Index pairs (i < j) of chips sharing at least one identical patch in `bands`."""
    w = np.random.default_rng(seed).random((len(bands), PATCH, PATCH))
    hs, cs = [], []
    for i, r in enumerate(recs):
        with np.load(r["path"]) as z:
            img = z["image"][..., bands].astype(np.float64).transpose(2, 0, 1)
        h, _, _ = patch_hashes(img, min_std, w)
        hs.append(h)
        cs.append(np.full(len(h), i))
    h, c = np.concatenate(hs), np.concatenate(cs)
    o = np.lexsort((c, h))
    h, c = h[o], c[o]
    same = np.flatnonzero((np.diff(h) == 0) & (np.diff(c) != 0))
    return {(int(c[j]), int(c[j + 1])) for j in same}


def connected_groups(n: int, pairs) -> list[list[int]]:
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in pairs:
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def assign(groups, n_inst, targets: dict[str, float], seed: int) -> dict[str, list[int]]:
    """Largest group first, into the bucket whose fill (chips and instances, relative to its
    target share) is lowest after adding it. Ties between equal-size groups are shuffled."""
    rng = np.random.default_rng(seed)
    order = sorted(range(len(groups)), key=lambda g: (-len(groups[g]), rng.random()))
    tot_c, tot_i = sum(len(g) for g in groups), sum(n_inst)
    fill = {b: [0, 0] for b in targets}
    out = {b: [] for b in targets}
    for g in order:
        gc, gi = len(groups[g]), sum(n_inst[i] for i in groups[g])
        b = min(targets, key=lambda b: ((fill[b][0] + gc) / (targets[b] * tot_c)
                                        + (fill[b][1] + gi) / (targets[b] * tot_i)))
        fill[b][0] += gc
        fill[b][1] += gi
        out[b] += groups[g]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="splits/grouped_folds.json")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    cfg["data"]["limit"] = None
    recs = load_records(cfg)
    idx = band_index(cfg)
    rgb = overlap_pairs(recs, [idx["red"], idx["green"], idx["blue"]], min_std=2.0, seed=args.seed)
    dem = overlap_pairs(recs, [idx["shaded_relief"]], min_std=0.5, seed=args.seed)
    pairs = rgb | dem
    groups = connected_groups(len(recs), pairs)

    n_inst = [len(r["anns"]) for r in recs]
    fold_share = (1 - args.test_frac) / args.n_folds
    targets = {"test": args.test_frac, **{f"fold{k}": fold_share for k in range(args.n_folds)}}
    buckets = assign(groups, n_inst, targets, args.seed)

    # every overlapping pair must land in one bucket
    where = {i: b for b, members in buckets.items() for i in members}
    crossing = [(a, b) for a, b in pairs if where[a] != where[b]]
    assert not crossing, f"{len(crossing)} overlapping pairs straddle two splits"

    ids = lambda b: sorted(recs[i]["id"] for i in buckets[b])
    out = {"n_folds": args.n_folds, "seed": args.seed, "method": "grouped by pixel overlap",
           "test_frac": args.test_frac, "test_ids": ids("test"),
           "val_ids": [ids(f"fold{k}") for k in range(args.n_folds)],
           "groups": sorted((sorted(recs[i]["id"] for i in g) for g in groups), key=lambda g: (-len(g), g[0]))}
    path = Path(__file__).resolve().parent / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))

    sizes = sorted((len(g) for g in groups), reverse=True)
    print(f"overlap pairs: {len(pairs)} (rgb {len(rgb)}, shaded_relief {len(dem)}) | "
          f"groups: {len(groups)}, largest {sizes[:5]}, singletons {sizes.count(1)}")
    for b, members in buckets.items():
        print(f"{b:6s} chips {len(members):4d} ({100 * len(members) / len(recs):4.1f}%)  "
              f"instances {sum(n_inst[i] for i in members):4d}  groups {sum(1 for g in groups if where[g[0]] == b)}")
    print("overlapping pairs crossing splits: 0 ->", path)


if __name__ == "__main__":
    main()
