"""Data for E1-E4: release loading, fixed folds, per-fold stats, NaN-fill cache, augmentation.

Preprocessing itself is `rts_preprocess.py`, used unchanged:
    raw tile --fill_nan_neighbor_mean--> filled --normalize(train stats)--> RTSDataset
    --transform (channel select + augment)--> pad_collate
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils
from sklearn.model_selection import KFold

from config import resolve_path, resolve_streams
from rts_preprocess import RTSDataset, compute_channel_stats, fill_nan_neighbor_mean, pad_collate

# ----------------------------------------------------------------------------- release


def release_root(cfg: dict) -> Path:
    return resolve_path(cfg["data"]["root"])


def band_index(cfg: dict) -> dict[str, int]:
    """Band name -> channel index, from the release's own metadata (never hard-coded)."""
    meta = json.loads((release_root(cfg) / "metadata/band_names.json").read_text())
    return {b["name"]: b["index"] for b in meta["bands"]}


def stream_indices(cfg: dict) -> dict[str, list[int]]:
    """Channel indices per stream. The model sees swin channels first, then terrain."""
    idx = band_index(cfg)
    return {k: [idx[b] for b in names] for k, names in resolve_streams(cfg).items()}


def load_coco(cfg: dict) -> dict:
    return json.loads((release_root(cfg) / "train/annotations/instances_train.json").read_text())


def load_records(cfg: dict) -> list[dict]:
    """One record per training image, sorted by id: id, path, height, width, anns."""
    coco = load_coco(cfg)
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    root = release_root(cfg)
    recs = [{"id": im["id"],
             # the JSON says images/x.npz, the manifest says train/images/x.npz: join on basename
             "path": root / "train/images" / Path(im["file_name"]).name,
             "height": im["height"], "width": im["width"],
             "anns": anns[im["id"]]} for im in coco["images"]]
    recs.sort(key=lambda r: r["id"])
    if cfg["data"].get("limit"):
        recs = recs[: cfg["data"]["limit"]]
    return recs


def load_raw(rec: dict) -> np.ndarray:
    """(C, H, W) float32 with NaNs. The npz stores HWC."""
    with np.load(rec["path"]) as z:
        return np.ascontiguousarray(z["image"].transpose(2, 0, 1), dtype=np.float32)


def decode_masks(rec: dict) -> np.ndarray:
    """(N, H, W) uint8 instance masks from the COCO RLEs."""
    h, w = rec["height"], rec["width"]
    if not rec["anns"]:
        return np.zeros((0, h, w), np.uint8)
    return np.stack([mask_utils.decode(a["segmentation"]) for a in rec["anns"]]).astype(np.uint8)


# ----------------------------------------------------------------------------- folds


def get_folds(cfg: dict, recs: list[dict]) -> list[list[int]]:
    """Val image ids per fold. Created once and saved, then reused by every experiment."""
    path = resolve_path(cfg["data"]["folds_file"])
    ids = [r["id"] for r in recs]
    if path.exists():
        folds = json.loads(path.read_text())["val_ids"]
        if cfg["data"].get("limit"):                 # debug subset: keep the saved assignment
            keep = set(ids)
            folds = [[i for i in f if i in keep] for f in folds]
        return folds
    if cfg["data"].get("limit"):
        raise RuntimeError(f"{path} missing; create folds on the full dataset first")
    kf = KFold(cfg["data"]["n_folds"], shuffle=True, random_state=cfg["data"]["fold_seed"])
    folds = [sorted(int(ids[i]) for i in val) for _, val in kf.split(ids)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n_folds": len(folds), "seed": cfg["data"]["fold_seed"],
                                "val_ids": folds}))
    return folds


def get_test_ids(cfg: dict) -> set[int]:
    """Held-out local test ids (grouped split only; the old random folds.json has none)."""
    path = resolve_path(cfg["data"]["folds_file"])
    return set(json.loads(path.read_text()).get("test_ids", [])) if path.exists() else set()


def split_records(cfg: dict, fold: int) -> tuple[list[dict], list[dict]]:
    """(train, val) for one fold. Local test chips are in neither."""
    recs = load_records(cfg)
    val_ids = set(get_folds(cfg, recs)[fold])
    test_ids = get_test_ids(cfg)
    recs = [r for r in recs if r["id"] not in test_ids]
    return [r for r in recs if r["id"] not in val_ids], [r for r in recs if r["id"] in val_ids]


def test_split_records(cfg: dict) -> list[dict]:
    test_ids = get_test_ids(cfg)
    if not test_ids:
        raise ValueError(f"{cfg['data']['folds_file']} has no test_ids; build it with make_grouped_split.py")
    return [r for r in load_records(cfg) if r["id"] in test_ids]


# ----------------------------------------------------------------------------- NaN fill + stats cache


def filled_tiles(cfg: dict, recs: list[dict]) -> dict[int, np.ndarray]:
    """NaN-filled tiles, computed once for the whole release and cached to disk.

    `fill_nan_neighbor_mean` is run with fallback=NaN, so pixels it cannot reach from a valid
    neighbour (all-NaN bands, giant voids) stay NaN in the cache. `fold_items` then sets them to
    that fold's train mean. That is exactly `fallback=train_mean`, without one cache per fold.
    """
    cache = resolve_path(cfg["data"]["cache_dir"]) / "filled.npz"
    if cache.exists():
        with np.load(cache) as z:
            have = {int(k): z[k] for k in z.files}
        if all(r["id"] in have for r in recs):
            return have
    out = {}
    # fill every release chip, not just `recs`: callers pass subsets (train+val, local test)
    for r in (recs if cfg["data"].get("limit") else load_records(cfg)):
        out[r["id"]], _ = fill_nan_neighbor_mean(load_raw(r), fallback=np.full(8, np.nan))
    if not cfg["data"].get("limit"):
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{str(k): v for k, v in out.items()})
    return out


def fold_stats(cfg: dict, fold: int, train_recs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std from this fold's TRAIN tiles only (raw, NaNs ignored)."""
    tag = f"_limit{cfg['data']['limit']}" if cfg["data"].get("limit") else ""
    split = Path(cfg["data"]["folds_file"]).stem             # stats differ per split file
    split = "" if split == "folds" else f"{split}_"          # keep old runs' stats_fold{k}.json names
    path = resolve_path(cfg["data"]["cache_dir"]) / f"stats_{split}fold{fold}{tag}.json"
    if path.exists():
        s = json.loads(path.read_text())
        return np.array(s["mean"], np.float32), np.array(s["std"], np.float32)
    mean, std = compute_channel_stats([load_raw(r) for r in train_recs])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist(),
                                "n_train": len(train_recs)}))
    return mean, std


def fold_items(recs, filled, mean):
    """(filled image, masks, labels) for RTSDataset. Leftover NaNs get the train-mean fallback."""
    items = []
    for r in recs:
        img = filled[r["id"]].copy()
        nan = np.isnan(img)
        if nan.any():
            img[nan] = np.broadcast_to(mean[:, None, None], img.shape)[nan]
        masks = decode_masks(r)
        items.append((img, masks, np.ones(len(masks), np.int64)))
    return items


# ----------------------------------------------------------------------------- transforms


class Augment:
    """Train-time augmentation on normalised (C, H, W) tensors and (N, H, W) masks.

    Geometric ops apply to every channel and to the masks together. Brightness/contrast
    apply only to `jitter_idx`, the positions of the optical bands in the selected channels.
    """

    def __init__(self, aug: dict, jitter_idx: list[int]):
        self.a, self.jitter_idx = aug, jitter_idx

    def __call__(self, img: torch.Tensor, masks: torch.Tensor):
        a = self.a
        crop = a.get("crop")
        H, W = img.shape[-2:]
        if crop and (H > crop or W > crop):
            ch, cw = min(H, crop), min(W, crop)
            y = int(torch.randint(0, H - ch + 1, ()))
            x = int(torch.randint(0, W - cw + 1, ()))
            img, masks = img[:, y:y + ch, x:x + cw], masks[:, y:y + ch, x:x + cw]
        if torch.rand(()) < a["hflip"]:
            img, masks = img.flip(-1), masks.flip(-1)
        if torch.rand(()) < a["vflip"]:
            img, masks = img.flip(-2), masks.flip(-2)
        if a["rot90"]:
            k = int(torch.randint(0, 4, ()))
            img, masks = torch.rot90(img, k, (-2, -1)), torch.rot90(masks, k, (-2, -1))
        if self.jitter_idx and torch.rand(()) < a["jitter_p"]:
            img = img.clone()
            c = 1 + (torch.rand(()) * 2 - 1) * a["contrast"]
            b = (torch.rand(()) * 2 - 1) * a["brightness"]
            img[self.jitter_idx] = img[self.jitter_idx] * c + b   # about 0 == the train mean
        return img.contiguous(), masks.contiguous()


class SelectThenAugment:
    """Pick the experiment's channels (swin first, then terrain), then optionally augment."""

    def __init__(self, channels: list[int], augment: Augment | None):
        self.channels, self.augment = torch.as_tensor(channels), augment

    def __call__(self, img, masks):
        img = img[self.channels]
        return self.augment(img, masks) if self.augment else (img, masks)


def jitter_positions(cfg: dict) -> list[int]:
    names = [*resolve_streams(cfg)["swin"], *resolve_streams(cfg)["terrain"]]
    return [i for i, n in enumerate(names) if n in set(cfg["aug"]["jitter_bands"])]


# ----------------------------------------------------------------------------- loaders


def build_loaders(cfg: dict, fold: int, seed: int):
    """Train/val DataLoaders for one fold, plus the val records (ids, sizes) for scoring."""
    train_recs, val_recs = split_records(cfg, fold)
    filled = filled_tiles(cfg, train_recs + val_recs)
    mean, std = fold_stats(cfg, fold, train_recs)
    s = stream_indices(cfg)
    channels = s["swin"] + s["terrain"]

    train_ds = RTSDataset(fold_items(train_recs, filled, mean), mean, std,
                          transform=SelectThenAugment(channels, Augment(cfg["aug"], jitter_positions(cfg))))
    val_ds = RTSDataset(fold_items(val_recs, filled, mean), mean, std,
                        transform=SelectThenAugment(channels, None))

    g = torch.Generator().manual_seed(seed)
    nw = cfg["data"]["num_workers"]
    common = dict(batch_size=cfg["train"]["batch_size"], num_workers=nw,
                  collate_fn=pad_collate, persistent_workers=nw > 0)
    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True, drop_last=True, generator=g, **common)
    val_dl = torch.utils.data.DataLoader(val_ds, shuffle=False, **common)
    return train_dl, val_dl, val_recs


def build_test_loader(cfg: dict, fold: int):
    """Local test DataLoader, normalised with this fold's train stats, plus its records."""
    train_recs, _ = split_records(cfg, fold)
    test_recs = test_split_records(cfg)
    filled = filled_tiles(cfg, test_recs)
    mean, std = fold_stats(cfg, fold, train_recs)
    s = stream_indices(cfg)
    ds = RTSDataset(fold_items(test_recs, filled, mean), mean, std,
                    transform=SelectThenAugment(s["swin"] + s["terrain"], None))
    dl = torch.utils.data.DataLoader(ds, shuffle=False, batch_size=cfg["train"]["batch_size"],
                                     num_workers=cfg["data"]["num_workers"], collate_fn=pad_collate)
    return dl, test_recs
