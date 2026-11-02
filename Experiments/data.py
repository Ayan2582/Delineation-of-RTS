"""Data for E1-E6: release loading, fixed folds, per-fold stats, NaN-fill cache, augmentation.

Preprocessing itself is `rts_preprocess.py`, used unchanged:
    raw tile --fill_nan_neighbor_mean--> filled --normalize(input_norm(train stats))--> RTSDataset
    --transform (channel select + augment)--> pad_collate
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils

from config import ALL_FOLD, resolve_path, resolve_streams
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


def _split_file(cfg: dict) -> dict:
    """The grouped split (val_ids per fold + test_ids), built once by make_grouped_split.py."""
    path = resolve_path(cfg["data"]["folds_file"])
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; build it with make_grouped_split.py")
    return json.loads(path.read_text())


def get_folds(cfg: dict, recs: list[dict]) -> list[list[int]]:
    """Val image ids per fold, restricted to `recs` (a debug `limit` keeps the saved assignment)."""
    keep = {r["id"] for r in recs}
    return [[i for i in f if i in keep] for f in _split_file(cfg)["val_ids"]]


def get_test_ids(cfg: dict) -> set[int]:
    """Held-out local test ids: never trained on, never used to pick a checkpoint."""
    return set(_split_file(cfg).get("test_ids", []))


def split_records(cfg: dict, fold: int | str) -> tuple[list[dict], list[dict]]:
    """(train, val) for one fold. Local test chips are in neither.
    fold "all": every labelled chip trains, the local test chips included, and there is no val."""
    recs = load_records(cfg)
    if fold == ALL_FOLD:
        return recs, []
    val_ids = set(get_folds(cfg, recs)[fold])
    test_ids = get_test_ids(cfg)
    recs = [r for r in recs if r["id"] not in test_ids]
    return [r for r in recs if r["id"] not in val_ids], [r for r in recs if r["id"] in val_ids]


def test_split_records(cfg: dict) -> list[dict]:
    test_ids = get_test_ids(cfg)
    if not test_ids:
        raise ValueError(f"{cfg['data']['folds_file']} has no test_ids; build it with make_grouped_split.py")
    return [r for r in load_records(cfg) if r["id"] in test_ids]


def test_records(root: Path) -> list[dict]:
    """The competition's unlabelled test chips, from metadata/test_manifest.csv (official ids)."""
    with (root / "metadata/test_manifest.csv").open() as f:
        return [{"id": int(r["image_id"]), "path": root / r["file_name"],
                 "height": int(r["height"]), "width": int(r["width"])} for r in csv.DictReader(f)]


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
        out[r["id"]] = fill_nan_neighbor_mean(load_raw(r), fallback=np.full(8, np.nan))
    if not cfg["data"].get("limit"):
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{str(k): v for k, v in out.items()})
    return out


def fold_stats(cfg: dict, fold: int, train_recs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std from this fold's TRAIN tiles only (raw, NaNs ignored)."""
    path = stats_path(cfg, fold)
    if path.exists():
        s = json.loads(path.read_text())
        return np.array(s["mean"], np.float32), np.array(s["std"], np.float32)
    mean, std = compute_channel_stats([load_raw(r) for r in train_recs])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist(),
                                "n_train": len(train_recs)}))
    return mean, std


RGB = ("red", "green", "blue")


def input_norm(cfg: dict, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The (mean, std) passed to `normalize`, per band, from the fold's raw train stats.

    data.swin_norm = zscore: (x - mean) / std for every band.
    data.swin_norm = satlas: the Swin bands get the input Swin_V2_B_Weights.NAIP_RGB_SI_SATLAS was
        pretrained on, RGB / 255 (the release's RGB is 0-255). Other Swin bands (NIR, indices, DEM)
        are z-scored and then given the RGB bands' mean/std in that /255 space, so they sit in the
        same range. Terrain bands (trained from scratch) stay z-scored.
        Padding (0) is then black for the RGB bands, not their mean.
        data.rgb_ref = geoai (DARTS pretraining): the RGB is not 8-bit (Sentinel-2 reflectance 0-10000), so
        each RGB band is first z-scored with its own train stats and given the GeoAI RGB mean/std
        (data.geoai_rgb), then /255. The Swin weights see the same input range in pretraining and
        fine-tuning. With GeoAI's own stats as the reference this is exactly RGB / 255.
    """
    mode = cfg["data"].get("swin_norm", "zscore")
    if mode == "zscore":
        return mean, std
    if mode != "satlas":
        raise ValueError(f"data.swin_norm must be 'zscore' or 'satlas', got {mode!r}")
    idx = band_index(cfg)
    rgb = [idx[b] for b in RGB]
    ref = cfg["data"].get("rgb_ref")
    if ref is None:
        r_mean, r_std = mean[rgb].astype(np.float64), std[rgb].astype(np.float64)
    elif ref == "geoai":
        r_mean, r_std = np.array(cfg["data"]["geoai_rgb"]["mean"]), np.array(cfg["data"]["geoai_rgb"]["std"])
    else:
        raise ValueError(f"data.rgb_ref must be null or 'geoai', got {ref!r}")
    m_rgb, s_rgb = float(r_mean.mean()) / 255, float(r_std.mean()) / 255
    mean_n, std_n = mean.copy(), std.copy()
    for b in resolve_streams(cfg)["swin"]:
        i = idx[b]
        if b in RGB:
            k = RGB.index(b)       # ((x - mean) / std * r_std + r_mean) / 255
            mean_n[i] = mean[i] - r_mean[k] * std[i] / r_std[k]
            std_n[i] = 255.0 * std[i] / r_std[k]
        else:
            mean_n[i], std_n[i] = mean[i] - std[i] * m_rgb / s_rgb, std[i] / s_rgb
    return mean_n, std_n


def jitter_ref(cfg: dict, mean: np.ndarray, std: np.ndarray, mean_n: np.ndarray, std_n: np.ndarray,
               positions: list[int] | None = None):
    """Train mean and std of each jittered band (or of `positions`) in normalised space, as (n, 1, 1) tensors.

    (0, 1) under z-score. Brightness/contrast use them, so the jitter strength stays in
    train-std units about the train mean whatever `data.swin_norm` is; band dropout fills with the mean.
    """
    names = [*resolve_streams(cfg)["swin"], *resolve_streams(cfg)["terrain"]]
    idx = band_index(cfg)
    b = [idx[names[p]] for p in (jitter_positions(cfg) if positions is None else positions)]
    mu = (mean[b] - mean_n[b]) / std_n[b]
    s = std[b] / std_n[b]
    return torch.as_tensor(mu, dtype=torch.float32)[:, None, None], torch.as_tensor(s, dtype=torch.float32)[:, None, None]


def fold_items(recs, filled, mean):
    """(filled image, masks) for RTSDataset. Leftover NaNs get the train-mean fallback."""
    items = []
    for r in recs:
        img = filled[r["id"]].copy()
        nan = np.isnan(img)
        if nan.any():
            img[nan] = np.broadcast_to(mean[:, None, None], img.shape)[nan]
        items.append((img, decode_masks(r)))
    return items


# ----------------------------------------------------------------------------- transforms


class InstanceBank:
    """Every train instance cut to its bbox: (selected-channel image crop, bool mask crop)."""

    def __init__(self, cache, channels: list[int]):
        self.items = []
        for img, masks in cache:
            for m in masks:
                ys, xs = np.nonzero(m)
                if len(ys) == 0:
                    continue
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                self.items.append((torch.from_numpy(img[channels, y0:y1, x0:x1].copy()),
                                   torch.from_numpy(m[y0:y1, x0:x1] > 0)))

    def sample(self):
        return self.items[int(torch.randint(0, len(self.items), ()))]


class Augment:
    """Train-time augmentation on normalised (C, H, W) tensors and (N, H, W) masks.

    Geometric ops apply to every channel and to the masks together. Brightness/contrast
    apply only to `jitter_idx`, the positions of the optical bands in the selected channels.
    Copy-paste needs `bank` (set by build_loaders) and may add instances.
    `ref` is (mean, std) of the jittered bands in normalised space (`jitter_ref`); None = (0, 1).
    Band dropout replaces one of `drop_idx` (positions of aug.band_drop_bands) by its train mean
    `drop_val` ((n, 1, 1), normalised space; None = 0). It runs before pad_collate, so padding stays 0.
    """

    def __init__(self, aug: dict, jitter_idx: list[int], bank: InstanceBank | None = None, ref=None,
                 drop_idx: list[int] | None = None, drop_val=None):
        self.a, self.jitter_idx, self.bank = aug, jitter_idx, bank
        self.ref = ref if ref is not None else (0.0, 1.0)
        self.drop_idx = drop_idx or []
        self.drop_val = drop_val if drop_val is not None else torch.zeros(len(self.drop_idx), 1, 1)

    def copy_paste(self, img: torch.Tensor, masks: torch.Tensor):
        """Paste 1..copy_paste_max random train instances (all channels, mask pixels only).

        Pasted pixels occlude whatever is below; an instance left with < 30% of its area is dropped.
        """
        H, W = img.shape[-2:]
        img, masks = img.clone(), masks.clone()            # the cache must stay untouched
        area = masks.flatten(1).sum(1)
        for _ in range(int(torch.randint(1, self.a["copy_paste_max"] + 1, ()))):
            crop, m = self.bank.sample()
            k = int(torch.randint(0, 4, ()))
            crop, m = torch.rot90(crop, k, (-2, -1)), torch.rot90(m, k, (-2, -1))
            h, w = m.shape
            if h > H or w > W:
                continue
            y = int(torch.randint(0, H - h + 1, ()))
            x = int(torch.randint(0, W - w + 1, ()))
            img[:, y:y + h, x:x + w] = torch.where(m, crop, img[:, y:y + h, x:x + w])
            full = torch.zeros(1, H, W, dtype=masks.dtype)
            full[0, y:y + h, x:x + w] = m.to(masks.dtype)
            masks = torch.cat([masks * (1 - full), full])
            area = torch.cat([area, full.flatten(1).sum(1)])
        keep = masks.flatten(1).sum(1) >= 0.3 * area
        return img, masks[keep]

    def __call__(self, img: torch.Tensor, masks: torch.Tensor):
        a = self.a
        if self.bank is not None and torch.rand(()) < a.get("copy_paste_p", 0.0):
            img, masks = self.copy_paste(img, masks)
        if a.get("scale_jitter"):
            lo, hi = a["scale_jitter"]
            s = float(lo + torch.rand(()) * (hi - lo))
            size = (max(1, round(img.shape[-2] * s)), max(1, round(img.shape[-1] * s)))
            img = F.interpolate(img[None], size=size, mode="bilinear", align_corners=False)[0]
            masks = F.interpolate(masks[None].float(), size=size, mode="nearest")[0].to(torch.uint8)
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
            mu, s = self.ref                                       # (0, 1) under z-score
            img[self.jitter_idx] = (img[self.jitter_idx] - mu) * c + mu + b * s   # about the train mean
        if self.drop_idx and torch.rand(()) < a.get("band_drop_p", 0.0):
            img = img.clone()
            k = int(torch.randint(0, len(self.drop_idx), ()))
            img[self.drop_idx[k]] = self.drop_val[k]
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


def drop_positions(cfg: dict) -> list[int]:
    """Positions of aug.band_drop_bands among the selected channels (bands the experiment lacks are skipped)."""
    names = [*resolve_streams(cfg)["swin"], *resolve_streams(cfg)["terrain"]]
    return [i for i, n in enumerate(names) if n in set(cfg["aug"].get("band_drop_bands") or [])]


# ----------------------------------------------------------------------------- loaders


def build_loaders(cfg: dict, fold: int | str, seed: int):
    """Train/val DataLoaders for one fold, plus the val records (ids, sizes) for scoring.
    fold "all" has no val: the val loader is None and the records are []."""
    train_recs, val_recs = split_records(cfg, fold)
    filled = filled_tiles(cfg, train_recs + val_recs)
    mean, std = fold_stats(cfg, fold, train_recs)
    s = stream_indices(cfg)
    channels = s["swin"] + s["terrain"]
    norm = input_norm(cfg, mean, std)                     # fold_items still fills NaNs with the raw mean

    drop = drop_positions(cfg)
    aug = Augment(cfg["aug"], jitter_positions(cfg), ref=jitter_ref(cfg, mean, std, *norm),
                  drop_idx=drop, drop_val=jitter_ref(cfg, mean, std, *norm, positions=drop)[0])
    train_ds = RTSDataset(fold_items(train_recs, filled, mean), *norm,
                               transform=SelectThenAugment(channels, aug))
    if cfg["aug"].get("copy_paste_p"):
        aug.bank = InstanceBank(train_ds.cache, channels)   # train fold only, so nothing leaks from val
    val_ds = RTSDataset(fold_items(val_recs, filled, mean), *norm,
                        transform=SelectThenAugment(channels, None))

    g = torch.Generator().manual_seed(seed)
    nw = cfg["data"]["num_workers"]
    common = dict(batch_size=cfg["train"]["batch_size"], num_workers=nw,
                  collate_fn=pad_collate, persistent_workers=nw > 0)
    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True, drop_last=True, generator=g, **common)
    val_dl = torch.utils.data.DataLoader(val_ds, shuffle=False, **common) if val_recs else None
    return train_dl, val_dl, val_recs


def build_test_loader(cfg: dict, fold: int | str, split: str = "test"):
    """Unaugmented, unshuffled DataLoader over the local test chips (or, with split="train",
    this fold's train chips), normalised with the fold's train stats, plus its records."""
    if fold == ALL_FOLD and split != "train":
        raise ValueError("fold 'all' trained on the local test chips; only split='train' can be scored")
    train_recs, _ = split_records(cfg, fold)
    recs = train_recs if split == "train" else test_split_records(cfg)
    filled = filled_tiles(cfg, recs)
    mean, std = fold_stats(cfg, fold, train_recs)
    s = stream_indices(cfg)
    ds = RTSDataset(fold_items(recs, filled, mean), *input_norm(cfg, mean, std),
                    transform=SelectThenAugment(s["swin"] + s["terrain"], None))
    dl = torch.utils.data.DataLoader(ds, shuffle=False, batch_size=cfg["train"]["batch_size"],
                                     num_workers=cfg["data"]["num_workers"], collate_fn=pad_collate)
    return dl, recs


def build_monitor_loader(cfg: dict):
    """GeoAI chips scored during DARTS pretraining, as a second val set (cfg["monitor"] set).

    `monitor` gives the GeoAI data.root / cache_dir / folds_file. The chips are every labelled GeoAI chip
    outside its test_ids, normalised with their own stats like a fine-tuning run (data.rgb_ref off).
    They only report: the checkpoint is still picked on the DARTS val fold, and the local test chips
    are never touched. Returns (loader, records, the GeoAI cfg used for scoring)."""
    import copy
    g = copy.deepcopy(cfg)
    g["data"].update(cfg["monitor"], rgb_ref=None, limit=cfg["data"].get("limit"))
    test_ids = get_test_ids(g)
    recs = [r for r in load_records(g) if r["id"] not in test_ids]
    filled = filled_tiles(g, recs)
    mean, std = fold_stats(g, "trainval", recs)
    s = stream_indices(g)
    ds = RTSDataset(fold_items(recs, filled, mean), *input_norm(g, mean, std),
                    transform=SelectThenAugment(s["swin"] + s["terrain"], None))
    dl = torch.utils.data.DataLoader(ds, shuffle=False, batch_size=g["train"]["batch_size"],
                                     num_workers=g["data"]["num_workers"], collate_fn=pad_collate)
    return dl, recs, g


def stats_path(cfg: dict, fold: int) -> Path:
    """Where `fold_stats` caches this fold's stats (train.py copies it into the run folder)."""
    tag = f"_limit{cfg['data']['limit']}" if cfg["data"].get("limit") else ""
    split = Path(cfg["data"]["folds_file"]).stem             # stats differ per split file
    return resolve_path(cfg["data"]["cache_dir"]) / f"stats_{split}_fold{fold}{tag}.json"
