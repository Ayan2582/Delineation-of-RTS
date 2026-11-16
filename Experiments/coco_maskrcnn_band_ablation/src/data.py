from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import convolve
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as mask_utils


NEIGHBOUR_KERNEL = np.array(
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float64
)


def read_band_indices(data_root: Path) -> dict[str, int]:
    """Read the official band order from metadata; never assume channel positions."""
    metadata = json.loads((data_root / "metadata" / "band_names.json").read_text())
    return {item["name"]: int(item["index"]) for item in metadata["bands"]}


def load_records(data_root: Path) -> tuple[list[dict], dict]:
    """Read labelled COCO image/annotation metadata from the competition release."""
    annotation_path = data_root / "train" / "annotations" / "instances_train.json"
    coco = json.loads(annotation_path.read_text())
    annotations_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    records = []
    for image in coco["images"]:
        image_id = int(image["id"])
        filename = Path(image["file_name"]).name
        records.append({
            "id": image_id,
            "path": data_root / "train" / "images" / filename,
            "height": int(image["height"]),
            "width": int(image["width"]),
            "anns": annotations_by_image.get(image_id, []),
        })
    records.sort(key=lambda x: x["id"])
    return records, coco


def load_grouped_split(
    records: list[dict], split_path: Path, fold: int
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Use the project's existing grouped split unchanged.

    The split file stores validation IDs per fold and one fixed local-test set.
    Training IDs are the remaining labelled IDs, matching the existing project loader.
    """
    split = json.loads(split_path.read_text())
    if fold < 0 or fold >= len(split["val_ids"]):
        raise ValueError(f"fold={fold} is invalid; split contains {len(split['val_ids'])} folds")

    all_by_id = {int(r["id"]): r for r in records}
    all_ids = set(all_by_id)
    test_ids = set(map(int, split["test_ids"]))
    val_ids = set(map(int, split["val_ids"][fold]))

    missing = (test_ids | val_ids) - all_ids
    if missing:
        raise ValueError(f"Split file contains IDs absent from annotations: {sorted(missing)[:10]}")
    overlap = test_ids & val_ids
    if overlap:
        raise ValueError(f"Local test and fold validation overlap: {sorted(overlap)[:10]}")

    # The JSON does not store a separate train_ids list. This is the same
    # complement rule used by the existing RTS project loader.
    train_ids = all_ids - test_ids - val_ids

    # Meaningful spatial-group audit: each connected component in `groups`
    # must belong wholly to train, this fold's validation set, or local test.
    # Unlike checking train_ids against val_ids after a set subtraction, this
    # checks the saved pixel-overlap components independently.
    groups = split.get("groups", [])
    seen_group_ids: set[int] = set()
    duplicate_group_ids: set[int] = set()
    crossing_groups = []
    for group_index, group in enumerate(groups):
        members = set(map(int, group))
        duplicate_group_ids |= (seen_group_ids & members)
        seen_group_ids |= members
        assignments = set()
        for image_id in members:
            if image_id not in all_ids:
                continue
            if image_id in test_ids:
                assignments.add("local_test")
            elif image_id in val_ids:
                assignments.add("validation")
            else:
                assignments.add("training")
        if len(assignments) > 1:
            crossing_groups.append(group_index)

    if duplicate_group_ids:
        raise ValueError(
            f"Pixel-overlap groups repeat image IDs: {sorted(duplicate_group_ids)[:10]}"
        )
    missing_group_ids = all_ids - seen_group_ids
    unknown_group_ids = seen_group_ids - all_ids
    if missing_group_ids or unknown_group_ids:
        raise ValueError(
            "Saved pixel-overlap groups do not match the labelled annotation IDs. "
            f"Missing IDs: {sorted(missing_group_ids)[:10]}; "
            f"unknown IDs: {sorted(unknown_group_ids)[:10]}"
        )
    if crossing_groups:
        raise ValueError(
            "The saved grouped split separates pixel-overlapping chips across partitions. "
            f"Crossing group indices: {crossing_groups[:10]}"
        )

    train_records = [all_by_id[i] for i in sorted(train_ids)]
    val_records = [all_by_id[i] for i in sorted(val_ids)]
    local_test_records = [all_by_id[i] for i in sorted(test_ids)]

    audit = {
        "method": split.get("method", "unspecified"),
        "fold": int(fold),
        "n_folds": len(split["val_ids"]),
        "n_labelled": len(all_ids),
        "n_train": len(train_records),
        "n_val": len(val_records),
        "n_local_test": len(local_test_records),
        "n_pixel_overlap_groups": len(groups),
        "n_ids_in_pixel_overlap_groups": len(seen_group_ids),
        "n_ids_not_listed_in_groups": len(all_ids - seen_group_ids),
        "groups_crossing_splits": len(crossing_groups),
        "duplicate_group_ids": len(duplicate_group_ids),
        "validation_local_test_overlap": len(val_ids & test_ids),
    }
    return train_records, val_records, local_test_records, audit


def _load_raw_hwc(record: dict) -> np.ndarray:
    with np.load(record["path"]) as archive:
        if "image" not in archive:
            raise KeyError(f"NPZ has no 'image' array: {record['path']}")
        image = archive["image"]
    if image.ndim != 3 or image.shape[-1] != 8:
        raise ValueError(f"Expected HxWx8 tile at {record['path']}, got {image.shape}")
    if image.shape[:2] != (record["height"], record["width"]):
        raise ValueError(
            f"NPZ shape {image.shape[:2]} does not match annotation size "
            f"{(record['height'], record['width'])} for image {record['id']}"
        )
    return np.asarray(image, dtype=np.float32)


def compute_band_scaling(
    train_records: list[dict],
    n_bands: int = 8,
    samples_per_tile: int = 4096,
    seed: int = 42,
    lower_pct: float = 2.0,
    upper_pct: float = 98.0,
) -> dict:
    """Compute train-only robust per-band scaling statistics.

    Each selected band is mapped to [0, 1] using its train-split 2nd and 98th
    percentiles. Sampling bounds RAM while using the same scaler for all 20 runs.
    """
    rng = np.random.default_rng(seed)
    sampled: list[list[np.ndarray]] = [[] for _ in range(n_bands)]
    for record in train_records:
        image = _load_raw_hwc(record)
        for band in range(n_bands):
            values = image[..., band].reshape(-1)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            if values.size > samples_per_tile:
                take = rng.choice(values.size, size=samples_per_tile, replace=False)
                values = values[take]
            sampled[band].append(values.astype(np.float32, copy=False))

    low, median, high = [], [], []
    for band in range(n_bands):
        if not sampled[band]:
            raise ValueError(f"Band index {band} has no finite values in training images")
        values = np.concatenate(sampled[band])
        q = np.percentile(values, [lower_pct, 50.0, upper_pct])
        low.append(float(q[0]))
        median.append(float(q[1]))
        high.append(float(q[2]))

    return {
        "method": "train-only percentile stretch",
        "lower_percentile": lower_pct,
        "upper_percentile": upper_pct,
        "low": low,
        "median": median,
        "high": high,
        "n_train_images": len(train_records),
        "samples_per_tile_per_band": samples_per_tile,
    }


def fill_nan_neighbour_mean(image_chw: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Iteratively fill NaNs from valid 8-neighbours, with train-median fallback."""
    image = np.asarray(image_chw, dtype=np.float64).copy()
    for channel in range(image.shape[0]):
        band = image[channel]
        missing = ~np.isfinite(band)
        # Treat infinities as missing too.
        for _ in range(200):
            if not missing.any():
                break
            valid = (~missing).astype(np.float64)
            sums = convolve(np.where(missing, 0.0, band), NEIGHBOUR_KERNEL, mode="constant", cval=0.0)
            counts = convolve(valid, NEIGHBOUR_KERNEL, mode="constant", cval=0.0)
            fillable = missing & (counts > 0)
            if not fillable.any():
                break
            band[fillable] = sums[fillable] / counts[fillable]
            missing = ~np.isfinite(band)
        if missing.any():
            band[missing] = float(fallback[channel])
        image[channel] = band
    return image.astype(np.float32)


def decode_instance_mask(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((height, width), dtype=np.uint8)
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            converted = mask_utils.frPyObjects(rle, height, width)
            rle = mask_utils.merge(converted) if isinstance(converted, list) else converted
    else:
        raise TypeError(f"Unsupported COCO segmentation type: {type(segmentation)}")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(np.uint8)


def _boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    boxes = []
    for mask in masks:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


class RTSBandDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        band_names: list[str],
        band_indices: dict[str, int],
        scaling: dict,
        augment: bool = False,
    ):
        self.records = records
        self.band_names = list(band_names)
        self.band_indices = [band_indices[name] for name in self.band_names]
        self.low = np.asarray(scaling["low"], dtype=np.float32)[self.band_indices]
        self.median = np.asarray(scaling["median"], dtype=np.float32)[self.band_indices]
        self.high = np.asarray(scaling["high"], dtype=np.float32)[self.band_indices]
        self.augment = bool(augment)

    def __len__(self):
        return len(self.records)

    def _load_image(self, record: dict) -> np.ndarray:
        raw = _load_raw_hwc(record)
        selected = raw[..., self.band_indices].transpose(2, 0, 1)
        selected = fill_nan_neighbour_mean(selected, self.median)
        denom = np.maximum(self.high - self.low, 1e-6)
        scaled = np.clip((selected - self.low[:, None, None]) / denom[:, None, None], 0.0, 1.0)
        return scaled.astype(np.float32)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = self._load_image(record)
        masks = [
            decode_instance_mask(a["segmentation"], record["height"], record["width"])
            for a in record["anns"]
        ]
        if masks:
            masks_np = np.stack(masks, axis=0).astype(np.uint8)
        else:
            masks_np = np.zeros((0, record["height"], record["width"]), dtype=np.uint8)

        image_t = torch.from_numpy(np.ascontiguousarray(image))
        masks_t = torch.from_numpy(np.ascontiguousarray(masks_np))

        if self.augment:
            # Rotations and flips preserve masks and are suitable for orientation-agnostic chips.
            k = random.randint(0, 3)
            if k:
                image_t = torch.rot90(image_t, k, dims=(1, 2))
                masks_t = torch.rot90(masks_t, k, dims=(1, 2))
            if random.random() < 0.5:
                image_t = torch.flip(image_t, dims=(2,))
                masks_t = torch.flip(masks_t, dims=(2,))
            if random.random() < 0.5:
                image_t = torch.flip(image_t, dims=(1,))
                masks_t = torch.flip(masks_t, dims=(1,))

        masks_np = masks_t.numpy().astype(np.uint8, copy=False)
        # Drop any empty decoded masks and create boxes from the final masks.
        keep = masks_np.reshape(len(masks_np), -1).any(axis=1) if len(masks_np) else np.zeros(0, bool)
        masks_np = masks_np[keep]
        boxes_np = _boxes_from_masks(masks_np)

        if len(masks_np):
            areas = masks_np.reshape(len(masks_np), -1).sum(axis=1)
        else:
            areas = np.zeros((0,), dtype=np.float32)

        target = {
            "boxes": torch.as_tensor(boxes_np, dtype=torch.float32),
            "labels": torch.ones((len(boxes_np),), dtype=torch.int64),
            "masks": torch.as_tensor(masks_np, dtype=torch.uint8),
            "image_id": torch.tensor([int(record["id"])], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes_np),), dtype=torch.int64),
        }
        return image_t, target


def detection_collate(batch):
    return tuple(zip(*batch))


def make_loaders(
    train_records: list[dict],
    val_records: list[dict],
    band_names: list[str],
    band_indices: dict[str, int],
    scaling: dict,
    batch_size: int = 2,
    num_workers: int = 2,
):
    train_ds = RTSBandDataset(train_records, band_names, band_indices, scaling, augment=True)
    val_ds = RTSBandDataset(val_records, band_names, band_indices, scaling, augment=False)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": detection_collate,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, train_ds, val_ds
