from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_official_scorer(data_root: Path):
    """Load the competition's own scorer so AP settings exactly match the challenge."""
    scorer_path = data_root / "tools" / "evaluate_coco.py"
    if not scorer_path.exists():
        raise FileNotFoundError(
            f"Official evaluator not found at {scorer_path}. "
            "Check that DATA_ROOT points to competition_release."
        )
    spec = importlib.util.spec_from_file_location("rts_official_evaluate_coco", scorer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import official scorer from {scorer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_coco_ground_truth(annotation_path: Path, image_ids: list[int]) -> COCO:
    data = json.loads(annotation_path.read_text())
    keep = set(map(int, image_ids))
    subset = {
        "images": [im for im in data["images"] if int(im["id"]) in keep],
        "annotations": [ann for ann in data["annotations"] if int(ann["image_id"]) in keep],
        "categories": data["categories"],
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
    }
    gt = COCO()
    gt.dataset = subset
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def _run_cocoeval(gt: COCO, predictions: list[dict], scorer=None) -> COCOeval:
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes([dict(item) for item in predictions])
        evaluator = COCOeval(gt, dt, iouType="segm")
        if scorer is not None:
            evaluator.params.maxDets = list(scorer.DEFAULT_MAX_DETECTIONS)
            evaluator.params.areaRng = [list(x) for x in scorer.DEFAULT_AREA_RANGES]
            evaluator.params.areaRngLbl = list(scorer.DEFAULT_AREA_LABELS)
        evaluator.evaluate()
        evaluator.accumulate()
    return evaluator


def official_mask_ap(gt: COCO, predictions: list[dict], scorer) -> dict[str, float]:
    """Challenge mask AP using official maxDet and object-area settings."""
    keys = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
    if not predictions:
        return {key: 0.0 for key in keys}
    evaluator = _run_cocoeval(gt, predictions, scorer)
    top_det = evaluator.params.maxDets[-1]

    def metric(iou=None, area="all"):
        value = scorer.summarize_metric(
            evaluator, ap=True, iou_thr=iou, area_label=area, max_det=top_det
        )
        return max(0.0, float(value))

    return {
        "AP": metric(),
        "AP50": metric(0.50),
        "AP75": metric(0.75),
        "AP_small": metric(area="small"),
        "AP_medium": metric(area="medium"),
        "AP_large": metric(area="large"),
    }


def _decode_coco_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    """Decode either polygon or RLE COCO segmentation into an HxW mask."""
    if isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((height, width), dtype=bool)
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
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
    return decoded.astype(bool)


def mean_iou(gt: COCO, per_image: dict, score_threshold: float = 0.5) -> float:
    """Mean image-level IoU of union(predicted instances) against union(GT instances)."""
    image_ious = []
    for image_id, (pred_masks, scores) in per_image.items():
        info = gt.imgs[image_id]
        gt_union = np.zeros((info["height"], info["width"]), dtype=bool)
        for ann in gt.imgToAnns[image_id]:
            gt_union |= _decode_coco_segmentation(
                ann["segmentation"], info["height"], info["width"]
            )
        selected = scores >= score_threshold
        pred_union = pred_masks[selected].any(axis=0) if selected.any() else np.zeros_like(gt_union)
        union = np.logical_or(gt_union, pred_union).sum()
        image_ious.append(1.0 if union == 0 else np.logical_and(gt_union, pred_union).sum() / union)
    return float(np.mean(image_ious)) if image_ious else 0.0


@torch.inference_mode()
def predict_and_score(
    model,
    loader,
    records: list[dict],
    data_root: Path,
    device: torch.device,
    amp: bool = True,
    mask_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> tuple[dict[str, float], list[dict], dict]:
    """Predict every image in a non-shuffled loader and calculate official metrics."""
    model.eval()
    results: list[dict] = []
    per_image: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    record_iter = iter(records)

    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        with torch.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
            outputs = model(images)

        for output, target in zip(outputs, targets):
            record = next(record_iter)
            target_id = int(target["image_id"].reshape(-1)[0].item())
            if target_id != int(record["id"]):
                raise RuntimeError(
                    f"Prediction/record order mismatch: target {target_id}, record {record['id']}"
                )
            height, width = int(record["height"]), int(record["width"])
            masks_tensor = output["masks"][:, 0]
            if masks_tensor.shape[-2:] != (height, width):
                raise ValueError(
                    f"Prediction mask size {tuple(masks_tensor.shape[-2:])} != "
                    f"original image size {(height, width)} for image {record['id']}"
                )
            masks = (masks_tensor > mask_threshold).detach().cpu().numpy().astype(np.uint8)
            scores = output["scores"].detach().float().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()

            # The adapted two-class model should only return label 1 (RTS).
            keep = labels == 1
            masks, scores = masks[keep], scores[keep]
            per_image[int(record["id"])] = (masks, scores)

            for mask, score in zip(masks, scores):
                if not mask.any():
                    continue
                rle = mask_utils.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                results.append({
                    "image_id": int(record["id"]),
                    "category_id": 1,
                    "segmentation": rle,
                    "score": float(score),
                })

    expected_ids = {int(r["id"]) for r in records}
    if set(per_image) != expected_ids:
        raise RuntimeError("Prediction loader order/count did not cover the expected image IDs")

    annotation_path = data_root / "train" / "annotations" / "instances_train.json"
    gt = build_coco_ground_truth(annotation_path, [int(r["id"]) for r in records])
    scorer = load_official_scorer(data_root)
    metrics = official_mask_ap(gt, results, scorer)
    metrics["AP_default"] = 0.0
    if results:
        default_eval = _run_cocoeval(gt, results, scorer=None)
        with contextlib.redirect_stdout(io.StringIO()):
            default_eval.summarize()
        metrics["AP_default"] = max(0.0, float(default_eval.stats[0]))
    metrics["mIoU"] = mean_iou(gt, per_image, score_threshold)
    metrics["n_predictions"] = float(len(results))
    metrics["n_images"] = float(len(records))
    metrics["n_images_with_predictions"] = float(sum(len(v[1]) > 0 for v in per_image.values()))
    return metrics, results, per_image
