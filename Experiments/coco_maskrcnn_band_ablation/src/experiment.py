from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import (
    load_records,
    load_grouped_split,
    read_band_indices,
    compute_band_scaling,
    make_loaders,
)
from .model import build_coco_maskrcnn, count_parameters
from .metrics import predict_and_score
from .visualize import save_prediction_samples


RESULT_FIELDS = [
    "experiment", "name", "bands", "fold", "seed", "best_epoch",
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AP_default", "mIoU", "n_predictions", "n_images_with_predictions",
    "train_loss_at_best_ap", "training_seconds", "epochs_completed", "early_stopped",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _save_json_predictions(path: Path, predictions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(predictions))


def _save_history(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _train_one_epoch(model, loader, optimizer, scaler, device, amp: bool, grad_clip: float = 5.0):
    model.train()
    loss_total = 0.0
    loss_parts: dict[str, float] = {}
    steps = 0

    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [
            {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
             for key, value in target.items()}
            for target in targets
        ]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {steps}: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip
        )
        scaler.step(optimizer)
        scaler.update()

        loss_total += float(loss.detach().item())
        for key, value in loss_dict.items():
            loss_parts[key] = loss_parts.get(key, 0.0) + float(value.detach().item())
        steps += 1

    if steps == 0:
        raise RuntimeError("Training DataLoader yielded no batches")

    result = {"train_loss": loss_total / steps}
    result.update({f"loss_{key}": value / steps for key, value in loss_parts.items()})
    return result


def _metric_row(combo: dict, fold: int, seed: int, best_epoch: int, metrics: dict,
                train_loss_at_best_ap: float, training_seconds: float) -> dict:
    row = {
        "experiment": combo["id"],
        "name": combo["name"],
        "bands": ", ".join(combo["bands"]),
        "fold": int(fold),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "train_loss_at_best_ap": float(train_loss_at_best_ap),
        "training_seconds": float(training_seconds),
    }
    for key in RESULT_FIELDS:
        if key in metrics:
            value = metrics[key]
            row[key] = int(value) if key in ("n_predictions", "n_images_with_predictions") else float(value)
    return row


def _write_results_table(output_dir: Path) -> Path:
    rows = []
    for metrics_path in sorted((output_dir / "runs").glob("*/metrics.json")):
        rows.append(json.loads(metrics_path.read_text()))
    # A run is saved at runs/Cxx/metrics.json; also support nested run folders.
    if not rows:
        for metrics_path in sorted((output_dir / "runs").glob("**/metrics.json")):
            rows.append(json.loads(metrics_path.read_text()))
    result_path = output_dir / "results.csv"
    if rows:
        # Stable C01...C20 ordering.
        rows.sort(key=lambda x: x["experiment"])
        with result_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return result_path


def run_experiments(
    data_root: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    combinations_path: str | Path,
    fold: int = 1,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 2,
    num_workers: int = 2,
    learning_rate: float = 0.002,
    weight_decay: float = 0.0005,
    momentum: float = 0.9,
    amp: bool = True,
    patience: int = 6,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_size: int = 256,
    max_size: int = 512,
    detections_per_image: int = 10,
    trainable_backbone_layers: int = 3,
    force: bool = False,
    combo_ids: list[str] | None = None,
) -> list[dict]:
    """Run the selected band-combination screening experiments.

    The fixed grouped split is used exactly as stored: one local test set is held
    aside, and one validation fold is selected for the first ablation screen.
    The local test set is deliberately NOT used to rank the 20 combinations.
    """
    data_root = Path(data_root)
    split_path = Path(split_path)
    output_dir = Path(output_dir)
    combinations_path = Path(combinations_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, coco_data = load_records(data_root)
    train_records, val_records, local_test_records, split_audit = load_grouped_split(
        records, split_path, fold
    )
    band_indices = read_band_indices(data_root)

    print("DATA SPLIT AUDIT")
    print(json.dumps(split_audit, indent=2))
    print(f"Device: {device}")
    print(f"Torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    scaling_path = output_dir / "preprocessing" / f"fold{fold}_seed{seed}_train_band_scaling.json"
    if scaling_path.exists():
        scaling = json.loads(scaling_path.read_text())
        if int(scaling.get("n_train_images", -1)) != len(train_records):
            raise ValueError("Cached band scaling does not match the current training split")
        print("Using cached train-only band scaling:", scaling_path)
    else:
        print("Computing train-only robust scaling statistics for all 8 bands...")
        scaling = compute_band_scaling(train_records, seed=seed)
        scaling_path.parent.mkdir(parents=True, exist_ok=True)
        _save_json(scaling_path, scaling)

    combinations = json.loads(combinations_path.read_text())
    if combo_ids is not None:
        requested = set(combo_ids)
        combinations = [combo for combo in combinations if combo["id"] in requested]
        missing = requested - {combo["id"] for combo in combinations}
        if missing:
            raise ValueError(f"Unknown combination IDs: {sorted(missing)}")
    if not combinations:
        raise ValueError("No combinations selected")

    config_snapshot = {
        "model": "torchvision Mask R-CNN ResNet50-FPN, full COCO_V1 pretrained",
        "pretrained_weights": "MaskRCNN_ResNet50_FPN_Weights.COCO_V1",
        "fold": fold,
        "seed": seed,
        "epochs_max": epochs,
        "batch_size": batch_size,
        "optimizer": "SGD",
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "trainable_backbone_layers": trainable_backbone_layers,
        "frozen_backbone_parts": "ResNet stem + layer1 (Torchvision trainable_backbone_layers=3)",
        "image_min_size": min_size,
        "image_max_size": max_size,
        "detections_per_image": detections_per_image,
        "mask_threshold": mask_threshold,
        "score_threshold_for_mIoU": score_threshold,
        "band_scaling": "train-only 2nd-98th percentile to [0,1]; Torchvision pretrained transform applies its standard mean/std",
        "local_test_used_for_selection": False,
        "split_audit": split_audit,
    }
    _save_json(output_dir / "experiment_config.json", config_snapshot)

    all_rows = []
    for combo_number, combo in enumerate(combinations, start=1):
        run_dir = output_dir / "runs" / combo["id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists() and not force:
            saved = json.loads(metrics_path.read_text())
            print(f"[{combo_number}/{len(combinations)}] {combo['id']} already complete; skipping.")
            all_rows.append(saved)
            continue

        # Reset every RNG before each combination so the shared seed controls
        # head initialisation, augmentation and sampler order consistently.
        set_seed(seed)
        print("\n" + "=" * 76)
        print(f"[{combo_number}/{len(combinations)}] {combo['id']} — {combo['name']}")
        print("Bands:", combo["bands"])
        print("=" * 76)

        train_loader, val_loader, train_ds, val_ds = make_loaders(
            train_records=train_records,
            val_records=val_records,
            band_names=combo["bands"],
            band_indices=band_indices,
            scaling=scaling,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        model = build_coco_maskrcnn(
            detections_per_image=detections_per_image,
            min_size=min_size,
            max_size=max_size,
            trainable_backbone_layers=trainable_backbone_layers,
        ).to(device)
        params = count_parameters(model)
        print(
            f"Parameters: {params['total'] / 1e6:.2f}M total | "
            f"{params['trainable'] / 1e6:.2f}M trainable | "
            f"{params['frozen'] / 1e6:.2f}M frozen"
        )

        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        milestone_1 = max(1, int(epochs * 0.60))
        milestone_2 = max(milestone_1 + 1, int(epochs * 0.85))
        milestones = sorted(set(m for m in (milestone_1, milestone_2) if m < epochs))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        scaler = _make_grad_scaler(bool(amp and device.type == "cuda"))

        best_path = run_dir / "best.pth"
        last_path = run_dir / "last_checkpoint.pth"
        history_path = run_dir / "history.csv"
        history: list[dict] = []
        best_ap = -1.0
        best_epoch = -1
        best_metrics: dict = {}
        best_train_loss = float("inf")
        start_epoch = 0
        epochs_without_improvement = 0
        elapsed_before_resume = 0.0

        if last_path.exists() and not force:
            state = torch.load(last_path, map_location="cpu", weights_only=False)
            if state.get("experiment") == combo["id"] and state.get("bands") == combo["bands"]:
                model.load_state_dict(state["model_state"])
                optimizer.load_state_dict(state["optimizer_state"])
                scheduler.load_state_dict(state["scheduler_state"])
                scaler.load_state_dict(state["scaler_state"])
                history = state.get("history", [])
                best_ap = float(state.get("best_ap", -1.0))
                best_epoch = int(state.get("best_epoch", -1))
                best_metrics = state.get("best_metrics", {})
                best_train_loss = float(state.get("best_train_loss", float("inf")))
                start_epoch = int(state.get("next_epoch", 0))
                epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
                elapsed_before_resume = float(state.get("elapsed_seconds", 0.0))
                print(f"Resuming {combo['id']} from epoch {start_epoch + 1}")

        start_time = time.time()
        stopped_early = False
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            train_stats = _train_one_epoch(model, train_loader, optimizer, scaler, device, amp)
            scheduler.step()
            val_metrics, val_predictions, per_image = predict_and_score(
                model=model,
                loader=val_loader,
                records=val_records,
                data_root=data_root,
                device=device,
                amp=amp,
                mask_threshold=mask_threshold,
                score_threshold=score_threshold,
            )
            row = {
                "epoch": int(epoch + 1),
                **{key: float(value) for key, value in train_stats.items()},
                **{f"val_{key}": float(value) for key, value in val_metrics.items()},
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": float(time.time() - epoch_start),
            }
            history.append(row)
            _save_history(history_path, history)

            current_ap = float(val_metrics["AP"])
            improved = current_ap > best_ap + 1e-8
            if improved:
                best_ap = current_ap
                best_epoch = epoch + 1
                best_metrics = dict(val_metrics)
                best_train_loss = float(train_stats["train_loss"])
                epochs_without_improvement = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "experiment": combo["id"],
                    "name": combo["name"],
                    "bands": combo["bands"],
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics,
                    "fold": fold,
                    "seed": seed,
                    "pretrained": "MaskRCNN_ResNet50_FPN_Weights.COCO_V1",
                }, best_path)
                _save_json_predictions(run_dir / "val_predictions.json", val_predictions)
                save_prediction_samples(
                    dataset=val_ds,
                    records=val_records,
                    per_image=per_image,
                    output_path=run_dir / "samples_val.png",
                    band_names=combo["bands"],
                    n_samples=8,
                    score_threshold=score_threshold,
                    seed=seed,
                )
            else:
                epochs_without_improvement += 1

            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"loss {train_stats['train_loss']:.4f} | "
                f"val AP {val_metrics['AP'] * 100:.2f} | "
                f"AP50 {val_metrics['AP50'] * 100:.2f} | "
                f"mIoU {val_metrics['mIoU'] * 100:.2f} | "
                f"best AP {best_ap * 100:.2f} @ {best_epoch} | "
                f"{row['epoch_seconds']:.1f}s"
            )

            elapsed = elapsed_before_resume + (time.time() - start_time)
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "history": history,
                "best_ap": best_ap,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "best_train_loss": best_train_loss,
                "next_epoch": epoch + 1,
                "epochs_without_improvement": epochs_without_improvement,
                "elapsed_seconds": elapsed,
                "experiment": combo["id"],
                "bands": combo["bands"],
            }, last_path)

            if epochs_without_improvement >= patience:
                stopped_early = True
                print(f"Early stopping: no AP improvement for {patience} epochs.")
                break

        total_seconds = elapsed_before_resume + (time.time() - start_time)
        if best_epoch < 0 or not best_path.exists():
            raise RuntimeError(f"No best checkpoint was saved for {combo['id']}")

        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_checkpoint["model_state"])
        row = _metric_row(
            combo, fold, seed, best_epoch, best_metrics, best_train_loss, total_seconds
        )
        row["early_stopped"] = bool(stopped_early)
        row["epochs_completed"] = len(history)
        _save_json(metrics_path, row)
        all_rows.append(row)

        # Remove the last checkpoint once the experiment has completed successfully.
        if last_path.exists():
            last_path.unlink()
        _save_history(history_path, history)

        del model, optimizer, scheduler, scaler, train_loader, val_loader, train_ds, val_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _write_results_table(output_dir)
        print(
            f"COMPLETED {combo['id']} | best validation AP={row['AP'] * 100:.2f}% "
            f"| AP50={row['AP50'] * 100:.2f}% | mIoU={row['mIoU'] * 100:.2f}%"
        )

    _write_results_table(output_dir)
    return all_rows
