"""Train one (experiment, fold, seed) run; keep the checkpoint with the best val mask AP.

    python train.py --config configs/e4.yaml --fold 0 --seed 0
    python train.py --config configs/e2.yaml --fold 0 --seed 0 --override flags.drop_ndwi=true
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from config import load_config, resolve_path, resolve_streams, run_name
from data import build_loaders, stream_indices
from eval import append_row, evaluate
from models.build import build_model, param_groups


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def warmup_cosine(steps_per_epoch: int, epochs: int, warmup_epochs: float):
    warm = max(1, int(steps_per_epoch * warmup_epochs))
    total = steps_per_epoch * epochs

    def f(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--override", nargs="*", default=[], help="dotted key=value, e.g. train.epochs=50")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    name = run_name(cfg)
    run_dir = resolve_path(cfg["out_dir"]) / "runs" / name / f"fold{args.fold}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_info = {"experiment": name, "fold": args.fold, "seed": args.seed, "device": str(device),
                "amp": amp, "streams": resolve_streams(cfg), "channel_indices": stream_indices(cfg),
                "torch": str(torch.__version__)}
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump({**cfg, "run": run_info}, sort_keys=False))
    print(json.dumps(run_info))

    train_dl, val_dl, val_recs = build_loaders(cfg, args.fold, args.seed)
    model = build_model(cfg).to(device)

    t = cfg["train"]
    opt = torch.optim.AdamW(param_groups(model, t["lr"]), weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine(len(train_dl), t["epochs"], t["warmup_epochs"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    tb = SummaryWriter(run_dir / "tb")
    tb.add_text("run", f"```\n{yaml.safe_dump({**cfg, 'run': run_info}, sort_keys=False)}\n```")

    best, best_epoch, step = -1.0, -1, 0
    for epoch in range(t["epochs"]):
        model.train()
        t0, running = time.time(), {}
        for images, targets in train_dl:
            images = list(images.to(device, non_blocking=True))
            targets = [{k: v.to(device) for k, v in tg.items() if k in ("boxes", "labels", "masks")}
                       for tg in targets]
            with torch.autocast(device.type, enabled=amp):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}: "
                                         f"{ {k: float(v) for k, v in losses.items()} }")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip_grad"])
            scaler.step(opt)
            scaler.update()
            sched.step()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()
            tb.add_scalar("train/loss", loss.item(), step)
            tb.add_scalar("train/grad_norm", float(gnorm), step)
            for g in opt.param_groups:
                tb.add_scalar(f"lr/{g['name']}", g["lr"], step)
            step += 1

        for k, v in running.items():
            tb.add_scalar(f"train_epoch/{k}", v / len(train_dl), epoch)
        m = evaluate(model, val_dl, val_recs, cfg, device, amp)
        for k, v in m.items():
            tb.add_scalar(f"val/{k}", float(v), epoch)
        improved = m["AP"] > best
        if improved:
            best, best_epoch = float(m["AP"]), epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": m,
                        "run": run_info}, run_dir / "best.pth")
        print(f"[{name} f{args.fold} s{args.seed}] epoch {epoch + 1}/{t['epochs']} "
              f"loss {sum(running.values()) / len(train_dl):.3f} "
              f"(mask {running['loss_mask'] / len(train_dl):.3f} cls {running['loss_classifier'] / len(train_dl):.3f}) "
              f"val AP {m['AP']:.4f} "
              f"AP50 {m['AP50']:.4f} mIoU {m['mIoU']:.4f} {'*' if improved else ''} "
              f"({time.time() - t0:.0f}s)", flush=True)
    tb.close()

    # Final row: re-score the best checkpoint, so the CSV holds best-val metrics.
    ckpt = torch.load(run_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    m = evaluate(model, val_dl, val_recs, cfg, device, amp)
    row = {"experiment": name, "fold": args.fold, "seed": args.seed, "best_epoch": best_epoch,
           **{k: float(v) for k, v in m.items()}}
    print("best:", row)
    print("->", append_row(cfg, row))


if __name__ == "__main__":
    main()
