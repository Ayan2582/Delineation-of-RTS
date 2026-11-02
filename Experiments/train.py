"""Train one (experiment, fold, seed) run; keep the checkpoint with the best val mask AP.

    python train.py --config configs/e4.yaml --fold 0 --seed 0
    python train.py --config configs/e2.yaml --fold 0 --seed 0 --override flags.drop_ndwi=true
    python train.py --config configs/e2.yaml --fold all --seed 0 --override train.epochs=40

    python train.py --config configs/e6.yaml --fold 0 --init <pretrain run>/best.pth   # fine-tune
    python train.py --config configs/e6.yaml --fold 1 --resume <run>/last.pth           # continue a run

--fold all trains on every labelled chip (local test included) for the final submission model.
There is no val, so nothing picks an epoch: best.pth is the LAST epoch (fix train.epochs from the
CV plateau), and no row is written to results.csv, which holds cross-validation scores only.

Long runs (DARTS pretraining) span Kaggle sessions: every epoch writes last.pth (model, optimiser,
schedule, history), --stop_after_h ends the run cleanly after the epoch that crosses the budget and
leaves an INCOMPLETE marker in the run folder, and --resume continues from last.pth. A finished run
deletes last.pth and the marker.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from config import ALL_FOLD, load_config, parse_fold, resolve_path, resolve_streams, run_name
from data import build_loaders, build_monitor_loader, stats_path, stream_indices
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
    ap.add_argument("--fold", type=parse_fold, required=True, help=f"grouped-CV fold index, or {ALL_FOLD!r}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--override", nargs="*", default=[], help="dotted key=value, e.g. train.epochs=50")
    ap.add_argument("--init", default=None, help="checkpoint whose model weights start this run (fine-tuning)")
    ap.add_argument("--resume", default=None, help="last.pth of an unfinished run; continues it here")
    ap.add_argument("--stop_after_h", type=float, default=None, help="stop after the epoch that passes this many hours")
    args = ap.parse_args()
    t_start = time.time()

    cfg = load_config(args.config, args.override)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    name = run_name(cfg)
    run_dir = resolve_path(cfg["out_dir"]) / "runs" / name / f"fold{args.fold}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_info = {"experiment": name, "fold": args.fold, "seed": args.seed, "device": str(device),
                "selection": "last epoch (no val)" if args.fold == ALL_FOLD else "best val AP",
                "amp": amp, "streams": resolve_streams(cfg), "channel_indices": stream_indices(cfg),
                "torch": str(torch.__version__), "init": args.init}
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump({**cfg, "run": run_info}, sort_keys=False))
    print(json.dumps(run_info))

    train_dl, val_dl, val_recs = build_loaders(cfg, args.fold, args.seed)
    monitor = build_monitor_loader(cfg) if cfg.get("monitor") else None   # GeoAI chips, report only
    shutil.copy(stats_path(cfg, args.fold), run_dir / "stats.json")   # normalisation, for inference
    if args.init or args.resume:
        cfg["model"]["pretrained"] = False                 # weights come from the checkpoint
    model = build_model(cfg).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["model"])
        print("init from", args.init)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params {n_all / 1e6:.1f}M, trainable {n_train / 1e6:.1f}M "
          f"(optical {cfg['model'].get('optical', 'swin_v2_b')}, frozen stages {cfg['model'].get('freeze_optical', 0)})")

    t = cfg["train"]
    opt = torch.optim.AdamW(param_groups(model, t["lr"]), weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine(len(train_dl), t["epochs"], t["warmup_epochs"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    log_dir = resolve_path(cfg["log_dir"]) / name / run_dir.name if cfg.get("log_dir") else run_dir / "tb"
    tb = SummaryWriter(log_dir)
    history = resolve_path(cfg["out_dir"]) / "results" / "history" / f"{name}_{run_dir.name}.csv"
    history.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    tb.add_text("run", f"```\n{yaml.safe_dump({**cfg, 'run': run_info}, sort_keys=False)}\n```")

    best, best_epoch, step, start = -1.0, -1, 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        best, best_epoch, step, start, rows = ck["best"], ck["best_epoch"], ck["step"], ck["epoch"] + 1, ck["rows"]
        prev_best = Path(args.resume).parent / "best.pth"
        if prev_best.resolve() != (run_dir / "best.pth").resolve() and prev_best.exists():
            shutil.copy(prev_best, run_dir / "best.pth")  # the previous session's best checkpoint
        print(f"resumed from {args.resume}: epoch {start}/{t['epochs']} done, best epoch {best_epoch + 1}")
    eval_every = int(t.get("eval_every", 1))
    (run_dir / "INCOMPLETE").write_text("training not finished; resume from last.pth")
    for epoch in range(start, t["epochs"]):
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
        scored = val_dl is not None and ((epoch + 1) % eval_every == 0 or epoch + 1 == t["epochs"])
        m = evaluate(model, val_dl, val_recs, cfg, device, amp) if scored else {}
        g = evaluate(model, *monitor, device, amp) if scored and monitor else {}
        for k, v in m.items():
            tb.add_scalar(f"val/{k}", float(v), epoch)
        for k, v in g.items():
            tb.add_scalar(f"geoai/{k}", float(v), epoch)
        rows.append({"epoch": epoch + 1, "time_s": round(time.time() - t0, 1),
                     **{f"lr_{g['name']}": g["lr"] for g in opt.param_groups},
                     **{f"train_{k}": v / len(train_dl) for k, v in running.items()},
                     "train_loss": sum(running.values()) / len(train_dl),
                     **{f"val_{k}": float(v) for k, v in m.items()},
                     **{f"geoai_{k}": float(v) for k, v in g.items()}})
        with history.open("w", newline="") as f:              # rewritten each epoch: survives a crash
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
        improved = m["AP"] > best if m else val_dl is None    # no val: every epoch replaces the last
        if improved:
            best, best_epoch = float(m.get("AP", -1.0)), epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": m or None,
                        "run": run_info}, run_dir / "best.pth")
        parts = " ".join(f"{k.removeprefix('loss_')} {v / len(train_dl):.3f}" for k, v in running.items())
        val = (f"val AP {m['AP']:.4f} AP50 {m['AP50']:.4f} mIoU {m['mIoU']:.4f} {'*' if improved else ''} "
               if m else "no val " if val_dl is None else "")
        if g:
            val += f"| GeoAI AP {g['AP']:.4f} AP50 {g['AP50']:.4f} mIoU {g['mIoU']:.4f} "
        print(f"[{name} f{args.fold} s{args.seed}] epoch {epoch + 1}/{t['epochs']} "
              f"loss {sum(running.values()) / len(train_dl):.3f} "
              f"({parts}) {val}"
              f"({time.time() - t0:.0f}s)", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "best_epoch": best_epoch,
                    "step": step, "rows": rows}, run_dir / "last.pth")
        hours = (time.time() - t_start) / 3600
        if args.stop_after_h and hours > args.stop_after_h and epoch + 1 < t["epochs"]:
            tb.close()
            print(f"stopped after epoch {epoch + 1}/{t['epochs']} ({hours:.1f} h > {args.stop_after_h} h). "
                  f"Resume with --resume {run_dir / 'last.pth'}")
            return
    tb.close()
    if not t.get("keep_last"):
        (run_dir / "last.pth").unlink()
    (run_dir / "INCOMPLETE").unlink()

    if val_dl is None:
        print(f"fold {ALL_FOLD}: trained on {len(train_dl.dataset)} chips; best.pth = epoch {best_epoch + 1} "
              f"(the last). No results.csv row: there is no val to score.")
        print("->", run_dir / "best.pth")
        return

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
