"""Smoke test: a random 8x333x417 batch through every experiment, in train and eval mode.

    python smoke_test.py                 # random weights (fast, offline)
    python smoke_test.py --pretrained    # also downloads the Satlas Swin weights

Asserts: expected channels per experiment, C2-C5 shapes, finite losses, eval output cropped
to orig_size, internal resize disabled, stem expansion maths, augmentation channel rules.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from config import load_config, resolve_streams, run_name
from data import Augment, band_index, jitter_positions, stream_indices
from models.backbones import SWIN_CHANNELS, TERRAIN_CHANNELS, expand_stem
from models.build import build_model
from rts_preprocess import pad_collate

H, W = 333, 417
HP, WP = 352, 448                       # pad_collate -> multiple of 32
STRIDES = (4, 8, 16, 32)

OPT4 = ["red", "green", "blue", "nir"]
TER3 = ["relative_elevation", "shaded_relief", "ndvi"]
ALL8 = ["red", "green", "blue", "ndvi", "relative_elevation", "shaded_relief", "nir", "ndwi"]

# (config, overrides, expected swin bands, expected terrain bands, expected head maps)
CASES = [
    ("configs/e1.yaml", [], ["green", "nir", "relative_elevation"], [], "fpn"),
    ("configs/e2.yaml", [], ALL8, [], "fpn"),
    ("configs/e2.yaml", ["flags.drop_ndwi=true"], ALL8[:-1], [], "fpn"),
    ("configs/e3.yaml", [], OPT4, TER3, "c5"),
    ("configs/e3.yaml", ["flags.ndwi_to_terrain=true"], OPT4, TER3 + ["ndwi"], "c5"),
    ("configs/e4.yaml", [], OPT4, TER3, "fpn"),
    ("configs/e4.yaml", ["flags.ndwi_to_terrain=true"], OPT4, TER3 + ["ndwi"], "fpn"),
]


def fake_batch(n=2):
    """Random normalised 8-band tiles with two rectangular instances each, through pad_collate."""
    batch = []
    for i in range(n):
        img = torch.randn(8, H, W)
        masks = torch.zeros(2, H, W, dtype=torch.uint8)
        masks[0, 40:120, 50:160] = 1
        masks[1, 200:300, 250:400] = 1
        boxes = torch.tensor([[50, 40, 160, 120], [250, 200, 400, 300]], dtype=torch.float32)
        batch.append((img, {"boxes": boxes, "labels": torch.ones(2, dtype=torch.int64), "masks": masks}))
    return pad_collate(batch)


def check_expand_stem():
    conv = nn.Conv2d(3, 128, 4, 4)
    for k in (3, 4, 7, 8):
        new = expand_stem(conv, k)
        assert new.in_channels == k
        w0, w1 = conv.weight.data, new.weight.data
        if k > 3:
            assert torch.allclose(w1[:, :3], w0 * 3 / k)
            assert torch.allclose(w1[:, 3:], (w0.mean(1, keepdim=True) * 3 / k).expand_as(w1[:, 3:]))
        # sum over input channels is preserved -> a constant input gives the same output
        x = torch.full((1, 3, 8, 8), 0.7)
        assert torch.allclose(new(x.new_full((1, k, 8, 8), 0.7)), conv(x), atol=1e-5)
    print("expand_stem: ok")


def check_augment(cfg):
    names = [*resolve_streams(cfg)["swin"], *resolve_streams(cfg)["terrain"]]
    jit = jitter_positions(cfg)
    assert [names[i] for i in jit] == [n for n in names if n in cfg["aug"]["jitter_bands"]]
    # jitter only: non-optical channels must be untouched
    a = dict(cfg["aug"], hflip=0, vflip=0, rot90=False, jitter_p=1.0)
    img = torch.randn(len(names), 50, 60)
    out, _ = Augment(a, jit)(img, torch.zeros(1, 50, 60, dtype=torch.uint8))
    other = [i for i in range(len(names)) if i not in jit]
    assert torch.equal(out[other], img[other]), "jitter touched a non-optical band"
    # geometry only: the image and masks must move together
    a = dict(cfg["aug"], hflip=1.0, vflip=1.0, rot90=True, jitter_p=0.0, crop=40)
    m = (torch.rand(1, 50, 60) > 0.5).to(torch.uint8)
    img = m.float().expand(len(names), -1, -1).clone()
    out, mo = Augment(a, jit)(img, m)
    assert out.shape[-2:] == mo.shape[-2:] == (40, 40)
    assert torch.equal(out[0], mo[0].float()), "geometric aug desynced image and masks"


def run_case(path, overrides, exp_swin, exp_ter, head, pretrained, images, targets, extra=()):
    cfg = load_config(path, [*extra, *overrides, f"model.pretrained={str(pretrained).lower()}"])
    name = run_name(cfg)
    s = resolve_streams(cfg)
    assert s["swin"] == exp_swin and s["terrain"] == exp_ter, (name, s)
    idx = band_index(cfg)
    si = stream_indices(cfg)
    assert si["swin"] == [idx[b] for b in exp_swin] and si["terrain"] == [idx[b] for b in exp_ter]
    channels = si["swin"] + si["terrain"]
    check_augment(cfg)

    model = build_model(cfg)
    assert type(model.transform).__name__ == "NoResize"
    bb = model.backbone
    assert bb.swin.stem.in_channels == len(exp_swin), name
    if exp_ter:
        assert bb.terrain.stem.in_channels == len(exp_ter), name
    x = images[:, channels]
    assert x.shape[1] == len(exp_swin) + len(exp_ter)

    model.eval()
    with torch.no_grad():
        sw, te = bb.stages(x)
        for f, c, st in zip(sw, SWIN_CHANNELS, STRIDES):
            assert tuple(f.shape[1:]) == (c, HP // st, WP // st), (name, "swin", tuple(f.shape))
        if exp_ter:
            for f, c, st in zip(te, TERRAIN_CHANNELS, STRIDES):
                assert tuple(f.shape[1:]) == (c, HP // st, WP // st), (name, "terrain", tuple(f.shape))
        feats = bb(x)
    shapes = {k: tuple(v.shape[1:]) for k, v in feats.items()}
    if head == "fpn":
        want = {"0": (256, 88, 112), "1": (256, 44, 56), "2": (256, 22, 28), "3": (256, 11, 14),
                "pool": (256, 6, 7)}
    else:
        want = {"0": (256, 11, 14)}
    assert shapes == want, (name, shapes)

    model.train()
    losses = model(list(x), [{k: v for k, v in t.items() if k != "orig_size"} for t in targets])
    loss = sum(losses.values())
    assert all(torch.isfinite(v) for v in losses.values()), (name, losses)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    model.eval()
    with torch.no_grad():
        outs = model(list(x))
    for out, t in zip(outs, targets):
        h, w = (int(v) for v in t["orig_size"])
        assert (h, w) == (H, W)
        m = out["masks"][:, :, :h, :w]
        assert m.shape[1:] == (1, H, W) and out["masks"].shape[-2:] == (HP, WP), m.shape

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{name:22s} swin={exp_swin} terrain={exp_ter}\n"
          f"{'':22s} swin C2-C5={[tuple(f.shape[1:]) for f in sw]}"
          + (f"\n{'':22s} terr C2-C5={[tuple(f.shape[1:]) for f in te]}" if exp_ter else "")
          + f"\n{'':22s} to head={shapes}\n{'':22s} losses={ {k: round(float(v), 3) for k, v in losses.items()} } "
          f"params={n_params:.1f}M dets={[len(o['scores']) for o in outs]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--override", nargs="*", default=[], help="e.g. data.root=/kaggle/input/...")
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    check_expand_stem()
    images, targets = fake_batch()
    assert images.shape == (2, 8, HP, WP)
    for case in CASES:
        run_case(*case, args.pretrained, images, targets, args.override)
    print(f"\nsmoke test passed: {len(CASES)} configurations")


if __name__ == "__main__":
    main()
