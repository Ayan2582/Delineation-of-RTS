"""Smoke test: a random 8x333x417 batch through every experiment, in train and eval mode.

    python smoke_test.py                 # random weights (fast, offline)
    python smoke_test.py --pretrained    # also downloads the Satlas Swin weights

Asserts: expected channels per experiment, C2-C5 shapes, finite losses, eval output cropped
to orig_size, fixed-scale resize (model.input_scale) with targets and outputs mapped correctly,
input normalisation (data.swin_norm), stem expansion maths, augmentation channel rules.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from config import ALL_FOLD, load_config, parse_fold, resolve_streams, run_name
from torchvision.ops import masks_to_boxes

from data import (Augment, InstanceBank, band_index, build_test_loader, drop_positions, get_test_ids, input_norm,
                  jitter_positions, jitter_ref, load_records, release_root, split_records, stream_indices)
from models.backbones import TERRAIN_CHANNELS, expand_stem
from models.boundary import BoundaryMaskPredictor, BoundaryRoIHeads, boundary_targets
from models.build import ScaledTransform, build_model
from models.gate import BandGate
from models.ohem import OHEMRoIHeads
from models.seg import SegModel, gap_weights
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
    ("configs/e4.yaml", ["model.input_scale=2.0", "data.swin_norm=satlas"], OPT4, TER3, "fpn"),
    ("configs/e4.yaml", ["model.ohem=true"], OPT4, TER3, "fpn"),
    ("configs/e5.yaml", [], OPT4, TER3, "fpn"),
    ("configs/e5.yaml", ["model.input_scale=2.0", "data.swin_norm=satlas"], OPT4, TER3, "fpn"),
    ("configs/e5.yaml", ["model.optical=resnet18", "model.freeze_optical=1", "data.swin_norm=zscore"], OPT4, TER3, "fpn"),
    ("configs/e5.yaml", ["model.optical=swin_v2_t", "model.freeze_optical=2", "data.swin_norm=zscore"], OPT4, TER3, "fpn"),
    ("configs/e4.yaml", ["model.optical=resnet50", "data.swin_norm=zscore"], OPT4, TER3, "fpn"),
    ("configs/e2_gate.yaml", ["aug.band_drop_p=0.1"], ALL8, [], "fpn"),
    ("configs/e2_boundary.yaml", [], ALL8, [], "fpn"),
    ("configs/e6.yaml", ["model.input_scale=2.0", "data.swin_norm=satlas", "aug.band_drop_p=0.1"], ALL8, [], "fpn"),
    ("configs/e6.yaml", ["model.ohem=true"], ALL8, [], "fpn"),
    ("configs/e4.yaml", ["model.band_gate=true", "model.boundary_head=true"], OPT4, TER3, "fpn"),
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
    base = dict(cfg["aug"], scale_jitter=None, copy_paste_p=0.0)   # each check enables only its own op
    # jitter only: non-optical channels must be untouched
    a = dict(base, hflip=0, vflip=0, rot90=False, jitter_p=1.0)
    img = torch.randn(len(names), 50, 60)
    out, _ = Augment(a, jit)(img, torch.zeros(1, 50, 60, dtype=torch.uint8))
    other = [i for i in range(len(names)) if i not in jit]
    assert torch.equal(out[other], img[other]), "jitter touched a non-optical band"
    # geometry only: the image and masks must move together
    a = dict(base, hflip=1.0, vflip=1.0, rot90=True, jitter_p=0.0, crop=40)
    m = (torch.rand(1, 50, 60) > 0.5).to(torch.uint8)
    img = m.float().expand(len(names), -1, -1).clone()
    out, mo = Augment(a, jit)(img, m)
    assert out.shape[-2:] == mo.shape[-2:] == (40, 40)
    assert torch.equal(out[0], mo[0].float()), "geometric aug desynced image and masks"
    # scale jitter: image and masks resized together
    a = dict(base, hflip=0, vflip=0, rot90=False, jitter_p=0.0, crop=None, scale_jitter=[0.5, 0.5])
    out, mo = Augment(a, jit)(img, m)
    assert out.shape[-2:] == mo.shape[-2:] == (25, 30), "scale jitter shape"
    # copy-paste: a 1-value instance pasted on a zero image adds a mask that matches its pixels
    bank = InstanceBank([(np.ones((len(names), 12, 10), np.float32), np.ones((1, 12, 10), np.uint8))],
                        list(range(len(names))))
    a = dict(a, scale_jitter=None, copy_paste_p=1.0, copy_paste_max=1)
    out, mo = Augment(a, jit, bank)(torch.zeros(len(names), 50, 60), torch.zeros(0, 50, 60, dtype=torch.uint8))
    assert len(mo) == 1 and int(mo.sum()) == 120 and torch.equal(out[0], mo[0].float()), "copy-paste"
    # band dropout: exactly one droppable band becomes its fill value, nothing else changes, never RGB
    drop = drop_positions(cfg)
    assert not {names[i] for i in drop} & {"red", "green", "blue"}
    if drop:
        a = dict(base, hflip=0, vflip=0, rot90=False, jitter_p=0.0, crop=None, band_drop_p=1.0)
        img = torch.randn(len(names), 50, 60)
        fill = torch.arange(len(drop), dtype=torch.float32)[:, None, None] + 7
        out, _ = Augment(a, jit, drop_idx=drop, drop_val=fill)(img, torch.zeros(1, 50, 60, dtype=torch.uint8))
        changed = [i for i in range(len(names)) if not torch.equal(out[i], img[i])]
        assert len(changed) == 1 and changed[0] in drop, "band dropout"
        assert torch.all(out[changed[0]] == fill[drop.index(changed[0])])


def check_input_norm(cfg):
    """zscore leaves the stats alone; satlas maps Swin RGB 0-255 to 0-1, the other Swin bands to the RGB
    range, and leaves terrain z-scored. Either way the jitter reference is the train mean/std."""
    idx = band_index(cfg)
    mean = np.array([94, 97, 78, 0.37, 0.0, 165, 2172, -0.36], np.float32)
    std = np.array([40, 33, 32, 0.38, 1.3, 11, 876, 0.40], np.float32)
    mn, sn = input_norm(cfg, mean, std)
    s = resolve_streams(cfg)
    if cfg["data"]["swin_norm"] == "zscore":
        assert np.array_equal(mn, mean) and np.array_equal(sn, std)
    else:
        for b in s["swin"]:
            i = idx[b]
            if b in ("red", "green", "blue"):
                assert (mn[i], sn[i]) == (0.0, 255.0), b
            else:   # the band's train mean/std land on the RGB bands' mean/std in /255 space
                assert np.isclose((mean[i] - mn[i]) / sn[i], mean[:3].mean() / 255, atol=1e-4), b
                assert np.isclose(std[i] / sn[i], std[:3].mean() / 255, atol=1e-4), b
        for b in s["terrain"]:
            assert (mn[idx[b]], sn[idx[b]]) == (mean[idx[b]], std[idx[b]]), b
    mu, sd = jitter_ref(cfg, mean, std, mn, sn)
    names = [*s["swin"], *s["terrain"]]
    b = [idx[names[p]] for p in jitter_positions(cfg)]
    z = torch.as_tensor((mean[b] - mn[b]) / sn[b])[:, None, None]
    assert torch.allclose(mu, z) and torch.allclose(sd, torch.as_tensor(std[b] / sn[b])[:, None, None])


def check_scaled_transform(t):
    """Training targets are resized with the image; masks stay binary and match their boxes."""
    t = t.train()
    img = torch.zeros(3, 96, 128)
    m = torch.zeros(1, 96, 128, dtype=torch.uint8)
    m[0, 10:30, 20:60] = 1
    tgt = {"boxes": torch.tensor([[20.0, 10.0, 60.0, 30.0]]), "labels": torch.ones(1, dtype=torch.int64), "masks": m}
    il, (out,) = t([img], [tgt])
    H, W = round(96 * t.scale), round(128 * t.scale)
    assert il.image_sizes[0] == (H, W) and out["masks"].shape[-2:] == (H, W)
    assert set(out["masks"].unique().tolist()) <= {0, 1}
    assert torch.allclose(out["boxes"], torch.tensor([[20.0, 10.0, 60.0, 30.0]]) * t.scale)
    assert torch.equal(masks_to_boxes(out["masks"].bool()) + torch.tensor([0, 0, 1, 1]), out["boxes"])


def spy_ohem(model, cfg):
    """With model.ohem: check the RoI heads are OHEM and record how many RoIs each image keeps.

    Returns the list of per-image counts, one entry per training forward (None without OHEM).
    """
    rh = model.roi_heads
    if not cfg["model"].get("ohem"):
        assert not isinstance(rh, OHEMRoIHeads)
        return None
    assert isinstance(rh, OHEMRoIHeads) and rh.ohem_batch == cfg["model"]["ohem_rois"]
    picked, select = [], rh.select_training_samples

    def spy(proposals, targets):
        out = select(proposals, targets)
        picked.append([len(p) for p in out[0]])
        return out

    rh.select_training_samples = spy
    return picked


def check_ohem_topk():
    """OHEM without NMS keeps exactly the k highest-loss RoIs; with NMS no kept pair overlaps above the threshold."""
    from torchvision.ops import box_iou
    from torchvision.models.detection.roi_heads import RoIHeads
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
    from torchvision.ops import MultiScaleRoIAlign
    rh = RoIHeads(MultiScaleRoIAlign(["0"], 7, 2), TwoMLPHead(8 * 49, 32), FastRCNNPredictor(32, 2),
                  0.5, 0.5, 512, 0.25, None, 0.05, 0.5, 100)
    feats = {"0": torch.randn(1, 8, 32, 32)}
    props = [torch.tensor([[x, y, x + w, y + w] for x in range(0, 100, 7) for y in range(0, 100, 9) for w in (12, 30)],
                          dtype=torch.float32)]
    tgt = [{"boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]]), "labels": torch.ones(1, dtype=torch.int64)}]
    for nms_thresh in (1.0, 0.7):
        OHEMRoIHeads.from_roi_heads(rh, 16, nms_thresh)
        rh._ohem_inputs = (feats, [(128, 128)])
        all_p = rh.add_gt_proposals([props[0].clone()], [tgt[0]["boxes"]])
        m, lab = rh.assign_targets_to_proposals(all_p, [tgt[0]["boxes"]], [tgt[0]["labels"]])
        reg = list(rh.box_coder.encode([tgt[0]["boxes"][m[0]]], all_p))
        loss = rh.roi_losses(all_p, lab, reg)[0]
        p, _, _, _ = rh.select_training_samples([props[0].clone()], tgt)
        assert len(p[0]) == 16
        if nms_thresh == 1.0:
            top = all_p[0][loss.argsort(descending=True)[:16]]
            assert torch.equal(p[0], top), "OHEM did not keep the top-k losses"
        else:
            iou = box_iou(p[0], p[0]).fill_diagonal_(0)
            assert iou.max() <= 0.7 + 1e-6, "OHEM NMS left overlapping RoIs"
        rh._ohem_inputs = None
    print("ohem: ok")


def check_seg():
    """E5 pieces: the gap weight sits only on background between two instances; postprocess turns
    blobs into one detection each (tiny ones dropped), highest mean probability first."""
    m = np.zeros((2, 40, 60), np.uint8)
    m[0, 10:30, 5:25] = 1
    m[1, 10:30, 28:50] = 1                                   # 3 px gap at columns 25-27
    w = gap_weights(m, 10.0, 5.0)
    assert w[20, 26] > 5 and w[0, 0] < 1.01 and np.all(w[m.any(0)] == 1), "gap weights"
    assert np.all(gap_weights(m[:1], 10.0, 5.0) == 1), "one instance has no gaps"
    cfg = load_config("configs/e5.yaml")
    stub = nn.Identity()
    stub.out_channels = 256
    model = SegModel(stub, 7, cfg["model"])
    prob = torch.zeros(40, 60)
    prob[5:15, 5:15] = 0.9
    prob[20:35, 30:55] = 0.7
    prob[0:2, 58:60] = 0.95                                  # 4 px: below seg_min_px
    out = model.postprocess(prob)
    assert np.allclose(out["scores"].tolist(), [0.9, 0.7]) and out["masks"].shape == (2, 1, 40, 60)
    assert int(out["masks"][0].sum()) == 100 and out["boxes"][0].tolist() == [5, 5, 14, 14]
    prob = torch.full((40, 60), 0.05)
    prob[10:20, 10:20] = 0.3                                 # nothing >= 0.5: fall back to the best blob
    assert len(model.postprocess(prob)["scores"]) == 1
    print("seg: ok")


def check_gate_boundary():
    """Band gate is the identity at init and keeps padding 0; boundary targets are the mask outline;
    the boundary predictor equals a plain mask predictor at init (zero fusion)."""
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    g = BandGate(8)
    x = torch.randn(2, 8, 20, 30)
    x[1, :, 15:] = 0                                         # padding rows
    y = g(x)
    assert torch.allclose(y, x, atol=1e-6) and g.last_gate.shape == (2, 8) and torch.all(y[1, :, 15:] == 0)
    with torch.no_grad():
        g.mlp[-1].bias.fill_(-20.0)                          # gate ~0: every band -> its neutral value
        g.neutral.copy_(torch.arange(8.0))
    y = g(x)
    assert torch.allclose(y[0], torch.arange(8.0)[:, None, None].expand(8, 20, 30), atol=1e-4)
    assert torch.all(y[1, :, 15:] == 0), "gate filled the padding"

    m = torch.zeros(1, 28, 28)
    m[0, 7:21, 7:21] = 1
    box = torch.tensor([[0.0, 0.0, 28.0, 28.0]])
    t = boundary_targets([m], [box], [torch.zeros(1, dtype=torch.int64)], 28)[0]
    # a ~2 px ring on the square's edge (torchvision's RoIAlign projection shifts it by up to half a pixel)
    assert t[14, 14] == 0 and t[0, 0] == 0 and t[27, 27] == 0, "boundary target off the outline"
    assert t[14, 6:9].any() and t[14, 19:22].any() and t[6:9, 14].any() and t[19:22, 14].any()
    assert 2 <= int(t[14].sum()) <= 6, "boundary ring too thick"

    bp, mp = BoundaryMaskPredictor(16, 16, 2, 8), MaskRCNNPredictor(16, 16, 2)
    mp.load_state_dict({k: v for k, v in bp.state_dict().items() if k.split(".")[0] in ("conv5_mask", "mask_fcn_logits")})
    f = torch.randn(3, 16, 14, 14)
    assert torch.allclose(bp(f), mp(f), atol=1e-5) and bp.last_boundary.shape == (3, 28, 28)
    print("gate + boundary: ok")


def check_fold_all(extra=()):
    """--fold all: every labelled chip trains (local test included), no val, and the local test
    split can no longer be scored. The split part needs the release, so it is skipped without it."""
    assert parse_fold(ALL_FOLD) == ALL_FOLD and parse_fold("3") == 3
    try:
        parse_fold("three")
        raise AssertionError("parse_fold accepted 'three'")
    except ValueError:
        pass
    cfg = load_config("configs/e2.yaml", list(extra))
    if not (release_root(cfg) / "train" / "annotations").exists():
        print("fold all: release not found, split check skipped")
        return
    recs = load_records(cfg)
    train, val = split_records(cfg, ALL_FOLD)
    assert len(train) == len(recs) and not val, "fold all must train on every chip with no val"
    test_ids = get_test_ids(cfg)
    assert test_ids <= {r["id"] for r in train}, "fold all must include the local test chips"
    assert not test_ids & {r["id"] for r in split_records(cfg, 0)[0]}, "CV folds must still hold out local test"
    try:
        build_test_loader(cfg, ALL_FOLD, "test")
        raise AssertionError("fold all must refuse to score the local test chips")
    except ValueError:
        pass
    print(f"fold all: {len(train)} train chips (incl. {len(test_ids)} local test), no val")


def check_freeze(optical, k):
    """freeze_optical=k: the stem and the first k stages have no trainable params, the rest all do."""
    stem = [p.requires_grad for p in optical.stem.parameters()]
    last = [p.requires_grad for p in list(optical.parameters())[-4:]]
    assert all(last) and (not any(stem) if k else all(stem)), (k, stem, last)
    if k == 0:
        assert all(p.requires_grad for p in optical.parameters())


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

    check_input_norm(cfg)
    model = build_model(cfg)
    seg = cfg["experiment"].startswith("seg_")
    if seg:
        assert isinstance(model, SegModel) and model.scale == float(cfg["model"]["input_scale"])
    else:
        assert isinstance(model.transform, ScaledTransform)
        assert model.transform.scale == float(cfg["model"]["input_scale"])
        check_scaled_transform(model.transform)
    bb = model.backbone
    assert bb.swin.stem.in_channels == len(exp_swin), name
    check_freeze(bb.swin, int(cfg["model"]["freeze_optical"]))
    if exp_ter:
        assert bb.terrain.stem.in_channels == len(exp_ter), name
    x = images[:, channels]
    assert x.shape[1] == len(exp_swin) + len(exp_ter)

    model.eval()
    with torch.no_grad():
        sw, te = bb.stages(x)
        for f, c, st in zip(sw, bb.swin.out_channels_list, STRIDES):
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

    assert (bb.gate is not None) == bool(cfg["model"].get("band_gate")), name
    if cfg["model"].get("boundary_head"):
        assert isinstance(model.roi_heads, BoundaryRoIHeads), name
    picked = None if seg else spy_ohem(model, cfg)
    model.train()
    losses = model(list(x), [{k: v for k, v in t.items() if k != "orig_size"} for t in targets])
    assert ("loss_boundary" in losses) == bool(cfg["model"].get("boundary_head")), (name, list(losses))
    if picked is not None:
        assert picked and all(0 < n <= cfg["model"]["ohem_rois"] for n in picked[-1]), (name, picked)
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
    check_ohem_topk()
    check_seg()
    check_gate_boundary()
    check_fold_all([o for o in args.override if o.startswith("data.")])
    try:
        build_model(load_config("configs/e5.yaml", ["model.optical=resnet18", "data.swin_norm=satlas",
                                                     "model.pretrained=false"]))
        raise AssertionError("satlas norm with an ImageNet optical network must be rejected")
    except ValueError:
        pass
    images, targets = fake_batch()
    assert images.shape == (2, 8, HP, WP)
    for case in CASES:
        run_case(*case, args.pretrained, images, targets, args.override)
    print(f"\nsmoke test passed: {len(CASES)} configurations")


if __name__ == "__main__":
    main()
