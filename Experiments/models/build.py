"""build_model(cfg): one factory for E1-E4.

    single_fpn (E1) / early_fpn (E2):  Swin ------------------------------> FPN P2-P6 -> head
    late_c5 (E3):                      Swin + ResNet18 -> fuse C5 -> 256-ch map -----> head
    late_fpn (E4):                     Swin + ResNet18 -> fuse C2-C5 ---> FPN P2-P6 -> head

The head is the same for every experiment. Only two things follow from FPN vs single map:
the anchors per location (3 vs 15, so RPNHead's conv widths) and the RoIAlign featmap names.
"""
from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from config import resolve_streams
from models.backbones import SwinStages, TerrainResNet18
from models.fusion import C5Fusion, MultiLevelFusion
from rts_preprocess import disable_internal_resize

EXPERIMENTS = ("single_fpn", "early_fpn", "late_c5", "late_fpn")


class RTSBackbone(nn.Module):
    """Input (B, n_swin + n_terrain, H, W) -> OrderedDict of `out_channels`-wide maps."""

    def __init__(self, swin: SwinStages, terrain: TerrainResNet18 | None = None,
                 fusion: nn.Module | None = None, fpn: FeaturePyramidNetwork | None = None,
                 out_channels: int = 256):
        super().__init__()
        self.swin, self.terrain, self.fusion, self.fpn = swin, terrain, fusion, fpn
        self.n_swin = swin.in_chans
        self.out_channels = out_channels

    def stages(self, x):
        """Raw per-stream C2-C5 (before fusion/FPN); used by the smoke test."""
        s = self.swin(x[:, : self.n_swin])
        t = self.terrain(x[:, self.n_swin:]) if self.terrain is not None else None
        return s, t

    def forward(self, x):
        s, t = self.stages(x)
        feats = self.fusion(s, t) if self.fusion is not None else s
        feats = OrderedDict((str(i), f) for i, f in enumerate(feats))
        return self.fpn(feats) if self.fpn is not None else feats


def build_backbone(cfg: dict) -> tuple[RTSBackbone, bool]:
    """Returns (backbone, uses_fpn)."""
    exp, m = cfg["experiment"], cfg["model"]
    if exp not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {exp!r}; expected one of {EXPERIMENTS}")
    streams = resolve_streams(cfg)
    two_stream = exp in ("late_c5", "late_fpn")
    if two_stream != bool(streams["terrain"]):
        raise ValueError(f"{exp} needs {'a' if two_stream else 'no'} terrain stream, got {streams}")

    swin = SwinStages(len(streams["swin"]), pretrained=m["pretrained"])
    terrain = TerrainResNet18(len(streams["terrain"]), m["terrain_gn_groups"]) if two_stream else None
    fusion = None
    if exp == "late_c5":
        fusion = C5Fusion(swin.out_channels_list[-1], terrain.out_channels_list[-1],
                          m["fpn_channels"], m["fuse_gn_groups"])
    elif exp == "late_fpn":
        fusion = MultiLevelFusion(swin.out_channels_list, terrain.out_channels_list, m["fuse_gn_groups"])

    uses_fpn = exp != "late_c5"
    fpn = None
    if uses_fpn:
        in_list = fusion.out_channels_list if fusion is not None else swin.out_channels_list
        fpn = FeaturePyramidNetwork(in_list, m["fpn_channels"], extra_blocks=LastLevelMaxPool())
    return RTSBackbone(swin, terrain, fusion, fpn, m["fpn_channels"]), uses_fpn


def build_model(cfg: dict) -> MaskRCNN:
    m = cfg["model"]
    backbone, uses_fpn = build_backbone(cfg)
    ch = backbone.out_channels
    ratios = tuple(float(r) for r in m["aspect_ratios"])
    sizes = tuple(int(s) for s in m["anchor_sizes"])

    if uses_fpn:   # one size per level on P2-P6
        anchors = AnchorGenerator(tuple((s,) for s in sizes), (ratios,) * len(sizes))
        roi_maps = ["0", "1", "2", "3"]   # P2-P5, as in torchvision's FPN Mask R-CNN
    else:          # all sizes on the single fused C5 map
        anchors = AnchorGenerator((sizes,), (ratios,))
        roi_maps = ["0"]

    model = MaskRCNN(
        backbone,
        num_classes=None,
        rpn_anchor_generator=anchors,
        rpn_head=RPNHead(ch, anchors.num_anchors_per_location()[0]),
        box_roi_pool=MultiScaleRoIAlign(roi_maps, m["box_roi"], sampling_ratio=2),
        box_head=TwoMLPHead(ch * m["box_roi"] ** 2, m["box_fc"]),
        box_predictor=FastRCNNPredictor(m["box_fc"], m["num_classes"]),
        mask_roi_pool=MultiScaleRoIAlign(roi_maps, m["mask_roi"], sampling_ratio=2),
        mask_head=MaskRCNNHeads(ch, [256] * m["mask_convs"], 1),
        mask_predictor=MaskRCNNPredictor(256, 256, m["num_classes"]),   # 14 -> 28 deconv
    )
    if m.get("mask_logits_init_std"):
        # torchvision inits the 1x1 mask logits with kaiming fan_out (std 1 for 2 classes),
        # so the initial mask loss is about 10. Opt-in detectron2-style init; same for every experiment.
        logits = model.roi_heads.mask_predictor.mask_fcn_logits
        nn.init.normal_(logits.weight, std=float(m["mask_logits_init_std"]))
        nn.init.zeros_(logits.bias)
    disable_internal_resize(model)
    assert type(model.transform).__name__ == "NoResize", "internal resize must be disabled"
    return model


def param_groups(model: MaskRCNN, lr: dict) -> list[dict]:
    """Swin at lr.swin, terrain ResNet18 at lr.terrain, everything else at lr.other."""
    groups = {"swin": [], "terrain": [], "other": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = ("swin" if name.startswith("backbone.swin.") else
               "terrain" if name.startswith("backbone.terrain.") else "other")
        groups[key].append(p)
    return [{"name": k, "params": v, "lr": float(lr[k])} for k, v in groups.items() if v]
