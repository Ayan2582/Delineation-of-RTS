"""build_model(cfg): one factory for E1-E6.

    single_fpn (E1) / early_fpn (E2):  Swin ------------------------------> FPN P2-P6 -> head
    late_c5 (E3):                      Swin + ResNet18 -> fuse C5 -> 256-ch map -----> head
    late_fpn (E4):                     Swin + ResNet18 -> fuse C2-C5 ---> FPN P2-P6 -> head
    seg_late_fpn (E5):                 E4's backbone -> semantic head -> connected components (models/seg.py)

Two optional modules on top of any Mask R-CNN experiment (e2_gate / e2_boundary / e6 configs):
    model.band_gate:     learned per-image band trust in front of the backbone (models/gate.py)
    model.boundary_head: boundary branch fused into the mask predictor + boundary loss (models/boundary.py)

"Swin" is the optical stream: model.optical picks its network (Satlas Swin-V2-B by default, or a
smaller ImageNet Swin-V2-T / ResNet), model.freeze_optical how many of its stages stay fixed.

The head is the same for every experiment. Only two things follow from FPN vs single map:
the anchors per location (3 vs 15, so RPNHead's conv widths) and the RoIAlign featmap names.
"""
from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform, resize_boxes
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from config import resolve_streams
from models.backbones import TerrainResNet18, optical_backbone
from models.boundary import BoundaryMaskPredictor, attach as attach_boundary
from models.fusion import C5Fusion, MultiLevelFusion
from models.gate import BandGate
from models.ohem import OHEMRoIHeads
from models.seg import SegModel

EXPERIMENTS = ("single_fpn", "early_fpn", "late_c5", "late_fpn")
SEG_EXPERIMENTS = tuple(f"seg_{e}" for e in EXPERIMENTS if e != "late_c5")   # the seg head needs P2-P5


class ScaledTransform(GeneralizedRCNNTransform):
    """torchvision's transform with a fixed resize by `scale` instead of min_size=800, and no
    normalisation (inputs are already normalised). scale=1 leaves the chip untouched.

    postprocess maps boxes and masks back to the input size, so predictions, scoring and the
    submission stay at the chip's own resolution whatever the scale.
    """

    def __init__(self, scale: float, size_divisible: int = 32):
        super().__init__(min_size=1, max_size=10_000, image_mean=[0.0] * 3, image_std=[1.0] * 3,
                         size_divisible=size_divisible)
        if scale <= 0:
            raise ValueError(f"model.input_scale must be > 0, got {scale}")
        self.scale = float(scale)

    def normalize(self, image):
        return image

    def resize(self, image, target=None):
        if self.scale == 1.0:
            return image, target
        h, w = image.shape[-2:]
        size = (round(h * self.scale), round(w * self.scale))
        image = F.interpolate(image[None], size=size, mode="bilinear", align_corners=False)[0]
        if target is not None:
            target["boxes"] = resize_boxes(target["boxes"], (h, w), size)
            if "masks" in target:
                m = target["masks"]
                target["masks"] = (F.interpolate(m[:, None].float(), size=size, mode="nearest")[:, 0].to(m.dtype)
                                   if len(m) else m.new_zeros((0, *size)))
        return image, target


class RTSBackbone(nn.Module):
    """Input (B, n_swin + n_terrain, H, W) -> OrderedDict of `out_channels`-wide maps."""

    def __init__(self, swin: nn.Module, terrain: TerrainResNet18 | None = None,
                 fusion: nn.Module | None = None, fpn: FeaturePyramidNetwork | None = None,
                 out_channels: int = 256, gate: BandGate | None = None):
        super().__init__()
        self.swin, self.terrain, self.fusion, self.fpn, self.gate = swin, terrain, fusion, fpn, gate
        self.n_swin = swin.in_chans
        self.out_channels = out_channels

    def stages(self, x):
        """Raw per-stream C2-C5 (before fusion/FPN); used by the smoke test."""
        if self.gate is not None:                      # every band of both streams
            x = self.gate(x)
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
    if exp not in EXPERIMENTS + SEG_EXPERIMENTS:
        raise ValueError(f"unknown experiment {exp!r}; expected one of {EXPERIMENTS + SEG_EXPERIMENTS}")
    exp = exp.removeprefix("seg_")                     # E5 reuses a Mask R-CNN experiment's backbone
    streams = resolve_streams(cfg)
    two_stream = exp in ("late_c5", "late_fpn")
    if two_stream != bool(streams["terrain"]):
        raise ValueError(f"{exp} needs {'a' if two_stream else 'no'} terrain stream, got {streams}")

    arch = m.get("optical", "swin_v2_b")
    if arch != "swin_v2_b" and cfg["data"].get("swin_norm") == "satlas":
        raise ValueError(f"data.swin_norm=satlas is the Satlas Swin-V2-B input; use zscore with {arch}")
    swin = optical_backbone(arch, len(streams["swin"]), m["pretrained"], int(m.get("freeze_optical", 0)))
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
    gate = (BandGate(len(streams["swin"]) + len(streams["terrain"]), int(m.get("band_gate_hidden", 32)))
            if m.get("band_gate") else None)
    return RTSBackbone(swin, terrain, fusion, fpn, m["fpn_channels"], gate), uses_fpn


def build_model(cfg: dict) -> MaskRCNN | SegModel:
    m = cfg["model"]
    backbone, uses_fpn = build_backbone(cfg)
    if cfg["experiment"] in SEG_EXPERIMENTS:
        if m.get("boundary_head"):
            raise ValueError("model.boundary_head is a Mask R-CNN mask-head option; E5 has no mask head")
        s = resolve_streams(cfg)
        return SegModel(backbone, len(s["swin"]) + len(s["terrain"]), m)
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
        mask_predictor=(BoundaryMaskPredictor(256, 256, m["num_classes"], int(m.get("boundary_dim", 128)))
                        if m.get("boundary_head") else MaskRCNNPredictor(256, 256, m["num_classes"])),  # 14 -> 28 deconv
    )
    if m.get("mask_logits_init_std"):
        # torchvision inits the 1x1 mask logits with kaiming fan_out (std 1 for 2 classes),
        # so the initial mask loss is about 10. Opt-in detectron2-style init; same for every experiment.
        logits = model.roi_heads.mask_predictor.mask_fcn_logits
        nn.init.normal_(logits.weight, std=float(m["mask_logits_init_std"]))
        nn.init.zeros_(logits.bias)
    if m.get("ohem"):   # hardest RoIs instead of a random 512; same weights, so checkpoints are interchangeable
        OHEMRoIHeads.from_roi_heads(model.roi_heads, m["ohem_rois"], m["ohem_nms"])
    if m.get("boundary_head"):   # after OHEM: wraps whatever RoI heads class is there
        attach_boundary(model.roi_heads, m.get("boundary_weight", 1.0))
    # replaces torchvision's min_size=800 resize + ImageNet normalisation (same for every experiment)
    model.transform = ScaledTransform(m.get("input_scale", 1.0))
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
