"""E5: semantic segmentation instead of Mask R-CNN; instances are connected components.

In the training labels every RTS is one connected component and no two RTS touch, so
"RTS / not RTS" per pixel, split into connected components, recovers the instances exactly.

    input -> (x input_scale) -> backbone (same RTSBackbone as E1/E2/E4: C2-C5 [+ fusion] -> FPN P2-P5)
          -> SemanticFPNHead: P2-P5 to stride 4, summed -> + stride-2 detail branch from the input
          -> 1 logit per pixel -> back to chip size
    train: BCE (U-Net gap weights between nearby RTS) + soft Dice
    eval:  sigmoid >= seg_thresh -> 8-connected components -> one detection each, score = mean prob

forward() has torchvision's detection interface (loss dict in train mode; per image boxes / labels /
scores / masks (N, 1, H, W) in eval mode), so train.py, eval.py and submission.py run unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torchvision.ops import masks_to_boxes


def _conv_gn(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.GroupNorm(min(32, cout // 4), cout), nn.ReLU(inplace=True))


class SemanticFPNHead(nn.Module):
    """Panoptic-FPN semantic branch (Kirillov et al. 2019) plus a stride-2 detail branch.

    P_k (stride 2^k) gets max(1, k-2) conv-GN-ReLU blocks, each followed by a 2x upsample until stride 4;
    the four maps are summed. The sum is upsampled to stride 2, concatenated with two convs on the
    input image (stride 2), fused by one conv and turned into logits at stride 2.
    """

    def __init__(self, in_channels: int, in_chans_image: int, channels: int = 128, detail: int = 32):
        super().__init__()
        self.levels = nn.ModuleList()
        for k in range(4):                                       # P2..P5 -> stride 4
            blocks = [_conv_gn(in_channels if i == 0 else channels, channels) for i in range(max(1, k))]
            self.levels.append(nn.ModuleList(blocks))
        self.detail = nn.Sequential(_conv_gn(in_chans_image, detail, 2), _conv_gn(detail, detail))
        self.fuse = _conv_gn(channels + detail, channels // 2)
        self.logits = nn.Conv2d(channels // 2, 1, 1)
        nn.init.normal_(self.logits.weight, std=0.01)
        nn.init.constant_(self.logits.bias, -2.0)                # p ~ 0.12 at start (RTS cover ~6 %)

    def forward(self, feats, image):
        out = 0
        for k, blocks in enumerate(self.levels):
            x = feats[str(k)]
            for i, b in enumerate(blocks):
                x = b(x)
                if k > 0:
                    x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            out = out + x
        d = self.detail(image)
        out = F.interpolate(out, size=d.shape[-2:], mode="bilinear", align_corners=False)
        return self.logits(self.fuse(torch.cat([out, d], 1)))       # stride 2


def gap_weights(masks: np.ndarray, w0: float, sigma: float) -> np.ndarray:
    """U-Net weight map (Ronneberger et al. 2015): background pixels between two RTS get
    1 + w0 * exp(-(d1 + d2)^2 / (2 sigma^2)), d1/d2 = distances to the nearest two instances."""
    h, w = masks.shape[-2:]
    out = np.ones((h, w), np.float32)
    if len(masks) < 2 or w0 <= 0:
        return out
    d = np.stack([ndimage.distance_transform_edt(~m.astype(bool)) for m in masks])
    d.sort(0)
    bg = ~masks.astype(bool).any(0)
    out[bg] += w0 * np.exp(-((d[0] + d[1]) ** 2) / (2 * sigma ** 2))[bg]
    return out


class SegModel(nn.Module):
    def __init__(self, backbone: nn.Module, n_channels: int, cfg_model: dict):
        super().__init__()
        m = cfg_model
        self.backbone = backbone
        self.head = SemanticFPNHead(backbone.out_channels, n_channels, m["seg_channels"])
        self.scale = float(m.get("input_scale", 1.0))
        if self.scale <= 0:
            raise ValueError(f"model.input_scale must be > 0, got {self.scale}")
        self.thresh, self.min_px = float(m["seg_thresh"]), int(m["seg_min_px"])
        self.dice_weight = float(m["seg_dice_weight"])
        self.gap_w0, self.gap_sigma = float(m["seg_gap_w0"]), float(m["seg_gap_sigma"])
        self.max_dets = 100                                          # as torchvision's detections_per_img

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) normalised chips -> (B, 1, H, W) logits at the same size."""
        h, w = x.shape[-2:]
        if self.scale != 1.0:
            x = F.interpolate(x, size=(round(h * self.scale), round(w * self.scale)),
                              mode="bilinear", align_corners=False)
        sh, sw = x.shape[-2:]
        x = F.pad(x, (0, -sw % 32, 0, -sh % 32))                     # 0 = mean padding (inputs normalised)
        y = self.head(self.backbone(x), x)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)[..., :sh, :sw]
        return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False) if (sh, sw) != (h, w) else y

    def forward(self, images, targets=None):
        hw = [im.shape[-2:] for im in images]
        H, W = max(h for h, _ in hw), max(w for _, w in hw)
        x = torch.stack([F.pad(im, (0, W - im.shape[-1], 0, H - im.shape[-2])) for im in images])
        logits = self.logits(x)[:, 0]
        if self.training:
            if targets is None:
                raise ValueError("targets are required in training mode")
            return self.losses(logits, targets)
        return [self.postprocess(p) for p in torch.sigmoid(logits)]

    def losses(self, logits, targets):
        H, W = logits.shape[-2:]
        gts, ws = [], []
        for t in targets:
            m = t["masks"]
            m = F.pad(m, (0, W - m.shape[-1], 0, H - m.shape[-2])) if len(m) else m.new_zeros((0, H, W))
            gts.append(m.bool().any(0) if len(m) else torch.zeros((H, W), dtype=torch.bool, device=logits.device))
            ws.append(torch.from_numpy(gap_weights(m.cpu().numpy(), self.gap_w0, self.gap_sigma)))
        gt = torch.stack(gts).float()
        wt = torch.stack(ws).to(logits.device)
        bce = (F.binary_cross_entropy_with_logits(logits, gt, reduction="none") * wt).sum() / wt.sum()
        p = torch.sigmoid(logits)
        inter, total = (p * gt).flatten(1).sum(1), (p + gt).flatten(1).sum(1)
        dice = 1 - ((2 * inter + 1) / (total + 1)).mean()
        return {"loss_seg_bce": bce, "loss_seg_dice": self.dice_weight * dice}

    @torch.no_grad()
    def postprocess(self, prob: torch.Tensor) -> dict:
        """(H, W) probabilities -> detections, one per 8-connected component >= seg_min_px."""
        p = prob.float().cpu().numpy()
        lab, k = ndimage.label(p >= self.thresh, structure=np.ones((3, 3)))
        if k == 0 and p.max() >= 0.1:            # every chip has an RTS: fall back to the most likely blob
            lab, k = ndimage.label(p >= p.max() / 2, structure=np.ones((3, 3)))
        masks, scores = [], []
        for c in range(1, k + 1):
            cc = lab == c
            if cc.sum() >= self.min_px:
                masks.append(cc)
                scores.append(float(p[cc].mean()))
        dev = prob.device
        if not masks:
            return {"boxes": torch.zeros((0, 4), device=dev), "labels": torch.zeros(0, dtype=torch.int64, device=dev),
                    "scores": torch.zeros(0, device=dev), "masks": torch.zeros((0, 1, *p.shape), device=dev)}
        order = np.argsort(scores)[::-1][: self.max_dets]
        m = torch.from_numpy(np.stack([masks[i] for i in order])).to(dev)
        return {"boxes": masks_to_boxes(m).float(), "labels": torch.ones(len(m), dtype=torch.int64, device=dev),
                "scores": torch.tensor([scores[i] for i in order], device=dev), "masks": m[:, None].float()}
