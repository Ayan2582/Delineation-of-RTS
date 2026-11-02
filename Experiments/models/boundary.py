"""Boundary-refined mask head (model.boundary_head), after BMask R-CNN (Cheng et al., ECCV 2020).

On the local test set AP50 is ~0.3 but AP75 ~0.08: RTS are found but outlined loosely, and the
scorer averages IoU 0.5:0.95. A boundary branch next to the mask predictor learns the outline
explicitly and feeds it back into the mask:

    mask_head features f (256 x 14 x 14)
      boundary branch: b = 2 x (conv3x3 + ReLU)(f)          -> deconv -> 1 logit (28 x 28, class-agnostic)
      mask branch:     f + conv1x1(b) (zero-initialised)    -> deconv -> per-class logits (28 x 28)

    loss_boundary = boundary_weight * (BCE + Dice) against the GT mask's outline in each RoI
    (dilate(m) - erode(m), 3x3, on the 28 x 28 target torchvision's mask loss already uses)

The fusion conv starts at zero, so at initialisation the mask branch equals torchvision's
MaskRCNNPredictor (same `conv5_mask` / `mask_fcn_logits` names). The mask loss and inference are
unchanged; the boundary logits are only used in training.

`attach(roi_heads, weight)` turns existing RoIHeads (plain or OHEM) into boundary ones in place.
torchvision computes the mask loss inside RoIHeads.forward with the module-level `maskrcnn_loss`;
during our forward it is swapped for a version that also scores the boundary branch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.detection.roi_heads as tv_roi_heads
from torchvision.models.detection.roi_heads import RoIHeads, maskrcnn_loss, project_masks_on_boxes


class BoundaryMaskPredictor(nn.Module):
    def __init__(self, in_channels: int, dim: int, num_classes: int, boundary_dim: int = 128):
        super().__init__()
        self.boundary = nn.Sequential(nn.Conv2d(in_channels, boundary_dim, 3, padding=1), nn.ReLU(inplace=True),
                                      nn.Conv2d(boundary_dim, boundary_dim, 3, padding=1), nn.ReLU(inplace=True))
        self.fuse = nn.Conv2d(boundary_dim, in_channels, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        self.boundary_up = nn.ConvTranspose2d(boundary_dim, boundary_dim, 2, 2)
        self.boundary_logits = nn.Conv2d(boundary_dim, 1, 1)
        self.conv5_mask = nn.ConvTranspose2d(in_channels, dim, 2, 2)    # as MaskRCNNPredictor
        self.mask_fcn_logits = nn.Conv2d(dim, num_classes, 1)
        for m in (self.conv5_mask, self.mask_fcn_logits, self.boundary_up, self.boundary_logits):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(m.bias)
        self.last_boundary = None

    def forward(self, x):
        b = self.boundary(x)
        self.last_boundary = self.boundary_logits(F.relu(self.boundary_up(b)))[:, 0]   # (N, 28, 28)
        return self.mask_fcn_logits(F.relu(self.conv5_mask(x + self.fuse(b))))


def boundary_targets(gt_masks, proposals, matched_idxs, size: int) -> torch.Tensor:
    """GT outline per positive RoI, (N, size, size) in {0, 1}: dilate - erode of the projected mask."""
    m = torch.cat([project_masks_on_boxes(g, p, i, size) for g, p, i in zip(gt_masks, proposals, matched_idxs)])
    m = (m >= 0.5).float()[:, None]
    dil = F.max_pool2d(m, 3, 1, 1)
    ero = -F.max_pool2d(-m, 3, 1, 1)
    return (dil - ero)[:, 0]


def boundary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + soft Dice over all RoIs, as in BMask R-CNN."""
    if target.numel() == 0:
        return logits.sum() * 0
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    dice = 1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)
    return bce + dice.mean()


class BoundaryRoIHeads:
    """Mixin put in front of the RoI heads' own class by `attach`."""

    def forward(self, features, proposals, image_shapes, targets=None):
        if not self.training:
            return super().forward(features, proposals, image_shapes, targets)
        extra = {}

        def mask_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs):
            target = boundary_targets(gt_masks, mask_proposals, pos_matched_idxs, mask_logits.shape[-1])
            extra["loss_boundary"] = self.boundary_weight * boundary_loss(self.mask_predictor.last_boundary, target)
            return maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)

        tv_roi_heads.maskrcnn_loss = mask_loss
        try:
            result, losses = super().forward(features, proposals, image_shapes, targets)
        finally:
            tv_roi_heads.maskrcnn_loss = maskrcnn_loss
        losses.update(extra)
        return result, losses


def attach(heads: RoIHeads, weight: float) -> RoIHeads:
    """Make `heads` (RoIHeads or a subclass such as OHEMRoIHeads) compute the boundary loss, in place.
    Its mask_predictor must be a BoundaryMaskPredictor."""
    if not isinstance(heads.mask_predictor, BoundaryMaskPredictor):
        raise TypeError("boundary RoI heads need a BoundaryMaskPredictor")
    if not isinstance(heads, BoundaryRoIHeads):
        heads.__class__ = type(f"Boundary{type(heads).__name__}", (BoundaryRoIHeads, type(heads)), {})
    heads.boundary_weight = float(weight)
    return heads
