"""Online hard example mining (OHEM, Shrivastava et al., CVPR 2016) for the RoI heads.

torchvision trains the box head on 512 RoIs per image drawn at random (at most 25% foreground).
With OHEM, every proposal plus the GT boxes is scored by the box head without gradients, and per
image the `batch` RoIs with the highest loss (classification + box regression) are kept. NMS over
the candidate boxes, scored by their loss, runs first so near-duplicates of one hard region do not
fill the batch.

Only the choice of RoIs changes. The losses, the mask branch (trained on the selected foreground
RoIs), the weights and inference are torchvision's, so checkpoints load with or without OHEM.
OHEM applies to the RoI heads only; the RPN keeps its own 256-anchor sampler.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import nms


class OHEMRoIHeads(RoIHeads):
    """RoIHeads whose training RoIs are the hardest ones instead of a random sample."""

    @classmethod
    def from_roi_heads(cls, heads: RoIHeads, batch: int, nms_thresh: float) -> "OHEMRoIHeads":
        """Turn an existing RoIHeads into OHEM in place (same modules and weights)."""
        if batch < 1:
            raise ValueError(f"model.ohem_rois must be >= 1, got {batch}")
        heads.__class__ = cls
        heads.ohem_batch, heads.ohem_nms = int(batch), float(nms_thresh)
        heads._ohem_inputs = None
        return heads

    def forward(self, features, proposals, image_shapes, targets=None):
        self._ohem_inputs = (features, image_shapes)        # select_training_samples needs them
        try:
            return super().forward(features, proposals, image_shapes, targets)
        finally:
            self._ohem_inputs = None

    @torch.no_grad()
    def roi_losses(self, proposals, labels, regression_targets) -> list[torch.Tensor]:
        """Per-RoI loss, the same terms as fastrcnn_loss: cross-entropy + smooth L1 (foreground only)."""
        features, image_shapes = self._ohem_inputs
        logits, deltas = self.box_predictor(self.box_head(self.box_roi_pool(features, proposals, image_shapes)))
        labels_all = torch.cat(labels)
        cls = F.cross_entropy(logits, labels_all.clamp(min=0), reduction="none")
        deltas = deltas.reshape(len(labels_all), -1, 4)[torch.arange(len(labels_all)), labels_all.clamp(min=0)]
        box = F.smooth_l1_loss(deltas, torch.cat(regression_targets), beta=1 / 9, reduction="none").sum(1)
        loss = cls + box * (labels_all > 0)
        loss[labels_all < 0] = -float("inf")                  # ignored RoIs are never picked
        return list(loss.split([len(p) for p in proposals]))

    def select_training_samples(self, proposals, targets):
        """torchvision's version with the random subsample replaced by hardest-RoI selection."""
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype, device = proposals[0].dtype, proposals[0].device
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        proposals = self.add_gt_proposals(proposals, gt_boxes)
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        matched_gt_boxes = [(g if g.numel() else torch.zeros((1, 4), dtype=dtype, device=device))[m]
                            for g, m in zip(gt_boxes, matched_idxs)]
        regression_targets = list(self.box_coder.encode(matched_gt_boxes, proposals))

        losses = self.roi_losses(proposals, labels, regression_targets)
        for i, loss in enumerate(losses):
            keep = torch.where(torch.isfinite(loss))[0]
            order = (nms(proposals[i][keep], loss[keep], self.ohem_nms) if self.ohem_nms < 1.0   # by loss, descending
                     else loss[keep].argsort(descending=True))
            keep = keep[order][: self.ohem_batch]
            proposals[i], labels[i] = proposals[i][keep], labels[i][keep]
            matched_idxs[i], regression_targets[i] = matched_idxs[i][keep], regression_targets[i][keep]
        return proposals, matched_idxs, labels, regression_targets
