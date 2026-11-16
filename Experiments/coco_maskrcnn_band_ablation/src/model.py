from __future__ import annotations

import torch
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator


def build_coco_maskrcnn(
    detections_per_image: int = 10,
    min_size: int = 256,
    max_size: int = 512,
    trainable_backbone_layers: int = 3,
    small_object_anchors: bool = True,
):
    """Build full COCO-pretrained Mask R-CNN and adapt its final predictors to RTS.

    COCO_V1 initializes the complete detector (backbone, FPN, RPN, box and mask
    feature heads). The class-specific box predictor and final mask-logit convolution
    are adapted to two classes: background and RTS. The pretrained mask upsampler is kept.

    trainable_backbone_layers=3 freezes the ResNet stem and first residual stage,
    while later backbone stages remain trainable.
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # Replace the 91-class COCO box classifier/regressor with a 2-class RTS head.
    box_in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in_features, num_classes=2)

    # Keep the pretrained mask upsampling/deconvolution layer and replace only
    # the final 91-class logits convolution with a two-class RTS logits layer.
    old_mask_logits = model.roi_heads.mask_predictor.mask_fcn_logits
    model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(
        old_mask_logits.in_channels, 2, kernel_size=1
    )

    # The release chips are small; use a consistent resize for every combination.
    # Keep the standard normalization expected by the COCO-pretrained weights.
    model.transform.min_size = (int(min_size),)
    model.transform.max_size = int(max_size)

    # The competition permits at most ten instances per image.
    model.roi_heads.detections_per_img = int(detections_per_image)

    # Keep the same number of anchors per FPN level as the COCO model (3 ratios x 1
    # scale per level), but add smaller scales suited to the chip-level RTS objects.
    if small_object_anchors:
        model.rpn.anchor_generator = AnchorGenerator(
            sizes=((8,), (16,), (32,), (64,), (128,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )

    return model


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable)}
