"""Stage extractors: optical (model.optical: Satlas Swin-V2-B by default, or a smaller ImageNet
Swin-V2-T / ResNet) and timm ResNet18 with GroupNorm (terrain).

Both return [C2, C3, C4, C5] as NCHW tensors at strides 4/8/16/32.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

SWIN_CHANNELS = [128, 256, 512, 1024]
TERRAIN_CHANNELS = [64, 128, 256, 512]


def expand_stem(conv: nn.Conv2d, k: int) -> nn.Conv2d:
    """3-channel pretrained stem -> k channels.

    RGB weights are copied, each extra channel gets the mean of the RGB weights, then ALL
    weights are scaled by 3/k. The sum over input channels is unchanged, so a constant input
    produces the same activation as before. The bias is kept as is.
    """
    w = conv.weight.data
    if w.shape[1] != 3:
        raise ValueError(f"expected a 3-channel stem, got {w.shape[1]}")
    if k < 3:
        raise ValueError(f"cannot expand a stem to {k} < 3 channels")
    if k == 3:
        return conv
    extra = w.mean(1, keepdim=True).expand(-1, k - 3, -1, -1)
    new = nn.Conv2d(k, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    conv.dilation, conv.groups, conv.bias is not None)
    new.weight.data.copy_(torch.cat([w, extra], 1) * (3 / k))
    if conv.bias is not None:
        new.bias.data.copy_(conv.bias.data)
    return new


OPTICAL = ("swin_v2_b", "swin_v2_t", "resnet50", "resnet34", "resnet18")


def _freeze(modules) -> None:
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(False)


class SwinStages(nn.Module):
    """Swin-V2 stages. Stage outputs come from features[1], [3], [5], [7], NHWC -> NCHW.

    swin_v2_b: torchgeo, Swin_V2_B_Weights.NAIP_RGB_SI_SATLAS (aerial RGB), 87M params.
    swin_v2_t: torchvision, ImageNet-1k, 28M params.
    `freeze` = stages kept fixed (patch embed + stage 1 = 1, ... all four = 4).
    """

    STAGES = (1, 3, 5, 7)

    def __init__(self, in_chans: int, pretrained: bool = True, arch: str = "swin_v2_b", freeze: int = 0):
        super().__init__()
        if arch == "swin_v2_b":
            from torchgeo.models import Swin_V2_B_Weights, swin_v2_b
            m = swin_v2_b(weights=Swin_V2_B_Weights.NAIP_RGB_SI_SATLAS if pretrained else None)
            self.out_channels_list = SWIN_CHANNELS
        elif arch == "swin_v2_t":
            from torchvision.models import Swin_V2_T_Weights, swin_v2_t
            m = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None)
            self.out_channels_list = [96, 192, 384, 768]
        else:
            raise ValueError(f"unknown Swin arch {arch!r}")
        m.features[0][0] = expand_stem(m.features[0][0], in_chans)
        self.features = m.features                      # norm/avgpool/head are not used
        self.in_chans = in_chans
        _freeze(self.features[: 2 * freeze])

    @property
    def stem(self) -> nn.Conv2d:
        return self.features[0][0]

    def forward(self, x):
        outs = []
        for i, f in enumerate(self.features):
            x = f(x)
            if i in self.STAGES:
                outs.append(x.permute(0, 3, 1, 2).contiguous())
        return outs


def frozen_bn(module: nn.Module) -> nn.Module:
    """Every BatchNorm2d -> torchvision FrozenBatchNorm2d with the same (ImageNet) statistics.
    Batch size 2 is far too small to train BatchNorm, as in torchvision's own detection backbones."""
    from torchvision.ops.misc import FrozenBatchNorm2d

    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            f = FrozenBatchNorm2d(child.num_features, child.eps)
            for k in ("weight", "bias", "running_mean", "running_var"):
                getattr(f, k).copy_(getattr(child, k).data)
            setattr(module, name, f)
        else:
            frozen_bn(child)
    return module


class OpticalResNet(nn.Module):
    """timm ImageNet ResNet (18/34/50) for the optical stream, FrozenBatchNorm, stem expanded
    like the Swin stem. `freeze` = stages kept fixed (stem + layer1 = 1, ... all four = 4)."""

    def __init__(self, in_chans: int, pretrained: bool = True, arch: str = "resnet50", freeze: int = 0):
        super().__init__()
        self.body = frozen_bn(timm.create_model(arch, pretrained=pretrained, features_only=True,
                                                out_indices=(1, 2, 3, 4)))
        self.body.conv1 = expand_stem(self.body.conv1, in_chans)
        self.in_chans = in_chans
        self.out_channels_list = self.body.feature_info.channels()
        stages = [nn.ModuleList([self.body.conv1, self.body.bn1]), self.body.layer1, self.body.layer2,
                  self.body.layer3, self.body.layer4]
        _freeze(stages[: freeze + 1] if freeze else [])

    @property
    def stem(self) -> nn.Conv2d:
        return self.body.conv1

    def forward(self, x):
        return self.body(x)


def optical_backbone(arch: str, in_chans: int, pretrained: bool, freeze: int = 0) -> nn.Module:
    if arch not in OPTICAL:
        raise ValueError(f"model.optical must be one of {OPTICAL}, got {arch!r}")
    if not 0 <= freeze <= 4:
        raise ValueError(f"model.freeze_optical must be 0-4, got {freeze}")
    cls = SwinStages if arch.startswith("swin") else OpticalResNet
    return cls(in_chans, pretrained, arch, freeze)


def _group_norm(groups: int):
    def make(num_channels, **kwargs):                  # timm also passes device=/dtype=
        return nn.GroupNorm(groups, num_channels, **kwargs)
    return make


class TerrainResNet18(nn.Module):
    """timm resnet18 trained from scratch, with GroupNorm(16) instead of BatchNorm."""

    def __init__(self, in_chans: int, gn_groups: int = 16):
        super().__init__()
        self.body = timm.create_model("resnet18", pretrained=False, features_only=True,
                                      out_indices=(1, 2, 3, 4), in_chans=in_chans,
                                      norm_layer=_group_norm(gn_groups))
        assert self.body.feature_info.channels() == TERRAIN_CHANNELS
        assert not any(isinstance(m, nn.BatchNorm2d) for m in self.body.modules())
        self.in_chans = in_chans
        self.out_channels_list = TERRAIN_CHANNELS

    @property
    def stem(self) -> nn.Conv2d:
        return self.body.conv1

    def forward(self, x):
        return self.body(x)
