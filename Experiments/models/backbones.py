"""Stage extractors: torchgeo Swin-V2-B (optical) and timm ResNet18 with GroupNorm (terrain).

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


class SwinStages(nn.Module):
    """torchgeo swin_v2_b. Stage outputs come from features[1], [3], [5], [7], NHWC -> NCHW."""

    STAGES = (1, 3, 5, 7)

    def __init__(self, in_chans: int, pretrained: bool = True):
        super().__init__()
        from torchgeo.models import Swin_V2_B_Weights, swin_v2_b

        m = swin_v2_b(weights=Swin_V2_B_Weights.NAIP_RGB_SI_SATLAS if pretrained else None)
        m.features[0][0] = expand_stem(m.features[0][0], in_chans)
        self.features = m.features                      # norm/avgpool/head are not used
        self.in_chans = in_chans
        self.out_channels_list = SWIN_CHANNELS

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
