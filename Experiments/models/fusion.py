"""Late-fusion blocks: F = ReLU(GN(Conv1x1(concat[O, T])))."""
from __future__ import annotations

import torch
import torch.nn as nn


class FuseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, gn_groups: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.GroupNorm(gn_groups, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, optical, terrain):
        return self.act(self.norm(self.conv(torch.cat([optical, terrain], 1))))


class MultiLevelFusion(nn.Module):
    """E4: one FuseBlock per level C2-C5. Out channels = in channels (192/384/768/1536)."""

    def __init__(self, optical_channels, terrain_channels, gn_groups: int = 32):
        super().__init__()
        self.blocks = nn.ModuleList(FuseBlock(o + t, o + t, gn_groups)
                                    for o, t in zip(optical_channels, terrain_channels))
        self.out_channels_list = [o + t for o, t in zip(optical_channels, terrain_channels)]

    def forward(self, optical, terrain):
        return [b(o, t) for b, o, t in zip(self.blocks, optical, terrain)]


class C5Fusion(nn.Module):
    """E3: fuse only C5 (1024 + 512) into a single 256-channel map."""

    def __init__(self, optical_c5: int, terrain_c5: int, out_channels: int = 256, gn_groups: int = 32):
        super().__init__()
        self.block = FuseBlock(optical_c5 + terrain_c5, out_channels, gn_groups)
        self.out_channels_list = [out_channels]

    def forward(self, optical, terrain):
        return [self.block(optical[-1], terrain[-1])]
