"""Band gate (model.band_gate): learned, per-image trust in each input band.

The competition test sites differ from the train sites in NIR and the DEM bands, not in RGB
(adversarial validation AUC 0.76). The gate looks at each band's per-image statistics and scales
the band about a learned neutral value before the backbone:

    stats  = [mean_b, std_b] per band over the chip's real pixels (padding excluded)
    g      = 2 * sigmoid(MLP(stats))          in (0, 2); the last layer is zero-initialised, so g = 1
    x_b'   = n_b + g_b * (x_b - n_b)          n_b learned, one per band

At initialisation g = 1 and the gate is the identity, so the pretrained backbone sees exactly the
input it saw before. g -> 0 replaces a band by its neutral value (a band the model learns to
distrust); g > 1 sharpens it. Padding (0 in every band, from pad_collate) stays 0.
Trained together with aug.band_drop_p, which teaches it what a missing band looks like.
`last_gate` holds the latest (B, C) gate values, detached, for the band-weight plots.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BandGate(nn.Module):
    def __init__(self, n_bands: int, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * n_bands, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, n_bands))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.neutral = nn.Parameter(torch.zeros(n_bands))
        self.last_gate = None

    def forward(self, x):
        valid = (x != 0).any(1, keepdim=True).to(x.dtype)               # (B, 1, H, W); padding is 0 everywhere
        n = valid.sum((2, 3)).clamp(min=1)
        mean = (x * valid).sum((2, 3)) / n
        var = (((x - mean[..., None, None]) ** 2) * valid).sum((2, 3)) / n
        g = 2 * torch.sigmoid(self.mlp(torch.cat([mean, (var + 1e-6).sqrt()], 1)))   # (B, C)
        self.last_gate = g.detach()
        nb = self.neutral[None, :, None, None]
        return (nb + g[..., None, None] * (x - nb)) * valid
