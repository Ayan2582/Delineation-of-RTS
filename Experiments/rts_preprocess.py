"""
Preprocessing for variable-size, NaN-containing multiband tiles (RTS instance segmentation).

Pipeline per tile (order matters):
    1. fill_nan_neighbor_mean   -> NaN replaced by the mean of valid 3x3 neighbours, iterated inward
    2. normalize                -> per-channel (x - mean) / std, stats from TRAIN split only
    3. pad_collate              -> pad to batch max (multiple of 32) with 0
                                   (0 after normalisation == the channel mean, so this IS mean padding)

Shapes: images are (C, H, W) float32 numpy arrays; masks are (N, H, W) uint8.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import convolve

# ----------------------------------------------------------------------------- 1. NaN fill

_KERNEL = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]], dtype=np.float64)   # 8 neighbours, centre excluded


def fill_nan_neighbor_mean(img: np.ndarray,
                           max_iter: int = 200,
                           fallback: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Replace NaNs with the mean of their valid 8-neighbours.

    One pass only fills NaNs that touch a valid pixel, so it repeats, filling
    holes from the edge inward (like ice closing over a pond).

    Args:
        img:      (C, H, W) array with NaNs.
        max_iter: cap on passes; leftovers (huge voids) go to `fallback`.
        fallback: (C,) per-channel value for pixels still NaN after max_iter,
                  or for an all-NaN channel. Default: the train-set channel mean
                  you pass in; if None, the tile's own channel mean, else 0.

    Returns:
        filled: (C, H, W) float32, no NaNs.
        valid:  (C, H, W) bool, True where the ORIGINAL value was real.
    """
    img = img.astype(np.float64, copy=True)
    valid = ~np.isnan(img)

    for c in range(img.shape[0]):
        band = img[c]
        nan = np.isnan(band)
        if not nan.any():
            continue
        for _ in range(max_iter):
            if not nan.any():
                break
            v = (~nan).astype(np.float64)
            s = convolve(np.where(nan, 0.0, band), _KERNEL, mode="constant", cval=0.0)
            n = convolve(v, _KERNEL, mode="constant", cval=0.0)
            fillable = nan & (n > 0)
            if not fillable.any():          # channel is entirely NaN
                break
            band[fillable] = s[fillable] / n[fillable]
            nan = nan & ~fillable

        if nan.any():                       # leftover: all-NaN band or giant void
            if fallback is not None:
                fv = fallback[c]
            elif (~nan).any():
                fv = band[~nan].mean()
            else:
                fv = 0.0
            band[nan] = fv
        img[c] = band

    return img.astype(np.float32), valid


# ----------------------------------------------------------------------------- 2. stats + normalise

def compute_channel_stats(train_images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over the TRAIN split, ignoring NaNs. Run on RAW (unfilled) tiles."""
    C = train_images[0].shape[0]
    total = np.zeros(C); sq = np.zeros(C); count = np.zeros(C)
    for im in train_images:
        x = im.reshape(C, -1).astype(np.float64)
        m = ~np.isnan(x)
        total += np.where(m, x, 0).sum(1)
        sq += np.where(m, x * x, 0).sum(1)
        count += m.sum(1)
    mean = total / np.maximum(count, 1)
    std = np.sqrt(np.maximum(sq / np.maximum(count, 1) - mean ** 2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((img - mean[:, None, None]) / std[:, None, None]).astype(np.float32)


# ----------------------------------------------------------------------------- 3. mean padding (collate)

def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def pad_collate(batch, size_divisible: int = 32):
    """Pad every image in the batch to the same (H, W), bottom/right only.

    Bottom/right padding keeps box and mask coordinates unchanged (the origin
    stays at the top-left). Images are already normalised, so padding with 0
    is padding with the channel mean.

    batch: list of (image (C,H,W) tensor, target dict with 'boxes','labels','masks')
    """
    H = _round_up(max(im.shape[1] for im, _ in batch), size_divisible)
    W = _round_up(max(im.shape[2] for im, _ in batch), size_divisible)

    images, targets = [], []
    for im, t in batch:
        ph, pw = H - im.shape[1], W - im.shape[2]
        images.append(F.pad(im, (0, pw, 0, ph), value=0.0))          # mean pad
        t = dict(t)
        if "masks" in t and t["masks"].numel():
            t["masks"] = F.pad(t["masks"], (0, pw, 0, ph), value=0)  # padded area = background
        t["orig_size"] = torch.tensor([im.shape[1], im.shape[2]])    # to crop predictions back
        targets.append(t)
    return torch.stack(images), targets


# ----------------------------------------------------------------------------- torchvision Mask R-CNN gotcha

def disable_internal_resize(model, size_divisible: int = 32):
    """torchvision's Mask R-CNN resizes every image to min_size=800 and applies
    ImageNet mean/std by default. That would (a) rescale your 4 m pixels and
    (b) re-normalise already-normalised terrain bands. Turn both off; the
    model's own padding (zeros) is still mean padding because inputs are normalised.
    """
    from torchvision.models.detection.transform import GeneralizedRCNNTransform

    class NoResize(GeneralizedRCNNTransform):
        def resize(self, image, target=None):
            return image, target

    C = 3  # only used to build placeholder mean/std; overridden below
    t = NoResize(min_size=1, max_size=10_000,
                 image_mean=[0.0] * C, image_std=[1.0] * C,
                 size_divisible=size_divisible)
    t.normalize = lambda image: image        # inputs already standardised
    model.transform = t
    return model


# ----------------------------------------------------------------------------- dataset

class RTSDataset(torch.utils.data.Dataset):
    """Expects items as (raw_image (C,H,W) with NaNs, masks (N,H,W), labels (N,)).

    NaN filling is deterministic, so do it ONCE and cache (here: at init).
    Augmentations go in `transform`, applied after fill + normalise.
    """

    def __init__(self, items, mean, std, transform=None):
        self.mean, self.std, self.transform = mean, std, transform
        self.cache = []
        for img, masks, labels in items:
            filled, _ = fill_nan_neighbor_mean(img, fallback=mean)
            self.cache.append((normalize(filled, mean, std), masks, labels))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        img, masks, labels = self.cache[i]
        img = torch.from_numpy(img)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        if self.transform:
            img, masks = self.transform(img, masks)
        keep = masks.flatten(1).bool().any(1)                     # drop instances cropped away
        masks, labels = masks[keep], torch.as_tensor(labels)[keep]
        boxes = _masks_to_boxes(masks)
        return img, {"boxes": boxes, "labels": labels.long(), "masks": masks}


def _masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    if masks.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    from torchvision.ops import masks_to_boxes
    b = masks_to_boxes(masks.bool())
    b[:, 2:] += 1                                            # make width/height >= 1 px
    return b
