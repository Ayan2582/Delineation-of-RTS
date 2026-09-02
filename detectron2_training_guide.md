# Training Cascade Mask R-CNN on Kaggle — an RGB-only guide

A working guide to fine-tuning a COCO-pretrained **Cascade Mask R-CNN** (detectron2) on the
2026 GeoAI Arctic Challenge release, using **only bands 0–2 (red, green, blue)**, on Kaggle.

**This document hands you decisions, API names and config keys — you write the code.** There
is no complete pipeline to paste. Where a detail is genuinely non-obvious you get a short
fragment showing the shape of ug 2026the call, not a finished implementation.

Every dataset number quoted here was measured over the actual release, not recalled. Every
detectron2 default quoted was read from `detectron2/config/defaults.py` and
`configs/Base-RCNN-FPN.yaml` on `main`.

**Contents**

1. [Kaggle and installing detectron2](#1-kaggle-and-installing-detectron2)
2. [The data layer — DatasetCatalog](#2-the-data-layer--datasetcatalog)
3. [The custom DatasetMapper](#3-the-custom-datasetmapper--mandatory-even-for-rgb)
4. [Pixel normalisation and the BGR trap](#4-pixel-normalisation-and-the-bgr-trap)
5. [Config for this dataset](#5-config-for-this-dataset)
6. [The training loop](#6-the-training-loop)
7. [Validating against the official metric](#7-validating-against-the-official-metric)
8. [Inference and the submission file](#8-inference-and-the-submission-file)
9. [Debug playbook](#9-debug-playbook)
10. [A suggested order of attack](#10-a-suggested-order-of-attack)
11. [Appendix: adding the other five channels](#appendix-adding-the-other-five-channels)

---

## Why RGB-only is a good place to start

The three visible bands are 8-bit digital numbers spanning the **full `0..255` range**, with a
corpus mean of `(92.31, 96.00, 77.26)` over all 756 training chips. That is the same scale
COCO pretraining expects. So the pretrained stem loads **clean** — no channel surgery, no
`conv1` inflation, no frozen-stem trap. Those are the hardest and most silent failure modes of
the 8-channel version, and this scope removes all of them. The appendix records what changes
when you add the rest.

### Three things that will still silently break

| | assumption detectron2 makes | reality here |
|---|---|---|
| **1** | images are JPEG/PNG, read via PIL | chips are `.npz`; `detection_utils.read_image()` calls `PIL.Image.open` and cannot open them |
| **2** | `mask_format="polygon"` | labels are **COCO RLE dicts**; polygon mode does not handle them |
| **3** | the COCO scorer's defaults | this challenge scores `maxDets=10` with area bins `300` / `2000`, not COCO's `32²` / `96²` |

None of these three raises a clear error at the point of the mistake. Each surfaces later as
"loss falls but AP is zero", which is why they get their own sections.

---

## 1. Kaggle and installing detectron2

**There is no single blessed recipe, and anyone who gives you one without caveats is guessing.**
detectron2's last tagged release is **v0.6 (2021)**, whose prebuilt wheels stop at **torch
1.10**. Kaggle's image ships torch 2.x. So there is no wheel for you: it must be **built from
source**, and the build compiles CUDA extensions against whatever torch is installed *at that
moment*.

### Check before you build

The build bakes in the torch and CUDA versions. Print them first, and note them — if Kaggle
later bumps its base image, your build breaks and this is the first thing to re-check.

```python
import torch, numpy
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(numpy.__version__)
```

Two constraints follow:

- **`numpy < 2`.** Extensions compiled against the NumPy 1.x C API abort at import under NumPy
  2.x with *"A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0"*. Pin it
  before building, not after.
- **`pycocotools` is already on the Kaggle image.** Don't rebuild it; you'll only pull in a
  version compiled against different NumPy headers.

The install itself is the source build:

```
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Expect roughly ten minutes and a wall of dependency-resolver complaints about Kaggle's
preinstalled `google-cloud-*`, `tensorflow` and `cudf` pins. Those warnings are noise here —
detectron2 does not use those packages. What matters is whether the extension compiled.

### The verification that actually means something

```python
import detectron2, detectron2._C   # the second line is the real test
```

`from detectron2 import model_zoo` succeeds even when the CUDA extension failed to build,
because the Python package imports fine on its own. `detectron2._C` is the compiled module;
if it imports, NMS and ROIAlign will work on GPU. Test that, not the former.

### Build once, not every session

This is the Kaggle-specific move that saves you hours. A ten-minute build on every session is
wasted time, and it forces internet on for every run.

1. In one notebook with **internet enabled**, build detectron2 as above.
2. Copy the installed package out of `site-packages` into `/kaggle/working`.
3. **Save that as a Kaggle Dataset.**
4. In subsequent training notebooks, **attach the dataset and put it on `sys.path`** — no
   internet needed, no rebuild.

The same trick handles the pretrained weights: download the model-zoo checkpoint once, save it
in the dataset, and point `cfg.MODEL.WEIGHTS` at the local path. Kaggle notebooks that run with
internet disabled cannot reach `dl.fbaipublicfiles.com`, and this is the usual cause of a run
dying at minute one.

> The version triple — torch / CUDA / numpy — is the thing to check whenever anything breaks
> mysteriously after a Kaggle image update. A cached build from a previous image will import
> and then segfault or throw undefined-symbol errors.

---

## 2. The data layer — `DatasetCatalog`

detectron2 gets data through a registered function returning a **list of dicts**:

```python
DatasetCatalog.register("rts_train", lambda: get_rts_dicts("train"))
MetadataCatalog.get("rts_train").set(thing_classes=["rts"])
```

Each dict describes one image:

| key | value |
|---|---|
| `file_name` | absolute path to the `.npz` |
| `image_id` | int |
| `height`, `width` | ints, the chip's true size |
| `annotations` | list of per-instance dicts |

And each annotation:

| key | value |
|---|---|
| `bbox` | `[x, y, w, h]` from the COCO file, unchanged |
| `bbox_mode` | `BoxMode.XYWH_ABS` — the COCO file is XYWH, and getting this wrong shifts every box |
| `category_id` | **`0`** — see below |
| `segmentation` | the **RLE dict exactly as it appears in the JSON** |

### Pass the RLE through untouched

Do not decode masks at registration time. `annotations_to_instances` decodes them later, in the
mapper, after transforms are known. Decoding 1,783 masks up front wastes memory and produces
masks in the wrong (pre-augmentation) geometry.

### The `category_id` off-by-one

This one silently produces an **empty submission**, so it is worth stating twice.

- Internally, detectron2 wants **contiguous ids starting at 0**. With one class,
  `NUM_CLASSES = 1`, the only valid `category_id` is **`0`**.
- The submission format requires **`category_id: 1`**.

So you map `1 → 0` when building the dataset dicts, and `0 → 1` when writing predictions. If
you register with `category_id: 1` while `NUM_CLASSES = 1`, you are labelling every instance as
class index 1 in a model with only index 0 — which either asserts or trains on nothing.

### Two joins that will bite you

Both are real inconsistencies in the release, not hypotheticals:

- **Path prefixes differ.** `instances_train.json` writes `file_name` as
  `images/train_000001.npz`; `metadata/train_manifest.csv` writes the same chip as
  `train/images/train_000001.npz`. **Join on the basename.**
- **Test ids restart at 1.** `test_000001` has `image_id` 1, and so does `train_000001`. Train
  and test ids collide. Always take submission ids from `metadata/test_manifest.csv`.

### Splitting 756 images

No official validation split is provided. With only 756 chips, a plain random split gives a
noisy validation signal. Instances per image range from **1 to 10** (mean 2.36), and that
distribution is skewed — 291 chips have exactly one instance while only 2 have ten. **Stratify
the split by instance count** so your validation fold isn't accidentally all easy single-slump
chips. Register the two splits as separate datasets (`rts_train`, `rts_val`).

Useful sanity fact: **every training chip has at least one instance.** There are no empty
images, so if your dataloader ever yields an image with zero ground-truth boxes, your code —
not the data — is at fault.

---

## 3. The custom `DatasetMapper` — mandatory even for RGB

The default mapper calls `detection_utils.read_image(dataset_dict["file_name"])`, which does
`PIL.Image.open(f)`. PIL cannot open `.npz`. There is no config flag that fixes this: you must
write a mapper.

A mapper is a callable taking one dataset dict and returning a dict with:

- `"image"` — a **CHW** float tensor
- `"instances"` — an `Instances` object
- `"height"`, `"width"` — the **original** size, used to rescale predictions at inference

### Order of operations

The sequence matters, and step 2 is the one people skip:

1. **Load and slice.** `with np.load(path) as d: arr = d["image"][..., :3]` — bands 0, 1, 2 are
   red, green, blue, in that order.
2. **Kill the NaNs.** `np.nan_to_num(arr, nan=0.0)`. **65 of the 756 chips (8.6%) contain at
   least one NaN in the RGB bands.** One NaN reaching the loss makes the loss NaN, which makes
   every gradient NaN, which makes every weight NaN — the run is dead and the traceback points
   nowhere near the cause. This is not a rare edge case at 8.6%.
3. **Channel order.** RGB or BGR — see [§4](#4-pixel-normalisation-and-the-bgr-trap). Decide
   once, apply identically here and at inference.
4. **Augment**, then build `Instances`.

### Augmentation with a float array

Geometric augmentations are channel-agnostic and safe:

```python
augs = T.AugmentationList([
    T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, "choice"),
    T.RandomFlip(horizontal=True),
])
aug_input = T.AugInput(arr)
transforms = augs(aug_input)          # mutates aug_input in place
arr = aug_input.image
```

`transforms` is the record of what happened, and you must replay it onto the annotations —
`transforms.apply_box(...)` for boxes and `transforms.apply_segmentation(...)` for decoded
masks. Skipping this is the classic "boxes drift away from objects" bug.

Two notes specific to this data. **Vertical flips and rotations are physically reasonable
here** — unlike photographs of the natural world, nadir satellite imagery has no canonical "up",
so you get a free doubling of augmentation strength that a COCO recipe would not use. But be
careful with the **shaded-relief-derived intuition**: illumination direction *is* directional,
which matters if you later add band 5. For RGB it is fine.

And detectron2's photometric augmentations (`RandomBrightness`, `RandomContrast`) assume a
particular value range. On raw `0..255` floats they are usable, but verify the output range
rather than assuming — it is easy to push values negative and quietly clip them later.

### Building `Instances` — the `mask_format` pairing

```python
instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
```

The signature is `annotations_to_instances(annos, image_size, mask_format="polygon")`, and
**the default is wrong for this dataset**. With `mask_format="bitmask"` it calls
`mask_util.decode(segm)` on dict segmentations, which is exactly what RLE labels need. Left on
`"polygon"` it will not handle the RLE dicts.

This must be paired with:

```python
cfg.INPUT.MASK_FORMAT = "bitmask"
```

The config key and the mapper argument are two separate settings that both have to say
`bitmask`. Setting one and not the other is a top cause of "loss decreases, mask AP stays at
zero".

---

## 4. Pixel normalisation and the BGR trap

This is the one place an RGB-only run can still silently lose accuracy.

detectron2's COCO configs assume **BGR** input:

```
_C.INPUT.FORMAT    = "BGR"
_C.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]   # in BGR order
_C.MODEL.PIXEL_STD  = [1.0, 1.0, 1.0]
```

The dataset's bands 0–2 are in **RGB** order. `GeneralizedRCNN.preprocess_image` computes
`(x - pixel_mean) / pixel_std` positionally — it has no idea what your channels mean. Feed RGB
data to a model configured for BGR and you subtract 103.5 from red and 123.7 from blue. Nothing
raises. You just lose accuracy for no reason.

Pick **one** of these and keep it consistent between training and inference:

**Option A — reorder the data, keep the config.** Reverse to BGR in the mapper (`arr[..., ::-1]`)
and leave `INPUT.FORMAT` and `PIXEL_MEAN` at their defaults. Closest to the pretrained setup;
fewest moving parts.

**Option B — keep RGB, reorder the config.** Set `cfg.INPUT.FORMAT = "RGB"` and reverse
`PIXEL_MEAN` to `[123.675, 116.280, 103.530]`.

> `INPUT.FORMAT` is consumed by `read_image`, which your mapper bypasses. Set it correctly
> anyway — it documents intent, and other detectron2 utilities read it.

### COCO statistics or this dataset's?

Measured over all 756 training chips, RGB only, NaNs excluded:

| | R | G | B |
|---|---|---|---|
| **mean** | 92.31 | 96.00 | 77.26 |
| **std** | 37.50 | 30.93 | 30.73 |

Compare the COCO means in RGB order: `(123.675, 116.280, 103.530)`. This imagery is
consistently **darker** than COCO — by roughly 30 DN in every channel — but the same order of
magnitude.

Either choice is defensible. Using the dataset's own mean centres the input slightly better;
using COCO's keeps the input distribution closest to what the pretrained weights saw. The one
thing that is *not* defensible is mixing them, or changing your mind between training and
inference.

Note that `PIXEL_STD` defaults to `[1, 1, 1]` — detectron2's COCO models centre but do **not**
scale. If you switch to the measured stds (~31–38) you are changing the input scale by a factor
of ~35, and you must retune the learning rate. **Recommendation: start with mean-subtraction
only, `PIXEL_STD = [1, 1, 1]`, exactly as the pretrained model expects.**

> **Do not feed the percentile-stretched `[0, 1]` image** from `00_data_basics.ipynb` §9. That
> stretch exists to make a picture legible on screen. The model wants raw `0..255`-scale values
> with the mean subtracted. Feeding it `[0, 1]` data with `PIXEL_MEAN` around 100 subtracts 100
> from values that never exceed 1.

---

## 5. Config for this dataset

Start from the model zoo:

```python
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")
```

That config sets `ROI_HEADS.NAME: CascadeROIHeads` and `CLS_AGNOSTIC_BBOX_REG: True`, and
inherits `configs/Base-RCNN-FPN.yaml`. Reference COCO performance is box AP 44.3 / mask AP 38.5.
`Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml` is the cheaper 1x-schedule alternative.

### The keys that matter here

**`MODEL.ROI_HEADS.NUM_CLASSES = 1`** (default 80). Cascade R-CNN has **three** sequential box
heads at IoU thresholds 0.5/0.6/0.7. `NUM_CLASSES` propagates to all three automatically — this
is why hand-rolled "replace the classifier layer" recipes written for plain Mask R-CNN break on
Cascade. Set the config key and let detectron2 build the heads.

The three heads are also why the checkpoint will report shape mismatches on the class-prediction
layers (81 classes → 2). **Those particular skips are expected and correct.** Read the log
anyway, and confirm the only skipped tensors are the final class/box predictors — anything else
skipped is a bug.

**`INPUT.MASK_FORMAT = "bitmask"`** — see [§3](#3-the-custom-datasetmapper--mandatory-even-for-rgb).

**`INPUT.MIN_SIZE_TRAIN` / `MAX_SIZE_TRAIN`.** The base config uses
`MIN_SIZE_TRAIN: (640, 672, 704, 736, 768, 800)` and `MAX_SIZE_TRAIN: 1333`. Chips here are
tiny — width `150..297`, height `106..295`, in **87 distinct `(w, h)` pairs**. So the default
**upsamples by about 4.2× on average**.

That sounds wasteful, and it is expensive, but it is not wrong — see the anchor discussion
below. It is the single biggest lever on both memory and accuracy, so make it a deliberate
choice and ablate it.

**`MODEL.ANCHOR_GENERATOR.SIZES`.** The FPN base sets `[[32], [64], [128], [256], [512]]` on
P2–P6. **Anchors and input resolution are one coupled decision, not two.** Here is the measured
instance size distribution, in pixels of `sqrt(area)`, under each resize:

| resize | avg scale | p5 | median | p95 | max |
|---|---|---|---|---|---|
| **default 800 / 1333** | **×4.16** | **30** | **113** | **400** | **901** |
| 512 / 853 | ×2.66 | 19 | 72 | 256 | 577 |
| native, no resize | ×1.00 | 8 | 27 | 92 | 197 |

At the default resize the instances span roughly 30–400 px, which lands almost perfectly on the
`32 / 64 / 128 / 256 / 512` anchor ladder. **So leave the anchors alone if you train at the
default resolution** — the stock configuration is already well matched, and the upsampling is
doing useful work by placing objects in the size range the pretrained weights understand.

The corollary matters more: **if you shrink `MIN_SIZE_TRAIN` to save memory, you must shrink
the anchors too.** Training at native resolution with default anchors leaves a median object of
27 px against a smallest anchor of 32 — most of the pyramid does nothing, and recall collapses.
Scale the anchor ladder by the same factor you scaled the input.

**`TEST.DETECTIONS_PER_IMAGE`** (default 100). The scorer uses `maxDets=10` and the data never
exceeds 10 instances per image. Set it to ~10–20. Keeping it at 100 doesn't hurt the metric
(only the top 10 by score count) but wastes time and makes prediction dumps larger.

**`MODEL.ROI_HEADS.SCORE_THRESH_TEST`** (default 0.05). Low is right for AP — see
[§8](#8-inference-and-the-submission-file).

**`SOLVER`.** The base config's `BASE_LR: 0.02` assumes `IMS_PER_BATCH: 16` across 16 GPUs. On
one Kaggle T4 you will run `IMS_PER_BATCH` of 2–4. Scale the LR linearly: at `IMS_PER_BATCH: 2`,
`BASE_LR ≈ 0.0025`. Fine-tuning a pretrained model on 756 images generally wants less than that,
so treat it as an upper bound.

`MAX_ITER` is in **iterations, not epochs**. At 756 images and `IMS_PER_BATCH: 2` one epoch is
378 iterations. Decide your epoch budget, then multiply. Set `STEPS` (the LR decay milestones)
to roughly 2/3 and 8/9 of `MAX_ITER`, mirroring the base config's 60k/80k out of 90k. Keep
`WARMUP_ITERS` — with a small batch and a fresh head, skipping warmup is a common cause of an
early loss explosion.

**`SOLVER.AMP.ENABLED = True`** (default `False`). Roughly halves memory on a T4 and is close
to free. Worth turning on before you reduce resolution.

**`MODEL.BACKBONE.FREEZE_AT = 2`** (the default) freezes the stem and `res2`. Unlike the
8-channel case, this is now **legitimate** — the stem is genuinely pretrained and matches your
input. At 756 images, freezing early layers is a reasonable regulariser. Worth ablating against
`FREEZE_AT = 0`.

**`DATALOADER.NUM_WORKERS = 4`** (the default) is usually about right for Kaggle's CPU
allocation. Raising it on a 4-core box makes throughput worse, not better.

---

## 6. The training loop

### `DefaultTrainer` or your own

`DefaultTrainer` gives you the LR scheduler, checkpointing, logging, and the hook system for
free. Subclass it and override:

```python
class RTSTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=RTSMapper(cfg, is_train=True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=RTSMapper(cfg, is_train=False))
```

Those two overrides are the entire integration point for your mapper. A hand-written loop is
more transparent, but you re-implement warmup, checkpoint resumption and EMA yourself — and on
Kaggle, checkpoint resumption is not optional (below). **Recommendation: subclass
`DefaultTrainer`.**

### Validation loss needs a hook

`DefaultTrainer` evaluates metrics at `TEST.EVAL_PERIOD` but **never computes validation
loss**. Since overfitting is the main risk at 756 images, you want that curve. The standard
solution is a `LossEvalHook`: a `HookBase` subclass whose `after_step` runs the model in
training mode (so it returns a loss dict) over the val loader every N iterations, with
`torch.no_grad()`, and writes the result to the event storage. Register it via
`trainer.register_hooks()` — and insert it **before** the periodic writer hook, or your logged
value lands one step late.

### Kaggle's 12-hour ceiling

The session dies at 12 hours whether or not training finished. Design for that from the start:

- `cfg.OUTPUT_DIR = "/kaggle/working/output"` — only `/kaggle/working` persists as output.
- `SOLVER.CHECKPOINT_PERIOD` well under the limit. A checkpoint every ~30–60 minutes costs
  little and bounds your worst-case loss to that window.
- `trainer.resume_or_load(resume=True)` — with `resume=True` this loads `last_checkpoint` from
  `OUTPUT_DIR` and **restores the iteration count and optimizer state**; with `resume=False` it
  loads `cfg.MODEL.WEIGHTS` and starts from iteration 0. The distinction is what makes chained
  sessions work.
- **Chain sessions**: save `OUTPUT_DIR` as a Kaggle Dataset at the end of a run, attach it next
  session, copy it back into `/kaggle/working`, and resume. This is the same pattern as the
  detectron2 build in [§1](#1-kaggle-and-installing-detectron2).

### On the second GPU

Kaggle offers 2×T4. detectron2's `launch()` spawns processes for DDP, which interacts badly
with notebook kernels. Unless you are prepared to debug that, **use one GPU** and put the
effort into resolution and batch size instead. If you do go multi-GPU, remember `IMS_PER_BATCH`
is the *total* across GPUs, and the LR scaling applies to that total.

---

## 7. Validating against the official metric

**`COCOEvaluator` does not report this challenge's metric.** It reports stock COCO numbers.
The differences are not cosmetic:

| | COCO default | this challenge |
|---|---|---|
| `maxDets` | `[1, 10, 100]` | `[1, 5, 10]` |
| small | area < 32² = 1024 | area < **300** |
| medium | 32²–96² | **300–2000** |
| large | > 96² = 9216 | > **2000** |

Under the official bins the split is **25.3% small / 50.6% medium / 24.1% large**. Under COCO's
bins, most of this dataset would read as "small". So `AP_small` from `COCOEvaluator` is not
`AP_small` from the leaderboard, and the headline AP differs too because of `maxDets`.

Use both, for different purposes:

- **`COCOEvaluator` in-loop** — cheap, automatic, good for watching a training curve. Treat the
  absolute value as arbitrary; watch the trend.
- **`tools/evaluate_coco.py` for decisions** — it already hardcodes `DEFAULT_MAX_DETECTIONS =
  [1, 5, 10]` and `DEFAULT_AREA_RANGES = [[0,1e10], [0,300], [300,2000], [2000,1e10]]`. Dump
  predictions to a COCO-results JSON and score them with it. This is the number to compare
  across experiments.

To use it you need a ground-truth JSON for your validation fold: filter
`instances_train.json`'s `images` and `annotations` to your val ids, keep `categories`
unchanged, and write it out. Then:

```
python tools/evaluate_coco.py --ground-truth val_gt.json --submission val_preds.json
```

The primary ranking metric is the first row it prints:
`AP @[IoU=0.50:0.95 | area=all | maxDets=10]`.

---

## 8. Inference and the submission file

### Don't use `DefaultPredictor`

`DefaultPredictor` reads the image from disk with `read_image` and applies its own
normalisation — it bypasses your mapper entirely, so it cannot read `.npz`. Either run the
model directly over `build_detection_test_loader` with your mapper, or replicate the mapper's
exact preprocessing by hand. **Any divergence between training and inference preprocessing —
channel order, NaN handling, scale — costs accuracy silently.** Reusing the mapper is the safer
of the two.

### Reading the output

Each image yields an `Instances` with:

| field | shape / meaning |
|---|---|
| `pred_masks` | `(N, H, W)` bool tensor, already rescaled to the **original chip size** |
| `scores` | `(N,)` float, descending |
| `pred_classes` | `(N,)` int — all `0` here |

detectron2 rescales masks back to the original resolution using the `height` / `width` you put
in the dataset dict. If those are wrong, RLE encoding fails validation with a size mismatch.

### Building the JSON

For each prediction: threshold on score, keep the **top 10** by score, convert
`mask.cpu().numpy().astype(np.uint8)`, and encode. Reuse the release helper rather than
hand-rolling the RLE:

```python
from coco_utils import encode_binary_mask   # tools/coco_utils.py
rle = encode_binary_mask(mask)              # asfortranarray + mask_utils.encode + utf-8 decode
```

That function already handles the two things people get wrong: `np.asfortranarray` before
encoding, and decoding `counts` from bytes to `str` so the result is JSON-serialisable.

Each entry:

```python
{"image_id": <int from test_manifest.csv>, "category_id": 1,
 "segmentation": rle, "score": float(score)}
```

**On the score threshold**: AP rewards recall at low precision, because low-scoring false
positives only hurt once you exceed `maxDets=10`. Keep `SCORE_THRESH_TEST` low (the 0.05
default) and let the top-10 cut do the filtering. Do not threshold at 0.5 — that discards
detections the metric would have credited.

### The three killers

1. **`image_id` from `metadata/test_manifest.csv`** — not from a filename, not from your
   internal index. Test ids restart at 1 and collide with train ids.
2. **`category_id` must be `1`** — your model outputs class `0`. Map it back.
3. **RLE `size` must equal `[height, width]`** of *that* chip. 87 distinct sizes; there is no
   single correct value.

### Validate before every upload

```
python tools/validate_submission.py --submission submission.json
```

It checks JSON structure, id membership, `category_id`, score range and finiteness, and that
every RLE decodes to exactly the manifest's `(height, width)`. It also warns when an image has
more than 10 predictions. A clean run prints `validation=ok`. This takes seconds and catches
the entire class of format errors that would otherwise cost you a submission slot.

---

## 9. Debug playbook

| symptom | likely cause | fix |
|---|---|---|
| `PIL.UnidentifiedImageError` / cannot open `.npz` | default mapper still in use | override `build_train_loader` **and** `build_test_loader` |
| Loss is `NaN` from iteration 1 | unhandled NaNs — 8.6% of chips have them in RGB | `np.nan_to_num` in the mapper, before anything else |
| Loss `NaN` after a few hundred iterations | LR too high for the batch size | scale LR linearly from `0.02 @ 16`; keep `WARMUP_ITERS` |
| Loss falls, mask AP stays ~0 | `mask_format` mismatch | `INPUT.MASK_FORMAT = "bitmask"` **and** `mask_format="bitmask"` in `annotations_to_instances` |
| Boxes drift off the objects | transforms not replayed onto annotations | `transforms.apply_box` / `apply_segmentation`; check `BoxMode.XYWH_ABS` |
| Submission scores 0.0, file is valid | `category_id` left at `0` | map class `0 → 1` when writing |
| Empty submission | registered `category_id: 1` with `NUM_CLASSES = 1` | register as `0` internally |
| CUDA OOM | 4.16× upsampling × batch size | `AMP.ENABLED = True` first, then lower `IMS_PER_BATCH`, then `MIN_SIZE_TRAIN` (and shrink anchors to match) |
| Recall collapses after lowering `MIN_SIZE_TRAIN` | anchors no longer match object sizes | scale `ANCHOR_GENERATOR.SIZES` by the same factor |
| `ImportError: detectron2._C` | build doesn't match current torch | rebuild from source; re-check the torch/CUDA/numpy triple |
| Numpy ABI error on import | NumPy 2.x against a 1.x-compiled extension | pin `numpy<2` **before** building |
| Run dies immediately, offline notebook | `cfg.MODEL.WEIGHTS` is a URL | pre-download weights into an attached Dataset |
| Validator: "RLE size does not match" | wrong `height`/`width` in the dataset dict | take them from the manifest |
| Checkpoint log lists skipped tensors | expected for class predictors (81 → 2) | confirm **only** the final class/box predictors are skipped |

---

## 10. A suggested order of attack

Each step isolates one assumption, so a failure tells you where the problem is.

**1 — Overfit 8 images.** Register a dataset of 8 chips, train until loss approaches zero,
predict on those same 8 and look at the masks. This proves the entire data layer end to end:
`.npz` loading, NaN handling, RLE decode, `bbox_mode`, `mask_format`, transform replay, and the
`category_id` mapping. If a model cannot memorise 8 images, nothing downstream will work.
**Do not skip this**; it is twenty minutes that saves a day.

**2 — Full run at defaults.** Default resolution, default anchors, `AMP` on, a modest LR.
Score it with `tools/evaluate_coco.py`. This is your baseline number, and the first honest read
on the metric.

**3 — Make a valid submission early.** Even from a mediocre model. It exercises the id mapping,
RLE encoding and validator while you still have time to fix format problems.

**4 — Then tune.** In rough order of expected value: input resolution (with matching anchors),
LR and schedule, augmentation strength (vertical flips and rotations are free here), then
`FREEZE_AT`.

**5 — Then consider more channels.** Only once the RGB pipeline is solid and you have a
baseline to beat.

---

## Appendix: adding the other five channels

Forward-looking notes for when RGB stops being enough. `channels_explained.md` covers which
bands carry signal — start there to decide *which* to add, since NDVI (band 3) and NDWI
(band 7) are derived from the same measurements and may add less than their count suggests.

**The mechanism is one config key.** In `detectron2/modeling/backbone/build.py`:

```python
def build_backbone(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
```

So setting `PIXEL_MEAN` and `PIXEL_STD` to 8-element lists is what makes the network
8-channel. There is no separate "number of input channels" setting.

**Two consequences to plan for:**

1. **The pretrained stem will be silently discarded.** `DetectionCheckpointer` skips
   `backbone.bottom_up.stem.conv1.weight` on a shape mismatch — `(64, 3, 7, 7)` vs
   `(64, 8, 7, 7)` — and **logs it rather than raising**. Your stem is then randomly
   initialised while everything else is pretrained, and the run just trains worse for no
   visible reason. The fix is to load the checkpoint yourself and inflate the tensor: copy RGB
   into channels 0–2, fill the other five with the mean across the RGB axis, and rescale by
   `3/8` to keep activation magnitudes roughly unchanged. Save the patched checkpoint and point
   `MODEL.WEIGHTS` at it.
2. **`MODEL.BACKBONE.FREEZE_AT = 2` would freeze that stem.** Harmless in the RGB case;
   actively wrong once you have modified conv1. Set `FREEZE_AT = 0`.

**Data notes.** Per-band standardisation becomes mandatory — the bands are on incompatible
scales (NIR in the thousands, NDVI in hundredths; see `00_data_basics.ipynb` §7). And the NaN
problem is much worse outside RGB: bands 4 (`relative_elevation`) and 5 (`shaded_relief`) carry
roughly **82,000 and 84,000 NaNs across 200 chips**, versus a few hundred in RGB. Whatever
NaN strategy you chose for RGB, revisit it — filling ~80k pixels with zeros is a much stronger
statement than filling 400.

---

## Reference summary

| thing | value |
|---|---|
| config | `Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml` (COCO box 44.3 / mask 38.5) |
| train / test | 756 / 138 chips; 1,783 instances; 1 category |
| array | `.npz`, key `"image"`, HWC `(H, W, 8)`, `float32`; RGB = `[..., :3]` |
| chip sizes | W `150..297`, H `106..295`, 87 distinct pairs |
| RGB mean / std | `(92.31, 96.00, 77.26)` / `(37.50, 30.93, 30.73)` |
| RGB NaNs | 65 of 756 chips (8.6%) |
| instances / image | mean 2.36, min 1, max 10 |
| instance size | median `sqrt(area)` 27 px native, 113 px at the default resize |
| must set | `NUM_CLASSES=1`, `MASK_FORMAT="bitmask"`, `DETECTIONS_PER_IMAGE≈10`, `AMP.ENABLED=True` |
| metric | `segm AP @[IoU=0.50:0.95, area=all, maxDets=10]` |
| submission | `image_id` from test manifest, `category_id: 1`, RLE sized to the chip |
