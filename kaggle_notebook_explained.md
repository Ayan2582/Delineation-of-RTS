# The Kaggle notebook, cell by cell

A line-by-line walkthrough of `kaggle_cascade_mask_rcnn_rts.ipynb`. Every code cell appears here verbatim, with numbered lines, followed by an explanation of what each line does and why.

The reasoning behind the *choices* lives in `detectron2_training_guide.md`; this document explains the *code*.

The chips were converted to PNG by `prepare_png_dataset.py`, which is what lets detectron2's built-in data machinery do most of the work here.

---


## Cell 1 — Header *(markdown)*

This cell is prose, not code. It reads:

> # Cascade Mask R-CNN on the RTS challenge — RGB baseline
>
> Fine-tunes a COCO-pretrained **Cascade Mask R-CNN** (detectron2) on retrogressive thaw
> slump chips, using the **red/green/blue bands only**, and writes a validated
> `submission.json`.
>
> ### Before you run this
>
> 1. Run `prepare_png_dataset.py` locally to build `data_png/` from the release, then
>    **upload that folder as a Kaggle Dataset** and attach it here. It is ~64 MB, against
>    ~547 MB for the raw `.npz` release.
> 2. Settings → **Accelerator: GPU**, and **Internet: On** for the first run (detectron2
>    has to be built from source).
> 3. Leave `SMOKE_TEST = True` for the first run. It proves the whole pipeline in about
>    fifteen minutes. Only then set it to `False` for the real ~2–3 hour run.
>
> The design decisions behind every setting here are argued in
> `detectron2_training_guide.md`.


## Cell 2 — Configuration

Every knob lives in this one cell so you never go hunting through the notebook to change
a hyperparameter. The `SMOKE_TEST` block at the bottom is the important part: it lets the
same notebook run as a fifteen-minute correctness check or a three-hour training run.

```python
 1  # ---------------------------------------------------------------------------
 2  # CONFIG - every knob for this notebook lives here.
 3  # ---------------------------------------------------------------------------
 4  SMOKE_TEST = True          # short run that proves the pipeline end to end
 5  RUN_TRAIN  = True
 6  RUN_INFER  = True
 7  SEED       = 42
 8
 9  # Solver - sized for ~2-3 h on one Kaggle T4 (~40 epochs over 642 training chips)
10  IMS_PER_BATCH     = 2
11  BASE_LR           = 0.0025
12  MAX_ITER          = 15000
13  STEPS             = (10000, 13500)
14  WARMUP_ITERS      = 500
15  CHECKPOINT_PERIOD = 2000
16  EVAL_PERIOD       = 2000
17  NUM_WORKERS       = 2
18
19  # Inference
20  DETECTIONS_PER_IMAGE = 20
21  SCORE_THRESH_TEST    = 0.05
22
23  CONFIG_YAML = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
24  OUTPUT_DIR  = "/kaggle/working/output"
25
26  if SMOKE_TEST:
27      MAX_ITER, STEPS, WARMUP_ITERS = 200, (150,), 50
28      CHECKPOINT_PERIOD = EVAL_PERIOD = 100
29      print("SMOKE_TEST is on - short run to prove the pipeline, not a competitive model")
30
31  print(f"iters={MAX_ITER}  batch={IMS_PER_BATCH}  lr={BASE_LR}")
```

**Line by line**

- **L1–3** — A banner comment. Everything you would want to tune is between here and the end of the cell — nothing is hidden further down.
- **L4** — `SMOKE_TEST` is the single most useful switch in the notebook. When `True`, the block near the bottom shrinks the schedule to ~200 iterations so a complete pass — data, model, training, evaluation, submission — finishes in minutes. Run it this way **first**; discovering a broken submission writer two hours into a real run is the failure this prevents.
- **L5–6** — Stage switches. Set `RUN_TRAIN = False` to re-run only inference against an existing checkpoint, which is what you want when iterating on the submission without retraining.
- **L7** — One seed, applied to `random`, `numpy` and `torch` in cell 4. Note this makes the run *reproducible*, not deterministic — cuDNN kernel selection still varies.
- **L8–9** — Blank line, then a comment recording the budget these numbers were chosen for. 642 is the training fold after the 15% validation split, not all 756 chips.
- **L10** — Images per iteration. Two is what fits alongside the default 800 px resize on a 16 GB T4 with AMP on. This is the number to lower first if you hit OOM.
- **L11** — `0.0025` is the reference `0.02` linearly rescaled from a batch of 16 to a batch of 2. Linear scaling with batch size is the standard rule; treat this as an upper bound, since fine-tuning 642 images usually wants less.
- **L12** — Total iterations, **not epochs** — detectron2 counts iterations everywhere. At 642 images and batch 2, one epoch is 321 iterations, so 15,000 is roughly 47 epochs.
- **L13** — The two milestones where the learning rate drops by 10×. Set at about 2/3 and 9/10 of `MAX_ITER`, mirroring the reference schedule's 60k/80k out of 90k.
- **L14** — Iterations spent ramping the learning rate up from near zero. With a small batch and freshly initialised heads, removing warmup is a reliable way to make the loss explode in the first hundred steps.
- **L15** — How often a checkpoint is written. Kaggle kills a session at 12 hours regardless of progress, so this bounds how much work a dead session can cost you.
- **L16** — How often validation runs. Each evaluation costs real time, so evaluating too often just slows training down.
- **L17** — Dataloader worker processes. Kaggle gives you few CPU cores; oversubscribing them starves the GPU rather than feeding it faster.
- **L18–19** — Blank line and a comment separating the inference settings.
- **L20** — How many detections the model may return per image. The official scorer uses `maxDets=10` and no training chip has more than 10 instances, so 20 is ample headroom — anything beyond the top 10 by score cannot improve the metric.
- **L21** — Detections below this confidence are discarded. Deliberately **low**: average precision rewards recall, and low-scoring false positives are nearly free once the top-10 cut is applied. Raising this to 0.5 throws away detections the metric would have credited.
- **L22–23** — Blank line, then the model-zoo config. `Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml` is the R-50 Cascade Mask R-CNN at the 3× schedule (COCO box AP 44.3, mask AP 38.5).
- **L24** — Checkpoints and logs go here. On Kaggle only `/kaggle/working` survives the session, so this path is not arbitrary.
- **L25–26** — Blank line, then the smoke-test override block.
- **L27** — Collapses the schedule to 200 iterations with a single LR drop at 150 and 50 warmup steps — enough for the loss to move, far too few to learn anything.
- **L28** — Checkpoint and evaluate every 100 iterations, so both code paths actually execute during the short run. A smoke test that never triggers evaluation has not tested evaluation.
- **L29** — States plainly what mode you are in. Mistaking a smoke-test checkpoint for a trained model is an easy and expensive confusion.
- **L30–31** — Blank line, then an echo of the effective settings, so the notebook's output records what was actually run.


## Cell 3 — Environment and detectron2 install

detectron2's last release (v0.6, 2021) ships wheels only up to torch 1.10, and Kaggle
runs torch 2.x — so there is no wheel to install and it must be **built from source**.
This cell checks the environment first, builds only if needed, and verifies the build in
the one way that actually proves anything.

```python
 1  import importlib
 2  import subprocess
 3  import sys
 4
 5  import numpy as np
 6  import torch
 7
 8  print("torch :", torch.__version__, "| cuda:", torch.version.cuda,
 9        "| gpu:", torch.cuda.is_available())
10  print("numpy :", np.__version__)
11
12  if int(np.__version__.split(".")[0]) >= 2:
13      print("WARNING: numpy 2.x - detectron2's extensions need numpy<2.")
14      print("         Run: pip install 'numpy<2'  then RESTART the kernel.")
15
16  try:
17      import detectron2
18      import detectron2._C                      # the compiled extension - the real test
19      print("detectron2:", detectron2.__version__, "(already present)")
20  except ImportError:
21      print("building detectron2 from source, ~10 min ...")
22      subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
23                             "git+https://github.com/facebookresearch/detectron2.git"])
24      importlib.invalidate_caches()
25      import detectron2
26      import detectron2._C
27      print("detectron2:", detectron2.__version__, "(built)")
```

**Line by line**

- **L1–3** — Standard-library imports used only by this cell: `importlib` to refresh the module cache after a fresh install, `subprocess` to shell out to pip, `sys` to get the interpreter that is actually running this kernel.
- **L4–6** — Blank line, then numpy and torch. These are imported *before* detectron2 on purpose — their versions determine whether a detectron2 build will work at all.
- **L7–9** — Prints the version triple that governs everything. A source build compiles against exactly this torch and CUDA; if Kaggle later bumps its base image, a cached build breaks and this line is the first place to look.
- **L10** — numpy's version, printed separately because it has its own constraint.
- **L11–12** — Blank line, then the numpy 2.x guard. Extensions compiled against the NumPy 1.x C API abort at import under 2.x.
- **L13–14** — Warn rather than silently `pip install` a downgrade: changing numpy under a running kernel leaves already-imported modules linked against the old ABI, so a **restart** is genuinely required. Telling you beats pretending it was fixed.
- **L15–16** — Blank line, then `try:` — attempt the import before spending ten minutes on a build. Re-running this notebook then costs seconds instead of rebuilding every session.
- **L17** — Imports the Python package.
- **L18** — **This is the line that matters.** `detectron2._C` is the compiled CUDA extension. The pure-Python package imports fine even when the extension failed to build, so testing only `import detectron2` gives a false pass and you find out later, mid-training, when NMS is called.
- **L19** — Report the version already present.
- **L20–21** — If either import failed, fall through to a build and say so — ten silent minutes looks like a hang.
- **L22–23** — Install from the GitHub `main` branch. `-q` keeps the output readable; expect a wall of dependency-resolver complaints about Kaggle's preinstalled `google-cloud-*` and `tensorflow` pins, which are noise here since detectron2 does not use them.
- **L24** — Clear Python's import caches so the interpreter notices packages that appeared after it started. Without this the imports below can still fail on a fresh install.
- **L25–27** — Import again — including `_C` — so a failed build raises **here**, at the install step, rather than somewhere confusing later. Then confirm the version.

> **Watch out.** `from detectron2 import model_zoo` succeeding proves nothing about the CUDA extension. Always verify with `import detectron2._C`.


## Cell 4 — Imports, seeding and locating the data

Pulls in everything the rest of the notebook uses, fixes the random seeds, and finds
`data_png/` whether the notebook is running on Kaggle or on your own machine.

```python
 1  import csv
 2  import json
 3  import os
 4  import random
 5  from pathlib import Path
 6
 7  import cv2
 8  import matplotlib.pyplot as plt
 9  from pycocotools import mask as mask_utils
10  from pycocotools.coco import COCO
11  from pycocotools.cocoeval import COCOeval
12
13  from detectron2 import model_zoo
14  from detectron2.checkpoint import DetectionCheckpointer
15  from detectron2.config import get_cfg
16  from detectron2.data import (DatasetCatalog, DatasetMapper, MetadataCatalog,
17                               build_detection_test_loader,
18                               build_detection_train_loader)
19  from detectron2.data import transforms as T
20  from detectron2.data.datasets import register_coco_instances
21  from detectron2.engine import DefaultTrainer, HookBase
22  from detectron2.evaluation import COCOEvaluator
23  from detectron2.modeling import build_model
24  from detectron2.utils.logger import setup_logger
25  from detectron2.utils.visualizer import Visualizer
26
27  setup_logger()
28  random.seed(SEED)
29  np.random.seed(SEED)
30  torch.manual_seed(SEED)
31
32
33  def find_data_root() -> Path:
34      """Locate data_png/ on Kaggle or locally, by its annotations file."""
35      roots = [Path("/kaggle/input"), Path(".")]
36      for root in roots:
37          if not root.exists():
38              continue
39          for hit in sorted(root.glob("**/annotations/instances_train_fold.json")):
40              return hit.parent.parent
41      raise FileNotFoundError(
42          "data_png not found. Attach it as a Kaggle Dataset, or run "
43          "prepare_png_dataset.py locally."
44      )
45
46
47  DATA = find_data_root()
48  print("data root:", DATA)
```

**Line by line**

- **L1–5** — Standard library: `csv` reads the test manifest, `json` reads and writes COCO files, `os` joins output paths, `random` is seeded below, `Path` is used for every filesystem operation in the notebook.
- **L6–7** — Blank line, then OpenCV — used to read PNGs for the sanity check. detectron2 uses it internally too.
- **L8** — matplotlib, only for displaying the sanity-check figure.
- **L9–11** — pycocotools, three separate entry points: `mask_utils` encodes and decodes RLE, `COCO` loads a ground-truth file, `COCOeval` computes average precision. The last two are what let us reproduce the challenge's exact metric locally.
- **L12–13** — Blank line, then `model_zoo`, which resolves a config name to both a YAML path and a pretrained-checkpoint URL.
- **L14** — `DetectionCheckpointer` loads weights into a model built outside the trainer — needed for the inference-only path.
- **L15** — `get_cfg()` returns a fresh config populated with detectron2's defaults.
- **L16–19** — The data API. `DatasetCatalog`/`MetadataCatalog` are the registries; `DatasetMapper` is the built-in mapper we can now use unchanged because the images are PNGs; the two loader builders turn a registered dataset into a torch `DataLoader`.
- **L20** — The augmentation module, aliased `T` by convention.
- **L21** — `register_coco_instances` is the function that makes the whole data layer three lines instead of a hundred.
- **L22** — `DefaultTrainer` supplies the training loop; `HookBase` is the base class for the validation-loss hook in cell 7.
- **L23** — `COCOEvaluator` gives cheap in-loop metrics. Note these are *stock COCO* numbers, not this challenge's — cell 9 handles the official metric.
- **L24** — `build_model` constructs a model from a config without a trainer, used by the inference path.
- **L25** — `setup_logger` makes detectron2's internal logging visible; without it, useful messages such as checkpoint shape mismatches are swallowed.
- **L26** — `Visualizer` draws ground-truth and predicted masks over an image.
- **L27–28** — Blank line, then activate detectron2's logger.
- **L29–31** — Seed all three generators. `torch.manual_seed` covers CUDA too. This makes runs comparable; it does not make them bit-identical, since cuDNN still picks kernels non-deterministically.
- **L32–34** — Blank lines, then the data locator.
- **L35** — Its docstring: the search is anchored on the annotations file rather than on a directory name, so it works no matter what Kaggle names the attached dataset folder.
- **L36** — Two search roots: the Kaggle input mount first, then the current directory for local runs.
- **L37–39** — Skip a root that does not exist — `/kaggle/input` is absent when running locally, and globbing it would simply find nothing.
- **L40–41** — Recursively search for `annotations/instances_train_fold.json` and return the directory two levels above it, which is `data_png/`. `sorted` keeps the choice stable when several copies are attached.
- **L42–45** — If nothing matched, fail immediately with an actionable message. A missing dataset should stop the notebook here, not surface as a confusing error inside the dataloader.
- **L46–48** — Blank lines, then resolve the path once and print it, so the notebook's output records which copy of the data was used.


## Cell 5 — Registering the datasets

The entire data layer. Because `prepare_png_dataset.py` wrote real PNGs and standard COCO
JSONs, detectron2's own `register_coco_instances` does all the work — including two
remappings that would otherwise be easy to get silently wrong.

```python
 1  for name, fold in (("rts_train", "train"), ("rts_val", "val")):
 2      if name in DatasetCatalog.list():
 3          DatasetCatalog.remove(name)
 4          MetadataCatalog.remove(name)
 5      register_coco_instances(
 6          name,
 7          {},
 8          str(DATA / "annotations" / f"instances_{fold}_fold.json"),
 9          str(DATA / "train"),
10      )
11      MetadataCatalog.get(name).thing_classes = ["rts"]
12
13  train_dicts = DatasetCatalog.get("rts_train")
14  val_dicts = DatasetCatalog.get("rts_val")
15  print(f"train: {len(train_dicts):>4} images, "
16        f"{sum(len(d['annotations']) for d in train_dicts):>5} instances")
17  print(f"val  : {len(val_dicts):>4} images, "
18        f"{sum(len(d['annotations']) for d in val_dicts):>5} instances")
```

**Line by line**

- **L1** — Loop over the two folds, pairing the dataset name detectron2 will know with the JSON suffix written by `prepare_png_dataset.py`.
- **L2–4** — Registration raises if a name already exists, which makes re-running this cell fail. Removing first makes the cell idempotent — worth it in a notebook you will re-run many times.
- **L5** — `register_coco_instances(name, metadata, json_file, image_root)` — it does not load anything now, it registers a lazy loader.
- **L6** — The dataset name.
- **L7** — An empty metadata dict. `load_coco_json` fills in `thing_classes` from the file's `categories` automatically.
- **L8** — Path to this fold's COCO JSON.
- **L9** — The image root. `file_name` inside the JSON is a bare `train_000001.png`, so detectron2 joins it against this directory. Both folds point at the same PNG folder — the split lives in the JSONs, not in the filesystem.
- **L10** — Closing paren of the registration call.
- **L11** — Set the class name explicitly. Only cosmetic — it is what `Visualizer` prints on a mask — but it makes the sanity-check figure readable.
- **L12–13** — Blank line, then materialise the training fold. This is where the JSON is actually parsed into detectron2's list-of-dicts format.
- **L14** — The same for validation.
- **L15–18** — Print image and instance counts for both folds. Expect 642/1515 and 114/268. Checking these against the conversion report is a two-second guard against having attached a stale or partial dataset.

> **Watch out.** `load_coco_json` **remaps `category_id` to contiguous ids automatically** (`1 → 0`) and sets `bbox_mode = BoxMode.XYWH_ABS` for you. On the raw `.npz` path both were manual, and the `category_id` off-by-one silently produced a valid submission that scored zero. Converting to PNG deleted that entire class of bug.


## Cell 6 — Sanity check the data layer

Cheap insurance. Every assertion here fails in seconds on the CPU; the same mistakes found
during training cost a GPU session. Run this before you ever set `RUN_TRAIN = True`.

```python
 1  d = train_dicts[0]
 2  print("record keys     :", sorted(d))
 3  print("annotation keys :", sorted(d["annotations"][0]))
 4  print("category_id     :", d["annotations"][0]["category_id"], " (remapped 1 -> 0)")
 5  print("bbox_mode       :", d["annotations"][0]["bbox_mode"])
 6
 7  img = cv2.imread(d["file_name"])[:, :, ::-1]
 8  assert img is not None, f"could not read {d['file_name']}"
 9  assert img.shape[:2] == (d["height"], d["width"]), "PNG size disagrees with the COCO record"
10  for a in d["annotations"]:
11      assert a["segmentation"]["size"] == [d["height"], d["width"]], "RLE size mismatch"
12      assert a["category_id"] == 0, "category_id must be 0 internally"
13
14  n_inst = sum(len(x["annotations"]) for x in train_dicts + val_dicts)
15  assert n_inst == 1783, f"expected 1783 instances across both folds, got {n_inst}"
16  print(f"\nall checks passed - {n_inst} instances across both folds")
17
18  vis = Visualizer(img, metadata=MetadataCatalog.get("rts_train"), scale=3.0)
19  plt.figure(figsize=(11, 7))
20  plt.imshow(vis.draw_dataset_dict(d).get_image())
21  plt.axis("off")
22  plt.title(f"{Path(d['file_name']).name} - {len(d['annotations'])} instance(s)")
23  plt.show()
```

**Line by line**

- **L1** — Take the first training record. Everything below inspects this one chip.
- **L2** — The keys detectron2 built for us: `file_name`, `height`, `width`, `image_id`, `annotations`.
- **L3** — The per-instance keys: `bbox`, `bbox_mode`, `category_id`, `segmentation`, `iscrowd`.
- **L4** — Prints `0`. The COCO file on disk says `1`; `load_coco_json` remapped it. Seeing the `0` here is what confirms the remap happened.
- **L5** — Prints `BoxMode.XYWH_ABS`, set automatically. If this were wrong, every box would be mis-shaped and training would quietly fail to converge.
- **L6–7** — Blank line, then read the PNG. `cv2.imread` returns BGR, so `[:, :, ::-1]` flips it to RGB for display — this is a *display* concern only; the training pipeline handles channel order itself in cell 7.
- **L8** — Guard against a path that resolved but does not exist; `cv2.imread` signals failure by returning `None` rather than raising.
- **L9** — The PNG's pixel dimensions must match what the COCO record claims. A mismatch means the JSON and the images came from different conversions.
- **L10** — Loop over this chip's instances.
- **L11** — **The check that matters most.** Each RLE carries its own `size`, and it must equal the image's. If these ever disagree, masks decode into the wrong geometry and mask AP sits near zero while the loss looks healthy.
- **L12** — Re-assert the contiguous class id at the annotation level.
- **L13–14** — Blank line, then count instances across both folds.
- **L15** — Assert the total is 1,783 — the release's known instance count. This catches a truncated or partially-copied dataset, which otherwise trains happily on less data than you think.
- **L16** — Report success.
- **L17–18** — Blank line, then build a `Visualizer` over the RGB image. `scale=3.0` because these chips are only ~150–300 px across and unscaled masks are hard to judge by eye.
- **L19** — Open a figure.
- **L20** — `draw_dataset_dict` overlays the ground-truth masks and boxes exactly as detectron2 parsed them. This is the visual counterpart to the assertions: if masks land on plausible terrain features, the label pipeline is right.
- **L21–23** — Hide the axes, title the figure with the chip name and instance count, and render.


## Cell 7 — Building the config

Turns the model-zoo Cascade Mask R-CNN config into one adapted to this dataset. Only a
handful of keys change; the comments record which defaults were deliberately *kept*, which
matters as much as the overrides.

```python
 1  def build_cfg() -> "CfgNode":
 2      cfg = get_cfg()
 3      cfg.merge_from_file(model_zoo.get_config_file(CONFIG_YAML))
 4      cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_YAML)
 5
 6      cfg.DATASETS.TRAIN = ("rts_train",)
 7      cfg.DATASETS.TEST = ("rts_val",)
 8      cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
 9
10      cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1        # propagates to all three cascade heads
11      cfg.INPUT.MASK_FORMAT = "bitmask"          # labels are RLE, not polygons
12
13      cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
14      cfg.SOLVER.BASE_LR = BASE_LR
15      cfg.SOLVER.MAX_ITER = MAX_ITER
16      cfg.SOLVER.STEPS = STEPS
17      cfg.SOLVER.WARMUP_ITERS = WARMUP_ITERS
18      cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD
19      cfg.SOLVER.AMP.ENABLED = True
20      cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
21
22      cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST
23      cfg.TEST.DETECTIONS_PER_IMAGE = DETECTIONS_PER_IMAGE
24
25      # Deliberately left at their defaults:
26      #   INPUT.FORMAT = "BGR" + COCO PIXEL_MEAN -> read_image does the RGB->BGR flip
27      #   MIN_SIZE_TRAIN = (640..800) -> upsamples these small chips ~4.2x
28      #   ANCHOR_GENERATOR.SIZES = 32..512 -> measured to match at that upsampling
29      # Shrink MIN_SIZE_TRAIN and you must shrink the anchors by the same factor.
30
31      cfg.OUTPUT_DIR = OUTPUT_DIR
32      os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
33      return cfg
34
35
36  cfg = build_cfg()
37  print("roi heads     :", cfg.MODEL.ROI_HEADS.NAME, "| classes:", cfg.MODEL.ROI_HEADS.NUM_CLASSES)
38  print("input format  :", cfg.INPUT.FORMAT, "| pixel mean:", cfg.MODEL.PIXEL_MEAN)
39  print("min_size_train:", cfg.INPUT.MIN_SIZE_TRAIN)
40  print("anchor sizes  :", cfg.MODEL.ANCHOR_GENERATOR.SIZES)
```

**Line by line**

- **L1** — A function rather than a bare block, because inference needs to rebuild the same config independently in cell 8.
- **L2** — `get_cfg()` returns detectron2's full default config — every key exists before anything is overridden.
- **L3** — Overlay the Cascade Mask R-CNN YAML. `get_config_file` resolves the name to a path inside the installed package, so no manual downloading.
- **L4** — The matching COCO-pretrained weights, as a URL. **On an offline Kaggle run this download fails** — pre-fetch the checkpoint into your attached dataset and point this at the local file instead.
- **L5–6** — Blank line, then the training dataset. It must be a **tuple** — a bare string would be read as a sequence of characters.
- **L7** — The dataset used by the periodic evaluation.
- **L8** — Dataloader workers, from cell 2.
- **L9–10** — Blank line, then the single most important override: one class. Cascade R-CNN has **three** sequential box heads (at IoU 0.5/0.6/0.7) and this value propagates to all of them — which is why hand-rolled 'replace the final layer' recipes written for plain Mask R-CNN break on Cascade.
- **L11** — **Must be `bitmask`.** The default is `polygon`, which cannot handle the RLE dict segmentations these labels use. This is the classic cause of 'loss falls but mask AP stays at zero'.
- **L12–13** — Blank line, then batch size.
- **L14** — Base learning rate.
- **L15** — Total iterations.
- **L16** — Learning-rate decay milestones.
- **L17** — Warmup length.
- **L18** — Checkpoint frequency.
- **L19** — Enable automatic mixed precision. Roughly halves activation memory on a T4 and is close to free in accuracy — turn this on before considering a smaller input size.
- **L20** — Evaluation frequency.
- **L21–22** — Blank line, then the test-time score floor, kept low so average precision can use the low-confidence tail.
- **L23** — Maximum detections returned per image.
- **L24–29** — A comment block recording the defaults that were kept **on purpose**, which is easy to lose track of later. `INPUT.FORMAT` stays `BGR` so `read_image` flips our RGB PNGs to match the COCO `PIXEL_MEAN` ordering; the resize stays large because it lifts these ~150–300 px chips into the size range the pretrained weights were trained on; and the anchors stay put because at that ~4.2× upsampling the instances span roughly 30–400 px, which is exactly the `32/64/128/256/512` ladder. The last line is the trap: resize and anchors are **one coupled decision**, so lowering the resize without shrinking the anchors collapses recall.
- **L30–31** — Blank line, then the output directory.
- **L32** — Create it if absent; `exist_ok=True` keeps re-runs quiet.
- **L33** — Return the finished config.
- **L34–36** — Blank lines, then build the config for training.
- **L37** — Print the head type — expect `CascadeROIHeads`, confirming the cascade config actually loaded — and the class count.
- **L38** — Print the channel format and pixel mean together, since they only make sense as a pair.
- **L39–40** — Print the resize and anchor settings, the two coupled values from the comment above, so the run's output records what they were.

> **Watch out.** `INPUT.MASK_FORMAT = "bitmask"` here and RLE labels in the JSON must agree. Everything trains normally if they do not — the loss decreases — and mask AP simply never rises.


## Cell 8 — Trainer and the validation-loss hook

`DefaultTrainer` supplies the loop, scheduler, checkpointing and logging. Three overrides
adapt it to this dataset, and one hook adds the validation-loss curve that `DefaultTrainer`
does not compute on its own — the main signal for overfitting on only 642 images.

```python
 1  TRAIN_AUGS = [
 2      T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, "choice"),
 3      T.RandomFlip(horizontal=True, vertical=False),
 4      T.RandomFlip(horizontal=False, vertical=True),
 5  ]
 6
 7
 8  class LossEvalHook(HookBase):
 9      """Periodically report the validation loss - DefaultTrainer never does."""
10
11      def __init__(self, period, model, loader):
12          self._period = period
13          self._model = model
14          self._loader = loader
15
16      def _do_loss_eval(self):
17          was_training = self._model.training
18          self._model.train()
19          totals, n = {}, 0
20          with torch.no_grad():
21              for batch in self._loader:
22                  for k, v in self._model(batch).items():
23                      totals[k] = totals.get(k, 0.0) + float(v)
24                  n += 1
25          self._model.train(was_training)
26          means = {f"val_{k}": v / max(n, 1) for k, v in totals.items()}
27          self.trainer.storage.put_scalars(val_total_loss=sum(means.values()), **means)
28
29      def after_step(self):
30          nxt = self.trainer.iter + 1
31          if self._period > 0 and nxt % self._period == 0 and nxt != self.trainer.max_iter:
32              self._do_loss_eval()
33
34
35  class RTSTrainer(DefaultTrainer):
36      @classmethod
37      def build_train_loader(cls, cfg):
38          mapper = DatasetMapper(cfg, is_train=True, augmentations=TRAIN_AUGS)
39          return build_detection_train_loader(cfg, mapper=mapper)
40
41      @classmethod
42      def build_test_loader(cls, cfg, dataset_name):
43          return build_detection_test_loader(
44              cfg, dataset_name, mapper=DatasetMapper(cfg, is_train=False))
45
46      @classmethod
47      def build_evaluator(cls, cfg, dataset_name, output_folder=None):
48          return COCOEvaluator(dataset_name, output_dir=output_folder or cfg.OUTPUT_DIR)
49
50      def build_hooks(self):
51          hooks = super().build_hooks()
52          loss_loader = build_detection_test_loader(
53              self.cfg, self.cfg.DATASETS.TEST[0],
54              mapper=DatasetMapper(self.cfg, is_train=True, augmentations=[
55                  T.ResizeShortestEdge(self.cfg.INPUT.MIN_SIZE_TEST,
56                                       self.cfg.INPUT.MAX_SIZE_TEST)]))
57          hooks.insert(-1, LossEvalHook(self.cfg.TEST.EVAL_PERIOD, self.model, loss_loader))
58          return hooks
```

**Line by line**

- **L1** — The training augmentation list, passed to the built-in mapper below.
- **L2** — Resize the short edge to one of `MIN_SIZE_TRAIN`'s six values, chosen at random (`"choice"`). This is multi-scale training, and it is also what performs the ~4.2× upsampling these small chips need.
- **L3–4** — Horizontal and vertical flips as two separate augmentations, each applied independently with probability 0.5. **Vertical flips are legitimate here** in a way they are not for ordinary photographs: nadir satellite imagery has no canonical 'up', so this is a free doubling of augmentation strength. (It would need reconsidering if you later add band 5, shaded relief, whose illumination direction *is* meaningful.)
- **L5** — Close the list.
- **L6–8** — Blank lines, then the hook class. `HookBase` gives access to `self.trainer` once registered.
- **L9** — Docstring stating why it exists — `DefaultTrainer` runs metric evaluation on a schedule but never computes a validation *loss*, and loss is the earliest overfitting signal.
- **L10–14** — Blank line, then the constructor: store the evaluation period, the model and the validation loader. Nothing heavy happens here — the loader is built once by the caller and reused every time the hook fires.
- **L15–16** — Blank line, then the evaluation body.
- **L17** — Remember whether the model was in training mode, so it can be restored exactly.
- **L18** — **Switch to train mode deliberately.** A detectron2 model returns a dict of losses in train mode and a list of predictions in eval mode. We want losses, so train mode it is — `torch.no_grad()` below is what prevents this from actually updating anything.
- **L19** — Accumulators for the loss totals and the batch count.
- **L20** — No gradients: this is measurement, not learning. Without it, memory balloons and the graph is retained.
- **L21** — Iterate the whole validation fold.
- **L22–23** — Call the model and accumulate each loss component by name (classification, box regression, mask, RPN).
- **L24** — Count batches for the mean.
- **L25** — Restore the model's previous mode. Leaving it in train mode would silently corrupt the next evaluation.
- **L26** — Average each component and prefix with `val_` so the keys do not collide with training losses in the logs.
- **L27** — Write to detectron2's event storage, adding a `val_total_loss` summary alongside the components. Anything put here shows up in the printed metrics and in TensorBoard.
- **L28–29** — Blank line, then the hook callback that detectron2 calls after every iteration.
- **L30** — Use `iter + 1` because `trainer.iter` is zero-based.
- **L31–32** — Fire only on the period, and skip the final iteration — the trainer runs its own evaluation there, and doing both just duplicates work.
- **L33–35** — Blank lines, then the trainer subclass.
- **L36–37** — Override the training loader. This is the entire integration point for augmentation.
- **L38** — The **built-in** `DatasetMapper`, with our augmentation list substituted. Passing `augmentations` as a keyword works because detectron2's `@configurable` decorator forwards kwargs its `from_config` does not recognise straight to `__init__`, overriding the cfg-derived value. No custom mapper class is needed — that was only required to read `.npz` files.
- **L39** — Build the loader. `mapper` is keyword-only.
- **L40–41** — Blank line, then the next `@classmethod` decorator.
- **L42** — The test-loader override.
- **L43–44** — The same built-in mapper with `is_train=False`: deterministic resize, no flips, and no ground-truth instances.
- **L45–46** — Blank line, then the next `@classmethod` decorator.
- **L47** — The evaluator override.
- **L48** — `COCOEvaluator` for in-loop monitoring. Remember these are **stock COCO** numbers — different area bins and `maxDets` from this challenge's scorer — so watch the trend, not the absolute value. Cell 11 computes the real metric.
- **L49–50** — Blank line, then hook registration.
- **L51** — Start from `DefaultTrainer`'s own hooks (checkpointing, LR scheduling, metric writing) rather than replacing them.
- **L52–56** — Build the loader for validation loss. **The subtlety: `is_train=True`.** The ordinary test loader's mapper produces no ground-truth instances, so the model could not compute a loss from it. We need training-style targets but deterministic geometry, so it is a *test* loader (no shuffling, single pass) with a *train* mapper restricted to a plain deterministic resize.
- **L57** — Insert the hook at position `-1`, immediately **before** the periodic writer. Append it instead and the writer runs first, so each validation loss is logged one interval late.
- **L58** — Return the augmented hook list.


## Cell 9 — Train

The whole training run. Short, because everything was arranged in the cells above — and
resumable, because Kaggle will kill the session at twelve hours regardless of progress.

```python
1  if RUN_TRAIN:
2      trainer = RTSTrainer(cfg)
3      trainer.resume_or_load(resume=True)
4      trainer.train()
5      print("training finished ->", os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
6  else:
7      print("RUN_TRAIN is False - skipping training")
```

**Line by line**

- **L1** — Guarded by the stage switch, so you can re-run the notebook for inference alone.
- **L2** — Construct the trainer. This builds the model, loads `cfg.MODEL.WEIGHTS`, and constructs the optimizer, scheduler and dataloaders. **Watch the log here**: it lists checkpoint tensors that were skipped on a shape mismatch. Skips on the final class predictors are expected and correct — the COCO checkpoint has 81 classes and this model has 2. Anything *else* being skipped is a bug worth chasing.
- **L3** — `resume=True` is what makes chained Kaggle sessions work: if `OUTPUT_DIR` holds a `last_checkpoint`, it restores the weights **and** the iteration counter and optimizer state. With `resume=False` it would load the pretrained weights and restart from iteration 0 — the difference between continuing a run and silently starting over.
- **L4** — Run the loop to `MAX_ITER`.
- **L5** — Report where the final weights landed.
- **L6–7** — The skip branch, which prints rather than staying silent so the notebook's output is unambiguous about what ran.

> **Watch out.** If the session dies, save `OUTPUT_DIR` as a Kaggle Dataset, attach it to the next session, copy it back into `/kaggle/working/output`, and re-run — `resume_or_load(resume=True)` picks up where it stopped.


## Cell 10 — Inference helpers

Loads a trained model and turns its output into COCO-format predictions. Note it does not
use `DefaultPredictor`: that helper re-reads images itself and would bypass the config used
during training.

```python
 1  def encode_binary_mask(mask):
 2      """Encode an H x W binary mask as compressed COCO RLE (JSON-safe)."""
 3      rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
 4      rle["counts"] = rle["counts"].decode("utf-8")
 5      return rle
 6
 7
 8  def load_trained_model(weights=None):
 9      cfg_i = build_cfg()
10      cfg_i.MODEL.WEIGHTS = weights or os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
11      model = build_model(cfg_i)
12      DetectionCheckpointer(model).load(cfg_i.MODEL.WEIGHTS)
13      model.eval()
14      return cfg_i, model
15
16
17  def predict(model, cfg_i, dicts, top_k=10):
18      """Run the model over a list of dataset dicts -> COCO results."""
19      loader = build_detection_test_loader(
20          dicts, mapper=DatasetMapper(cfg_i, is_train=False))
21      results = []
22      with torch.no_grad():
23          for batch in loader:
24              for inp, out in zip(batch, model(batch)):
25                  inst = out["instances"].to("cpu")
26                  order = inst.scores.argsort(descending=True)[:top_k]
27                  for i in order.tolist():
28                      results.append({
29                          "image_id": int(inp["image_id"]),
30                          "category_id": 1,
31                          "segmentation": encode_binary_mask(inst.pred_masks[i].numpy()),
32                          "score": float(inst.scores[i]),
33                      })
34      return results
```

**Line by line**

- **L1** — The RLE encoder, identical in behaviour to `encode_binary_mask` in the release's `tools/coco_utils.py`. It is inlined so the notebook needs nothing on `sys.path` when running on Kaggle.
- **L2** — Its docstring.
- **L3** — Two details that are both mandatory. `asfortranarray` gives column-major order, which is what COCO RLE assumes — pass a C-ordered array and the encoding is scrambled. `astype(uint8)` because pycocotools rejects bool.
- **L4** — `mask_utils.encode` returns `counts` as **bytes**, which `json.dump` cannot serialise. Decoding to a UTF-8 string is what makes the prediction writable.
- **L5** — Return the RLE dict.
- **L6–8** — Blank lines, then the model loader.
- **L9** — Rebuild the config from scratch rather than mutating the training one, so inference cannot be affected by leftover state.
- **L10** — Use an explicit checkpoint if given, otherwise the final weights from training.
- **L11** — `build_model` constructs the architecture from the config, with random weights.
- **L12** — `DetectionCheckpointer(...).load(...)` fills in the trained weights.
- **L13** — **Eval mode.** In eval mode the model returns predictions; in train mode it would return a loss dict. Forgetting this produces a confusing `KeyError` on `instances` later.
- **L14** — Return both, since prediction needs the config too.
- **L15–17** — Blank lines, then the prediction driver.
- **L18** — Its docstring. It takes a plain list of dataset dicts, which is what both callers have: `DatasetCatalog.get` returns one for the validation fold, and the unregistered test set is built as one in cell 12.
- **L19–20** — Build a test loader straight from that list — `build_detection_test_loader` accepts a list as its dataset, so no registration is needed. The mapper is the built-in one in eval mode: the **same** preprocessing as training, which is the whole reason for not using `DefaultPredictor` here.
- **L21** — Accumulator for the COCO-format results.
- **L22** — No gradients during inference.
- **L23** — Iterate batches.
- **L24** — Zip inputs with outputs so each prediction keeps the `image_id` of the image it came from.
- **L25** — Move the `Instances` to CPU once, rather than per-field.
- **L26** — Sort detections by score, highest first, and keep only the top `top_k`. The official metric uses `maxDets=10`, so anything past the tenth cannot help.
- **L27** — Loop over the surviving indices.
- **L28** — Open the prediction dict.
- **L29** — The image id, taken from the input record so it can never drift out of step with the image.
- **L30** — **Always `1`.** The model predicts contiguous class `0`; the submission format requires the original category id `1`. This is the mapping's other half, and getting it wrong yields a structurally valid file that scores zero.
- **L31** — `pred_masks[i]` is a bool tensor already rescaled to the chip's original size — detectron2 uses the `height`/`width` in the input record to do that. Encode it to RLE.
- **L32** — The confidence, cast to a plain float so it is JSON-serialisable.
- **L33** — Close the dict and append.
- **L34** — Return all predictions.

> **Watch out.** `DefaultPredictor` would read the image from disk with its own preprocessing, bypassing this config. Any divergence between training and inference preprocessing costs accuracy silently, which is why inference reuses the same mapper.


## Cell 11 — Scoring against the official metric

`COCOEvaluator` reports stock COCO numbers, whose area bins and `maxDets` differ from this
challenge's. This cell reproduces the official settings exactly, so the number you compare
across experiments is the one the leaderboard will report.

```python
 1  OFFICIAL_MAXDETS = [1, 5, 10]
 2  OFFICIAL_AREA_RNG = [[0, 1e10], [0, 300], [300, 2000], [2000, 1e10]]
 3  OFFICIAL_AREA_LBL = ["all", "small", "medium", "large"]
 4
 5
 6  def score_official(gt_json_path, predictions):
 7      """COCO segm AP using the challenge's maxDets and area ranges."""
 8      if not predictions:
 9          print("no predictions - nothing to score")
10          return None
11      coco_gt = COCO(str(gt_json_path))
12      coco_dt = coco_gt.loadRes(list(predictions))
13      ev = COCOeval(coco_gt, coco_dt, "segm")
14      ev.params.maxDets = OFFICIAL_MAXDETS
15      ev.params.areaRng = OFFICIAL_AREA_RNG
16      ev.params.areaRngLbl = OFFICIAL_AREA_LBL
17      ev.evaluate()
18      ev.accumulate()
19
20      area_i = ev.params.areaRngLbl.index("all")
21      det_i = ev.params.maxDets.index(10)
22      prec = ev.eval["precision"][:, :, :, area_i, det_i]
23      primary = float(np.mean(prec[prec > -1])) if (prec > -1).any() else -1.0
24      print(f"AP @[IoU=0.50:0.95 | area=all | maxDets=10] = {primary:.4f}   <- ranking metric")
25      return primary
26
27
28  if RUN_INFER:
29      cfg_i, model = load_trained_model()
30      val_preds = predict(model, cfg_i, val_dicts, top_k=10)
31      print(f"{len(val_preds)} predictions over {len(val_dicts)} validation images")
32      score_official(DATA / "annotations" / "instances_val_fold.json", val_preds)
```

**Line by line**

- **L1** — `maxDets` from the official scorer: `[1, 5, 10]`. COCO's default is `[1, 10, 100]`, so this alone changes the headline number.
- **L2** — The official area ranges. COCO's defaults are 32² = 1024 and 96² = 9216; this challenge uses **300** and **2000**, which reclassifies most of this dataset. Under the official bins the split is 25% small / 51% medium / 24% large.
- **L3** — The labels, in the same order as the ranges — `COCOeval` pairs them positionally.
- **L4–7** — Blank lines, then the scoring function and its docstring. It takes a ground-truth JSON path and an in-memory prediction list.
- **L8–10** — An empty prediction list is legal but `loadRes` chokes on it, so return early with a clear message instead of an opaque pycocotools error.
- **L11** — Load the ground truth for this fold. Its `category_id` values are the original `1`, matching what `predict` emits.
- **L12** — `loadRes` accepts a list of prediction dicts directly, so nothing needs writing to disk first.
- **L13** — Build the evaluator in **`segm`** mode — this challenge scores masks, not boxes.
- **L14–16** — Override the three parameters that differ from COCO's defaults. Setting these *after* construction is required, since `COCOeval` fills in defaults during `__init__`.
- **L17** — Per-image, per-category matching of predictions to ground truth.
- **L18** — Aggregate those matches into the precision/recall arrays.
- **L19–20** — Blank line, then locate the `all` area bin by label rather than by a hardcoded index, so it stays correct if the ordering ever changes.
- **L21** — Locate `maxDets=10` the same way.
- **L22** — Slice the precision array, which is indexed `[IoU, recall, category, area, maxDets]`. Keeping the first three axes averages over all ten IoU thresholds — that is what the `0.50:0.95` in the metric name means.
- **L23** — Average the valid entries. `COCOeval` writes `-1` where a cell has no data, so those must be filtered out or they drag the mean down.
- **L24** — Print the single number that decides the ranking.
- **L25** — Return it for programmatic comparison across runs.
- **L26–28** — Blank lines, then the inference block, guarded by its stage switch.
- **L29** — Load the trained weights.
- **L30** — Predict over the validation fold, capped at 10 detections per image to match the metric.
- **L31** — Report the prediction count — roughly 10× the image count, since the low score threshold lets most slots fill.
- **L32** — Score against the validation fold's ground truth. **This is the number to compare between experiments**, not the `COCOEvaluator` output printed during training.


## Cell 12 — Test inference and the submission file

Predicts on the 138 test chips and writes `submission.json`, then re-checks it with the
same rules the official validator applies — because a malformed submission costs a slot,
and every one of these errors is catchable locally in seconds.

```python
 1  if RUN_INFER:
 2      with open(DATA / "test_manifest.csv", newline="") as f:
 3          rows = list(csv.DictReader(f))
 4
 5      test_dicts = [{
 6          "file_name": str(DATA / "test" / f"{r['public_id']}.png"),
 7          "image_id": int(r["image_id"]),
 8          "height": int(r["height"]),
 9          "width": int(r["width"]),
10      } for r in rows]
11      print(f"{len(test_dicts)} test chips")
12
13      test_preds = predict(model, cfg_i, test_dicts, top_k=10)
14      out_path = "/kaggle/working/submission.json"
15      with open(out_path, "w") as f:
16          json.dump(test_preds, f)
17      print(f"wrote {out_path}  ({len(test_preds)} predictions)")
18
19      sizes = {int(r["image_id"]): (int(r["height"]), int(r["width"])) for r in rows}
20      seen = {}
21      for i, p in enumerate(test_preds):
22          assert set(p) == {"image_id", "category_id", "segmentation", "score"}, i
23          assert p["image_id"] in sizes, f"unknown image_id {p['image_id']}"
24          assert p["category_id"] == 1, "category_id must be 1"
25          assert 0.0 <= p["score"] <= 1.0 and np.isfinite(p["score"]), "bad score"
26          assert p["segmentation"]["size"] == list(sizes[p["image_id"]]), "RLE size mismatch"
27          assert mask_utils.decode(p["segmentation"]).shape == sizes[p["image_id"]]
28          seen[p["image_id"]] = seen.get(p["image_id"], 0) + 1
29
30      crowded = {k: v for k, v in seen.items() if v > 10}
31      print(f"validation=ok  images_with_predictions={len(seen)}/{len(rows)}")
32      if crowded:
33          print(f"note: {len(crowded)} images exceed maxDets=10")
```

**Line by line**

- **L1** — Guarded by the inference switch.
- **L2–3** — Read the test manifest. **This file is the only correct source of test `image_id`s** — they restart at 1 and therefore collide with training ids, so deriving them from a filename or a loop counter produces a valid file that scores zero.
- **L4** — Blank line.
- **L5–10** — Build plain detectron2 dicts for the test chips. The test set has no labels, so there is nothing to register as a COCO dataset — a list of dicts with `file_name`, `image_id`, `height` and `width` is all the mapper needs. Carrying the manifest's height and width here is what lets detectron2 rescale predicted masks back to each chip's true size.
- **L11** — Confirm 138 chips.
- **L12–13** — Blank line, then predict, again capped at the metric's 10 detections per image.
- **L14** — Kaggle only persists `/kaggle/working`, so the submission is written there.
- **L15–16** — Write the JSON. No `indent` — the file has thousands of RLE strings and pretty-printing bloats it for no benefit.
- **L17** — Report the path and prediction count.
- **L18–19** — Blank line, then a lookup from `image_id` to its true `(height, width)`, used by the size checks below.
- **L20** — Counter for detections per image.
- **L21** — Walk every prediction.
- **L22** — Exactly the four required fields — no more, no fewer.
- **L23** — The id must exist in the target manifest.
- **L24** — The category must be `1`, not the model's internal `0`.
- **L25** — The score must be finite and inside `[0, 1]`. `NaN` would pass a naive range check, which is why finiteness is tested explicitly.
- **L26** — The RLE's declared `size` must equal this chip's dimensions. With 87 distinct chip sizes in the release there is no single right answer, so this is checked per prediction.
- **L27** — Stronger than the previous line: actually **decode** the mask and confirm its shape. This catches a corrupt `counts` string that merely claims the right size.
- **L28** — Tally detections for this image.
- **L29–30** — Blank line, then find images carrying more than ten predictions.
- **L31** — Report success and the coverage. Images with no predictions are legal, so this is informational rather than an assertion.
- **L32–33** — Warn about crowded images. Allowed, but predictions past the top ten by score cannot improve the official metric.

> **Watch out.** The three classic killers are all asserted here: `image_id` taken from the test manifest, `category_id` mapped back to `1`, and RLE `size` matching that specific chip.


## Cell 13 — Next steps *(markdown)*

This cell is prose, not code. It reads:

> ## Next steps
>
> Once `SMOKE_TEST = False` has produced a real baseline, in rough order of expected value:
>
> 1. **Input resolution.** `INPUT.MIN_SIZE_TRAIN` is the biggest lever on both memory and
>    accuracy. If you lower it, scale `MODEL.ANCHOR_GENERATOR.SIZES` by the same factor —
>    they are one coupled decision.
> 2. **Learning rate and schedule.** `0.0025` is the linear-scaling upper bound; fine-tuning
>    642 images often prefers less.
> 3. **Augmentation.** Rotations are as defensible as the flips already here, for the same
>    reason — nadir imagery has no canonical orientation.
> 4. **`MODEL.BACKBONE.FREEZE_AT`.** Default `2`. On 642 images, freezing early layers is a
>    reasonable regulariser; worth an ablation against `0`.
> 5. **More channels.** Only once the RGB pipeline is solid. That means leaving the PNG path
>    and reinstating a custom mapper — see the appendix of `detectron2_training_guide.md`.
>
> Compare runs on the number from cell 11, not on the `COCOEvaluator` output printed during
> training: those use COCO's area bins and `maxDets`, not this challenge's.
