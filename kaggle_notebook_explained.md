# The Kaggle notebook, cell by cell

A line-by-line walkthrough of `kaggle_cascade_mask_rcnn_rts.ipynb`. Every code cell appears here verbatim, with numbered lines, followed by an explanation of what each line does and why.

The reasoning behind the *choices* lives in `detectron2_training_guide.md`; this document explains the *code*.

The chips were converted to PNG by `prepare_png_dataset.py`, which is what lets detectron2's built-in data machinery do most of the work here.

---


## Cell 1 — Header *(markdown)*

This cell is prose, not code. It reads:

> # Cascade Mask R-CNN on the RTS challenge — vanilla RGB baseline
>
> Fine-tunes a COCO-pretrained **Cascade Mask R-CNN** (detectron2) on retrogressive thaw
> slump chips, using the **red/green/blue bands only**, and writes a validated
> `submission.json`.
>
> The model is the stock model-zoo R50-FPN Cascade Mask R-CNN with detectron2's default
> augmentation. Training evaluates on val every `EVAL_PERIOD` iterations, keeps the best
> checkpoint as `model_best.pth`, and **stops early** once `segm/AP` has not improved for
> `EARLY_STOP_PATIENCE` evaluations.
>
> ### Before you run this
>
> 1. Run `prepare_png_dataset.py` locally to build `data_png/` from the release, then
>    **upload that folder as a Kaggle Dataset** and attach it here. It is ~64 MB, against
>    ~547 MB for the raw `.npz` release.
> 2. Settings → **Accelerator: GPU**, and **Internet: On** for the first run (detectron2
>    has to be built from source).
> 3. Leave `SMOKE_TEST = True` for the first run. It proves the whole pipeline in about
>    fifteen minutes. Only then set it to `False` for the real run (at most ~2–3 hours).
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
15  CHECKPOINT_PERIOD = 1000
16  EVAL_PERIOD       = 1000
17  NUM_WORKERS       = 2
18
19  # Early stopping - on segm/AP, which with maxDets=10 is the official ranking metric
20  EARLY_STOP_METRIC   = "segm/AP"
21  EARLY_STOP_PATIENCE = 3            # evaluations without a new best before stopping
22  EARLY_STOP_START    = STEPS[0]     # never stop before the first LR drop
23
24  # Inference
25  DETECTIONS_PER_IMAGE = 20
26  SCORE_THRESH_TEST    = 0.05
27
28  CONFIG_YAML = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
29  OUTPUT_DIR  = "/kaggle/working/output"
30
31  if SMOKE_TEST:
32      MAX_ITER, STEPS, WARMUP_ITERS = 200, (150,), 50
33      CHECKPOINT_PERIOD, EVAL_PERIOD = 100, 50
34      EARLY_STOP_PATIENCE, EARLY_STOP_START = 1, 0     # let the stop path actually run
35      print("SMOKE_TEST is on - short run to prove the pipeline, not a competitive model")
36
37  print(f"iters={MAX_ITER}  batch={IMS_PER_BATCH}  lr={BASE_LR}  "
38        f"early stop: {EARLY_STOP_METRIC}, patience {EARLY_STOP_PATIENCE} evals "
39        f"x {EVAL_PERIOD} iters, from iter {EARLY_STOP_START}")
```

**Line by line**

- **L1–3** — A banner comment. Everything you would want to tune is between here and the end of the cell — nothing is hidden further down.
- **L4** — `SMOKE_TEST` is the single most useful switch in the notebook. When `True`, the block near the bottom shrinks the schedule to ~200 iterations so a complete pass — data, model, training, evaluation, early stopping, submission — finishes in minutes. Run it this way **first**; discovering a broken submission writer two hours into a real run is the failure this prevents.
- **L5–6** — Stage switches. Set `RUN_TRAIN = False` to re-run only inference against an existing checkpoint, which is what you want when iterating on the submission without retraining.
- **L7** — One seed, applied to `random`, `numpy` and `torch` in cell 4. Note this makes the run *reproducible*, not deterministic — cuDNN kernel selection still varies.
- **L8–9** — Blank line, then a comment recording the budget these numbers were chosen for. 642 is the training fold after the 15% validation split, not all 756 chips.
- **L10** — Images per iteration. Two is what fits alongside the default 800 px resize on a 16 GB T4 with AMP on. This is the number to lower first if you hit OOM.
- **L11** — `0.0025` is the reference `0.02` linearly rescaled from a batch of 16 to a batch of 2. Linear scaling with batch size is the standard rule; treat this as an upper bound, since fine-tuning 642 images usually wants less.
- **L12** — The iteration **ceiling**, not a target — early stopping usually ends the run before it. detectron2 counts iterations everywhere: at 642 images and batch 2 one epoch is 321 iterations, so 15,000 is roughly 47 epochs.
- **L13** — The two milestones where the learning rate drops by 10×. Set at about 2/3 and 9/10 of `MAX_ITER`, mirroring the reference schedule's 60k/80k out of 90k. `MultiStepLR` is fixed to these iterations, so early stopping cannot move them — it can only end the run after them.
- **L14** — Iterations spent ramping the learning rate up from near zero. With a small batch and freshly initialised heads, removing warmup is a reliable way to make the loss explode in the first hundred steps.
- **L15** — How often a resumable checkpoint is written. Matched to `EVAL_PERIOD`, so if a Kaggle session dies you lose at most one evaluation interval, and the early-stopping history in `early_stop.json` never runs far ahead of the weights you resume from.
- **L16** — How often validation runs, and therefore the resolution of early stopping. Halved from the baseline's 2000, because at 2000 a patience of 3 would mean 6000 iterations — most of the run — before a stop could fire. Each evaluation of the 114 val chips costs extra time, so this is the knob to raise if training feels slow.
- **L17** — Dataloader worker processes. Kaggle gives you few CPU cores; oversubscribing them starves the GPU rather than feeding it faster.
- **L18–19** — Blank line, then the early-stopping block.
- **L20** — The metric to watch, by its key in detectron2's event storage. `segm/AP` is mask AP@[.50:.95]; because the evaluator in cell 8 runs with `max_dets_per_image=10`, it is **exactly the challenge's ranking metric**, not a stock-COCO proxy for it.
- **L21** — Patience, counted in evaluations: training stops after this many evaluations in a row fail to beat the best so far. At `EVAL_PERIOD = 1000` that is 3000 iterations without a new best. Raise it if the log shows AP still creeping up in small, noisy steps when the run stops.
- **L22** — The earliest iteration at which stopping is allowed. In the baseline the **biggest single gain came from the LR drop at 10000** (46.1 → 51.6 AP), and AP before a drop often plateaus. Stopping on that plateau would throw away the best part of the run, so the hook keeps tracking the best checkpoint from the start but may only stop after `STEPS[0]`.
- **L23–24** — Blank line and a comment separating the inference settings.
- **L25** — How many detections the model may return per image. The official scorer uses `maxDets=10` and no training chip has more than 10 instances, so 20 is ample headroom — anything beyond the top 10 by score cannot improve the metric.
- **L26** — Detections below this confidence are discarded. Deliberately **low**: average precision rewards recall, and low-scoring false positives are nearly free once the top-10 cut is applied. Raising this to 0.5 throws away detections the metric would have credited.
- **L27–28** — Blank line, then the model-zoo config. `Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml` is the vanilla R-50-FPN Cascade Mask R-CNN at the 3× schedule (COCO box AP 44.3, mask AP 38.5).
- **L29** — Checkpoints and logs go here. On Kaggle only `/kaggle/working` survives the session, so this path is not arbitrary.
- **L30–31** — Blank line, then the smoke-test override block.
- **L32** — Collapses the schedule to 200 iterations with a single LR drop at 150 and 50 warmup steps — enough for the loss to move, far too few to learn anything.
- **L33** — Checkpoint every 100 iterations and evaluate every 50, so both code paths actually execute during the short run. A smoke test that never triggers evaluation has not tested evaluation.
- **L34** — Patience 1 and no start delay, so the early-stopping hook gets real chances to fire in the short run. It still only stops if AP fails to improve between two evaluations, so a smoke test that runs to 200 is not a failure.
- **L35** — States plainly what mode you are in. Mistaking a smoke-test checkpoint for a trained model is an easy and expensive confusion.
- **L36–39** — Blank line, then an echo of the effective settings, so the notebook's output records what was actually run.


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
 3  import logging
 4  import os
 5  import random
 6  from pathlib import Path
 7
 8  import cv2
 9  import matplotlib.pyplot as plt
10  from pycocotools import mask as mask_utils
11  from pycocotools.coco import COCO
12  from pycocotools.cocoeval import COCOeval
13
14  from detectron2 import model_zoo
15  from detectron2.checkpoint import DetectionCheckpointer
16  from detectron2.config import get_cfg
17  from detectron2.data import (DatasetCatalog, DatasetMapper, MetadataCatalog,
18                               build_detection_test_loader)
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

- **L1–6** — Standard library: `csv` reads the test manifest, `json` reads and writes COCO files and the early-stopping state, `logging` lets the early-stopping hook write into detectron2's log, `os` joins output paths, `random` is seeded below, `Path` is used for every filesystem operation in the notebook.
- **L7–8** — Blank line, then OpenCV — used to read PNGs for the sanity check. detectron2 uses it internally too.
- **L9** — matplotlib, only for displaying the sanity-check figure.
- **L10–12** — pycocotools, three separate entry points: `mask_utils` encodes, decodes and intersects RLE masks, `COCO` loads a ground-truth file, `COCOeval` computes average precision. Together they reproduce the challenge's exact metric locally, plus the mask-IoU numbers in cell 11.
- **L13–14** — Blank line, then `model_zoo`, which resolves a config name to both a YAML path and a pretrained-checkpoint URL.
- **L15** — `DetectionCheckpointer` loads weights into a model built outside the trainer — needed for the inference-only path.
- **L16** — `get_cfg()` returns a fresh config populated with detectron2's defaults.
- **L17–18** — The data API. `DatasetCatalog`/`MetadataCatalog` are the registries; `DatasetMapper` is the built-in mapper we can use unchanged because the images are PNGs; `build_detection_test_loader` turns a dataset into a single-pass `DataLoader`. There is no `build_detection_train_loader` import any more: the vanilla run uses `DefaultTrainer`'s own training loader.
- **L19** — The augmentation module, aliased `T` by convention. Now only needed for the deterministic resize in the validation-loss loader.
- **L20** — `register_coco_instances` is the function that makes the whole data layer three lines instead of a hundred.
- **L21** — `DefaultTrainer` supplies the training loop; `HookBase` is the base class for the validation-loss and early-stopping hooks in cell 8.
- **L22** — `COCOEvaluator` computes the in-loop metrics that early stopping reads.
- **L23** — `build_model` constructs a model from a config without a trainer, used by the inference path.
- **L24** — `setup_logger` makes detectron2's internal logging visible; without it, useful messages such as checkpoint shape mismatches — and the early-stopping progress lines — are swallowed.
- **L25** — `Visualizer` draws ground-truth and predicted masks over an image.
- **L26–27** — Blank line, then activate detectron2's logger.
- **L28–30** — Seed all three generators. `torch.manual_seed` covers CUDA too. This makes runs comparable; it does not make them bit-identical, since cuDNN still picks kernels non-deterministically.
- **L31–33** — Blank lines, then the data locator.
- **L34** — Its docstring: the search is anchored on the annotations file rather than on a directory name, so it works no matter what Kaggle names the attached dataset folder.
- **L35** — Two search roots: the Kaggle input mount first, then the current directory for local runs.
- **L36–38** — Skip a root that does not exist — `/kaggle/input` is absent when running locally.
- **L39–40** — Recursively search for `annotations/instances_train_fold.json` and return the directory two levels above it, which is `data_png/`. `sorted` keeps the choice stable when several copies are attached.
- **L41–44** — If nothing matched, fail immediately with an actionable message. A missing dataset should stop the notebook here, not surface as a confusing error inside the dataloader.
- **L45–48** — Blank lines, then resolve the path once and print it, so the notebook's output records which copy of the data was used.


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

Turns the model-zoo Cascade Mask R-CNN config into one adapted to this dataset. The model
stays vanilla — only dataset, solver and test-time keys change; the comments record which defaults were deliberately *kept*, which
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
- **L7** — The dataset used by the periodic evaluation — and therefore the one early stopping selects on.
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
- **L20** — Evaluation frequency, which is also how often early stopping gets to decide.
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


## Cell 8 — Trainer, validation loss and early stopping

`DefaultTrainer` supplies the loop, scheduler, checkpointing and logging. This cell keeps the
model and its training augmentation **vanilla**, and adds two hooks: one that computes the
validation loss `DefaultTrainer` never reports, and one that keeps the best checkpoint and ends
the run once validation AP stops improving — on 642 training images, overfitting (not the
iteration budget) is what limits this model.

```python
  1  class LossEvalHook(HookBase):
  2      """Periodically report the validation loss - DefaultTrainer never does."""
  3
  4      def __init__(self, period, model, loader):
  5          self._period = period
  6          self._model = model
  7          self._loader = loader
  8
  9      def _do_loss_eval(self):
 10          was_training = self._model.training
 11          self._model.train()
 12          totals, n = {}, 0
 13          with torch.no_grad():
 14              for batch in self._loader:
 15                  for k, v in self._model(batch).items():
 16                      totals[k] = totals.get(k, 0.0) + float(v)
 17                  n += 1
 18          self._model.train(was_training)
 19          means = {f"val_{k}": v / max(n, 1) for k, v in totals.items()}
 20          self.trainer.storage.put_scalars(val_total_loss=sum(means.values()), **means)
 21
 22      def after_step(self):
 23          nxt = self.trainer.iter + 1
 24          if self._period > 0 and nxt % self._period == 0 and nxt != self.trainer.max_iter:
 25              self._do_loss_eval()
 26
 27
 28  class EarlyStopping(Exception):
 29      """Raised by EarlyStoppingHook to end training before MAX_ITER."""
 30
 31
 32  def best_iteration(history):
 33      """Iteration (str key) with the highest score; ties go to the earliest."""
 34      return max(history, key=lambda k: (history[k], -int(k)))
 35
 36
 37  class EarlyStoppingHook(HookBase):
 38      """Save model_best.pth on `metric`; stop after `patience` evals with no new best.
 39
 40      Must run after EvalHook, whose result it reads from event storage. The per-eval
 41      history lives in early_stop.json, so a resumed Kaggle session keeps counting
 42      instead of starting over. model_best.pth is written with torch.save rather than
 43      the trainer's checkpointer, which would repoint `last_checkpoint` at it.
 44      """
 45
 46      def __init__(self, metric, patience, start_iter, output_dir):
 47          self._metric = metric
 48          self._patience = patience
 49          self._start_iter = start_iter
 50          self._state_path = Path(output_dir) / "early_stop.json"
 51          self._best_path = Path(output_dir) / "model_best.pth"
 52          self.state = {"history": {}, "stopped_at": None}
 53          if self._state_path.exists():
 54              self.state = json.loads(self._state_path.read_text())
 55
 56      def _check(self):
 57          """Record a fresh evaluation. Returns True once patience has run out."""
 58          latest = self.trainer.storage.latest().get(self._metric)
 59          if latest is None or latest[1] != self.trainer.storage.iter:
 60              return False                                  # no evaluation this step
 61          value, it = float(latest[0]), int(latest[1])
 62          hist = self.state["history"]
 63          hist[str(it)] = value if np.isfinite(value) else -1.0
 64
 65          best_it = best_iteration(hist)
 66          if best_it == str(it):
 67              model = getattr(self.trainer.model, "module", self.trainer.model)
 68              torch.save({"model": model.state_dict(), "iteration": it}, self._best_path)
 69          stale = sum(int(k) > int(best_it) for k in hist)
 70          self._state_path.write_text(json.dumps(self.state, indent=1))
 71
 72          logging.getLogger("detectron2").info(
 73              f"early stopping: {self._metric}={value:.2f} @ {it}, best {hist[best_it]:.2f} "
 74              f"@ {best_it}, {stale}/{self._patience} evals without a new best")
 75          return stale >= self._patience and it + 1 >= self._start_iter
 76
 77      def after_step(self):
 78          if self._check():
 79              self.state["stopped_at"] = self.trainer.iter
 80              self._state_path.write_text(json.dumps(self.state, indent=1))
 81              raise EarlyStopping(
 82                  f"early stop at iter {self.trainer.iter + 1}: no new best {self._metric} "
 83                  f"in {self._patience} evaluations")
 84
 85      def after_train(self):
 86          if self.trainer.iter + 1 >= self.trainer.max_iter:   # ran to MAX_ITER
 87              self._check()                                     # score the final eval too
 88
 89
 90  class RTSTrainer(DefaultTrainer):
 91      # build_train_loader is deliberately not overridden: vanilla DatasetMapper
 92      # augmentation (ResizeShortestEdge over INPUT.MIN_SIZE_TRAIN + horizontal flip).
 93
 94      @classmethod
 95      def build_test_loader(cls, cfg, dataset_name):
 96          return build_detection_test_loader(
 97              cfg, dataset_name, mapper=DatasetMapper(cfg, is_train=False))
 98
 99      @classmethod
100      def build_evaluator(cls, cfg, dataset_name, output_folder=None):
101          # max_dets_per_image=10 makes segm AP, AP50 and AP75 the official numbers.
102          # APs/APm/APl here still use COCO's area bins - cell 11 has the official ones.
103          return COCOEvaluator(dataset_name, output_dir=output_folder or cfg.OUTPUT_DIR,
104                               max_dets_per_image=10)
105
106      def build_hooks(self):
107          hooks = super().build_hooks()
108          loss_loader = build_detection_test_loader(
109              self.cfg, self.cfg.DATASETS.TEST[0],
110              mapper=DatasetMapper(self.cfg, is_train=True, augmentations=[
111                  T.ResizeShortestEdge(self.cfg.INPUT.MIN_SIZE_TEST,
112                                       self.cfg.INPUT.MAX_SIZE_TEST)]))
113          # Both go just before PeriodicWriter, and so after EvalHook.
114          hooks.insert(-1, LossEvalHook(self.cfg.TEST.EVAL_PERIOD, self.model, loss_loader))
115          hooks.insert(-1, EarlyStoppingHook(EARLY_STOP_METRIC, EARLY_STOP_PATIENCE,
116                                             EARLY_STOP_START, self.cfg.OUTPUT_DIR))
117          return hooks
```

**Line by line**

- **L1–2** — The validation-loss hook. `HookBase` gives access to `self.trainer` once registered. `DefaultTrainer` runs metric evaluation on a schedule but never computes a validation *loss*, and loss is the earliest overfitting signal.
- **L3–7** — Blank line, then the constructor: store the evaluation period, the model and the validation loader. Nothing heavy happens here — the loader is built once by the caller and reused every time the hook fires.
- **L8–9** — Blank line, then the evaluation body.
- **L10** — Remember whether the model was in training mode, so it can be restored exactly.
- **L11** — **Switch to train mode deliberately.** A detectron2 model returns a dict of losses in train mode and a list of predictions in eval mode. We want losses, so train mode it is — `torch.no_grad()` below is what prevents this from actually updating anything.
- **L12** — Accumulators for the loss totals and the batch count.
- **L13** — No gradients: this is measurement, not learning. Without it, memory balloons and the graph is retained.
- **L14–17** — Iterate the whole validation fold, accumulate each loss component by name (classification, box regression, mask, RPN) and count batches for the mean.
- **L18** — Restore the model's previous mode. Leaving it in train mode would silently corrupt the next evaluation.
- **L19** — Average each component and prefix with `val_` so the keys do not collide with training losses in the logs.
- **L20** — Write to detectron2's event storage, adding a `val_total_loss` summary alongside the components. Anything put here shows up in the printed metrics and in TensorBoard.
- **L21–25** — The callback detectron2 calls after every iteration. `iter + 1` because `trainer.iter` is zero-based; fire only on the period and skip the final iteration, which only duplicates the trainer's own end-of-run evaluation.
- **L26–29** — Blank lines, then a dedicated exception type. detectron2's loop is a plain `for self.iter in range(start_iter, max_iter)`, so there is no flag a hook can set to leave it — raising is the only clean exit, and a named type lets cell 9 catch *this* exception without swallowing real crashes.
- **L30–34** — A helper shared with cell 9: the iteration with the highest score. The sort key `(score, -iteration)` breaks ties toward the **earliest** iteration, so an equal score later on neither overwrites `model_best.pth` nor resets the patience count.
- **L35–44** — The early-stopping hook and its docstring, which records the two design constraints explained below.
- **L45–46** — Blank line, then the constructor arguments: the metric key, patience in evaluations, the earliest iteration at which stopping is allowed, and the output directory.
- **L47–51** — Store them and derive the two files this hook owns: `early_stop.json` (state) and `model_best.pth` (weights).
- **L52–54** — Fresh state, **unless `early_stop.json` already exists** — which is the resume case. Kaggle kills sessions; `resume_or_load(resume=True)` restores the weights, optimizer and iteration counter, but a hook's in-memory counters would restart from nothing. Reloading the history from disk is what lets patience keep counting across sessions. (detectron2's built-in `BestCheckpointer` has exactly this gap, which is one reason it is not used.)
- **L55–57** — Blank line, then `_check`, records one evaluation and reports whether patience has run out.
- **L58** — Read the most recent value of the metric. `storage.latest()` maps each key to `(value, iteration)`.
- **L59–60** — **The freshness test.** `latest()` returns the last value ever written, so on the 999 iterations between evaluations it would hand back a stale score. Only when the value's iteration equals the current storage iteration did `EvalHook` just write it. This also covers an evaluation that produced nothing (for instance zero predictions early in training, when `COCOEvaluator` returns no `segm` results).
- **L61** — Unpack as plain Python types, so they serialise to JSON.
- **L62–63** — Record the score under its iteration. Keying by iteration (as a string — JSON keys must be strings) makes a re-run evaluation after a resume **overwrite** its earlier entry instead of being counted twice. A `NaN` score is stored as `-1`, so it can never become the best.
- **L64–65** — Blank line, then find the best iteration across the whole history.
- **L66–68** — If this evaluation *is* the best, save its weights. Two deliberate choices. First, `torch.save` of `{"model": state_dict}` rather than `trainer.checkpointer.save`: the checkpointer rewrites the `last_checkpoint` pointer on every save, so a resume would restart from the best iteration instead of the latest one. The `{"model": ...}` layout is what `DetectionCheckpointer.load` expects, so cell 10 loads it like any other checkpoint. Second, `getattr(..., "module", ...)` unwraps a `DistributedDataParallel` model if there is one, so the saved keys have no `module.` prefix.
- **L69** — Patience is counted, not incremented: the number of evaluations **after** the best one. Deriving it from the history each time is what keeps it correct after a resume, where evaluations may be repeated.
- **L70** — Persist the state after every evaluation.
- **L71–74** — Blank line, then one progress line per evaluation in detectron2's log, so you can watch the count approach the patience.
- **L75** — Stop only when both conditions hold: patience has run out **and** the run is past `EARLY_STOP_START`. Before that point the best checkpoint is still tracked — it just cannot end the run.
- **L76–83** — After each iteration, run the check; on a stop, record the iteration in `early_stop.json` and raise. The `stopped_at` marker is what stops cell 9 from resuming a finished run.
- **L84–87** — `after_train` runs in a `finally` block after the loop, whether it completed or raised. `EvalHook` does its final evaluation there at `MAX_ITER`, so this hook scores that evaluation too — otherwise the last 1000 iterations could never produce `model_best.pth`. After an early stop, `iter + 1 < max_iter`, so nothing happens.
- **L88–92** — The trainer subclass. `build_train_loader` is **not overridden**, which is what "vanilla" means here: `DefaultTrainer` builds the stock `DatasetMapper(cfg, is_train=True)`, whose augmentation is `ResizeShortestEdge` over `INPUT.MIN_SIZE_TRAIN` plus a random horizontal flip. The baseline's extra vertical flip is gone. The mapper still reads `INPUT.MASK_FORMAT = "bitmask"` from the config, so the RLE labels are handled as before.
- **L93–97** — The test-loader override: the same built-in mapper with `is_train=False`, meaning deterministic resize, no flips, and no ground-truth instances.
- **L98–104** — The evaluator. **`max_dets_per_image=10` is the important argument.** detectron2 expands it to COCO `maxDets=[1, 10, 10]` and then reports AP at the last entry, so `segm/AP`, `AP50` and `AP75` become the challenge's own numbers — the stock `[1, 10, 100]` would let up to 20 detections per image count. The `all` area range is identical to the official one. Only `APs`/`APm`/`APl` still use COCO's area bins (32² and 96², not 300 and 2000), so read those from cell 11 instead.
- **L105–107** — Hook registration, starting from `DefaultTrainer`'s own list: timer, LR scheduler, periodic checkpointer, `EvalHook`, and `PeriodicWriter` last.
- **L108–112** — Build the loader for validation loss. **The subtlety: `is_train=True`.** The ordinary test loader's mapper produces no ground-truth instances, so the model could not compute a loss from it. We need training-style targets but deterministic geometry, so it is a *test* loader (no shuffling, single pass) with a *train* mapper restricted to a plain resize.
- **L113–116** — Insert both hooks at `-1`, immediately **before** `PeriodicWriter`. Hooks run in list order, so this places them after `EvalHook` — the early-stopping hook must run *after* the evaluation it reads, or it would always see the previous one — and before the writer, so their values are logged in the same interval.
- **L117** — Return the hook list.

> **Watch out.** Early stopping selects `model_best.pth` using the val fold, so that checkpoint's val AP is a slightly **optimistic** estimate — it is the best of up to 15 looks at the same 114 chips. Comparing two runs on val is still fair; the test leaderboard is the unbiased number.


## Cell 9 — Train

The whole training run. Short, because everything was arranged in the cells above — and
resumable, because Kaggle will kill the session at twelve hours regardless of progress.

```python
 1  stop_file = Path(OUTPUT_DIR) / "early_stop.json"
 2  already_stopped = (stop_file.exists()
 3                     and json.loads(stop_file.read_text()).get("stopped_at") is not None)
 4
 5  if not RUN_TRAIN:
 6      print("RUN_TRAIN is False - skipping training")
 7  elif already_stopped:
 8      print(f"this run already early-stopped ({stop_file}) - "
 9            f"clear {OUTPUT_DIR} to start a new one")
10  else:
11      trainer = RTSTrainer(cfg)
12      trainer.resume_or_load(resume=True)
13      try:
14          trainer.train()
15          print("training ran to MAX_ITER ->", os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
16      except EarlyStopping as e:
17          # detectron2 logs this as "Exception during training" first - that is expected.
18          print(e)
19
20  if stop_file.exists():
21      hist = json.loads(stop_file.read_text())["history"]
22      best_it = best_iteration(hist)
23      print(f"best {EARLY_STOP_METRIC} = {hist[best_it]:.2f} at iter {best_it} "
24            f"-> {os.path.join(OUTPUT_DIR, 'model_best.pth')}")
```

**Line by line**

- **L1** — The state file written by the early-stopping hook.
- **L2–3** — Has this output directory already **finished** by early stopping? Without this check, re-running the cell would call `resume_or_load(resume=True)`, pick up the last periodic checkpoint, and quietly start training again from where the run was stopped.
- **L4–6** — Blank line, then the stage switch, so you can re-run the notebook for inference alone.
- **L7–9** — A finished run is left alone, with instructions to clear the output directory to start a fresh one. This also catches the easy mistake of flipping `SMOKE_TEST` off in a session whose `/kaggle/working/output` still holds the smoke test.
- **L10–11** — Construct the trainer. This builds the model, loads `cfg.MODEL.WEIGHTS`, and constructs the optimizer, scheduler, dataloaders and hooks. **Watch the log here**: it lists checkpoint tensors that were skipped on a shape mismatch. Skips on the final class predictors are expected and correct — the COCO checkpoint has 81 classes and this model has 2. Anything *else* being skipped is a bug worth chasing.
- **L12** — `resume=True` is what makes chained Kaggle sessions work: if `OUTPUT_DIR` holds a `last_checkpoint`, it restores the weights **and** the iteration counter and optimizer state, and the early-stopping hook reloads its history from `early_stop.json`. With `resume=False` it would load the pretrained weights and restart from iteration 0.
- **L13–15** — Train. If this returns normally, the run reached `MAX_ITER` without triggering early stopping.
- **L16–18** — Early stopping arrives as the `EarlyStopping` exception. detectron2's loop logs any exception with a full traceback (`Exception during training:`) before re-raising it, so **that traceback is expected** — this `except` is what turns it into a normal end of training. Any other exception still propagates.
- **L19–24** — Blank line, then report the best evaluation and where its weights are, whichever way training ended. This is also the number to note down for the run.

> **Watch out.** If the session dies, save `OUTPUT_DIR` as a Kaggle Dataset, attach it to the next session, copy it back into `/kaggle/working/output`, and re-run — `resume_or_load(resume=True)` picks up the weights and `early_stop.json` picks up the patience count. Copy the **whole** directory: `last_checkpoint`, the `model_*.pth` files, `model_best.pth` and `early_stop.json` belong together.


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
10      if weights is None:                        # best val checkpoint, else the last one
11          best = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
12          weights = best if os.path.exists(best) else os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
13      cfg_i.MODEL.WEIGHTS = weights
14      print("weights:", weights)
15      model = build_model(cfg_i)
16      DetectionCheckpointer(model).load(cfg_i.MODEL.WEIGHTS)
17      model.eval()
18      return cfg_i, model
19
20
21  def predict(model, cfg_i, dataset, top_k=10):
22      """Run the model over a dataset name or a list of dicts -> COCO results."""
23      loader = build_detection_test_loader(
24          dataset if isinstance(dataset, list) else cfg_i,
25          **({} if isinstance(dataset, list) else {"dataset_name": dataset}),
26          mapper=DatasetMapper(cfg_i, is_train=False),
27      )
28      results = []
29      with torch.no_grad():
30          for batch in loader:
31              for inp, out in zip(batch, model(batch)):
32                  inst = out["instances"].to("cpu")
33                  order = inst.scores.argsort(descending=True)[:top_k]
34                  for i in order.tolist():
35                      results.append({
36                          "image_id": int(inp["image_id"]),
37                          "category_id": 1,
38                          "segmentation": encode_binary_mask(inst.pred_masks[i].numpy()),
39                          "score": float(inst.scores[i]),
40                      })
41      return results
```

**Line by line**

- **L1** — The RLE encoder, identical in behaviour to `encode_binary_mask` in the release's `tools/coco_utils.py`. It is inlined so the notebook needs nothing on `sys.path` when running on Kaggle.
- **L2** — Its docstring.
- **L3** — Two details that are both mandatory. `asfortranarray` gives column-major order, which is what COCO RLE assumes — pass a C-ordered array and the encoding is scrambled. `astype(uint8)` because pycocotools rejects bool.
- **L4** — `mask_utils.encode` returns `counts` as **bytes**, which `json.dump` cannot serialise. Decoding to a UTF-8 string is what makes the prediction writable.
- **L5** — Return the RLE dict.
- **L6–8** — Blank lines, then the model loader.
- **L9** — Rebuild the config from scratch rather than mutating the training one, so inference cannot be affected by leftover state.
- **L10–12** — With no explicit checkpoint, prefer `model_best.pth` — the early-stopping pick — and fall back to `model_final.pth` for a run trained before early stopping existed. After an early stop there *is* no `model_final.pth`, so the old default would have failed.
- **L13–14** — Set the weights, and print which file was chosen so the output records what was scored.
- **L15** — `build_model` constructs the architecture from the config, with random weights.
- **L16** — `DetectionCheckpointer(...).load(...)` fills in the trained weights. It reads the `{"model": state_dict}` layout that both detectron2's checkpoints and `model_best.pth` use.
- **L17** — **Eval mode.** In eval mode the model returns predictions; in train mode it would return a loss dict. Forgetting this produces a confusing `KeyError` on `instances` later.
- **L18** — Return both, since prediction needs the config too.
- **L19–21** — Blank lines, then the prediction driver.
- **L22** — Its docstring. It accepts either a registered dataset name (the val fold) or a plain list of dicts (the unregistered test set built in cell 12).
- **L23–27** — Build a test loader for either case: a list is passed as the dataset directly, while a name goes through the config-based form with `dataset_name`. The mapper is the built-in one in eval mode: the **same** preprocessing as training, which is the whole reason for not using `DefaultPredictor` here.
- **L28** — Accumulator for the COCO-format results.
- **L29** — No gradients during inference.
- **L30** — Iterate batches.
- **L31** — Zip inputs with outputs so each prediction keeps the `image_id` of the image it came from.
- **L32** — Move the `Instances` to CPU once, rather than per-field.
- **L33** — Sort detections by score, highest first, and keep only the top `top_k`. The official metric uses `maxDets=10`, so anything past the tenth cannot help.
- **L34** — Loop over the surviving indices.
- **L35** — Open the prediction dict.
- **L36** — The image id, taken from the input record so it can never drift out of step with the image.
- **L37** — **Always `1`.** The model predicts contiguous class `0`; the submission format requires the original category id `1`. This is the mapping's other half, and getting it wrong yields a structurally valid file that scores zero.
- **L38** — `pred_masks[i]` is a bool tensor already rescaled to the chip's original size — detectron2 uses the `height`/`width` in the input record to do that. Encode it to RLE.
- **L39** — The confidence, cast to a plain float so it is JSON-serialisable.
- **L40** — Close the dict and append.
- **L41** — Return all predictions.

> **Watch out.** `DefaultPredictor` would read the image from disk with its own preprocessing, bypassing this config. Any divergence between training and inference preprocessing costs accuracy silently, which is why inference reuses the same mapper.


## Cell 11 — Scoring: official AP breakdown and mask IoU

`COCOEvaluator`'s area bins differ from this challenge's. This cell reproduces the official
scorer's settings exactly — its six AP numbers were checked to match
`competition_release/tools/evaluate_coco.py` to three decimals — and adds two mask-IoU numbers
that AP does not show directly.

```python
  1  OFFICIAL_MAXDETS = [1, 5, 10]
  2  OFFICIAL_AREA_RNG = [[0, 1e10], [0, 300], [300, 2000], [2000, 1e10]]
  3  OFFICIAL_AREA_LBL = ["all", "small", "medium", "large"]
  4
  5
  6  def _ap(ev, area="all", iou_thr=None, max_det=10):
  7      """Mean precision for one row of the challenge's summary table."""
  8      prec = ev.eval["precision"]
  9      if iou_thr is not None:
 10          prec = prec[np.isclose(ev.params.iouThrs, iou_thr)]
 11      prec = prec[:, :, :, ev.params.areaRngLbl.index(area), ev.params.maxDets.index(max_det)]
 12      valid = prec[prec > -1]
 13      return float(valid.mean()) if valid.size else -1.0
 14
 15
 16  def iou_metrics(coco_gt, predictions, max_det=10, match_thr=0.5, fg_score=0.5):
 17      """Two mask-IoU numbers that AP does not show directly.
 18
 19      matched_iou    - mean IoU of prediction/GT pairs, greedily matched by score
 20                       (top `max_det` per image, IoU >= `match_thr`), as COCOeval does.
 21      foreground_iou - pixel IoU of the union of predictions scoring >= `fg_score`
 22                       against the union of GT masks, summed over all images.
 23      """
 24      by_img = {}
 25      for p in predictions:
 26          by_img.setdefault(p["image_id"], []).append(p)
 27
 28      matched, inter, union = [], 0, 0
 29      for img_id in coco_gt.getImgIds():
 30          info = coco_gt.imgs[img_id]
 31          h, w = info["height"], info["width"]
 32          gts = [coco_gt.annToRLE(a) for a in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))]
 33          dts = sorted(by_img.get(img_id, []), key=lambda p: -p["score"])[:max_det]
 34
 35          if gts and dts:
 36              ious = mask_utils.iou([d["segmentation"] for d in dts], gts, [0] * len(gts))
 37              taken = set()
 38              for row in np.atleast_2d(ious):
 39                  best, best_j = match_thr, -1
 40                  for j, v in enumerate(row):
 41                      if j not in taken and v >= best:
 42                          best, best_j = v, j
 43                  if best_j >= 0:
 44                      taken.add(best_j)
 45                      matched.append(best)
 46
 47          gt_fg = np.zeros((h, w), bool)
 48          for r in gts:
 49              gt_fg |= mask_utils.decode(r).astype(bool)
 50          dt_fg = np.zeros((h, w), bool)
 51          for d in dts:
 52              if d["score"] >= fg_score:
 53                  dt_fg |= mask_utils.decode(d["segmentation"]).astype(bool)
 54          inter += int((gt_fg & dt_fg).sum())
 55          union += int((gt_fg | dt_fg).sum())
 56
 57      return {
 58          "matched_iou": float(np.mean(matched)) if matched else 0.0,
 59          "n_matched": len(matched),
 60          "foreground_iou": inter / union if union else 0.0,
 61      }
 62
 63
 64  def score_official(gt_json_path, predictions):
 65      """COCO segm AP using the challenge's maxDets and area ranges, plus mask IoU."""
 66      if not predictions:
 67          print("no predictions - nothing to score")
 68          return None
 69      coco_gt = COCO(str(gt_json_path))
 70      # copies: loadRes writes bbox/area into each dict, which corrupts a second scoring
 71      coco_dt = coco_gt.loadRes([dict(p) for p in predictions])
 72      ev = COCOeval(coco_gt, coco_dt, "segm")
 73      ev.params.maxDets = OFFICIAL_MAXDETS
 74      ev.params.areaRng = OFFICIAL_AREA_RNG
 75      ev.params.areaRngLbl = OFFICIAL_AREA_LBL
 76      ev.evaluate()
 77      ev.accumulate()
 78
 79      metrics = {
 80          "AP":        _ap(ev),
 81          "AP50":      _ap(ev, iou_thr=0.50),
 82          "AP75":      _ap(ev, iou_thr=0.75),
 83          "AP_small":  _ap(ev, area="small"),
 84          "AP_medium": _ap(ev, area="medium"),
 85          "AP_large":  _ap(ev, area="large"),
 86      }
 87      n_gt = len(coco_gt.getAnnIds())
 88      metrics.update(iou_metrics(coco_gt, predictions))
 89
 90      print(f"AP        @[IoU=0.50:0.95 | area=all    | maxDets=10] = {metrics['AP']:.4f}   <- ranking metric")
 91      print(f"AP50      @[IoU=0.50      | area=all    | maxDets=10] = {metrics['AP50']:.4f}")
 92      print(f"AP75      @[IoU=0.75      | area=all    | maxDets=10] = {metrics['AP75']:.4f}")
 93      print(f"AP_small  @[IoU=0.50:0.95 | area=small  | maxDets=10] = {metrics['AP_small']:.4f}")
 94      print(f"AP_medium @[IoU=0.50:0.95 | area=medium | maxDets=10] = {metrics['AP_medium']:.4f}")
 95      print(f"AP_large  @[IoU=0.50:0.95 | area=large  | maxDets=10] = {metrics['AP_large']:.4f}")
 96      print(f"matched mask IoU (IoU>=0.5)                        = {metrics['matched_iou']:.4f}"
 97            f"   ({metrics['n_matched']}/{n_gt} GT instances matched)")
 98      print(f"foreground IoU   (score>=0.5, all images)          = {metrics['foreground_iou']:.4f}")
 99      return metrics
100
101
102  if RUN_INFER:
103      cfg_i, model = load_trained_model()
104      val_preds = predict(model, cfg_i, "rts_val", top_k=10)
105      print(f"{len(val_preds)} predictions over {len(val_dicts)} validation images")
106      val_metrics = score_official(DATA / "annotations" / "instances_val_fold.json", val_preds)
```

**Line by line**

- **L1** — `maxDets` from the official scorer: `[1, 5, 10]`. COCO's default is `[1, 10, 100]`.
- **L2** — The official area ranges. COCO's defaults are 32² = 1024 and 96² = 9216; this challenge uses **300** and **2000**, which reclassifies most of this dataset. Under the official bins the split is 25% small / 51% medium / 24% large.
- **L3** — The labels, in the same order as the ranges — `COCOeval` pairs them positionally.
- **L4–6** — Blank lines, then `_ap`: one row of the official summary table, mirroring `summarize_metric` in `evaluate_coco.py`.
- **L7** — Its docstring.
- **L8** — `eval["precision"]` is indexed `[IoU threshold, recall point, category, area range, maxDets]`.
- **L9–10** — For AP50 or AP75, keep only that IoU threshold. `np.isclose` rather than `==`, because the thresholds come from `np.linspace` and `0.75` is not exactly representable. With no threshold, all ten (0.50:0.05:0.95) stay — which is what `AP@[.50:.95]` means.
- **L11** — Select the area range and `maxDets` **by label and value** rather than by hardcoded index, so this cannot silently read the wrong column if either list changes.
- **L12–13** — Average the valid cells. `COCOeval` writes `-1` where a cell has no data — for example an area bin with no ground truth — and those must be excluded or they drag the mean down. `-1` is returned if nothing is valid, as the official scorer does.
- **L14–16** — Blank lines, then the IoU function. Its two outputs answer different questions from AP.
- **L17–23** — The docstring defines both. **Matched IoU** asks *how good are the masks the model got right* — AP75 versus AP50 hints at this, but this reports it directly. **Foreground IoU** ignores instances altogether and asks *how much of the thaw-slump area does the model cover* — the number to quote if the downstream use is mapping disturbed ground rather than counting slumps.
- **L24–26** — Group predictions by image.
- **L27–29** — Blank line, then accumulators: the IoU of every matched pair, and pooled pixel intersection and union, then a loop over every **ground-truth** image — so an image the model predicted nothing on still contributes its full ground-truth area to the union.
- **L30–31** — The image size, needed to build empty masks.
- **L32** — The image's ground-truth masks as RLE. `annToRLE` normalises polygons or uncompressed RLE, though this dataset only has compressed RLE.
- **L33** — The top `max_det` predictions by score — the same cut the official metric applies.
- **L34–36** — Blank line, then the pairwise mask-IoU matrix, predictions × ground truth, computed directly on RLE. The `iscrowd` flags are all `0`: this data has no crowd annotations, and a `1` would switch pycocotools to intersection-over-prediction-area.
- **L37–45** — **Greedy matching, in score order**, which is how `COCOeval` matches too: each prediction, highest score first, claims the unclaimed ground-truth instance it overlaps most, provided that overlap is at least `match_thr`. `atleast_2d` guards the single-prediction case. Each ground-truth instance is matched at most once, so duplicate detections of one slump do not inflate the average.
- **L46–53** — Blank line, then foreground IoU. Merge the image's ground-truth masks into one boolean map and the predictions scoring at least `fg_score` into another. The 0.5 threshold matters here in a way it does not for AP: AP ranks by score, whereas this is a hard yes/no per pixel, so the low-confidence tail kept for AP would otherwise paint large false-positive areas.
- **L54–55** — Accumulate intersection and union **as pixel counts across all images**, not as a mean of per-image IoUs. A per-image mean would let a 150-pixel chip weigh as much as a 290×290 one, and an image with no prediction and a tiny slump would pull it down as hard as a large miss.
- **L56–61** — Blank line, then return the mean matched IoU, the number of matches (a high IoU over few matches is not a good model), and the pooled foreground IoU. Each guards its empty case.
- **L62–64** — Blank lines, then the scoring function.
- **L65** — Its docstring.
- **L66–68** — An empty prediction list is legal but `loadRes` chokes on it, so return early with a clear message instead of an opaque pycocotools error.
- **L69** — Load the ground truth for this fold. Its `category_id` values are the original `1`, matching what `predict` emits.
- **L70–71** — **Score copies of the predictions.** `loadRes` writes `bbox`, `area`, `id` and `iscrowd` *into each dict it is given*. Scoring the same list a second time — re-running the cell, or saving it and running `evaluate_coco.py` on it — would then hit pycocotools' `bbox` branch, which recomputes each area from the *box* instead of the mask, and every small/medium/large number shifts (AP_medium moved from 0.572 to 0.543 in testing). `dict(p)` is a shallow copy, which is enough: `loadRes` only adds top-level keys.
- **L72** — Build the evaluator in **`segm`** mode — this challenge scores masks, not boxes.
- **L73–75** — Override the three parameters that differ from COCO's defaults. Setting these *after* construction is required, since `COCOeval` fills in defaults during `__init__`.
- **L76–77** — Match predictions to ground truth, then aggregate into the precision/recall arrays.
- **L78–86** — Blank line, then the six AP numbers the leaderboard displays, each one `_ap` call. `AP` — IoU 0.50:0.95, all areas, top 10 — is the **ranking metric**; the other five explain it. From the baseline, AP50 flat while AP75 climbs means the model already *finds* slumps and is gaining on mask boundaries, and a low AP_small next to a high AP_large points at resolution and anchors.
- **L87–88** — Count ground-truth instances for the match ratio, and add the IoU numbers to the same dict.
- **L89–98** — Blank line, then print everything in one aligned table.
- **L99** — Return the dict, so runs can be compared programmatically.
- **L100–102** — Blank lines, then the inference block, guarded by its stage switch.
- **L103** — Load the trained weights — `model_best.pth` by default, per cell 10.
- **L104** — Predict over the validation fold, capped at 10 detections per image to match the metric.
- **L105** — Report the prediction count — roughly 10× the image count, since the low score threshold lets most slots fill.
- **L106** — Score against the validation fold's ground truth and keep the result as `val_metrics`. **These are the numbers to compare between experiments.**


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
> 3. **Augmentation.** This run is vanilla: detectron2's default resize + horizontal flip.
>    Vertical flips and 90° rotations are the first additions to try — nadir imagery has
>    no canonical orientation.
> 4. **`MODEL.BACKBONE.FREEZE_AT`.** Default `2`. On 642 images, freezing early layers is a
>    reasonable regulariser; worth an ablation against `0`.
> 5. **More channels.** Only once the RGB pipeline is solid. That means leaving the PNG path
>    and reinstating a custom mapper — see the appendix of `detectron2_training_guide.md`.
>
> Compare runs on the numbers from cell 11. During training, `segm/AP`, `AP50` and `AP75` already
> match the official scorer (the evaluator uses `maxDets=10`), but `APs/APm/APl` there still use
> COCO's area bins. Early stopping picks `model_best.pth` *on val*, so its val AP is slightly
> optimistic — the test leaderboard is the unbiased check.
