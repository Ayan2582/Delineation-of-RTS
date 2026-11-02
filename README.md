# Delineation of Retrogressive Thaw Slumps (GeoAI Arctic Challenge 2026)

Instance segmentation of retrogressive thaw slumps (RTS) in 8-band Arctic image chips at 4 m. The bands
are RGB, NIR, NDVI, NDWI, relative elevation and shaded relief. Every model is Mask R-CNN-style, built on
torchvision and a Satlas-pretrained Swin-V2-B, and they differ in how the optical and terrain bands are
fused. Scores use the challenge's COCO mask AP (maxDets = 10).

## Repository layout

```
Experiments/            the training / evaluation pipeline (everything current lives here)
  configs/              base.yaml + one YAML per experiment (e1 … e6); `_base_` inheritance
  models/               backbones, fusion, band gate, boundary head, OHEM, semantic-seg head, model factory
  splits/               grouped_folds.json (GeoAI), darts_grouped_folds.json (DARTS): leak-free CV + local test
  config.py             config loading, CLI overrides, band-stream resolution
  data.py               release loading, splits, NaN-fill cache, per-fold stats, augmentation, loaders
  rts_preprocess.py     NaN fill, normalisation, padding collate, dataset
  train.py              one (experiment, fold, seed) run; best checkpoint by val AP
  eval.py               scoring (official AP, COCO AP, mIoU), CSV rows, cross-fold summaries
  plots.py              training curves, train/val/test bars, summary figure
  submission.py         checkpoint -> validated submission.json
  make_grouped_split.py builds splits/grouped_folds.json
  export_split.py       writes one fold of the split to disk as train/val/test folders
  smoke_test.py         random-batch test of every experiment + unit checks
  kaggle_rts_fusion.ipynb   Kaggle driver (edit the first cell only)
DARTS_EDA/              building a GeoAI-format dataset from DARTS labels + compatibility study
EDA/                    data walkthrough notebook and EDA figures
archive/                earlier approaches (detectron2, XGBoost pixels, ViT); see archive/README.md
```

These are not in git and must be provided locally (see `.gitignore`):
- `competition_release/`: the challenge data.
- `results/` and `submissions/`: run outputs.
- `DARTS_EDA/{chips,dataset,kaggle_darts}`: the built DARTS data.

## Setup

```bash
pip install torch torchvision torchgeo timm pycocotools scipy numpy pandas matplotlib pyyaml tensorboard
```

Put the challenge release at `./competition_release/`. It needs `train/`, `test/`, `metadata/` and `tools/`.
Config paths are relative to `Experiments/`, so the default `data.root` is `../competition_release`.

Check the install (CPU is enough):

```bash
cd Experiments
python smoke_test.py --override data.root=../competition_release
```

## Data and splits

The release chips are overlapping windows cut from larger scenes: 634 of the 756 train chips share
pixels with another chip. A random split therefore puts the same ground in train and val, which is how
an early detectron2 model scored 52 val AP but about 10 on the leaderboard.
`make_grouped_split.py` handles this in three steps:
1. It finds overlapping chips through identical 8×8 patches in RGB or shaded relief.
2. It groups them into connected components.
3. It assigns whole groups to a held-out **local test** set (15 %) plus 5 CV folds.

The local test chips are never trained on and never used to pick a checkpoint. Local test AP is the
number that tracks the leaderboard.

NaNs are filled by iterated 8-neighbour means (cached once). Bands are z-scored with each fold's train
stats, or mapped to the Satlas RGB/255 input (`data.swin_norm: satlas`).

## Experiments

| Config | Model |
|---|---|
| `e1` | Single stream: 3 bands into Swin-V2-B, then FPN |
| `e2` | Early fusion: all 8 bands into Swin-V2-B (stem expanded 3 → 8), then FPN |
| `e3` | Late fusion at C5 only: Swin (RGB+NIR) + ResNet18-GN (terrain), no FPN |
| `e4` | Late fusion at every level C2–C5, then FPN |
| `e5` | E4 backbone + semantic-FPN head; instances = connected components |
| `e2_gate` | E2 + learned per-image band gate (`models/gate.py`) |
| `e2_boundary` | E2 + boundary branch in the mask head (`models/boundary.py`) |
| `e6` | E2 + band gate + boundary head |

Shared options are set in `base.yaml` or with `--override`: OHEM (`model.ohem`), copy-paste
(`aug.copy_paste_p`), band dropout, input upscaling, and the optical backbone (Swin-V2-B/T, ResNet).

### Local test results (mask AP, mean ± std over folds)

| Experiment | Folds | AP | AP50 | AP75 |
|---|---|---|---|---|
| e1 | 2 | 0.104 ± 0.013 | 0.268 | 0.058 |
| e2 | 5 | 0.120 ± 0.008 | 0.322 | 0.060 |
| e3 | 2 | 0.110 ± 0.007 | 0.282 | 0.066 |
| e4 | 2 | 0.112 ± 0.016 | 0.283 | 0.060 |
| e5 (+copy-paste) | 1 | 0.120 | 0.263 | 0.077 |
| e2_gate (+cp, OHEM) | 2 | 0.134 ± 0.005 | 0.343 | 0.067 |
| e2_boundary (+cp, OHEM) | 2 | 0.139 ± 0.001 | 0.339 | 0.088 |
| **e6** | 2 | **0.149 ± 0.020** | **0.361** | **0.094** |
| e6 (+cp, OHEM) | 2 | 0.128 ± 0.017 | 0.347 | 0.062 |
| e6, DARTS-pretrained | 1 | 0.122 | 0.290 | 0.080 |

Recipes differ slightly between rows (folds, seeds, augmentation). The per-run CSVs and configs are in
`results/` (not in git).

## Usage

All commands run from `Experiments/`.

```bash
# build the leak-free split (already committed as splits/grouped_folds.json)
python make_grouped_split.py

# train one run; best.pth is chosen on val AP
python train.py --config configs/e6.yaml --fold 0 --seed 0
python train.py --config configs/e2.yaml --fold 0 --override flags.drop_ndwi=true train.epochs=40
python train.py --config configs/e6.yaml --fold all --override train.epochs=40   # final model, no val

# score a run
python eval.py --config configs/e6.yaml --fold 0 --split test    # local test -> results_test.csv
python eval.py --summary --split test                            # mean ± std across folds
python plots.py --out_dir outputs                                # curves + summary figure

# submission (runs the official validator)
python submission.py --model outputs/runs/e6/fold0_seed0/best.pth --check_val

# DARTS pretraining, then fine-tuning (resumable across Kaggle sessions)
python train.py --config configs/e6.yaml --fold 1 --override data.root=<kaggle_darts> \
    data.cache_dir=cache/darts data.folds_file=splits/darts_grouped_folds.json \
    data.rgb_ref=geoai data.swin_norm=satlas out_dir=outputs/darts_pretrain name=e6_darts_pt
python train.py --config configs/e6.yaml --fold 1 --override data.swin_norm=satlas name=e6_darts_ft \
    --init outputs/darts_pretrain/runs/e6_darts_pt/fold1_seed0/best.pth
```

On Kaggle, upload `Experiments/` as a dataset and use `kaggle_rts_fusion.ipynb`. Only its first cell needs
editing. It runs the smoke test, then the chosen experiments, folds and seeds, with optional DARTS
pretraining, and prints the cross-fold summaries.

## Findings so far

- **Overfitting is the ceiling.** Train AP is about 0.65 against local test AP of about 0.12–0.15. Larger
  or extra modules move test AP by roughly the fold-to-fold noise (±0.02).
- **Fold 1 val is easy.** Every model scores about 0.20–0.24 val AP on fold 1 but about 0.12 on test.
  Compare runs on local test AP, not val.
- **DARTS pretraining did not help** (0.122 test). It sped up convergence but did not raise the ceiling.
  Known issues:
  - The DARTS NDVI/NDWI normalisation stats are corrupted (std ≈ 10⁴, probably nodata), so those bands
    carried no signal during pretraining.
  - Zero-shot GeoAI AP peaked mid-pretraining (about 0.05), but the checkpoint was picked on DARTS val.
  - DARTS RGB comes from Sentinel-2 and is easily told apart from GeoAI's Maxar RGB (domain AUC 0.92).

  See `DARTS_EDA/report_darts_compatibility.md`.
