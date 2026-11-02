# Archive

Earlier approaches, kept for reference. Nothing here is used by the current pipeline in `Experiments/`.

| Path | What it was | Why it was superseded |
|---|---|---|
| `detectron2/kaggle_cascade_mask_rcnn_rts.ipynb` | detectron2 Cascade Mask R-CNN, RGB only, on Kaggle | Val segm AP 52 on random folds, ~10 on the leaderboard: the random split leaked overlapping chips. Replaced by the torchvision multi-band pipeline with a grouped split. |
| `detectron2/Mask_r_cnn_base_model.ipynb` | First plain Mask R-CNN baseline (Colab) | Same as above. |
| `detectron2/rts_inference.ipynb` | Inference and visualisation for the detectron2 models | `Experiments/submission.py` and `Experiments/eval.py`. |
| `detectron2/prepare_png_dataset.py` | `.npz` chips to RGB PNGs so detectron2's mapper can read them | The current code reads all 8 bands from `.npz` directly. |
| `xgboost_pixel/` | Per-pixel band table (`prepare_dataset`), profiling (`eda`), XGBoost pixel classifier (`model`) | A pixel classifier gives no instances; used only to study band separability. |
| `vit/vit.ipynb` | ViT learning notebook (MNIST) | Learning exercise, not part of the RTS work. |
| `base_eda.ipynb` | First look at one chip | `EDA/00_data_basics.ipynb` covers it properly. |
