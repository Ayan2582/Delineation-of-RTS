## runnig  the base mask r cnn model

**Step 1:** Open this Google Drive link and click "Add shortcut to Drive". Make sure it is saved in your main Google Drive.
https://drive.google.com/drive/folders/1faffzPu8TLNH_enWa3wwT6mLWo6kYGFT?usp=drive_link

**step 2:**From the same Drive folder, also grab `model_0003019.pth` and `metrics.json`. Place them inside a folder at `/content/drive/MyDrive/geoai_arctic_checkpoints_d2_r101/` in your Drive — create that folder if it doesn't exist. This is where the notebook looks for the trained checkpoint when evaluating 
**Step 3:** Open my notebook (`Mask_r_cnn_base_model.ipynb`) using Google Colab.

**Step 4:** Run the first cell to mount your Drive (`drive.mount('/content/drive')`) and authorize it when prompted.

**Step 5:** Check that the paths in the config cell match where the shortcut landed. If you added the shortcut to the root of My Drive, these should work as-is:
```python
DATASET_ROOT = "/content/drive/MyDrive/competition_release"
OUTPUT_DIR   = "/content/drive/MyDrive/geoai_arctic_checkpoints_d2_r101"
```
If Drive put the shortcut somewhere else (e.g. it's nested in a folder), update these two lines to match the actual path — right-click the folder in Drive and "Copy path" if you're not sure.

**Step 6:** Run the cells top to bottom. To retrain from scratch, run the `trainer.train()` cell as-is. To skip straight to evaluating the existing checkpoint, go to the "LOAD MODEL WITH COMPETITION AP50 STRICTNESS" cell, keep `cfg.MODEL.WEIGHTS` pointing at `model_0003019.pth` in the checkpoints folder, and run from there.


Report: https://docs.google.com/document/d/1O5Nk7U8ja9J2FQog5ebQv7xDTiJyD64y2G_4pFmuNWQ/edit?usp=sharing
