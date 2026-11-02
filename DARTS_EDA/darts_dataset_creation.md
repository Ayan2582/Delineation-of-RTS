# How the DARTS dataset was built

This document explains how the extra training set in `DARTS_EDA/dataset/` (and its Kaggle copy `DARTS_EDA/kaggle_darts/`) was made from the DARTS retrogressive-thaw-slump labels, so that it looks like the GeoAI Arctic challenge data: same 8 bands in the same order, same 4 m grid, same chip sizes, same COCO annotation format.

**In one line:** DARTS gives us *where the slumps are* (hand-drawn polygons). It gives us **no images**. So every pixel was rebuilt from free public sources: Sentinel-2 for the optical bands and ArcticDEM for the terrain bands. The polygons were then burned into masks on that grid.

| | |
|---|---|
| Chips | **1,861** (out of 1,885 attempted) |
| Slump masks | **6,180** (about 3,450 distinct slumps, because chips overlap) |
| Chip size | copied from GeoAI, median 158 × 289 px at 4 m |
| Regions | 15 (Banks Island, Horton, Novaya Zemlya, Lena, Kolguev, ...) |
| Label years | 2018–2022 |
| Size on disk | 2.8 GB (`dataset/`), 2.5 GB (`kaggle_darts/`) |

---

## 1. Pipeline at a glance

```
DARTS v1.2 training labels (3,749 polygons, 162 PlanetScope images, 6 iterations)
        │  drop polygons < 900 m²                         → 3,450 polygons
        │  plan chips so every polygon is covered once    → 1,885 chips
        ▼
for each chip (fetch_darts_chips.py --dataset):
   1. grid        4 m UTM grid, size drawn from the GeoAI chip sizes, slump jittered off-centre
   2. Sentinel-2  least-cloudy L2A scene within ±30 days of the label date
                  → red, green, blue, nir, ndvi, ndwi        (skip if >10% cloud/shadow/snow)
   3. ArcticDEM   v4.1 2 m mosaic → relative_elevation, shaded_relief   (skip if >30% void)
   4. masks       every DARTS polygon from the same PlanetScope image, rasterised at 4 m
   5. save        darts_XXXXX.npz + a manifest row
        ▼
finalize_darts_dataset.py   → COCO json (RLE masks) + preview figures
make_darts_split.py         → leak-free folds  (Experiments/splits/darts_grouped_folds.json)
export_darts_kaggle.py      → kaggle_darts/  in the competition release's folder layout
```

Scripts, all in `DARTS_EDA/`:

| script | job |
|---|---|
| `fetch_darts_chips.py` | download and build the chips (runs in `DARTS_EDA/.venv`) |
| `finalize_darts_dataset.py` | COCO annotations and preview PNGs |
| `make_darts_split.py` | grouped CV folds and held-out test |
| `export_darts_kaggle.py` | the Kaggle-ready copy |

---

## 2. The source labels

**DARTS** (Nitze et al., v1.2) ships manually digitised RTS polygons in
`Darts_dataset/ML_training_labels/retrogressive_thaw_slumps/iteration001…006/TrainingLabel_*.gpkg`.

Each polygon carries:
- `image_id`: the PlanetScope scene it was drawn on
- `image_date`
- `region`
- the iteration it came from

**Only the hand-drawn training labels were used.** The much larger level-1/level-2 parquet files are model predictions, so they were left out.

Loading (`load_labels`):
1. Read all six iterations and concatenate them. That gives 3,749 polygons.
2. Drop empty geometries and repair invalid ones (`make_valid`).
3. Compute each polygon's true area in m² in the polar equal-area projection EPSG:6931.

**Size filter: drop polygons < 900 m².** 900 m² is 3 × 3 Sentinel-2 pixels. Anything smaller is a blob of one or two 10 m pixels and can't be seen. The dropped polygons are removed completely: they are neither chip targets nor masks, so the model is never asked to find something invisible. **3,450 polygons remain.**

---

## 3. Placing the chips

### 3.1 Chip size and grid (`chip_grid`)
- **Size.** For each chip, one (width, height) pair is drawn at random from the 756 GeoAI chips (`competition_release/metadata/train_manifest.csv`). The DARTS chip-size distribution therefore matches GeoAI's: the median is 158 × 289 px in both.
- **Projection.** The local UTM zone of the slump, with 4 m pixels snapped to multiples of 4 m.
- **Position.** The chip is centred on the slump's centroid, then shifted at random by up to ±25% of the chip width and height. Without the shift, every slump would sit dead centre, which the model could learn as a shortcut. GeoAI slumps sit anywhere in the chip.

### 3.2 Covering every slump once (`plan_cover`)
Instead of one chip per polygon (which would duplicate heavily), a greedy cover is used:

1. Shuffle the polygons (seed 0).
2. Take the next polygon that isn't covered yet, and place a chip on it as in 3.1.
3. Mark as covered every polygon **from the same PlanetScope image** that lies fully inside that chip.
4. Repeat until every polygon is covered.

The plan uses the same random seeds as the builder, so the planned chip is exactly the built chip. Result: **1,885 chips.**

**Chips can still overlap.** Polygons near a chip edge are only partly inside it, so they get their own chip later. The same ground is also often labelled on several images from different years. This overlap is why the folds must be grouped (section 8).

---

## 4. Optical bands: Sentinel-2 (`s2_layers`)

**Source:** Sentinel-2 Level-2A (surface reflectance) from Microsoft Planetary Computer's STAC API, collection `sentinel-2-l2a`.

### 4.1 Choosing the scene
1. **Search.** Scenes over the chip, within **±30 days of the DARTS label date**, with scene cloud cover < 60%.
2. **Rank.** Scenes with ≤ 20% cloud come first, then the closest date, then the lowest cloud cover.
3. **Test the top 8 in order.** For each one, read the scene classification layer (SCL) over the chip.
   - Skip the scene if over 30% of the chip falls outside it (edge of the scene footprint).
   - Count **bad pixels**: SCL classes 0 no-data, 1 saturated, 3 cloud shadow, 8/9 cloud medium/high, 10 cirrus, 11 snow/ice.
   - Skip the scene if over 10% of the chip is bad.
4. **Use the first scene that passes.** If none passes, the chip is dropped as `no_clear_s2` (19 chips).

How close the chosen scene is to the label date:
- Median 0 days; 58% of chips use an image from the same day as the label.
- 90th percentile 12 days, maximum 27 days.
- Median scene cloud cover 3.8%.

### 4.2 Reading and correcting the bands
Each band is read at its **native resolution** on a grid aligned to the chip: 10 m for B02/B03/B04/B08 and 20 m for B8A.
- **Zeros become NaN.** 0 is Sentinel-2's no-data value.
- **The BOA offset is removed.** Scenes with processing baseline ≥ 04.00 store reflectance + 1000, so 1000 is subtracted from them.

### 4.3 Making the GeoAI bands

| GeoAI band | DARTS recipe | resampled to 4 m with |
|---|---|---|
| red | B04 (10 m) | bilinear |
| green | B03 (10 m) | bilinear |
| blue | B02 (10 m) | bilinear |
| nir | **B8A (20 m)** | bilinear |
| ndvi | (B08 − B04) / (B08 + B04), computed **at 10 m** | bilinear |
| ndwi | (B03 − B08) / (B03 + B08), computed **at 10 m** | bilinear |

Why these choices (from the compatibility EDA, `report_darts_compatibility.md`):

- **ndvi and ndwi.** GeoAI's ndvi and ndwi are themselves Sentinel-2 products upsampled to 4 m. Computing the indices at the native 10 m first, then upsampling, copies that pipeline. Upsampling the bands first would create fake 4 m detail.
- **nir uses B8A (20 m), not B08 (10 m).** GeoAI's nir is blurry: its power spectrum shows a ~20–30 m source, not Maxar. B8A matches that sharpness.
- **red, green, blue.** GeoAI RGB is 4 m Maxar, and there is no free equivalent. Sentinel-2 at 10 m is the closest open source, so DARTS RGB is about 2.5× blurrier. This is the one band group that cannot be matched.

**Masking.** The bad-pixel mask from the SCL is moved onto the 4 m grid (nearest neighbour), and every optical band is set to **NaN** there. Clouds and shadows therefore show up as square NaN blocks (the "black pixels" in the previews).

Chips have a mean NaN fraction of 1.0% and a maximum of 8.9%. The training pipeline fills these holes the same way as in GeoAI (section 10).

---

## 5. Terrain bands: ArcticDEM (`dem_layers`)

**Source:** ArcticDEM v4.1 **2 m mosaic** from the Polar Geospatial Center's open bucket (`pgc-opendata-dems` on AWS). It needs no login.
- **Finding the tiles.** The chip corners are projected to EPSG:3413, which gives the row/column of the 100 km tile and the 50 km sub-tile.
- **Reading.** The tile is read directly over HTTPS as a Cloud-Optimised GeoTIFF, resampled onto a 2 m grid aligned with the chip.
- **Two tiles.** 110 chips straddle a tile boundary, so two tiles are merged.

**Steps:**
1. **Read with padding.** The DEM is read with a 60 m border around the chip, so the window filters below have real neighbours at the chip edges.
2. **Mark voids.** Values below −1000 are void and become NaN. If over 30% of the chip is void, the chip is dropped as `dem_void` (5 chips).
3. **Fill pinholes.** Small holes are filled with a 5 × 5 NaN-aware mean, used only for the hillshade gradients.

### 5.1 relative_elevation
```
relative_elevation = Z − mean_{50×50 px window at 2 m}(Z)        → then 2×2 block-averaged to 4 m
```
This is the height of each pixel above or below its surroundings. Slumps are bowls, so they show up as negative values.

**Window size: 100 m (50 px at 2 m).** The GeoAI source paper (Yang et al. 2023) describes a 15 px window. That is 30 m, and it gave 5× too small a spread compared with the real GeoAI band. Several window sizes were fitted to GeoAI's distribution in the EDA notebook, and a ~100 m window matched: interquartile range 0.529 vs GeoAI 0.535. The mean ignores NaNs (normalised convolution: the sum of valid values divided by the count of valid values).

### 5.2 shaded_relief (placeholder recipe)
GeoAI's shaded relief is an "enhanced shaded relief" (ESR) whose exact recipe isn't published. The rebuilt version is:

```
hillshade_45 = weighted multi-direction hillshade, sun altitude 45°
               (azimuth weights from Yang et al.: 0°:.1, 90°:.3, 135°:.1, 180°:.1, 225°:.3, 315°:.1)
slope_255    = 255 · (1 − min(slope°/45, 1))
grey_255     = Sentinel-2 grey (mean of R, G, B), 2–98% stretched to 0–255

shaded_relief = 0.5 · (0.5 · hillshade_45 + 0.5 · slope_255) + 0.5 · grey_255
```

The hillshade is computed at 2 m (Horn-style gradients, 255 · max(0, cos θ)) and block-averaged to 4 m.

**This band is the weakest match.** Its texture and its slump signal are similar to GeoAI's. But GeoAI's shaded relief is almost uncorrelated with its RGB (r ≈ −0.05), while the rebuild is ~0.5 correlated because of the grey term. GeoAI's also never goes below ~143. To allow a later refit without downloading again, each chip stores the ingredients: hillshades at 30°/45°/60° and slope, as `extra_*` float16 arrays in `dataset/`.

**Shared NaNs.** Wherever the DEM is NaN, all the DEM bands are set to NaN together.

---

## 6. Masks

For each chip:
1. Take **every DARTS polygon from the same PlanetScope `image_id`** that touches the chip. A slump that is only partly inside becomes a clipped mask, as in GeoAI.
2. Rasterise them at 4 m into an instance map `inst` (0 = background, k = k-th polygon).

Only polygons from the **same image** are used. A polygon drawn on a 2021 image is not a valid label for a 2019 image of the same place: the slump may have grown, or not existed yet.

In `finalize_darts_dataset.py`, each instance becomes a COCO annotation with an RLE mask, a bbox and an area. **Instances under 10 px are dropped**: they are slivers where a polygon only just touches the chip edge, and GeoAI has no instance that small. This gives 6,180 annotations.

About 32% of DARTS masks touch a chip edge, against 36% in GeoAI.

---

## 7. What was saved

**`dataset/darts_XXXXX.npz`** (one per chip):

| key | content |
|---|---|
| `image` | (H, W, 8) float32, **GeoAI band order**: red, green, blue, ndvi, relative_elevation, shaded_relief, nir, ndwi; NaN = cloud / void |
| `inst` | (H, W) int16 instance map |
| `meta` | JSON: region, image date, chosen S2 scene, UTM CRS, chip origin, ... |
| `extra_hs_alt30/45/60`, `extra_slope_deg` | float16 ingredients for refitting shaded_relief |

**`dataset/manifest.csv`** has one row per attempted chip:
- status: `ok`, `no_clear_s2` or `dem_void`
- source: the PlanetScope `image_id` and date, region and iteration
- the chosen S2 scene with its date, cloud cover and bad-pixel fraction
- the number of DEM tiles
- the chip's CRS and origin
- `n_inst`, foreground fraction and NaN fraction

**`dataset/annotations/instances_darts.json`** holds the COCO annotations. `file_name` is `<chip>.npz`.

**`dataset/previews/`** holds four figures:
- `00_size_comparison.png`: chip and slump sizes against GeoAI
- `01_gallery.png`: RGB with each slump outlined
- `02_bands.png`: RGB / mask / ndvi / relative elevation / shaded relief
- `03_geoai_vs_darts.png`: the two datasets side by side

### Units: the DARTS images are not 8-bit
- **red / green / blue / nir:** Sentinel-2 surface reflectance × 10,000 (typically a few hundred to a few thousand). GeoAI RGB is 0–255.
- **ndvi / ndwi:** −1 to 1, same as GeoAI.
- **relative_elevation:** metres, same as GeoAI.
- **shaded_relief:** 0–255, same as GeoAI.

Section 10 explains how training handles the RGB difference.

---

## 8. Leak-free folds (`make_darts_split.py`)

Chips overlap a lot:
- **15,009 overlapping chip pairs.**
- The chips add up to 1,510 km² but cover only 536 km² of distinct ground.

This happens because DARTS labels the same sites on several images and in several years. A random split would put the same ground in train and val.

How the split avoids that:
1. **Footprints.** Each chip's footprint (from its origin and size) is moved to one shared projection, EPSG:3413.
2. **Grouping.** Chips are linked when their footprints, grown by **500 m**, touch. The connected groups are units that are never split. The 500 m buffer also keeps neighbouring hillsides together, not just identical ground.
3. **Assignment.** Whole groups, largest first, go to the least-full bucket: a 15% held-out test set plus 5 folds, balanced on chips and slumps. This is the same method as `Experiments/make_grouped_split.py` for GeoAI.
4. **Checks.** The script asserts that no overlapping or nearby pair crosses two buckets.

| bucket | chips | slumps | regions |
|---|---|---|---|
| test | 279 | 920 | 10 |
| fold 0 | 220 | 1,418 | 1 (Novaya Zemlya only) |
| fold 1 | 312 | 1,054 | 13 |
| fold 2 | 334 | 982 | 13 |
| fold 3 | 323 | 1,021 | 9 |
| fold 4 | 393 | 785 | 11 |

The output is `Experiments/splits/darts_grouped_folds.json`, in the same format as `grouped_folds.json` (`val_ids`, `test_ids` = COCO image ids).

**Pretraining used fold 1:** 1,270 train chips (4,206 slumps), with 312 val chips choosing the checkpoint.

**One leak that can't be checked:** whether DARTS covers the same ground as the GeoAI test chips. GeoAI has no coordinates. The DARTS data is therefore only ever used for training or pretraining, and GeoAI is always scored on its own held-out chips.

---

## 9. The Kaggle copy (`export_darts_kaggle.py`)

`kaggle_darts/` copies the competition release's folder layout, so the training code reads it without any changes:

```
kaggle_darts/
  train/images/darts_XXXXX.npz           image only (float32, GeoAI band order)
  train/annotations/instances_train.json same as instances_darts.json, file_name = "images/<chip>.npz"
  metadata/band_names.json               copied from the release (same band order)
  tools/evaluate_coco.py, coco_utils.py  copied from the release (the official scorer)
```
The `inst` raster, `meta` and the `extra_*` layers are left out, because training reads the masks from the COCO file.

---

## 10. How training treats it (`Experiments/`)

- **Missing pixels.** NaN holes (cloud, DEM void) are filled by `fill_nan_neighbor_mean`: repeated 8-neighbour averaging from the hole edge inward, the same as for GeoAI. It is cached in `cache/darts/filled.npz`. Anything left is filled with the training mean.
- **Normalisation.** Per-band statistics are computed from the DARTS training chips only.
  - RGB is the exception. The Satlas Swin backbone expects 8-bit RGB / 255, so with `data.rgb_ref = geoai`, DARTS RGB is first z-scored and then given GeoAI's RGB mean and standard deviation:
    ```
    x_rgb  →  ((x − μ_DARTS) / σ_DARTS · σ_GeoAI + μ_GeoAI) / 255
    ```
  - In both pretraining and fine-tuning, the backbone therefore sees RGB in the same range.
- **Recipe.** Pretrain on DARTS (`e6_darts_darts_pt`), then fine-tune on GeoAI (`e6_darts_darts_ft`) from the pretrained `best.pth`.

---

## 11. DARTS compared with GeoAI

| | GeoAI | DARTS |
|---|---|---|
| chips | 756 | 1,861 |
| median chip | 158 × 289 px | 158 × 289 px |
| slump masks | 1,783 | 6,180 |
| median slump | 737 px | 410 px |
| slumps per chip | 2.36 | 3.32 |
| small < 300 px | 25% | 42% |
| medium 300–2000 px | 51% | 43% |
| large ≥ 2000 px | 24% | 16% |
| RGB source | Maxar 4 m, 8-bit | Sentinel-2 10 m, reflectance |

Band by band, from the compatibility EDA:

| bands | match | notes |
|---|---|---|
| ndvi, ndwi, nir | **good** | Same kind of source and sharpness, same correlations, same slump signal direction. |
| relative_elevation | **good** | Requires the 100 m window. |
| shaded_relief | **partial** | Placeholder recipe (section 5.2). |
| red, green, blue | **different** | A classifier can tell the two datasets apart (AUC 0.92 without texture features, 0.99 with). Mostly the 4 m vs 10 m sharpness gap. |

That's why DARTS is used for **pretraining and then fine-tuning**, not mixed in as if it were the same data.

---

## 12. Known limitations

1. **RGB is 2.5× coarser than GeoAI** and can't be fixed with open data.
2. **Shaded relief is a placeholder recipe.** The stored extras allow a refit.
3. **The imagery doesn't match exactly.** DARTS polygons were drawn on 3 m PlanetScope images. The pixels here come from the closest clear Sentinel-2 date (median 0 days apart, up to 27). A slump that grew between those dates can be slightly misaligned with its mask.
4. **Slumps are smaller than GeoAI's** (median 410 vs 737 px), and the < 900 m² slumps are missing altogether.
5. **Regions are unevenly covered.** Banks Island, Horton, Novaya Zemlya and Lena make up 57% of the chips. Labels come mostly from 2019 (1,035 chips) and from iteration 001 (1,226 chips).
6. **Cloud holes.** 25% of chips have more than 1% of pixels missing, filled during training.

---

## 13. Reproducing it

```bash
# 1. chips (≈ hours; downloads S2 + ArcticDEM windows over HTTPS)
DARTS_EDA/.venv/Scripts/python DARTS_EDA/fetch_darts_chips.py --dataset --out DARTS_EDA/dataset --workers 8

# 2. COCO annotations + previews
python DARTS_EDA/finalize_darts_dataset.py --dir DARTS_EDA/dataset

# 3. leak-free folds
python DARTS_EDA/make_darts_split.py            # --buffer 500 --test_frac 0.15 --n_folds 5

# 4. Kaggle copy
python DARTS_EDA/export_darts_kaggle.py         # -> DARTS_EDA/kaggle_darts/
```

Everything is seeded (`--seed 0`): chip sizes, offsets, the cover order and the fold assignment. The only thing that can change on a rerun is the Sentinel-2 scene chosen, if Planetary Computer's catalogue changes.

Python packages for step 1, in `DARTS_EDA/.venv`: `geopandas`, `rasterio`, `pystac-client`, `planetary-computer`, `pyproj`, `scipy`, `shapely`.
