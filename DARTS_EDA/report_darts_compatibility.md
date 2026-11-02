# Can DARTS be merged with the GeoAI Arctic training set?

*Report on the DARTS × GeoAI compatibility EDA, written 2026-10-27.*

Every equation and statistical tool used here is derived from first principles in
**`math_explained.md`**. References like **(M8)** point to its sections.

---

## 0. Short answer

| channel group | can it be merged? | why |
|---|---|---|
| **nir, ndvi, ndwi** | **Most likely, yes**, with normal per-channel normalisation | Same satellite (Sentinel-2) and the same scale in both datasets. Same band-to-band structure (ndvi↔nir r = 0.87 vs 0.89, ndvi↔ndwi −0.97 vs −0.98). Same direction of the slump signal. |
| **relative_elevation** | **Yes**, once it is rebuilt with a **~100 m window**, not the paper's 30 m | Same DEM. With the right window, the spread matches GeoAI almost exactly (IQR 0.535 m vs 0.529 m). |
| **shaded_relief** | **Partly.** The recipe is not fully reproduced | GeoAI's band is almost uncorrelated with its RGB (r ≈ −0.05). My best rebuild correlates at ≈ 0.5 with S2 RGB. GeoAI's band also has a hard floor at ~143 that I can't explain. |
| **red, green, blue** | **Not interchangeable.** Usable only with care | Different sensor (Maxar vs Sentinel-2), different units, and 2.5× less sharp. Normalisation makes the histograms match, but a simple classifier still tells the two apart with AUC ≥ 0.92. |
| **labels** | Usable, with a filter | DARTS slumps are smaller (43% under 300 px vs 25% in GeoAI). 8% are under 900 m², i.e. ≤ 3×3 Sentinel-2 pixels. |

**Recommendation:**
- Use DARTS as **pre-training** data, or as a separately-tagged extra source. Don't add it as if it were more GeoAI.
- If you mix the two, mix them on **nir/ndvi/ndwi/rel_elev**, the source-matched bands.
- For RGB, randomly blur GeoAI chips to 10 m during training, so sharpness can't act as a "which dataset" shortcut.
- Your current model is RGB-only, and RGB is the worst-matching group, so the gain from DARTS will come mostly from the other bands.

**Status of the numbers:** read §8 before relying on any table. The analysis ran fully once.
That run had three bugs, which I then fixed. The corrected rerun was stopped by Claude Code
for low system memory during the last stage (§6). Every number below is tagged with which
run it comes from and whether the bugs could have affected it.

---

## 1. What was done

### 1.1 The problem

DARTS (Nitze et al., v1.2) ships **polygons only**: 3 749 hand-digitised slump outlines over
15 Arctic regions, drawn on PlanetScope images that cannot be redistributed. There are no
pixels to compare. So the first job was to **make DARTS pixels that look like GeoAI pixels**.

### 1.2 Where the GeoAI bands come from

Yang et al. (2023), the paper behind the GeoAI data, plus a spectral analysis of the chips
(§3 below) give this picture. The analysis measures which source resolution each band has (M2).

| GeoAI band | source |
|---|---|
| red, green, blue | Maxar Vivid basemap, 4 m |
| ndvi, ndwi | Sentinel-2, 10 m |
| nir | a ~20–30 m source. Its scale and spectrum fit Sentinel-2 B8A (20 m). **Not** Maxar, which is a 3-band product. |
| relative_elevation | ArcticDEM 2 m minus a local mean |
| shaded_relief | "enhanced shaded relief": weighted 8-direction hillshade + slope, blended with greyscale Maxar |

All bands sit on a **4 m grid**.

### 1.3 Building DARTS chips (`fetch_darts_chips.py`)

For 300 DARTS polygons, about 20 per region, the script did the following:

1. **Grid.** A 4 m grid in the local UTM projection, centred near the polygon (randomly
   shifted by up to ±25% so the slump isn't always dead centre). Chip sizes are drawn from the
   real GeoAI chip sizes (median 159 × 289 px).
2. **Optical bands** from Sentinel-2 L2A (Microsoft Planetary Computer):
   - Scenes within ±30 days of the label date, with < 10% bad pixels in the chip. "Bad" means
     cloud, cloud shadow, cirrus, snow or saturation, from the SCL map.
   - The 2022+ offset of 1000 is removed (M6.3).
   - red/green/blue = B04/B03/B02 at 10 m, bilinear to 4 m (M3.1). nir = B8A at 20 m,
     bilinear to 4 m.
   - ndvi and ndwi are computed **at 10 m, then upsampled**, the same order GeoAI used (M6).
3. **Elevation bands** from ArcticDEM v4.1 2 m mosaics, read directly from the public AWS
   bucket. The tile index arithmetic is in the M appendix.
   - Relative elevation: several window sizes (M4).
   - Hillshade: at sun altitudes of 30/45/60° with the paper's 8 direction weights (M5).
   - Slope.
   - Everything was derived at 2 m and then block-averaged to 4 m (M3.2).
4. **Masks.** Every DARTS polygon from the same PlanetScope image that falls in the window
   is rasterised.

**Result:** 298 of 300 chips built, holding 1 193 slump instances. The 2 failures had no
clear Sentinel-2 scene. GeoAI side: 756 chips, 1 783 instances.

### 1.4 The analysis notebook (`darts_eda.ipynb`)

| § | question | figure |
|---|---|---|
| 1 | What do the DARTS labels look like? Are the slumps the same size? | `01_darts_labels.png` |
| 2 | Which sensor/resolution is behind each GeoAI band? | `02_band_source_spectra.png` |
| 3 | Can the elevation bands be rebuilt exactly? | `03_dem_recipe_match.png` |
| 4 | Per-channel distributions, band correlations, slump signal | `04`, `05`, `06` |
| 5 | How big is the resolution gap in RGB? | `07_resolution_gap.png` |
| 6 | Does normalisation make the datasets indistinguishable? | `08`, `09` |
| 7 | Visual check | `10_gallery.png` |
| 8 | Verdict table | — |

---

## 2. The labels (§1, figure 01)

**Source:** first run. These numbers don't depend on any of the bugs.

| instance size (4 m px) | GeoAI | DARTS |
|---|---|---|
| small, < 300 px (< 4 800 m²) | 25.3% | **43.3%** |
| medium, 300–2 000 px | 50.6% | 41.1% |
| large, ≥ 2 000 px (≥ 32 000 m²) | 24.1% | 15.6% |

- **DARTS slumps are physically smaller.** If both datasets had the same median slump size,
  the implied GSD would be 2.92 m instead of 4 m (M1.2). In area terms, the median DARTS slump
  is about half the median GeoAI slump.
- **Why it matters.** The competition scorer splits AP by these same 300 / 2 000 px bins.
  Adding DARTS shifts the training set towards small objects. That probably helps on small
  test slumps, but it changes what the model sees as "typical".
- **Small-object loss at 10 m (M14):**

  | area threshold | DARTS | GeoAI |
  |---|---|---|
  | < 100 m² (1 S2 pixel) | 0.2% | 0.0% |
  | < 400 m² (2×2 S2 px) | 2.1% | 1.9% |
  | < 900 m² (3×3 S2 px) | 8.0% | 4.3% |

  In S2-based DARTS RGB/nir, those slumps are a blob of a handful of pixels. Drop DARTS labels
  under ~900 m² from any optical training.

---

## 3. Which sensor is behind each GeoAI band? (§2, figure 02)

**Source:** first run; unaffected by the bugs.

The idea (M2.4): a band captured at GSD *g* and put on the 4 m grid contains no detail finer
than 2*g*. Its power spectrum shows a **knee** at a period of 2*g*/4 px. Measured on GeoAI:

| band | knee period | implied source GSD | consistent with |
|---|---|---|---|
| red, green, blue | 2.1–2.2 px | ~4 m (grid Nyquist) | Maxar 4 m ✔ |
| ndvi, ndwi | 5.2 px | ~10 m | Sentinel-2 10 m ✔ (the paper) |
| nir | **11.3 px** | **~23 m** | Sentinel-2 B8A 20 m. Not 4 m Maxar. |
| shaded_relief | 10.1 px | ~20 m | a smoothed product. Its 2 m DEM input would give a knee near 2 px, so smoothing happened in their pipeline. |
| relative_elevation | 8.3 px | (not meaningful) | RE is a high-pass filter (M4.2). Its spectrum is flat then falls, so the knee finder has no clean cliff to find. |

An earlier, slightly different knee estimate gave nir 13.3 px. Either way, nir sits between
the 10 m and 30 m signatures.

**The shape-distance comparison (M2.5) didn't discriminate.** "GeoAI nir vs DARTS B8A" scored
0.73 and "vs DARTS B08 (10 m)" scored 0.49. The pair known to share a source (ndvi vs S2 ndvi)
itself scored 0.66. So the 0.5–0.7 differences are terrain and scene content, not source.
**The knee is the reliable evidence; the shape distance is not.**

**Practical consequence.** nir, ndvi and ndwi already come from Sentinel-2-class data in
GeoAI. So DARTS chips built from Sentinel-2 are **source-matched** on these three bands. Only
red/green/blue carry a genuine sensor change. shaded_relief carries a partial one, through
its Maxar blend.

---

## 4. Rebuilding the elevation bands (§3, figures 03 and 04)

### 4.1 Relative elevation — the window is ~100 m, not 30 m

**Source:** corrected run (figure 03), plus a standalone check over the same 298 chips.

The paper says "DEM minus a 15-pixel mean filter" on the 2 m DEM: a 30 m window. That rebuild
had **about 5× too little spread**: IQR 0.096 m vs GeoAI's 0.530 m, p1/p99 ±0.8 m vs ±3.6 m.
The window size sets the amplitude. A wider window leaves more of the hillside in, so the
residual grows (M4.3). Scanning the window:

| window (4 m grid) | in metres | IQR (m) | p1 / p99 (m) |
|---|---|---|---|
| k = 15 | 60 m | 0.254 | −2.20 / 2.13 |
| **k = 25** | **100 m** | **0.535** | **−4.12 / 3.86** |
| k = 35 | 140 m | 0.886 | −5.99 / 5.46 |
| k = 51 | 204 m | 1.547 | −8.69 / 7.79 |
| **GeoAI target** | | **0.529** | **−3.58 / 3.75** |

**A ~100 m window reproduces GeoAI's relative elevation.** The notebook now picks the window
by a rank fit (M5.8) and writes it into the DARTS chips. In figure 04, the raw rel_elev
histograms overlap almost completely.

Two possible explanations: the paper's "15 px" may refer to a coarser working resolution
(15 px × ~6.7 m ≈ 100 m), or the window was changed later for the challenge release. Either
way, **use a ~100 m window for any external data.**

The slump signal agrees in direction (figure 06): RTS pixels sit lower than their
surroundings in both datasets. AUC (M10) is 0.455 for GeoAI and ≈ 0.48 for DARTS. The cue is
weak in both. Relative elevation on its own is not a strong slump detector at this scale.

### 4.2 Shaded relief — close, but not the same recipe

**Source:** corrected run (figures 03, 04, 05, 06).

The enhanced shaded relief has four unpublished knobs: sun altitude, slope weight, blend
opacity α and smoothing σ (M5.6–M5.7). The notebook searches all 72 combinations, scoring
each on distribution shape, spectrum, and correlation with grey RGB (M5.8). It then rescales
the best one to GeoAI's median and spread (M5.9).

What matches:
- Centre and spread match by construction (166 vs 166; σ 11.05 vs 11.09).
- The **spectrum matches very well**. The red dashed and blue solid "shaded" curves in figure
  03 (right) lie on top of each other, so the smoothing level is right.
- The slump signal now has the **same direction**: RTS slightly brighter in both. AUC ≈ 0.58
  in GeoAI, ≈ 0.565 in DARTS.

What does not match:
- **GeoAI's band is almost independent of its RGB** (r = −0.03 to −0.07, figure 05). The DARTS
  rebuild correlates **≈ 0.45–0.50** with its RGB. The "photo blended in" story from the paper
  does not show up in GeoAI's numbers. Either their blend used a heavily stretched or
  differently-dated greyscale, or the challenge version was made without it.
- **GeoAI's histogram has a hard floor at ~143** (figure 04, a cliff on the left). That looks
  like clipping or a rescale step, which I have not reproduced. The DARTS rebuild instead piles
  up at ~191 on the right.

The fit tries to match the correlation, but it also has to match the slump signal and the
spectrum. With pure hillshade (α = 0), RTS comes out *darker* (AUC 0.41), the reverse of
GeoAI. That was the buggy first run. Some photo blend is needed to recover GeoAI's "brighter
slumps" cue. That suggests GeoAI's cue in this band also comes partly from the optical blend,
even though the overall correlation is tiny.

**Verdict for shaded_relief:** usable for pre-training. Treat it as the least reliable of the
non-RGB bands until the recipe is pinned down.

---

## 5. The shared channels side by side (§4, figures 04–06)

### 5.1 Raw distributions (figure 04)

- **red/green/blue: completely different units.** GeoAI is 8-bit display values (medians
  86/90/70). DARTS is Sentinel-2 reflectance × 10 000 (medians 619/607/384). Raw, the
  Wasserstein distance (M8.3) is 13–19 GeoAI σ, and KS is 0.92–0.97. The histograms barely
  overlap. This is expected and is exactly why the question is about normalisation.
- **nir: the same scale.** Medians 2 530 (GeoAI) and 2 353 (DARTS), and the shape is similar,
  with a water/dark mode near 0–600 and a vegetation mode near 2 800. This is the scale
  evidence that GeoAI nir is Sentinel-2-like reflectance.
- **ndvi/ndwi: the same scale, very similar shape.** Medians 0.58 vs 0.53 and −0.58 vs −0.56.
  DARTS is a little more spread (it reaches lower ndvi and higher ndwi: more water and bare
  ground in the sample).
- **rel_elev:** matched after the window fix (§4.1).
- **shaded:** see §4.2.

### 5.2 Band-to-band correlations (figure 05, M12)

Correlation is unchanged by any per-channel rescaling. So these differences **cannot be fixed
by normalisation**. They are structural.

| pair | GeoAI | DARTS | comment |
|---|---|---|---|
| red–green–blue | 0.93–0.98 | 0.87–0.96 | Similar. S2's blue is noisier (haze), so slightly lower. |
| ndvi–nir | 0.89 | 0.87 | ✔ same physics, same sensor |
| ndvi–ndwi | −0.98 | −0.97 | ✔ |
| nir–ndwi | −0.87 | −0.86 | ✔ |
| RGB–ndvi | −0.22 to −0.43 | −0.14 to −0.28 | similar sign, weaker in DARTS |
| **RGB–nir** | −0.11 to −0.29 | **+0.10 to −0.06** | **Different.** In DARTS, RGB and nir come from the *same S2 scene*, so brightness changes (sun, haze) hit both together. In GeoAI, RGB is Maxar from one date and nir is S2 from another, so they are less coupled. |
| **RGB–shaded** | ≈ −0.05 | ≈ +0.47 | the ESR issue from §4.2 |
| rel_elev–anything | ≈ 0 | ≈ 0 | ✔ |

The nir/indices block is essentially the same in both datasets. That is the strongest single
argument that those bands are compatible.

### 5.3 Slump signal per channel (figure 06, M10)

**Source:** corrected run.

| channel | GeoAI AUC | DARTS AUC | slump pixels are… | agree? |
|---|---|---|---|---|
| red | 0.70 | 0.70 | brighter | ✔ |
| green | 0.69 | 0.63 | brighter | ✔ |
| blue | 0.70 | 0.69 | brighter | ✔ |
| ndvi | 0.32 | 0.30 | less vegetated | ✔ |
| rel_elev | 0.46 | 0.48 | lower than surroundings | ✔ (weak in both) |
| shaded | 0.58 | 0.56 | brighter | ✔ |
| nir | 0.35 | 0.31 | darker in NIR | ✔ |
| ndwi | 0.69 | 0.70 | higher (bare/wet) | ✔ |

**All 8 channels carry the same slump cue, with similar strength.** A slump looks like
"bright, bare, unvegetated ground" in both datasets. Physically, the same thing is being
labelled, even though RGB comes from different sensors. This is the necessary condition for
merging: it would be fatal if a channel said "brighter" in one dataset and "darker" in the
other.

---

## 6. The resolution gap in RGB (§5, figure 07)

**Source:** first run; unaffected by the bugs (RGB only).

Sharpness was measured with the Laplacian variance of standardised luminance (M13). Medians
over chips:

| version | median | interquartile range |
|---|---|---|
| GeoAI native (Maxar 4 m) | **0.511** | 0.30–0.73 |
| GeoAI simulated at 10 m (4→10→4 m, M3.2) | 0.021 | 0.011–0.036 |
| DARTS (real S2 10 m → 4 m) | **0.028** | 0.012–0.068 |

- GeoAI RGB carries **about 18× more fine-scale energy** than DARTS RGB.
- DARTS is **within the same range as GeoAI degraded to 10 m**. It is slightly sharper,
  because my simulation's extra 3×3 box over-blurs a little.
- The luminance spectrum of DARTS is closer to degraded-GeoAI (shape distance 0.50) than to
  native GeoAI (0.82).

**Conclusion: the RGB sharpness gap is almost entirely the 4 m → 10 m resolution
difference,** not some other processing difference. That is good news, because it can be
handled. Randomly degrading GeoAI chips to 10 m during training (with some probability)
removes sharpness as a source shortcut.

---

## 7. Does normalisation make the datasets indistinguishable? (§6, figures 08–09)

**Source:** first run. **Partly affected by the bugs; see the tags.** The corrected rerun
was killed in this section.

### 7.1 Matching the histograms — easy

Wasserstein distance in GeoAI σ (M8.4). Lower is closer; below ~0.1 is negligible.

| scheme (M9) | red | green | blue | nir | tags |
|---|---|---|---|---|---|
| raw | 15.85 | 19.08 | 12.98 | 0.28 | valid |
| dataset z | 0.14 | 0.16 | 0.15 | 0.15 | valid |
| **chip z** | **0.09** | **0.07** | **0.09** | **0.08** | valid |
| chip p2–98 | 0.26 | 0.25 | 0.33 | 0.13 | valid |
| hist-match | 0.02 | 0.03 | 0.04 | 0.03 | valid, but ~0 by construction (M9.1) |

ndvi and ndwi read 0.08 under chip z. Their raw and dataset-z values (0.14 / 6.9 and 105 /
432) were **destroyed by the outlier bug** (§8) and are not meaningful. With the outliers
masked they will be small, because the histograms in figure 04 already overlap well in raw
units. shaded and rel_elev in this table are from the old, wrong rebuilds.

**Per-chip z-scoring brings every optical channel within < 0.1 σ.** After normalisation,
the *histograms* of DARTS and GeoAI RGB are practically the same.

### 7.2 Telling the sources apart — still trivial

The domain classifier (M11) asks: given a chip's summary features, can a model tell which
dataset it came from? 0.5 means it can't; 1.0 means always.

| group | best scheme | AUC without texture | AUC with texture | tags |
|---|---|---|---|---|
| **RGB** | chip p2–98 / hist-match | **0.916** | 0.991 | valid (RGB unaffected by bugs) |
| nir + indices | chip p2–98 | 0.874 | 0.989 | provisional: index outliers affect the mean/std features |
| DEM (rel, shaded) | chip p2–98 | 0.946 | 0.995 | invalid: old RE window + broken ESR |
| all 8 | chip z | 0.991 | 1.000 | invalid (contains the above) |

Read the valid RGB row carefully:
- **Without any sharpness feature, AUC 0.92.** Even after the histograms are matched, the
  *shape* of each chip's distribution and the *inter-band correlations* give the source away.
  Sentinel-2 surface reflectance and Maxar's colour-balanced "Vivid" mosaic render colour
  differently. Correlation can't be changed by rescaling (M12).
- **Adding sharpness pushes it to 0.99.** Resolution alone is a near-perfect giveaway (§6).

So: **matching histograms is not the same as matching domains.** Normalisation fixes the
first-order difference (offset, gain, scale). It leaves sharpness and colour rendering, and a
CNN will see both.

**How harsh is this test?** Very. With ~1 000 chips and well-chosen features, a boosted model
can separate almost any two imagery sources. Also, GeoAI's chips overlap each other, which
pushes the AUC up (M11.5). The < 0.65 threshold in the notebook's verdict was too strict to
be useful: no group passes it. **The useful reading is comparative:** RGB is clearly separable
even without texture, while the nir+indices block is the least separable group. That matches
§5.2, where it is the block with identical internal structure.

### 7.3 Does normalisation keep the slump signal?

**Source:** first run. Valid for RGB, nir, ndvi and ndwi.

After chip z-scoring, DARTS RTS AUCs are red 0.65, green 0.61, blue 0.67, ndvi 0.33, nir 0.35
and ndwi 0.65. GeoAI's are 0.62, 0.60, 0.61, 0.38, 0.47 and 0.61. **Every direction is
preserved.** The cue gets weaker under chip z, because part of the "slumps are bright" signal
is chip-level brightness, and chip z removes that (M10.3). Notably, nir in GeoAI drops from
0.35 to 0.47: GeoAI's nir cue is mostly *between-chip*, not *within-chip*.

---

## 8. Bugs found, what was fixed, and what still needs a run

The notebook ran **fully once**. Its output exposed three bugs, all in my code, not in your
data:

1. **Shaded-relief fit picked a meaningless recipe.** The correlation with grey RGB was
   computed over arrays that contained NaN (S2 cloud masks), so it came out NaN for all 72
   candidates. Every score became NaN, and the "best" was simply the first row: pure
   hillshade, no blend. That flipped the shaded slump cue (AUC 0.41 vs GeoAI 0.58).
   *Fixed:* NaNs are masked before correlating.
2. **ndvi/ndwi outliers.** 0.02% of DARTS index pixels lay outside [−1, 1]. These are pixels
   where reflectance is ≤ 0 after the offset removal, so the denominator is ~0 (M6.1). They
   drove the DARTS ndvi std to 21 000. *Fixed:* masked as NaN at load.
3. **Relative elevation used the paper's 30 m window,** which gives 5× too little amplitude.
   *Fixed:* the window is now fitted (§4.1), and 100 m wins.

The **corrected rerun** finished §0–§5 and regenerated figures 01–07. It was then **stopped by
Claude Code because the computer was low on memory** while §6 was building normalised copies
of the whole GeoAI set. That isn't a code failure. So:

- Figures **01–07** and the §2–§6 findings above come from the **corrected** code, or are
  unaffected by the bugs.
- Figures **08–10** and the §7 tables come from the **first run**, tagged per row above.
- `darts_eda.ipynb` is currently **saved without outputs**, because nbconvert only writes
  outputs when it finishes.

I have since **rewritten §6 to stream.** Each chip is normalised, summarised, sampled, and
discarded, so only one chip's copy exists at a time. The band-correlation and luminance steps
were also slimmed down. Peak memory should drop by several GB. To finish the job, close
memory-heavy programs and run:

```
cd DARTS_EDA
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 darts_eda.ipynb
```

Expect the provisional numbers in §7.2 to change for the nir+indices and DEM groups. The
qualitative RGB conclusions should not change, since nothing about RGB was affected.

---

## 9. What I recommend doing with DARTS

1. **Pre-train on DARTS, fine-tune on GeoAI.** This is the safest use. The model learns
   "what a slump looks like" from 3 749 more examples, which matters given your overfitting
   ceiling. Fine-tuning then adapts it to Maxar RGB. It also keeps DARTS out of every GeoAI
   validation fold automatically. Keep that, because GeoAI has no coordinates and geographic
   overlap with its test set can't be ruled out.
2. **If you mix instead of pre-training:**
   - Normalise every band per chip (chip z), or use dataset-z for nir/indices.
   - **Apply random 10 m degradation to GeoAI RGB** (bilinear down by 2.5× and back up, with
     probability ~0.3–0.5), so blur stops meaning "DARTS".
   - Rebuild DARTS relative elevation with the **100 m window**.
   - Drop DARTS labels under ~900 m².
   - Add a source flag, or sample per source, so a full-size DARTS set doesn't
     swamp GeoAI's 756 chips.
3. **Band choice:** among your 20 valid 3-band combos, those built from **nir / ndvi / ndwi /
   rel_elev** transfer most cleanly. RGB-only combos are the ones most exposed to the sensor
   gap.
4. **Scale up the fetch** before training: 298 chips was enough for an EDA, not for training.
   The fetcher takes `--n`. Many of the 3 749 labels share a PlanetScope image and would land
   in the same chip, so expect fewer chips than labels. Use the corrected relative-elevation
   window when you do.

---

## 10. Files

| file | what |
|---|---|
| `fetch_darts_chips.py` | builds the DARTS chips from Sentinel-2 + ArcticDEM |
| `chips/` | 298 `.npz` chips + `manifest.csv` (git-ignored) |
| `darts_eda.ipynb` | the analysis (bugs fixed, §6 made memory-light, outputs pending a rerun) |
| `01…10_*.png` | figures (01–07 from the corrected run, 08–10 from the first run) |
| `math_explained.md` | every equation, derived from first principles |
| `report_darts_compatibility.md` | this file |
