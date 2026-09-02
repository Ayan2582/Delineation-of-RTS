# The Eight Channels, From Scratch

A first-principles explanation of every band in the 2026 GeoAI Arctic Challenge release —
what it physically is, where its formula comes from, and what it does in the presence of a
retrogressive thaw slump.

No remote-sensing background is assumed. Every dataset-specific number quoted here was
measured over all 756 training chips in [`base_eda.ipynb`](base_eda.ipynb) §6, §8 and §11 —
none of it is quoted from memory or from the documentation.

**Contents**

1. [What a satellite image actually is](#1-what-a-satellite-image-actually-is)
2. [Bands 0–2 — red, green, blue](#2-bands-02--red-green-blue-maxar)
3. [Band 6 — near-infrared](#3-band-6--near-infrared-planet)
4. [Band 3 — NDVI](#4-band-3--ndvi)
5. [Band 7 — NDWI](#5-band-7--ndwi)
6. [Band 4 — relative elevation](#6-band-4--relative-elevation)
7. [Band 5 — shaded relief](#7-band-5--shaded-relief-hillshade)
8. [Summary table](#8-summary-table)
9. [What this means for the model](#9-what-this-means-for-the-model)

---

## 1. What a satellite image actually is

A camera sensor does not record "colour". It records **energy**. For each ground cell and
each wavelength interval $\lambda$ the instrument measures a *spectral radiance*
$L_\lambda$ — watts arriving per unit area, per steradian, per unit wavelength.

Radiance is not a property of the ground alone. Photograph the same tundra at noon and at
dusk and $L_\lambda$ changes by an order of magnitude, because the illumination changed.
To describe the *surface*, the illumination must be divided out. That gives **reflectance**,
the fraction of incoming light the surface sends back:

$$\rho_\lambda \;=\; \frac{\pi\, L_\lambda\, d^2}{E_{0,\lambda}\cos\theta_s}$$

where $E_{0,\lambda}$ is the solar irradiance at the top of the atmosphere, $d$ the
Earth–Sun distance in astronomical units, and $\theta_s$ the solar zenith angle. The
$\cos\theta_s$ term accounts for sunlight striking the ground obliquely; the $\pi$ comes from
integrating a Lambertian (perfectly diffuse) reflector over the hemisphere. Reflectance is
dimensionless and bounded in $[0, 1]$ — a property of the material, not of the moment.

So **one pixel is a vector**, one entry per spectral band:
$\boldsymbol{\rho} = (\rho_{blue}, \rho_{green}, \rho_{red}, \rho_{NIR}, \dots)$. It is a
coarse sample of the material's reflectance spectrum. Different materials have different
spectral shapes, and that — not colour — is what makes the classification possible.

### Digital numbers vs reflectance

Data is rarely shipped as physical reflectance. It is usually quantised into integer
**digital numbers (DN)**, related to reflectance by a linear rescaling
$\text{DN} = a\rho + b$ whose coefficients live in metadata that this release deliberately
strips.

This is not a technicality here — it is the single most consequential fact about the
dataset. The measured corpus ranges are:

| band | source | observed range |
|---|---|---|
| `red`, `green`, `blue` | Maxar | $0 \ldots 255$ — 8-bit DN |
| `nir` | Planet | $73 \ldots 7311$, median $2439$ — scaled reflectance, a different scale entirely |

The visible bands and the NIR band are in **incompatible units from different sensors**.
Any formula that combines them arithmetically is invalid until they are cross-calibrated.
[§4 below](#4-band-3--ndvi) shows exactly how this breaks the NDVI identity.

---

## 2. Bands 0–2 — red, green, blue (Maxar)

The three visible bands, sampling roughly

- **blue** ≈ 0.45–0.51 µm
- **green** ≈ 0.51–0.58 µm
- **red** ≈ 0.63–0.69 µm

These are *measured* quantities: filtered buckets of reflected sunlight. They align with the
three human cone responses, which is why stacking them yields an image that looks natural —
but that alignment is a convenience, not a definition. To the sensor they are simply three
more samples of the spectrum.

**What they respond to.** Blue scatters most in the atmosphere (hence hazy blue distance) and
penetrates water furthest. Green peaks where vegetation is *least* absorbing in the visible,
which is why plants look green — chlorophyll absorbs blue and red hard and reflects a little
green. Red is strongly absorbed by chlorophyll, so it is dark over healthy vegetation and
bright over bare soil.

**In this dataset.** All three occupy the full 8-bit range $0\ldots255$ with medians of
$89 / 92 / 73$ and standard deviations near $31\text{–}38$. They are extremely
correlated with one another — $r = 0.98$ (green–blue), $0.94$ (red–green), $0.92$ (red–blue)
— because scene brightness dominates: a bright pixel tends to be bright in all three. The
information *between* them (colour) is a small residual on top of a large shared component.

**RTS signature.** All three read **brighter** inside a slump (AUC $0.59$ red, $0.58$ green,
$0.60$ blue), which is what exposed, unvegetated soil should do.

---

## 3. Band 6 — near-infrared (Planet)

Near-infrared sits just past the red end of human vision, around 0.85 µm. Nothing about it
is visible, and it is the most physically informative band in the stack.

### The red edge

A green leaf does two very different things to light:

1. **Chlorophyll absorbs red strongly** to drive photosynthesis. Red reflectance from healthy
   vegetation drops to a few percent.
2. **The internal structure of the leaf scatters NIR.** The spongy mesophyll is a jumble of
   cell walls and air spaces with mismatched refractive indices. NIR photons carry too little
   energy for chlorophyll to absorb, so they scatter and leave. NIR reflectance from healthy
   vegetation rises to 40–60%.

Between these two regimes, at about 0.7 µm, reflectance jumps by an order of magnitude over a
span of ~40 nm. This near-discontinuity is the **red edge**, and *only living vegetation
produces it*. Soil, rock, concrete and sand all have smooth, gently rising spectra with no
such step.

That is the whole basis of vegetation remote sensing, and the reason the next two bands
exist: an index that measures the *height of the red edge* measures how much living plant
matter is present.

**In this dataset.** `nir` ranges $73 \ldots 7311$ with median $2439$ — roughly $27\times$
the red band's median, a units artefact, not a physical statement. It is the **only band
with no NaN at all** (0.0000%).

**RTS signature.** NIR is **lower** inside a slump (AUC $0.426$): the vegetation that
produced the red edge has been stripped away.

---

## 4. Band 3 — NDVI

*Normalized Difference Vegetation Index.* **Derived, not measured:**

$$\boxed{\;\mathrm{NDVI} \;=\; \frac{\rho_{NIR} - \rho_{red}}{\rho_{NIR} + \rho_{red}}\;}$$

### Where the formula comes from

The numerator is the direct measurement of red-edge height: the gap between high NIR and
suppressed red. A large gap means lots of chlorophyll.

The denominator is what makes it *useful*, and it is worth being precise about why. Suppose a
pixel is in topographic shadow, or under thin cloud, or simply illuminated at a steeper
angle. To a good approximation this scales **both** bands by the same factor $k$:

$$\frac{k\rho_{NIR} - k\rho_{red}}{k\rho_{NIR} + k\rho_{red}} = \frac{\rho_{NIR} - \rho_{red}}{\rho_{NIR} + \rho_{red}}$$

The $k$ cancels exactly. The raw difference $\rho_{NIR} - \rho_{red}$ has no such property —
it would halve in shadow. **Normalising converts an illumination-dependent difference into an
illumination-invariant ratio**, which is why the *normalized* difference form appears
everywhere in this field rather than a plain difference.

### Range and interpretation

For non-negative reflectances, $|\rho_{NIR} - \rho_{red}| \le \rho_{NIR} + \rho_{red}$, so

$$\mathrm{NDVI} \in [-1, +1]$$

with the value undefined where $\rho_{NIR} + \rho_{red} = 0$ (a perfectly black pixel — in
practice, guard the division). Conventional interpretation:

| NDVI | surface |
|---|---|
| $< 0$ | water, snow, cloud (NIR absorbed or red-dominant) |
| $0 \ldots 0.2$ | bare soil, rock, sand, gravel |
| $0.2 \ldots 0.4$ | sparse vegetation — typical Arctic tundra |
| $> 0.5$ | dense, vigorous vegetation |

Two limitations are worth carrying. NDVI **saturates** at high biomass — once the canopy is
closed, adding leaves barely changes either band, so the index flattens near its ceiling.
And it is **non-linear** in biomass throughout, so differences in NDVI are not proportional
to differences in vegetation.

### In this dataset — and a warning

Corpus range $-0.996 \ldots 0.893$, median $0.551$, mean $0.387$.

**The supplied `ndvi` cannot be recomputed from the supplied `red` and `nir`.**
[`base_eda.ipynb`](base_eda.ipynb) §4b.1 measures this over 225,000 pixels:

- **Naively**, `nir` is $\sim\!27\times$ larger than `red` at every pixel because they are in
  different units. The normalized difference is then pinned near $+1$ (measured mean
  $+0.890$) and describes the unit mismatch rather than the vegetation. Mean absolute error
  against the supplied band: $0.514$.
- **After rescaling** red onto the NIR scale by the ratio of medians, the recomputed index
  spans a sensible range and correlates $r = +0.81$ with the supplied band — but still
  differs pixel-by-pixel (MAE $0.490$).

The conclusion is that band 3 was computed from a red band **matched to the Planet NIR
sensor**, which is not the Maxar red shipped in band 0. Two consequences follow:

1. `ndvi` carries real information that bands 0 and 6 cannot reproduce. Do not discard it as
   redundant.
2. `red` and `nir` must never be combined arithmetically without cross-calibration.

**RTS signature.** NDVI is **lower** inside a slump (AUC $0.422$) — a thaw slump destroys the
vegetation that NDVI measures. This is the most physically direct signature in the stack.

---

## 5. Band 7 — NDWI

*Normalized Difference Water Index.* Same algebra as NDVI, different band pair:

$$\boxed{\;\mathrm{NDWI} \;=\; \frac{\rho_{green} - \rho_{NIR}}{\rho_{green} + \rho_{NIR}}\;}$$

### Why this pair

Liquid water **absorbs NIR very strongly** — it is nearly black beyond ~0.7 µm — while still
reflecting a usable amount of green. So over water $\rho_{green} > \rho_{NIR}$ and the index
goes positive; over vegetation and soil, where NIR is high, it goes negative. The sign flip
is the detector. The denominator does the same illumination-cancelling job as in NDVI, and
the same $[-1, +1]$ bound follows by the same argument.

Two indices share this name and they are **not** interchangeable:

- **McFeeters (1996)**, green and NIR, as above — detects *open water bodies*.
- **Gao (1996)**, NIR and SWIR — measures *water content within vegetation*.

This release has no SWIR band, and the measured corpus values are consistent with the
McFeeters form.

### In this dataset

Corpus range $-0.809 \ldots 0.998$, median $-0.568$, mean $-0.373$: **negative nearly
everywhere**, which is exactly right for predominantly dry land, with a positive tail where
chips contain sea or lakes.

The same cross-sensor caveat as NDVI applies, and was measured the same way. Recomputing
$(\rho_{green}-\rho_{NIR})/(\rho_{green}+\rho_{NIR})$ from the shipped bands 1 and 6 gives a
median of $-0.925$ against the supplied band's $-0.569$ — the shipped green is $\sim\!9\times$
smaller than NIR in raw units, so the naive result is pushed toward $-1$. The correlation is
high ($r = +0.84$), so the *ordering* is largely preserved, but the values are not
reproducible. Band 7, like band 3, came from sensor-matched inputs not present in this
release.

**Redundancy warning.** Measured over 200,000 pixels, `ndvi` and `ndwi` correlate at
$r = -0.980$. Both are dominated by the same NIR-versus-visible contrast, so as model inputs
they are close to duplicates of each other with the sign flipped.

**RTS signature.** NDWI is **higher** inside a slump (AUC $0.613$ — the strongest single-band
signal in the dataset), consistent with the wet, saturated, freshly-thawed ground a slump
exposes.

> **A statistical subtlety.** For `ndwi` and `ndvi`, Cohen's $d$ and the AUC disagree in sign.
> That is not an error. Open water in the *background* class sits at the extreme end of both
> indices and drags the class **mean**, while the bulk of the distribution moves the other
> way. AUC is rank-based and describes the typical pixel; $d$ is mean-based and here is
> reporting the tail. Trust the AUC.

---

## 6. Band 4 — relative elevation

Fully derived from a Digital Elevation Model (DEM) — a raster $Z(x,y)$ of ground height. No
optical measurement is involved.

The band is not absolute height. It is height **relative to a local reference surface**:

$$\boxed{\;Z_{rel}(x,y) \;=\; Z(x,y) \;-\; \bar{Z}_{\Omega}(x,y)\;}$$

where $\bar{Z}_{\Omega}$ is the DEM smoothed over a neighbourhood $\Omega$ — a moving mean, a
low-pass filter, or a locally fitted plane. This is a **high-pass filter on topography**.

### Why detrend at all

Absolute elevation is nearly useless for landform detection. A slump low on a coastal plain and an identical
slump high on a plateau have wildly different $Z$ but identical *local shape*. Worse, on a regional
slope the absolute height varies far more across a single chip than the
local relief of the feature of interest does, swamping it.

Subtracting the local trend removes the regional slope and leaves only local departures:
positive on ridges and mounds, negative in hollows and channels, near zero on uniform ground.
Landform geometry becomes comparable across chips at different altitudes.

**In this dataset** the measured statistics confirm exactly this. The median is
$-0.0007$ and the mean $+0.0104$ — centred on zero, as a detrended field must be — with the
central bulk inside roughly $\pm 4$ units (p1 $-4.24$, p99 $+4.31$) — the release
documents no units for the DEM bands, so these are not asserted to be metres.

But the corpus extremes reach $-335.9$ and $+311.3$ against a standard deviation of only
$1.38$. Those are **outliers of over 200 standard deviations**, almost certainly DEM
artefacts (voids, edge effects, water-surface noise). Combined with this band having the
joint-highest NaN rate (0.56%), the practical consequence is concrete: normalise this band
with **robust** statistics (median/IQR or 1–99 percentile clipping), never mean/std.

**RTS signature.** Lower inside a slump (AUC $0.449$) — a thaw slump *is* a depression. But
it is the **weakest** band of the eight, because a bare per-pixel height says nothing about
the shape that actually defines a slump: a bowl bounded by a steep headwall. That shape is a
spatial pattern, and only a model with a receptive field can see it.

---

## 7. Band 5 — shaded relief (hillshade)

Also fully derived from the DEM, by **simulating** how the terrain would look lit from a
fixed sun. It is a rendering, not a measurement.

### Building it up

**Step 1 — gradients.** Take partial derivatives of the DEM by finite differences (in
practice, Horn's 3×3 kernel, which averages over eight neighbours for stability):

$$p = \frac{\partial Z}{\partial x}, \qquad q = \frac{\partial Z}{\partial y}$$

**Step 2 — slope and aspect.** Slope is how steep, aspect is which way it faces:

$$S = \arctan\sqrt{p^2 + q^2}, \qquad A = \operatorname{arctan2}(q,\, p)$$

**Step 3 — the surface normal.** A surface $z = Z(x,y)$ has upward normal

$$\mathbf{n} = \frac{(-p,\; -q,\; 1)}{\sqrt{p^2 + q^2 + 1}}$$

**Step 4 — Lambertian shading.** A perfectly diffuse surface reflects light in proportion to
the cosine of the angle between its normal and the light direction $\mathbf{l}$ — Lambert's
cosine law. With solar zenith $Z_s$ (measured from vertical) and azimuth $A_s$:

$$\boxed{\;H \;=\; 255 \cdot \max\!\bigl(0,\; \cos S \cos Z_s + \sin S \sin Z_s \cos(A_s - A)\bigr)\;}$$

which is exactly $255\,\max(0, \mathbf{n} \cdot \mathbf{l})$ written in slope/aspect terms.
The $\max(0,\cdot)$ clamps self-shadowed faces to black. Conventional illumination is 45°
altitude ($Z_s = 45°$) from the northwest ($A_s = 315°$) — a convention, chosen because human
depth perception assumes light from above-left.

**In this dataset** the observed range is $142.8 \ldots 188.8$ on a nominal $0\ldots255$
scale, with median $166.5$ and a standard deviation of only $11.1$. That narrow band centred
near $165$ is what **low-relief terrain** produces: for flat ground $S \approx 0$ and
$H \approx 255\cos Z_s \approx 180$, and gentle Arctic slopes perturb this only slightly.
Nothing here is steep.

### Does it add information?

Hillshade is a deterministic function of the DEM, so it adds nothing beyond what the DEM
already holds. But it is **not** redundant with band 4 — measured correlation between
`shaded_relief` and `relative_elevation` is $r = 0.00$.

That is not a coincidence, it is structural: hillshade depends on the DEM's **gradient**
$\nabla Z$, while relative elevation depends on its **height** residual $Z - \bar{Z}$. The
two extract orthogonal aspects of the same surface. Keep both.

**RTS signature.** Brighter inside a slump (AUC $0.571$) — consistent with the headwall
presenting freshly exposed, sunward-facing slopes.

---

## 8. Summary table

Statistics over every pixel of all 756 training chips
([`base_eda.ipynb`](base_eda.ipynb) §6); percentiles from a 2,000-pixel-per-chip sample;
AUC from §11.

| # | band | kind | formula | theoretical range | observed median | observed min … max | std | NaN % | RTS signature (AUC) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | `red` | measured (Maxar) | — | 8-bit DN | 89 | 0 … 255 | 37.5 | 0.003 | **brighter** (0.592) |
| 1 | `green` | measured (Maxar) | — | 8-bit DN | 92 | 0 … 255 | 30.9 | 0.003 | **brighter** (0.578) |
| 2 | `blue` | measured (Maxar) | — | 8-bit DN | 73 | 0 … 255 | 30.7 | 0.007 | **brighter** (0.597) |
| 3 | `ndvi` | derived | $\frac{\rho_{NIR}-\rho_{red}}{\rho_{NIR}+\rho_{red}}$ | $[-1, 1]$ | 0.551 | −0.996 … 0.893 | 0.378 | 0.001 | **lower** (0.422) |
| 4 | `relative_elevation` | derived (DEM) | $Z - \bar{Z}_\Omega$ | unbounded, centred 0 | −0.001 | −335.9 … 311.3 | 1.38 | 0.561 | lower (0.449) |
| 5 | `shaded_relief` | derived (DEM) | $255\max(0, \mathbf{n}\cdot\mathbf{l})$ | $[0, 255]$ | 166.5 | 142.8 … 188.8 | 11.1 | 0.578 | brighter (0.571) |
| 6 | `nir` | measured (Planet) | — | scaled reflectance | 2439 | 73 … 7311 | 872 | 0.000 | **lower** (0.426) |
| 7 | `ndwi` | derived | $\frac{\rho_{green}-\rho_{NIR}}{\rho_{green}+\rho_{NIR}}$ | $[-1, 1]$ | −0.568 | −0.809 … 0.998 | 0.401 | 0.001 | **higher** (0.613) |

AUC $> 0.5$ means the band reads higher inside an RTS; $< 0.5$ means lower. $0.5$ is chance.

**The composite physical signature of a thaw slump**, read straight off the table: visible
bands brighter, NIR and NDVI lower, NDWI higher, ground locally depressed. That is bare wet
soil in a hollow where vegetation used to be — precisely what a retrogressive thaw slump is.

---

## 9. What this means for the model

**Per-band normalisation is mandatory.** The widest band span exceeds the narrowest by
$\sim\!4000\times$ (`nir` vs `ndwi`). An unnormalised 8-channel stack is an NIR model with
seven decorative channels. Use **robust** statistics (median/IQR, or 1–99 percentile
clipping) rather than mean/std, because `relative_elevation` carries 200σ outliers that would
wreck a mean/std fit.

**The eight channels are about four independent groups**, measured in §8:

| group | members | internal correlation |
|---|---|---|
| visible brightness | red, green, blue | $r = 0.92\ldots0.98$ |
| vegetation / NIR contrast | nir, ndvi, ndwi | $\lvert r\rvert = 0.86\ldots0.98$ |
| local topographic height | relative_elevation | — |
| topographic gradient | shaded_relief | — |

`ndwi` at $r = -0.98$ with `ndvi` is the strongest candidate for removal — but test it rather
than assuming it, since NDWI has the single best RTS AUC of the eight.

**NaN needs a mask, not just a fill value.** 11.4% of training chips (and 11.6% of test
chips) contain NaN, concentrated in the two DEM-derived bands at ~0.56% of pixels — roughly
100× the optical rate. `nir` alone is completely clean. Filling with zero is actively
harmful for `relative_elevation`, where zero is the modal *valid* value and would be
indistinguishable from real flat ground. Carry an explicit validity mask as an extra channel
and exclude invalid pixels from the loss.

**Pretrained backbones need a modified stem.** An ImageNet-pretrained first convolution
expects 3 channels. Expand it to 8 by replicating the RGB filters and rescaling so the
response magnitude is preserved, then let training adapt the rest.

**No single band solves this.** The best individual band is only $0.113$ away from chance.
Every per-pixel signature above is real and physically coherent, but small. **The
discriminative information is in spatial structure** — the bowl shape, the headwall edge, the
texture contrast against intact tundra — not in pixel values. That is a statement about
architecture: this task needs a segmentation model with sufficient receptive field to see
the landform, and a per-pixel classifier on band values will fail regardless of how the bands
are combined.

---

## References

- Rouse et al. (1974) — NDVI, ERTS symposium.
- McFeeters (1996) — *The use of the Normalized Difference Water Index (NDWI) in the
  delineation of open water features*, Int. J. Remote Sensing 17(7).
- Gao (1996) — *NDWI: a normalized difference water index for remote sensing of vegetation
  liquid water from space*, Remote Sensing of Environment 58(3).
- Horn (1981) — *Hill shading and the reflectance map*, Proceedings of the IEEE 69(1).
- Yang et al. (2023) — the source dataset,
  [doi:10.1016/j.rse.2023.113495](https://doi.org/10.1016/j.rse.2023.113495).
