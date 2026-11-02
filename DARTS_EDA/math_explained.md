# The mathematics behind the DARTS × GeoAI compatibility EDA

This is the companion to `report_darts_compatibility.md`. Every equation the report or the
code relies on is derived here, starting from the simplest idea it depends on. The sections
are numbered **M1 … M14** and the report points to them.

Read it in order the first time: later sections reuse ideas from earlier ones.

Notation used throughout:

| symbol | meaning |
|---|---|
| $x$ | a single pixel value |
| $x_1, \dots, x_n$ | a sample of $n$ pixel values |
| $\bar x$ | the mean of a sample |
| $X$ | a random variable: "the value of a pixel picked at random" |
| $F(t)$ | a cumulative distribution function, $F(t) = P(X \le t)$ |
| $Z(r, c)$ | elevation at image row $r$, column $c$ |
| GSD | ground sampling distance, the width of one pixel on the ground in metres |

---

## M1. Pixels, grids and ground sampling distance

### M1.1 What a raster is

A satellite image is a grid of numbers. Each number is the average of some physical
quantity (reflected light, elevation) over a small square of ground. The side of that square
is the **ground sampling distance (GSD)**:

- Maxar Vivid basemap: 4 m
- Sentinel-2 bands B02, B03, B04, B08: 10 m
- Sentinel-2 band B8A: 20 m
- ArcticDEM mosaic: 2 m

The GeoAI chips are stored on a **4 m grid**: one array cell = 4 m × 4 m on the ground.

### M1.2 Areas in pixels ↔ areas in square metres

One pixel covers $\text{GSD}^2$ square metres. So an object of area $A$ m² covers

$$
N_{\text{px}} = \frac{A}{\text{GSD}^2}
$$

pixels. At 4 m, one pixel is 16 m². A DARTS slump of 6 271 m² therefore covers
$6271 / 16 \approx 392$ px.

The report also asks the reverse question: *if DARTS slumps and GeoAI slumps had the same
typical size, what GSD would make their pixel counts agree?* Solve the equation above for GSD:

$$
\text{GSD}_{\text{implied}} = \sqrt{\frac{\text{median area in m}^2 \text{ (DARTS)}}{\text{median area in px (GeoAI)}}}
$$

If this comes out near 4 m, the two populations have similar real-world sizes. If it comes out
smaller, DARTS slumps are physically smaller than GeoAI slumps.

### M1.3 A coarse band placed on a fine grid

To stack a 10 m Sentinel-2 band with a 4 m grid, each 4 m cell needs a value. The 10 m image
simply does not contain information at a finer scale than 10 m. Resampling fills the 4 m
cells by **interpolation** (M3), which invents smooth values in between and adds no new
detail. This is the key fact behind the spectral analysis in M2: a band that came from a
coarse source keeps the "fingerprint" of that source even after it is put on a fine grid.

---

## M2. Fourier analysis: reading the source resolution out of an image

### M2.1 Any signal is a sum of waves

Take one row of pixels, $x_0, x_1, \dots, x_{N-1}$. A central result of mathematics
(Fourier's theorem, discrete form) says any such list can be written *exactly* as a sum of
$N$ sine/cosine waves of different frequencies:

$$
x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \, e^{2\pi i \, k n / N}
$$

Here $e^{i\theta} = \cos\theta + i\sin\theta$ is just a compact way to write a cosine and a
sine together. The complex number $X_k$ says *how much* of the wave with frequency
$k/N$ cycles per pixel is present (its magnitude) and where it starts (its phase).

The coefficients are computed with the **discrete Fourier transform (DFT)**:

$$
X_k = \sum_{n=0}^{N-1} x_n \, e^{-2\pi i \, k n / N}
$$

Why does this work? The waves $e^{2\pi i kn/N}$ for different $k$ are *orthogonal*: multiply
two different ones together and sum over $n$, and you get zero. So multiplying the signal by
one wave and summing "picks out" exactly that wave's amount, the way projecting a vector on an
axis picks out one coordinate. `np.fft.fft2` computes the 2-D version (waves running in both
row and column directions) quickly with the Fast Fourier Transform algorithm.

### M2.2 Frequency and period

A frequency $f$ in cycles per pixel corresponds to a **period** of $1/f$ pixels: the length
of one full wave.

- $f = 0.5$ cycles/px is the fastest possible wave on a grid: up, down, up, down. Its period
  is 2 px. This limit is the **Nyquist frequency**. You need at least 2 samples per cycle to
  see a wave at all.
- $f = 0.1$ cycles/px is a slow wave with a period of 10 px.

### M2.3 The power spectrum

The **power** at frequency $k$ is $|X_k|^2$, the squared magnitude. It measures how much of
the image's variance (contrast) lives at that scale. A sharp image has power out to high
frequencies (fine edges). A blurry one has almost none there.

**Radial averaging.** In 2-D, each coefficient $X_{k_y, k_x}$ has a frequency in two
directions. Terrain has no preferred direction, so the code collapses the 2-D spectrum to 1-D:
the radial frequency is

$$
f = \frac{\sqrt{k_x^2 + k_y^2}}{N}
$$

and all coefficients with roughly the same $f$ are averaged. That is `radial_spectrum` in the
notebook: 48 rings from $f = 0$ to $f = 0.5$. It is then divided by its sum, so the spectra of
different bands (which have very different units) can be compared by *shape* alone.

**The Hann window.** The DFT assumes the image repeats forever, edge to edge. Real chips do
not wrap, so the jump at the border looks like a sharp edge and smears false power over all
frequencies. Multiplying the chip by a smooth bump that is zero at the edges removes the jump:

$$
w(n) = \frac{1}{2}\left(1 - \cos\frac{2\pi n}{N-1}\right)
$$

The 2-D window is the product $w(r)\,w(c)$ (`np.outer(np.hanning(n), np.hanning(n))`). The
mean is also subtracted first, so the $f = 0$ term (overall brightness) does not dominate.

### M2.4 Why a coarse source leaves a "knee"

Suppose a band was captured at 10 m and placed on the 4 m grid. On the 4 m grid, the 10 m
source pixel is $10/4 = 2.5$ px wide. By M2.2, the finest wave the source could ever record has
a period of 2 source pixels, which is

$$
\text{knee period} = 2 \times \frac{\text{source GSD}}{\text{grid GSD}} \;\text{px}
$$

- 10 m source → $2 \times 10/4 = 5$ px
- 20 m source → $2 \times 20/4 = 10$ px
- 4 m source (native) → 2 px, the grid's own Nyquist

Waves faster than this simply are not in the data. Interpolation (M3) does not create them.
So the spectrum falls along its natural slope and then **drops off a cliff** at the knee.

Natural images have spectra that fall roughly as a power law, $P(f) \propto f^{-\beta}$. On a
log-log plot this is a straight line with slope $-\beta$, because

$$
\log P = \text{const} - \beta \log f .
$$

A cliff is therefore a point where the **log-log slope suddenly becomes much steeper**. The
notebook's `knee` function computes the slope between neighbouring rings,

$$
s_j = \frac{\log P_{j+1} - \log P_j}{\log f_{j+1} - \log f_j},
$$

finds the ring where $s_j$ is most negative (steepest drop), and reports $1/f$ there as the
knee period in pixels. Inverting the knee formula gives the source GSD:

$$
\text{source GSD} \approx \frac{\text{knee period} \times 4\text{ m}}{2}.
$$

A knee of 13.3 px gives about 27 m, which is why nir is called a "~20–30 m source".

### M2.5 Spectral-shape distance

To ask "do these two bands come from the same kind of source?", the code compares whole
spectra, not just knees:

$$
d(P, Q) = \sqrt{\frac{1}{J} \sum_{j} \big(\log_{10} P_j - \log_{10} Q_j\big)^2 }
$$

This is the root-mean-square difference of the two curves on the log-log plot, over rings 2
to 40 (the extreme rings are noisy). Logs are used because power spans many orders of
magnitude, and on a log scale a constant factor between two spectra becomes a constant offset.
0 means identical shape. Larger means more different.

---

## M3. Resampling: bilinear interpolation, area averaging and blocks

### M3.1 Bilinear interpolation (going from coarse to fine)

To find a value at a point that lies between four known pixel centres, interpolate linearly
in one direction, then in the other. In 1-D, a point a fraction $t$ of the way from value
$a$ to value $b$ gets

$$
v = (1 - t)\,a + t\,b .
$$

This is a weighted average: the closer neighbour gets more weight. In 2-D with fractions
$t$ (horizontal) and $u$ (vertical) inside the square of four neighbours $a, b, c, d$:

$$
v = (1-u)\big[(1-t)a + t b\big] + u\big[(1-t)c + t d\big].
$$

The result is continuous and smooth, but every new value is an average of old values. So
**bilinear upsampling cannot add detail**. This is why S2 RGB at 4 m still looks like 10 m.

### M3.2 Area averaging (going from fine to coarse)

To simulate a coarser sensor, each coarse pixel should be the average of the fine pixels it
covers. Physically a sensor pixel integrates the light from its footprint, so averaging is
the right model. In the code:

- `block_mean(a, 2)`: the DEM is at 2 m and the grid is 4 m. Every 2 × 2 block of 2 m cells
  is averaged into one 4 m cell:
  $$ v_{RC} = \frac{1}{4}\sum_{r=2R}^{2R+1}\sum_{c=2C}^{2C+1} a_{rc} $$
- `down_up(a)`: simulates 10 m from GeoAI's 4 m. A 3 × 3 box average (≈ 12 m footprint) is
  followed by shrinking by 2.5× and then bilinear enlargement back to the 4 m grid. The result
  is "what Maxar would look like if it had been captured at 10 m".

### M3.3 Nearest neighbour

For category maps (the Sentinel-2 cloud classes, SCL) averaging makes no sense: the mean of
"cloud" and "snow" is meaningless. Each new cell takes the value of the single closest old
cell. That is `Resampling.nearest`.

---

## M4. Filters as moving averages: relative elevation

### M4.1 The box (mean) filter

A $k \times k$ mean filter replaces each pixel with the average of the $k \times k$ window
around it:

$$
\bar Z_k(r, c) = \frac{1}{k^2} \sum_{i=-h}^{h}\sum_{j=-h}^{h} Z(r+i, c+j), \qquad h = \frac{k-1}{2}.
$$

This is a **convolution** with a kernel of $k^2$ equal weights $1/k^2$. It keeps slow,
large-scale variation (a whole hillside) and wipes out anything smaller than the window.

### M4.2 Relative elevation is a high-pass filter

Yang et al. (2023) define relative elevation as

$$
\text{RE}(r, c) = Z(r, c) - \bar Z_{15}(r, c).
$$

Subtracting the smooth version leaves only what the smooth version removed: features smaller
than about 15 px. At 2 m that is a 30 m window. In the Fourier terms of M2, the mean filter is
a **low-pass** filter, and "original minus low-pass" is a **high-pass** filter.

Why this suits slumps: a hillside raises everything in a 30 m window, so it cancels out. A
slump floor is a local depression a few metres deep, so it stays. RE is negative inside a
depression ("lower than its surroundings"). That is why the RTS AUC of rel-elev is below 0.5
(see M10).

### M4.3 The order of operations matters

The paper filters at 2 m and the GeoAI grid is 4 m. Two orders are possible:

1. filter with $k = 15$ at 2 m (30 m window), then block-average to 4 m (paper order)
2. block-average to 4 m first, then filter at 4 m with $k = 15$ (60 m window) or
   $k = 8$ (32 m window)

The window size in metres differs, so the RE values differ: a wider window removes less, and
the RE spread grows. The notebook builds all three and keeps the one whose distribution and
spectrum match GeoAI best.

### M4.4 Mean filter with missing data (normalised convolution)

DEM tiles have holes (NaN). A plain average would turn every window touching a hole into NaN.
The fix is to average only the valid pixels:

$$
\bar Z(r,c) = \frac{\sum_{\text{window}} m \cdot Z}{\sum_{\text{window}} m}, \qquad
m = \begin{cases}1 & Z \text{ valid}\\ 0 & Z \text{ missing}\end{cases}
$$

Both sums are themselves box filters, so the code runs `uniform_filter` twice (numerator and
denominator) and divides. This is `nan_mean_filter`.

---

## M5. Hillshade, slope and the enhanced shaded relief

### M5.1 The surface gradient

Elevation is a surface $Z(x, y)$. Its steepness in each direction is a partial derivative.
On a grid, the derivative is approximated by a **central difference**: the change across two
neighbours divided by the distance between them,

$$
\frac{\partial Z}{\partial x} \approx \frac{Z(r, c+1) - Z(r, c-1)}{2\,\Delta}, \qquad
\Delta = 2 \text{ m}.
$$

That is `np.gradient(z, res)`.

### M5.2 Slope

The steepest rise at a point has magnitude
$\|\nabla Z\| = \sqrt{(\partial Z/\partial x)^2 + (\partial Z/\partial y)^2}$. It is metres
up per metre across, i.e. the tangent of the slope angle:

$$
\text{slope} = \arctan\!\Big(\sqrt{Z_x^2 + Z_y^2}\Big).
$$

Flat ground has slope 0°. A headwall can be 30–45°.

### M5.3 Aspect

Aspect is the compass direction the slope faces, measured clockwise from north. The code uses
`aspect = arctan2(-dZ/dx, dZ/dy)`. The signs come from the image convention: rows increase
southwards, so the row gradient is minus the northward gradient.

### M5.4 Lambert's law: why slopes facing the sun look bright

A matte surface receives light in proportion to the cosine of the angle $\gamma$ between its
**normal vector** $\mathbf n$ (pointing straight out of the surface) and the **direction to
the sun** $\mathbf l$:

$$
I = \max(0,\; \mathbf n \cdot \mathbf l) = \max(0, \cos\gamma).
$$

Intuition: a beam of light of fixed width spreads over more surface the more the surface is
tilted away from it, so each square metre gets less light.

Describe the sun by its **azimuth** $\phi$ (compass direction) and **altitude** $\alpha$
(angle above the horizon). The **zenith** angle is $\theta = 90° - \alpha$. Write the surface
by its slope $s$ and aspect $a$. Expanding the dot product in spherical coordinates gives the
standard hillshade formula:

$$
I = \cos\theta \cos s + \sin\theta \sin s \cos(\phi - a).
$$

Two checks:
- **Flat ground** ($s = 0$): $I = \cos\theta = \sin\alpha$. The flat brightness depends only
  on the sun's height. With the 0–255 scale, $255\sin\alpha = 166.5$ (GeoAI's median) gives
  $\alpha \approx 41°$. This is why altitudes of 30°, 45° and 60° were tried.
- **Slope facing the sun** ($a = \phi$): $I = \cos(\theta - s)$, bright. Facing away:
  $I = \cos(\theta + s)$, dark. Negative values mean self-shadow, clipped to 0.

The code scales to 0–255: `hs = 255 * clip(I, 0, None)`.

### M5.5 Multi-directional hillshade

A single sun direction hides ridges that run parallel to the light. The paper averages
hillshades from 8 azimuths with fixed weights $w_\phi$:

$$
H = \frac{\sum_\phi w_\phi I_\phi}{\sum_\phi w_\phi}, \quad
w = \{N\!:.1,\ NE\!:0,\ E\!:.3,\ SE\!:.1,\ S\!:.1,\ SW\!:.3,\ W\!:0,\ NW\!:.1\}.
$$

Dividing by the sum of weights keeps the result on the same 0–255 scale. It is a **weighted
average**.

### M5.6 Enhanced shaded relief (ESR) and alpha blending

Slope is converted to the same 0–255 scale, with steep ground dark:
$S_{255} = 255\,(1 - \min(s/45°, 1))$. Then:

$$
\text{base} = (1 - w_s)\,H + w_s\,S_{255}
$$

$$
\text{ESR} = (1-\alpha)\,\text{base} + \alpha\,G_{255}
$$

where $G_{255}$ is the greyscale optical image (mean of R, G, B) stretched to 0–255. Each
line is a **convex combination**: weights that are non-negative and sum to 1, so the result
always lies between the inputs. $\alpha$ is the "opacity" of the photo layer. With
$\alpha = 0$ ESR is pure terrain. With $\alpha = 1$ it is pure photo.

The paper does not publish $\alpha$, the altitude, or the slope weight, so the notebook
**grid-searches** them (M5.8).

### M5.7 Gaussian smoothing

`gaussian_filter(x, σ)` is a weighted average where the weight falls off like a bell curve:

$$
G_\sigma(i, j) = \frac{1}{2\pi\sigma^2} e^{-(i^2 + j^2)/(2\sigma^2)}.
$$

A larger $\sigma$ blurs more. It is in the search because GeoAI's shaded band has a knee at
9.5 px, i.e. it is smoother than the 2 m DEM it came from. Some smoothing happened somewhere
in their pipeline.

### M5.8 Fitting the recipe: rank aggregation

Each candidate (altitude, $w_s$, $\alpha$, $\sigma$) is scored against GeoAI three ways:

1. the Wasserstein distance (M8) between robust-z-scored distributions (shape match)
2. the spectral distance (M2.5) (sharpness match)
3. $|\text{corr}(\text{ESR}, \text{grey}) - \text{corr}_{\text{GeoAI}}|$ (how much photo is blended in)

The three are in different units, so they cannot be added. Instead each candidate gets its
**rank** in each list (1 = best), and the ranks are summed. The lowest total wins. Ranks are
unit-free, and one extreme value cannot swamp the others.

### M5.9 Affine matching

The fitted ESR has the right *shape* but arbitrary offset and scale. A linear map
$y = a x + b$ fixes both. Choose $a$ to match the spread and $b$ to match the centre:

$$
a = \frac{\sigma_{\text{GeoAI}}}{\sigma_{\text{DARTS}}}, \qquad
b = \text{median}_{\text{GeoAI}} - a \cdot \text{median}_{\text{DARTS}}.
$$

After this, $\text{median}(y) = \text{median}_{\text{GeoAI}}$, because a linear map moves the
median with it, and $\sigma(y) = |a|\,\sigma_{\text{DARTS}} = \sigma_{\text{GeoAI}}$.

---

## M6. Spectral indices: NDVI and NDWI

### M6.1 Normalised differences

Healthy vegetation absorbs red light (chlorophyll) and strongly reflects near-infrared (leaf
cell structure). Water absorbs NIR strongly. A **normalised difference** turns this into one
number in $[-1, 1]$:

$$
\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}, \qquad
\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}.
$$

**Why divide?** Scaling both bands by the same factor $k$ (brighter sun, different sensor
gain) cancels out:
$\frac{kN - kR}{kN + kR} = \frac{N - R}{N + R}$. A ratio of a difference to a sum is
insensitive to overall brightness. This is why the indices transfer between sensors far better
than the raw bands do.

The code adds $\varepsilon = 10^{-6}$ to the denominator to avoid dividing by zero.

### M6.2 Why ndvi and ndwi are almost mirror images

Both are driven by NIR, with opposite sign. When NIR is high, NDVI goes up and NDWI goes down.
That is why their correlation is about −0.98.

### M6.3 Sentinel-2 reflectance scaling

Sentinel-2 L2A stores surface reflectance $\rho$ as an integer $\text{DN} = 10000\,\rho$.
Since processing baseline 4.0 (January 2022), an offset of 1000 is added so that slightly
negative reflectances (from atmospheric correction noise) can still be stored as positive
integers:

$$
\rho = \frac{\text{DN} - 1000}{10000}.
$$

The fetcher subtracts 1000 for baseline ≥ 4.0. Otherwise images from 2022+ would read 1000
too bright, and NDVI would be biased, because the offset does not cancel in a ratio of
differences.

---

## M7. Describing a distribution: centre, spread, percentiles

### M7.1 Mean and standard deviation

$$
\bar x = \frac{1}{n}\sum_i x_i, \qquad
\sigma = \sqrt{\frac{1}{n}\sum_i (x_i - \bar x)^2}.
$$

The standard deviation is the typical distance from the mean, in the data's own units. Both
are sensitive to outliers: a single crazy value (a DEM spike of 500 m) can move them a lot.

### M7.2 Percentiles, median and IQR (robust statistics)

Sort the sample. The **$p$-th percentile** is the value below which $p\%$ of the data lie.
The median is the 50th percentile, $Q_1$ the 25th, $Q_3$ the 75th. The **interquartile
range** $\text{IQR} = Q_3 - Q_1$ is the width of the middle half of the data.

These are **robust**: moving the most extreme 1% of values anywhere you like does not change
the median or the IQR at all. Relative elevation has heavy tails (cliffs, lake shores), so
the notebook uses median and IQR wherever the mean and σ would be dominated by a few pixels.

For a normal distribution $\text{IQR} \approx 1.35\,\sigma$, so an IQR-based z-score is on a
scale similar to an ordinary one.

### M7.3 p1 and p99

The 1st and 99th percentiles bound "almost everything", ignoring the 2% most extreme pixels.
GeoAI's rel-elev has p1/p99 ≈ ±4.3 m, so 98% of pixels lie within 4.3 m of their local mean.

---

## M8. Comparing two distributions: Wasserstein and Kolmogorov–Smirnov

The question is "how different are the pixel distributions of the two datasets?". It needs a
single number per channel.

### M8.1 The cumulative distribution function

For a sample, the **empirical CDF** is

$$
F(t) = \frac{\#\{x_i \le t\}}{n},
$$

the fraction of values at or below $t$. It climbs from 0 to 1 as $t$ goes from left to right.
Two datasets with identical distributions have identical CDFs.

### M8.2 Kolmogorov–Smirnov (KS) statistic

$$
D_{KS} = \max_t \big| F_G(t) - F_D(t) \big|
$$

The largest vertical gap between the two CDFs. It is between 0 and 1. A KS of 0.3 means there
is a threshold $t$ where, say, 70% of GeoAI pixels but only 40% of DARTS pixels lie below it.

KS only sees the *single worst* point, and with 150 000 samples its p-value is always
astronomically small, so the notebook reports the statistic, not the p-value.

### M8.3 Wasserstein-1 (earth mover's) distance

Picture each histogram as a pile of sand. The Wasserstein distance is the **minimum work**
(amount of sand × distance moved) needed to reshape one pile into the other.

In one dimension, the optimal plan is simple: move the $q$-th quantile of one pile to the
$q$-th quantile of the other. This gives two equivalent formulas:

$$
W_1 = \int_0^1 \big| F_G^{-1}(q) - F_D^{-1}(q) \big| \, dq
    = \int_{-\infty}^{\infty} \big| F_G(t) - F_D(t) \big| \, dt .
$$

The first reads as "the average distance between matching quantiles". The second reads as
"the total area between the two CDF curves". Unlike KS, it accounts for the *whole* shape,
and it is in the data's units: shifting every value by 5 gives $W_1 = 5$ exactly.

### M8.4 Why divide by σ

Wasserstein in raw units cannot be compared across channels (NIR is in thousands, NDVI in
fractions). Dividing by GeoAI's standard deviation gives **"how many GeoAI standard deviations
the DARTS distribution sits away"**:

$$
\tilde W = \frac{W_1}{\sigma_{\text{GeoAI}}}.
$$

Rough reading: below 0.1 is negligible, 0.1–0.3 is small, above 0.5 is large.

---

## M9. Normalisation schemes

Normalisation applies a transformation so that the numbers a model sees are comparable. Each
scheme answers "relative to what?".

| scheme | formula per channel | removes |
|---|---|---|
| raw | $x$ | nothing |
| dataset z | $\dfrac{x - \text{median}_{\text{dataset}}}{\text{IQR}_{\text{dataset}}}$ | a constant offset and gain between the two datasets |
| chip z | $\dfrac{x - \bar x_{\text{chip}}}{\sigma_{\text{chip}}}$ | offset and gain *per chip* (illumination, season, haze) |
| chip p2–98 | $\text{clip}\!\left(\dfrac{x - p_2}{p_{98} - p_2}, 0, 1\right)$ | the same per chip, robust to outliers |
| hist-match | $F_G^{-1}(F_D(x))$ (M9.1) | the whole shape of the dataset distribution |

**Dataset z** is an affine map $y = (x - m)/s$ fitted per dataset. If DARTS were just GeoAI
with a different brightness and contrast, i.e. $x_D = a x_G + b$, dataset z would make them
identical.

**Chip z** goes further: each chip is centred and scaled by its own statistics. This removes
differences *between* chips (one hazy, one sunny), not just between datasets. The cost: the
absolute level is lost. A chip that is entirely water and one that is entirely tundra end up
with the same mean (0).

**Percentile stretch** maps the chip's 2nd percentile to 0 and its 98th to 1. It is the robust
cousin of chip z and is what the gallery uses for display.

### M9.1 Histogram matching (quantile mapping)

The **probability integral transform** theorem: if $X$ has continuous CDF $F$, then $F(X)$ is
uniformly distributed on $[0, 1]$. Proof sketch:
$P(F(X) \le u) = P(X \le F^{-1}(u)) = F(F^{-1}(u)) = u$, which is the CDF of a uniform.

The reverse also holds: if $U$ is uniform, $F^{-1}(U)$ has CDF $F$.

Chain them: $x \mapsto F_D(x)$ turns DARTS values into uniforms, and $u \mapsto F_G^{-1}(u)$
turns uniforms into GeoAI-distributed values. So

$$
T(x) = F_G^{-1}\big(F_D(x)\big)
$$

makes the DARTS pixel distribution *equal* to GeoAI's. In words: a DARTS pixel at its
dataset's 30th percentile is mapped to the GeoAI value at the 30th percentile.

In code: both quantile tables are computed at 1001 points (0.0%, 0.1%, …, 100%), and
`np.interp(x, D_Q, GEO_Q)` evaluates $T$ by linear interpolation between them.

**What it cannot fix.** $T$ is one fixed curve for the whole dataset, applied to each pixel on
its own. It matches the *pooled* histogram exactly, so its Wasserstein distance to GeoAI is
near 0 by construction. But it cannot change spatial texture (sharpness), the relationship
between bands (correlations), or how values vary chip to chip. A domain classifier (M11) that
looks at these will still separate the datasets. So a near-zero Wasserstein after histogram
matching is *expected*, not evidence of compatibility.

---

## M10. ROC AUC: how well one number separates two groups

### M10.1 Definition by ranking

Take one channel, e.g. rel-elev. Pick a random RTS pixel and a random background pixel. The
**AUC** is the probability that the RTS pixel has the larger value:

$$
\text{AUC} = P\big(x_{\text{RTS}} > x_{\text{bg}}\big) + \tfrac12 P\big(x_{\text{RTS}} = x_{\text{bg}}\big).
$$

- AUC = 0.5: the channel says nothing (a coin flip).
- AUC > 0.5: RTS pixels tend to be **higher**.
- AUC < 0.5: RTS pixels tend to be **lower**. This is just as informative, in the opposite
  direction. An AUC of 0.45 is as useful as 0.55.

### M10.2 Why it is called "area under the curve"

Slide a threshold $t$. Call everything above it "RTS". The **true positive rate** is the
fraction of RTS pixels above $t$. The **false positive rate** is the fraction of background
pixels above $t$. Plotting TPR against FPR for every $t$ gives the ROC curve, and its area
equals the probability in M10.1. (This is the Mann–Whitney identity: the area counts, over all
RTS/background pairs, the fraction correctly ordered.)

### M10.3 Why AUC and not mean difference

AUC uses only the *order* of values, so it is unchanged by any increasing transformation:
scaling, shifting, logs, histogram matching. That makes it ideal for comparing the RTS "cue"
across datasets with different units. The report's **RTS-cue agreement** test only asks whether
both datasets have the AUC on the same side of 0.5: *is the slump brighter/darker,
higher/lower, in both?*

One subtlety: per-chip normalisation (chip z) is increasing **within** a chip but different
**between** chips. Since the AUC pools pixels from all chips, it *can* change after chip z. The
notebook checks this explicitly.

---

## M11. The domain classifier: a learned two-sample test

### M11.1 The idea

The most direct test of "are two datasets compatible?" is: **can a model tell them apart?**
If a classifier given a chip cannot guess which dataset it came from better than chance, the
model you eventually train cannot use "which dataset" as a shortcut either.

This is a **classifier two-sample test**. Label GeoAI chips 0 and DARTS chips 1, train a
classifier, and measure its AUC (M10) on chips it did not train on:

- AUC ≈ 0.5: indistinguishable
- AUC ≈ 1.0: perfectly separable

The report treats **< 0.65** as "compatible enough". It is a judgement call: weak residual
differences that a strong augmentation can drown out.

### M11.2 Features per chip

Each chip is reduced to a fixed-length vector of numbers:

- per channel: mean, std, p10, p50, p90 (5 numbers describing the histogram)
- between channels in the group: every pairwise Pearson correlation (M12)
- optionally, per channel: $\log_{10}$ of the Laplacian variance (M13), a sharpness measure

Running the test **with and without** the texture feature splits the gap into two parts:
"different values" and "different sharpness".

### M11.3 Logistic regression

Logistic regression models the probability of class 1 as

$$
P(y = 1 \mid \mathbf f) = \sigma(\mathbf w^\top \mathbf f + b), \qquad
\sigma(z) = \frac{1}{1 + e^{-z}}.
$$

$\mathbf w^\top \mathbf f + b$ is a weighted sum of features: a straight line (hyperplane)
through feature space. The **sigmoid** $\sigma$ squashes any real number into $(0, 1)$, so it
can be read as a probability. Training picks $\mathbf w, b$ to maximise the likelihood of the
true labels, which is the same as minimising the **log loss**:

$$
\mathcal L = -\sum_i \big[y_i \log p_i + (1-y_i)\log(1-p_i)\big] + \frac{1}{2C}\|\mathbf w\|^2.
$$

The last term (L2 regularisation, $C = 0.5$) penalises large weights so the model does not
overfit to noise. Features are first standardised (`StandardScaler`: subtract mean, divide by
σ), so one large-unit feature (NIR in thousands) does not dominate.

### M11.4 Gradient-boosted trees

A **decision tree** splits on one feature at a time ("is mean NDVI < 0.4?"). It can capture
non-linear rules and interactions that a single hyperplane cannot. **Boosting** builds 200
small trees in sequence. Each new tree is fitted to the errors (the gradient of the log loss)
the previous trees still make, and its output is added with a small learning rate (0.05).
The sum of many weak trees is a strong, flexible classifier.

The notebook reports the **maximum** AUC of the two models: the strictest test, since the
datasets count as separable if *either* model can separate them.

### M11.5 Cross-validation

A classifier scored on its own training data can memorise and look perfect. **5-fold
stratified cross-validation** splits the chips into 5 equal parts, keeping the GeoAI:DARTS
ratio in each. Each part is predicted by a model trained on the other 4. Every chip gets an
out-of-sample prediction (`cross_val_predict`), and one AUC is computed over all of them.

**Caveat.** GeoAI chips overlap spatially (known from the earlier fusion study). Some
near-duplicates land on both sides of a fold, which makes the GeoAI side slightly easier to
recognise. This biases the domain AUC **up**, so the estimate is conservative.

---

## M12. Correlation between bands

The **Pearson correlation** between two bands $a$ and $b$ over the same pixels:

$$
r = \frac{\sum_i (a_i - \bar a)(b_i - \bar b)}
         {\sqrt{\sum_i (a_i - \bar a)^2}\sqrt{\sum_i (b_i - \bar b)^2}}.
$$

The numerator (covariance) is positive when $a$ and $b$ are above their means together. The
denominator rescales it into $[-1, 1]$. It equals the cosine of the angle between the two
centred vectors $\mathbf a - \bar a$ and $\mathbf b - \bar b$: $+1$ means they point the same
way (perfect linear increase), $-1$ opposite, 0 unrelated.

It is unchanged by any affine rescaling of either band. So dataset-z or chip-z normalisation
**cannot** change the correlation matrix. If the DARTS − GeoAI correlation difference is large,
no per-channel normalisation can fix it. It is a structural difference between sensors.

---

## M13. Laplacian variance: a sharpness meter

The **Laplacian** is the sum of second derivatives:

$$
\nabla^2 x = \frac{\partial^2 x}{\partial r^2} + \frac{\partial^2 x}{\partial c^2}
\approx x_{r+1,c} + x_{r-1,c} + x_{r,c+1} + x_{r,c-1} - 4x_{r,c}.
$$

The second form is "the sum of the four neighbours minus 4 × the centre". On a smooth patch
(constant or linear), it is exactly 0. At an edge or a fine dot, the centre differs from its
neighbours and the Laplacian is large. In Fourier terms (M2), the Laplacian multiplies a wave
of frequency $f$ by about $(2\pi f)^2$, so it **amplifies the high frequencies** and suppresses
the low ones.

The **variance of the Laplacian over a chip** therefore measures how much high-frequency
energy, i.e. fine detail, the chip contains. Blurring removes high frequencies, so its
Laplacian variance falls.

The chip is first standardised (divide by its σ). Otherwise a high-contrast chip would score
"sharper" just for being contrasty. After standardisation, the number measures the *fraction*
of the chip's variance that lives at fine scales.

**How the report uses it.** Compute it for GeoAI native, GeoAI degraded to 10 m (M3.2), and
DARTS (true S2). If DARTS matches the degraded GeoAI, then the sharpness gap between datasets
is *entirely* due to the 4 m vs 10 m resolution, and a blur augmentation can remove it.

---

## M14. Small-object loss at 10 m

At 10 m, one S2 pixel covers 100 m². By M1.2, an object of $A$ m² covers $A/100$ S2 pixels.
A slump under 100 m² is smaller than a single S2 pixel. It blends into its neighbours and
cannot be seen. Under 400 m² (4 px, a 2 × 2 block) or 900 m² (3 × 3), it is a blob of a few
pixels without visible shape.

The report counts the fraction of DARTS and GeoAI slumps below each threshold. Those labels
would teach a model to "find" slumps in pixels that do not show them, which is label noise.
They should be dropped from a merged set when the DARTS optical bands are S2.

---

## Appendix: the ArcticDEM tile arithmetic (from the fetcher)

ArcticDEM 2 m mosaics are cut into 100 km × 100 km tiles in the polar stereographic projection
(EPSG:3413, coordinates in metres), each split into 2 × 2 sub-tiles of 50 km. For a point
$(x, y)$ in that projection:

$$
\text{row} = \left\lfloor \frac{y}{100\,000} \right\rfloor + 41, \qquad
\text{col} = \left\lfloor \frac{x}{100\,000} \right\rfloor + 41
$$

$$
\text{subrow} = 1 + [\,y \bmod 100\,000 \ge 50\,000\,], \qquad
\text{subcol} = 1 + [\,x \bmod 100\,000 \ge 50\,000\,]
$$

$\lfloor\cdot\rfloor$ rounds down, so it counts how many whole 100 km tiles lie between the
origin and the point. The $+41$ shifts the index so that the grid's bottom-left corner is
tile 1. $[\cdot]$ is 1 if the condition holds, else 0: the point is in the upper/right half of
its tile. The four corners of a chip are converted, and every tile they touch is read.
