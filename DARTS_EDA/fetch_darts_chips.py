"""Build GeoAI-style 8-band chips over DARTS training labels.

Pixels come from public sources only, rebuilt with the Yang et al. (2023) recipe that produced
the GeoAI Arctic bands, on the same 4 m grid:

    red, green, blue   Sentinel-2 L2A B04/B03/B02 (10 m)  -> 4 m bilinear   [GeoAI: Maxar 4 m]
    ndvi, ndwi         computed from S2 at 10 m          -> 4 m bilinear   [GeoAI: S2 10 m]
    nir                S2 B8A (20 m)                      -> 4 m bilinear   [GeoAI: ~20-30 m source]
    relative_elevation ArcticDEM 2 m minus 50 px mean    -> 4 m area mean   (100 m window: matches
                       the GeoAI band; the paper's 15 px / 30 m gives 5x too little amplitude)
    shaded_relief      ESR placeholder (refit in the notebook from the stored components)

Each chip is written to <out>/darts_XXXXX.npz with
    image  (H, W, 8) float32, GeoAI band order
    inst   (H, W) int16, 0 = background, k = k-th DARTS polygon of the same image_id
    extra_* alternative layers for the notebook's source/recipe tests
and one row per attempt in <out>/manifest.csv (status = ok or a skip reason).

Run with the isolated venv (the base env's rasterio/numpy pair is broken):
    DARTS_EDA/.venv/Scripts/python DARTS_EDA/fetch_darts_chips.py --n 20                 # EDA sample
    DARTS_EDA/.venv/Scripts/python DARTS_EDA/fetch_darts_chips.py --dataset --out DARTS_EDA/dataset
--dataset covers every label (>= --min_area m2) once instead of sampling, and keeps only the
extras needed to refit shaded_relief (float16).
"""
import argparse
import glob
import json
import math
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject
from scipy import ndimage
from shapely.geometry import box

LABEL_DIR = r"C:\Users\moham\Downloads\Darts_dataset\ML_training_labels\retrogressive_thaw_slumps"
GEOAI_MANIFEST = "competition_release/metadata/train_manifest.csv"
ADEM_URL = ("https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/mosaics/v4.1/2m/"
            "{r:02d}_{c:02d}/{r:02d}_{c:02d}_{sr}_{sc}_2m_v4.1_dem.tif")
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

GSD = 4.0          # GeoAI chip grid (m)
DEM_GSD = 2.0      # ArcticDEM mosaic
RE_KERNEL = 50     # px at 2 m = 100 m window, fitted to GeoAI (Yang et al. say 15 px)
DEM_PAD = 60       # extra metres read around the chip so filters have no edge effects
SCL_BAD = {0, 1, 3, 8, 9, 10, 11}  # nodata, saturated, shadow, cloud med/high, cirrus, snow
# multi-directional hillshade weights from Yang et al. (azimuth deg -> weight)
HS_WEIGHTS = {0: .1, 45: 0., 90: .3, 135: .1, 180: .1, 225: .3, 270: 0., 315: .1}
HS_ALTITUDES = (30, 45, 60)

os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")


# ---------------------------------------------------------------- labels
def load_labels():
    files = sorted(glob.glob(os.path.join(LABEL_DIR, "*", "TrainingLabel_*.gpkg")))
    parts = []
    for f in files:
        g = gpd.read_file(f)
        g["iteration"] = os.path.basename(os.path.dirname(f))[-12:]
        parts.append(g[["image_id", "image_date", "region", "iteration", "geometry"]])
    g = pd.concat(parts, ignore_index=True)
    g = gpd.GeoDataFrame(g, geometry="geometry", crs=4326)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    g["geometry"] = g.geometry.make_valid()
    g["image_date"] = pd.to_datetime(g["image_date"])
    g["uid"] = np.arange(len(g))
    g["area_m2"] = g.to_crs(6931).area
    return g


def sample_targets(labels, n, rng):
    """Stratify by region: equal share per region, topped up from the rest at random."""
    regions = labels["region"].fillna("unknown")
    per = max(1, n // regions.nunique())
    picked = []
    for _, idx in labels.groupby(regions).groups.items():
        idx = np.array(idx)
        picked.extend(rng.choice(idx, size=min(per, len(idx)), replace=False))
    rest = np.setdiff1d(labels.index.values, picked)
    if len(picked) < n:
        picked.extend(rng.choice(rest, size=n - len(picked), replace=False))
    return labels.loc[rng.permutation(picked)[:n]]


def plan_cover(labels, sizes, seed):
    """Chips that together contain every label once. Greedy over the labels in random order: each
    new chip is centred on an uncovered label and covers every label of the same image that lies
    fully inside it. Uses the same per-chip rng as build_chip, so the planned grid is the built one."""
    order = labels.loc[np.random.default_rng(seed).permutation(labels.index)]
    covered, plan = set(), []
    for row in order.itertuples():
        if row.uid in covered:
            continue
        crs, tf, w, h = chip_grid(row, sizes, np.random.default_rng(seed + len(plan)))
        same = labels[labels.image_id == row.image_id]
        covered.update(same.uid[same.to_crs(crs).within(box(*bounds_of(tf, w, h)))])
        covered.add(row.uid)
        plan.append(row)
    return plan


# ---------------------------------------------------------------- grid helpers
def utm_crs(lon, lat):
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1)


def chip_grid(row, sizes, rng):
    """4 m grid in local UTM, GeoAI-sized, with the target polygon jittered off-centre."""
    c = row.geometry.centroid
    crs = utm_crs(c.x, c.y)
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(c.x, c.y)
    w, h = sizes[rng.integers(len(sizes))]
    x += rng.uniform(-.25, .25) * w * GSD
    y += rng.uniform(-.25, .25) * h * GSD
    x0 = math.floor((x - w * GSD / 2) / GSD) * GSD
    y1 = math.ceil((y + h * GSD / 2) / GSD) * GSD
    return crs, from_origin(x0, y1, GSD, GSD), int(w), int(h)


def bounds_of(transform, w, h):
    x0, y1 = transform.c, transform.f
    return x0, y1 + transform.e * h, x0 + transform.a * w, y1


def lonlat_bbox(crs, b):
    t = Transformer.from_crs(crs, 4326, always_xy=True)
    xs, ys = t.transform([b[0], b[0], b[2], b[2]], [b[1], b[3], b[1], b[3]])
    return [min(xs), min(ys), max(xs), max(ys)]


def to_grid(src_arr, src_tf, src_crs, dst_tf, dst_crs, w, h, how):
    dst = np.full((h, w), np.nan, np.float32)
    reproject(src_arr.astype(np.float32), dst, src_transform=src_tf, src_crs=src_crs,
              dst_transform=dst_tf, dst_crs=dst_crs, resampling=how,
              src_nodata=np.nan, dst_nodata=np.nan)
    return dst


def read_on_grid(href, crs, tf, w, h, how=Resampling.nearest):
    """Read a remote COG resampled onto an arbitrary grid (WarpedVRT does the windowing)."""
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=crs, transform=tf, width=w, height=h, resampling=how,
                       src_nodata=src.nodata, nodata=np.nan, dtype="float32") as vrt:
            return vrt.read(1)


# ---------------------------------------------------------------- Sentinel-2
def s2_layers(catalog, crs, tf, w, h, date):
    b = bounds_of(tf, w, h)
    # native-resolution grids aligned to the chip extent
    tf10 = from_origin(math.floor(b[0] / 10) * 10, math.ceil(b[3] / 10) * 10, 10, 10)
    w10, h10 = math.ceil((b[2] - tf10.c) / 10), math.ceil((tf10.f - b[1]) / 10)
    tf20 = from_origin(math.floor(b[0] / 20) * 20, math.ceil(b[3] / 20) * 20, 20, 20)
    w20, h20 = math.ceil((b[2] - tf20.c) / 20), math.ceil((tf20.f - b[1]) / 20)

    window = f"{(date - pd.Timedelta(days=30)).date()}/{(date + pd.Timedelta(days=30)).date()}"
    items = list(catalog.search(collections=["sentinel-2-l2a"], bbox=lonlat_bbox(crs, b),
                                datetime=window, query={"eo:cloud_cover": {"lt": 60}}).items())
    items.sort(key=lambda it: (it.properties["eo:cloud_cover"] > 20,
                               abs((pd.Timestamp(it.datetime).tz_localize(None) - date).days),
                               it.properties["eo:cloud_cover"]))
    for it in items[:8]:
        scl = read_on_grid(it.assets["SCL"].href, crs, tf20, w20, h20)
        if np.isnan(scl).mean() > .3:
            continue  # chip falls off this granule's footprint
        bad = np.isin(scl, list(SCL_BAD)) | np.isnan(scl)
        if bad.mean() > .10:
            continue
        offset = 1000 if float(it.properties.get("s2:processing_baseline", "0")) >= 4.0 else 0
        raw = {}
        for band, grid in [("B02", (tf10, w10, h10)), ("B03", (tf10, w10, h10)),
                           ("B04", (tf10, w10, h10)), ("B08", (tf10, w10, h10)),
                           ("B8A", (tf20, w20, h20))]:
            a = read_on_grid(it.assets[band].href, crs, *grid)
            a[a == 0] = np.nan
            raw[band] = a - offset
        eps = 1e-6
        ndvi10 = (raw["B08"] - raw["B04"]) / (raw["B08"] + raw["B04"] + eps)
        ndwi10 = (raw["B03"] - raw["B08"]) / (raw["B03"] + raw["B08"] + eps)
        up10 = lambda a: to_grid(a, tf10, crs, tf, crs, w, h, Resampling.bilinear)
        up20 = lambda a: to_grid(a, tf20, crs, tf, crs, w, h, Resampling.bilinear)
        badmask = to_grid(bad.astype(np.float32), tf20, crs, tf, crs, w, h, Resampling.nearest) > 0
        out = {"red": up10(raw["B04"]), "green": up10(raw["B03"]), "blue": up10(raw["B02"]),
               "ndvi": up10(ndvi10), "ndwi": up10(ndwi10),
               "nir": up20(raw["B8A"]), "extra_nir_b08": up10(raw["B08"])}
        for k in out:
            out[k][badmask] = np.nan
        meta = {"s2_id": it.id, "s2_date": str(pd.Timestamp(it.datetime).date()),
                "s2_cloud": it.properties["eo:cloud_cover"], "s2_bad_frac": float(bad.mean())}
        return out, meta
    return None, {"s2_candidates": len(items)}


# ---------------------------------------------------------------- ArcticDEM
def adem_tiles(crs, b):
    t = Transformer.from_crs(crs, 3413, always_xy=True)
    xs, ys = t.transform([b[0], b[0], b[2], b[2]], [b[1], b[3], b[1], b[3]])
    tiles = set()
    for x in (min(xs), max(xs)):
        for y in (min(ys), max(ys)):
            r, c = math.floor(y / 1e5) + 41, math.floor(x / 1e5) + 41
            sr, sc = 1 + int((y % 1e5) >= 5e4), 1 + int((x % 1e5) >= 5e4)
            tiles.add(ADEM_URL.format(r=r, c=c, sr=sr, sc=sc))
    return sorted(tiles)


def nan_mean_filter(z, k):
    """Mean over a k x k box ignoring NaNs (normalised convolution)."""
    ok = np.isfinite(z)
    num = ndimage.uniform_filter(np.where(ok, z, 0.), k, mode="nearest")
    den = ndimage.uniform_filter(ok.astype(float), k, mode="nearest")
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def hillshade(z, res, az, alt):
    dzdy, dzdx = np.gradient(z, res)  # rows run north->south, so dzdy is -dZ/dNorth
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdx, dzdy)  # radians from north, clockwise
    zen, azr = np.radians(90 - alt), np.radians(az)
    hs = np.cos(zen) * np.cos(slope) + np.sin(zen) * np.sin(slope) * np.cos(azr - aspect)
    return 255 * np.clip(hs, 0, None), np.degrees(slope)


def block_mean(a, f):
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    return np.nanmean(a[:h, :w].reshape(h // f, f, w // f, f), axis=(1, 3))


def dem_layers(crs, tf, w, h):
    b = bounds_of(tf, w, h)
    pad_px = int(DEM_PAD / DEM_GSD)
    tf2 = from_origin(b[0] - DEM_PAD, b[3] + DEM_PAD, DEM_GSD, DEM_GSD)
    w2, h2 = int(w * GSD / DEM_GSD) + 2 * pad_px, int(h * GSD / DEM_GSD) + 2 * pad_px
    z = np.full((h2, w2), np.nan, np.float32)
    for url in adem_tiles(crs, bounds_of(tf2, w2, h2)):
        try:
            part = read_on_grid(url, crs, tf2, w2, h2, Resampling.bilinear)
        except rasterio.errors.RasterioIOError:
            continue  # tile does not exist (ocean / outside ArcticDEM)
        part[part < -1000] = np.nan
        z = np.where(np.isnan(z), part, z)
    if np.isnan(z).mean() > .3:
        return None, {"dem_nan_frac": float(np.isnan(z).mean())}

    crop = lambda a: a[pad_px:pad_px + h * 2, pad_px:pad_px + w * 2]
    to4 = lambda a: block_mean(crop(a), 2)[:h, :w].astype(np.float32)
    zf = np.where(np.isnan(z), nan_mean_filter(z, 5), z)  # fill pinholes before gradients

    out = {"relative_elevation": to4(z - nan_mean_filter(z, RE_KERNEL))}
    # alternative orders/kernels for the notebook's recipe match
    z4 = block_mean(z, 2)
    p4 = pad_px // 2
    out["extra_re_4m_k8"] = (z4 - nan_mean_filter(z4, 8))[p4:p4 + h, p4:p4 + w].astype(np.float32)
    out["extra_re_4m_k15"] = (z4 - nan_mean_filter(z4, 15))[p4:p4 + h, p4:p4 + w].astype(np.float32)
    slope = None
    for alt in HS_ALTITUDES:
        acc = np.zeros_like(zf, dtype=float)
        for az, wt in HS_WEIGHTS.items():
            if wt:
                hs, slope = hillshade(zf, DEM_GSD, az, alt)
                acc += wt * hs
        out[f"extra_hs_alt{alt}"] = to4(acc / sum(HS_WEIGHTS.values()))
    out["extra_slope_deg"] = to4(slope)
    out["extra_dem"] = to4(z)
    for k in out:
        out[k][np.isnan(to4(z))] = np.nan
    return out, {"dem_tiles": len(adem_tiles(crs, b))}


# ---------------------------------------------------------------- one chip
LEAN_EXTRAS = ("extra_hs_alt30", "extra_hs_alt45", "extra_hs_alt60", "extra_slope_deg")


def build_chip(i, row, labels, sizes, seed, catalog, out_dir, lean=False):
    rng = np.random.default_rng(seed + i)
    rec = {"chip": f"darts_{i:05d}", "uid": int(row.uid), "image_id": row.image_id,
           "image_date": str(row.image_date.date()), "region": row.region,
           "iteration": row.iteration, "status": "ok"}
    try:
        crs, tf, w, h = chip_grid(row, sizes, rng)
        rec.update(crs=crs.to_epsg(), width=w, height=h, x0=tf.c, y1=tf.f)
        s2, meta = s2_layers(catalog, crs, tf, w, h, row.image_date)
        rec.update(meta)
        if s2 is None:
            rec["status"] = "no_clear_s2"
            return rec
        dem, meta = dem_layers(crs, tf, w, h)
        rec.update(meta)
        if dem is None:
            rec["status"] = "dem_void"
            return rec

        # every DARTS polygon digitised on the same PlanetScope image, as instances
        b = bounds_of(tf, w, h)
        same = labels[labels.image_id == row.image_id].to_crs(crs)
        same = same[same.intersects(box(*b))]
        shapes = [(g, k + 1) for k, g in enumerate(same.geometry)]
        inst = rasterize(shapes, out_shape=(h, w), transform=tf, fill=0, dtype="int16") if shapes \
            else np.zeros((h, w), np.int16)
        rec["n_inst"] = int(len(np.unique(inst)) - 1)
        rec["fg_frac"] = float((inst > 0).mean())

        # placeholder ESR: equal-weight hillshade + inverted slope, 50% blend with S2 grey;
        # the notebook refits these weights against the GeoAI distribution
        grey = np.nanmean(np.stack([s2["red"], s2["green"], s2["blue"]]), 0)
        g255 = 255 * np.clip((grey - np.nanpercentile(grey, 2)) /
                             (np.nanpercentile(grey, 98) - np.nanpercentile(grey, 2) + 1e-6), 0, 1)
        sl255 = 255 * (1 - np.clip(dem["extra_slope_deg"] / 45, 0, 1))
        esr = .5 * (.5 * dem["extra_hs_alt45"] + .5 * sl255) + .5 * g255

        layers = {"red": s2["red"], "green": s2["green"], "blue": s2["blue"], "ndvi": s2["ndvi"],
                  "relative_elevation": dem["relative_elevation"], "shaded_relief": esr,
                  "nir": s2["nir"], "ndwi": s2["ndwi"]}
        image = np.stack([layers[k] for k in BAND_ORDER], -1).astype(np.float32)
        extras = {k: v for d in (s2, dem) for k, v in d.items() if k.startswith("extra_")}
        if lean:  # only what refitting shaded_relief needs
            extras = {k: extras[k].astype(np.float16) for k in LEAN_EXTRAS}
        np.savez_compressed(os.path.join(out_dir, rec["chip"] + ".npz"), image=image, inst=inst,
                            meta=json.dumps(rec), **extras)
        rec["nan_frac"] = float(np.isnan(image).mean())
    except Exception as e:  # keep going; the manifest records why
        rec["status"] = f"error: {type(e).__name__}: {e}"[:300]
        traceback.print_exc()
    return rec


BAND_ORDER = ["red", "green", "blue", "ndvi", "relative_elevation", "shaded_relief", "nir", "ndwi"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="DARTS_EDA/chips")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dataset", action="store_true", help="cover every label once instead of sampling --n")
    ap.add_argument("--min_area", type=float, default=0.,
                    help="m2; smaller polygons are neither targets nor masks (900 = 3x3 S2 px; --dataset default)")
    args = ap.parse_args()
    if args.dataset and not args.min_area:
        args.min_area = 900.

    os.makedirs(args.out, exist_ok=True)
    labels = load_labels()
    labels = labels[labels.area_m2 >= args.min_area]
    m = pd.read_csv(GEOAI_MANIFEST)
    sizes = m[["width", "height"]].to_numpy()
    targets = (plan_cover(labels, sizes, args.seed) if args.dataset
               else list(sample_targets(labels, args.n, np.random.default_rng(args.seed)).itertuples()))
    catalog = pystac_client.Client.open(PC_STAC, modifier=pc.sign_inplace)
    print(f"{len(labels)} DARTS labels (>= {args.min_area:g} m2), building {len(targets)} chips -> {args.out}")

    recs = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(build_chip, i, row, labels, sizes, args.seed, catalog, args.out, args.dataset)
                for i, row in enumerate(targets)]
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            recs.append(r)
            print(f"[{k}/{len(futs)}] {r['chip']} {r['region']}: {r['status']}", flush=True)
    man = pd.DataFrame(recs).sort_values("chip")
    man.to_csv(os.path.join(args.out, "manifest.csv"), index=False)
    print(man["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
