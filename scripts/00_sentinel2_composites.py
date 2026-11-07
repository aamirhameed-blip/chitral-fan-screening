"""Step 00 (optional). Annual post-monsoon Sentinel-2 L2A median composites for Chitral.

Downloads from the Microsoft Planetary Computer STAC catalogue (no account needed).
Writes one GeoTIFF per year to CHITRAL_S2_DIR: 20 m, EPSG:32643, bands B03 B04 B08
B11 B12 (surface reflectance x 10000) and the count of valid observations. Cloud,
shadow and snow are masked with the scene classification (SCL), and the +1000 offset
of processing baseline 04.00 and later is removed.

Usage:  python scripts/00_sentinel2_composites.py [YEAR ...]     (default 2016-2025)
Needs the optional packages in environment.yml (pystac-client, planetary-computer, odc-stac, dask).
"""

from __future__ import annotations

import sys
import time
import warnings

import dask
import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from odc.stac import load
from shapely.ops import unary_union

from fanscreen import config
from fanscreen.config import CRS, raw

warnings.filterwarnings("ignore")

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
RES = 20
BANDS = ["B03", "B04", "B08", "B11", "B12"]
VALID_SCL = [2, 4, 5, 6, 7]  # dark area, vegetation, bare soil, water, unclassified
SCENES_PER_TILE = 4  # lowest-cloud scenes kept per MGRS tile


def search_items(catalogue, area, year: int) -> list:
    """Lowest-cloud post-monsoon scenes; the window and cloud limit are wider before 2019, when fewer scenes exist."""
    early = year <= 2018
    window = f"{year}-07-15/{year}-11-10" if early else f"{year}-08-15/{year}-10-31"
    items = catalogue.search(
        collections=["sentinel-2-l2a"],
        intersects=area.__geo_interface__,
        datetime=window,
        query={"eo:cloud_cover": {"lt": 50 if early else 20}},
    ).items()
    per_tile: dict = {}
    for item in sorted(items, key=lambda i: i.properties["eo:cloud_cover"]):
        tile = per_tile.setdefault(item.properties["s2:mgrs_tile"], [])
        if len(tile) < SCENES_PER_TILE:
            tile.append(item)
    return [item for tile in per_tile.values() for item in tile]


def composite(items: list, bbox) -> tuple[np.ndarray, object]:
    offset = np.array([1000 if float(i.properties.get("s2:processing_baseline", "0")) >= 4.0 else 0 for i in items])
    ds = load(
        items,
        bands=BANDS + ["SCL"],
        crs=CRS,
        resolution=RES,
        x=(bbox[0], bbox[2]),
        y=(bbox[1], bbox[3]),
        chunks={"x": 2048, "y": 2048},
        groupby=None,
        resampling={"SCL": "nearest", "*": "average"},
    )
    valid = ds.SCL.isin(VALID_SCL)
    off = ds.time.copy(data=offset)
    layers = {b: (ds[b].astype("float32") - off).where(valid & (ds[b] > 0)).median("time", skipna=True) for b in BANDS}
    layers["count"] = valid.sum("time")
    computed = dask.compute(*layers.values())
    stack = np.stack(
        [np.nan_to_num(c.values, nan=-9999).round().astype("int16") for c in computed[:-1]]
        + [computed[-1].values.astype("int16")]
    )
    return stack, ds.odc.geobox.transform


def main(years: list[int]) -> None:
    out_dir = config.sentinel2(years[0]).parent
    district = gpd.read_file(raw("boundary"))
    area = unary_union(district.to_crs(4326).geometry)
    bbox = district.to_crs(CRS).total_bounds
    catalogue = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    dask.config.set(scheduler="threads", num_workers=6)
    for year in years:
        path = out_dir / f"S2_{year}.tif"
        if path.exists():
            print(year, "exists, skipped", flush=True)
            continue
        start = time.time()
        items = search_items(catalogue, area, year)
        stack, transform = composite(items, bbox)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=stack.shape[1],
            width=stack.shape[2],
            count=stack.shape[0],
            dtype="int16",
            crs=CRS,
            transform=transform,
            nodata=-9999,
            compress="deflate",
            tiled=True,
            predictor=2,
            BIGTIFF="IF_SAFER",
        ) as dst:
            dst.write(stack)
            for k, name in enumerate(BANDS + ["count"], 1):
                dst.set_band_description(k, name)
        print(
            f"{year}: {len(items)} scenes, valid share of bounding box {(stack[-1] > 0).mean():.1%}, {time.time() - start:.0f} s",
            flush=True,
        )


if __name__ == "__main__":
    main([int(y) for y in sys.argv[1:]] or list(range(2016, 2026)))
