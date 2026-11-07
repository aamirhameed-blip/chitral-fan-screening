"""Step 01. Build the analysis-ready catchment table from the raw GIS layers.

Requires CHITRAL_RAW_DATA and CHITRAL_S2_DIR (see README). The outputs are already
in the repository, so this step is needed only to check or extend the dataset.

Outputs
    data/catchments.gpkg              catchment polygons with all attributes
    data/catchments.csv               the same table without geometry
    data/lithology_area_km2_raw.csv   area of each mapped rock unit per catchment
    data/chitral_boundary.gpkg        district boundary (figures)
    data/dem_125m.tif                 DEM resampled to 125 m (figure hillshade)
    results/data_audit.json           checks of the stored attributes and inputs
"""

from __future__ import annotations

import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.interpolate import griddata

from fanscreen import config
from fanscreen.config import CRS, DATA, RESULTS, raw
from fanscreen.features import (
    ALLUVIAL_FAN,
    CRYSTALLINE_UNITS,
    DEBRIS_FANS,
    NON_BEDROCK_UNITS,
    RECODED_FAN_CODES,
    SOURCE_CLASSES,
)

warnings.filterwarnings("ignore")

STORED_FIELDS = [
    "HydroID",
    "GridID",
    "area_km",
    "WS_length",
    "WS_width",
    "Relief_KM",
    "Melton_rat",
    "Relief_rat",
    "Str_Length",
    "Glacr_area",
    "Fan_Area",
    "D_F_Type",
    "streams_na",
    "Schools",
    "Houses",
    "hospitals",
    "road_lengt",
    "geometry",
]
GRID_RES = 100.0  # analysis grid for the coarse layers (m)
S2_YEARS = range(2016, 2026)
NDSI_SNOW = 0.4  # NDSI above which a pixel counts as snow or ice
DEM_FIGURE_FACTOR = 10  # 12.5 m DEM -> 125 m for the figure hillshade


def load_catchments() -> gpd.GeoDataFrame:
    """The 169 catchments of the 2021 hazard database, with their legacy rule label."""
    parts = [
        gpd.read_file(raw(layer))[STORED_FIELDS].assign(rule_label=label)
        for layer, label in [("debris_catchments", "debris_flow"), ("flood_catchments", "flood")]
    ]
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs).to_crs(CRS)
    g["geometry"] = g.geometry.make_valid().buffer(0)
    g["cid"] = np.arange(1, len(g) + 1)
    return g


def add_terrain(g: gpd.GeoDataFrame, audit: dict) -> None:
    """Elevation, slope and outlet of each catchment from the 12.5 m DEM."""
    with rasterio.open(raw("dem")) as src:
        dem = src.read(1).astype("float32")
        transform, nodata, res, shape = src.transform, src.nodata, src.res[0], src.shape
        audit["dem"] = dict(
            crs=str(src.crs),
            res_m=res,
            nodata=None if nodata is None else float(nodata),
            shape=list(shape),
            file="ChitralDEM/Chitral.tif",
        )
        # agreement with the district-clipped DEM used in earlier susceptibility work
        with rasterio.open(raw("dem_clipped")) as other:
            clipped = np.full(shape, np.nan, "float32")
            reproject(
                rasterio.band(other, 1),
                clipped,
                dst_transform=transform,
                dst_crs=src.crs,
                src_nodata=other.nodata,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
    if nodata is not None:
        dem[dem == nodata] = np.nan
    dem[dem <= 0] = np.nan
    both = np.isfinite(dem) & np.isfinite(clipped)
    audit["dem"]["vs_clipped_median_abs_diff_m"] = float(np.median(np.abs(dem[both][::97] - clipped[both][::97])))
    del clipped, both

    labels = rasterize(zip(g.geometry, g.cid), out_shape=shape, transform=transform, fill=0, dtype="int32")
    filled = np.where(np.isnan(dem), np.nanmean(dem), dem)
    gy, gx = np.gradient(filled, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")
    del gx, gy, filled
    ok = (labels > 0) & ~np.isnan(dem)
    L, Z, S = labels[ok], dem[ok], slope[ok]
    n = len(g) + 1
    count = np.bincount(L, minlength=n)
    mean = np.bincount(L, Z, n) / np.maximum(count, 1)
    std = np.sqrt(np.maximum(np.bincount(L, Z**2, n) / np.maximum(count, 1) - mean**2, 0))
    zmin, zmax = np.full(n, np.inf), np.full(n, -np.inf)
    np.minimum.at(zmin, L, Z)
    np.maximum.at(zmax, L, Z)

    # outlet = lowest DEM cell of each catchment
    rows, cols = np.nonzero(ok)
    lowest = np.lexsort((Z, L))
    _, first = np.unique(L[lowest], return_index=True)
    outlet = (
        pd.DataFrame({"cid": L[lowest][first], "r": rows[lowest][first], "c": cols[lowest][first]})
        .set_index("cid")
        .reindex(g.cid)
    )
    g["outlet_x"] = transform.c + (outlet.c.values + 0.5) * res
    g["outlet_y"] = transform.f - (outlet.r.values + 0.5) * res

    g["cov_dem"] = (count * res**2 / 1e6)[g.cid] / (g.geometry.area.values / 1e6)
    g["elev_min"], g["elev_max"], g["elev_mean"], g["elev_std"] = zmin[g.cid], zmax[g.cid], mean[g.cid], std[g.cid]
    g["slope_mean"] = (np.bincount(L, S, n) / np.maximum(count, 1))[g.cid]
    g["slope_gt30"] = (np.bincount(L, (S > 30).astype(float), n) / np.maximum(count, 1))[g.cid]
    g["slope_gt45"] = (np.bincount(L, (S > 45).astype(float), n) / np.maximum(count, 1))[g.cid]


def add_morphometry(g: gpd.GeoDataFrame, audit: dict) -> None:
    """Basin shape indices and drainage density; comparison with the stored 2021 values."""
    g["area_km2"] = g.geometry.area / 1e6
    g["relief_km"] = (g.elev_max - g.elev_min) / 1000
    g["melton"] = g.relief_km / np.sqrt(g.area_km2)
    g["length_km"] = g.WS_length / 1000
    g["relief_ratio"] = g.relief_km / g.length_km
    g["hypsometric"] = (g.elev_mean - g.elev_min) / (g.elev_max - g.elev_min)
    g["elongation"] = np.sqrt(4 * g.area_km2 / np.pi) / g.length_km
    g["circularity"] = 4 * np.pi * g.area_km2 / (g.geometry.length / 1000) ** 2
    g["glacier_frac_raw"] = g.Glacr_area / 1e6 / g.area_km2

    streams = gpd.read_file(raw("streams")).to_crs(CRS)
    hits = gpd.sjoin(streams[["geometry"]], g[["cid", "geometry"]], predicate="intersects").reset_index()
    pieces = gpd.GeoSeries(hits.geometry.values, crs=CRS).intersection(
        gpd.GeoSeries(g.set_index("cid").geometry.reindex(hits.cid).values, crs=CRS)
    )
    g["stream_km"] = (
        pd.Series(pieces.length.values / 1000).groupby(hits.cid.values).sum().reindex(g.cid).fillna(0).values
    )
    g["drainage_density"] = g.stream_km / g.area_km2

    audit["streams_n"] = len(streams)
    audit["stored_stream_vs_recomputed_spearman"] = float(g.Str_Length.corr(g.stream_km, method="spearman"))
    audit["recomputed_vs_stored"] = dict(
        melton_spearman=float(g.melton.corr(g.Melton_rat, method="spearman")),
        melton_median_abs_diff=float((g.melton - g.Melton_rat).abs().median()),
        melton_max_abs_diff=float((g.melton - g.Melton_rat).abs().max()),
        relief_km_median_abs_diff=float((g.relief_km - g.pop("Relief_KM")).abs().median()),
        area_max_abs_diff_km2=float((g.area_km2 - g.area_km).abs().max()),
        min_dem_coverage=float(g.cov_dem.min()),
    )


class Grid:
    """A 100 m grid over the catchments, used for zonal means of the coarse layers."""

    def __init__(self, g: gpd.GeoDataFrame):
        x0, y0, x1, y1 = g.total_bounds
        self.width, self.height = int((x1 - x0) // GRID_RES) + 2, int((y1 - y0) // GRID_RES) + 2
        self.transform = from_origin(x0 - GRID_RES, y1 + GRID_RES, GRID_RES, GRID_RES)
        self.labels = rasterize(
            zip(g.geometry, g.cid), out_shape=(self.height, self.width), transform=self.transform, fill=0, dtype="int32"
        )
        cols, rows = np.meshgrid(np.arange(self.width), np.arange(self.height))
        self.x = self.transform.c + (cols + 0.5) * GRID_RES
        self.y = self.transform.f - (rows + 0.5) * GRID_RES
        self.cids, self.n = g.cid, len(g) + 1

    def zonal_mean(self, a: np.ndarray) -> np.ndarray:
        ok = (self.labels > 0) & np.isfinite(a)
        count = np.bincount(self.labels[ok], minlength=self.n)
        return (np.bincount(self.labels[ok], a[ok], self.n) / np.where(count > 0, count, np.nan))[self.cids]

    def resample(self, src, band: int) -> np.ndarray:
        a = np.full((self.height, self.width), np.nan, "float32")
        reproject(
            rasterio.band(src, band),
            a,
            dst_transform=self.transform,
            dst_crs=CRS,
            dst_nodata=np.nan,
            src_nodata=src.nodata,
            resampling=Resampling.average,
        )
        a[a <= 0] = np.nan
        return a


def add_rainfall(g: gpd.GeoDataFrame, grid: Grid, audit: dict) -> None:
    """Mean annual TRMM rainfall 2010-2019, interpolated linearly between the grid-cell centres."""
    pts = gpd.read_file(raw("trmm")).to_crs(CRS)
    xy = np.c_[pts.geometry.x, pts.geometry.y]
    linear = griddata(xy, pts.grid_code.values, (grid.x, grid.y), method="linear")
    nearest = griddata(xy, pts.grid_code.values, (grid.x, grid.y), method="nearest")
    g["rain_trmm"] = grid.zonal_mean(np.where(np.isnan(linear), nearest, linear).astype("float32"))
    audit["trmm"] = dict(
        n_points=len(pts),
        min=float(pts.grid_code.min()),
        max=float(pts.grid_code.max()),
        spacing_km=float(np.median(pts.distance(pts.geometry.iloc[0]).sort_values().values[1:4]) / 1000),
    )


def add_sentinel2(g: gpd.GeoDataFrame, grid: Grid, audit: dict) -> None:
    """NDVI and snow/ice fraction from the median of the annual post-monsoon composites."""
    ndvi_years, ndsi_years, medians = [], [], {}
    for year in S2_YEARS:
        path = config.sentinel2(year)
        if not path.exists():
            print("Sentinel-2", year, "not found, skipped", flush=True)
            continue
        with rasterio.open(path) as src:
            band = {str(name): i + 1 for i, name in enumerate(src.descriptions)}
            if "s2" not in audit:
                audit["s2"] = dict(
                    bands=list(map(str, src.descriptions)),
                    crs=str(src.crs),
                    res=src.res[0],
                    dtype=src.dtypes[0],
                    nodata=src.nodata,
                )
            b = {name: grid.resample(src, band[name]) for name in ["B03", "B04", "B08", "B11"]}
        ndvi_years.append((b["B08"] - b["B04"]) / (b["B08"] + b["B04"]))
        ndsi_years.append((b["B03"] - b["B11"]) / (b["B03"] + b["B11"]))
        medians[year] = {name: float(np.nanmedian(a[grid.labels > 0])) for name, a in b.items()}
        print("Sentinel-2", year, medians[year], flush=True)
    ndvi = np.nanmedian(np.stack(ndvi_years), axis=0)
    ndsi = np.nanmedian(np.stack(ndsi_years), axis=0)
    g["ndvi"] = grid.zonal_mean(ndvi)
    g["snow_ice_frac"] = grid.zonal_mean(np.where(np.isfinite(ndsi), (ndsi > NDSI_SNOW).astype("float32"), np.nan))
    g["cov_s2"] = [np.isfinite(ndvi[grid.labels == c]).mean() for c in g.cid]
    audit["s2_median_dn_by_year"] = medians
    audit["s2_years_used"] = len(ndvi_years)
    audit["s2_min_coverage"] = float(g.cov_s2.min())
    audit["s2_coverage_lt_90pct"] = int((g.cov_s2 < 0.9).sum())
    audit["stored_glacier_vs_ndsi_spearman"] = float(g.glacier_frac_raw.corr(g.snow_ice_frac, method="spearman"))
    audit["stored_glacier_gt_100pct"] = int((g.glacier_frac_raw > 1).sum())


def add_geology(g: gpd.GeoDataFrame, audit: dict) -> None:
    """Rock-unit areas, crystalline share of the mapped bedrock, and fault density."""
    geo = gpd.read_file(raw("geology")).to_crs(CRS)
    geo["geometry"] = geo.geometry.make_valid().buffer(0)
    geo["lith_raw"] = geo.Short_Desp.fillna("unknown").astype(str).str.strip()
    audit["geology_units"] = {
        k: round(float(v), 1)
        for k, v in geo.assign(a=geo.area / 1e6).groupby("lith_raw").a.sum().sort_values(ascending=False).items()
    }
    x = gpd.overlay(geo[["lith_raw", "geometry"]], g[["cid", "geometry"]], how="intersection", keep_geom_type=True)
    x["a"] = x.area / 1e6
    units = x.pivot_table(index="cid", columns="lith_raw", values="a", aggfunc="sum", fill_value=0)
    units.to_csv(DATA / "lithology_area_km2_raw.csv")
    audit["geology_coverage_min"] = float((units.sum(axis=1).reindex(g.cid).values / g.area_km2.values).min())

    # crystalline share of the mapped bedrock; ice, snow and alluvium are excluded
    bedrock = geo.copy()
    bedrock["Formation"] = bedrock.Formation.fillna("Unknown")
    bedrock = bedrock[~bedrock.Formation.isin(NON_BEDROCK_UNITS)]
    bedrock["cryst"] = bedrock.Formation.str.contains("|".join(CRYSTALLINE_UNITS), case=False)
    x = gpd.overlay(bedrock[["cryst", "geometry"]], g[["cid", "geometry"]], how="intersection", keep_geom_type=True)
    x["a"] = x.area
    area = x.pivot_table(index="cid", columns="cryst", values="a", aggfunc="sum", fill_value=0)
    share = (area[True] / area.sum(axis=1)).reindex(g.cid)
    g["crystalline_frac"] = share.fillna(share.median()).values
    g["crystalline_missing"] = share.isna().values

    faults = gpd.read_file(raw("faults")).to_crs(CRS)
    xf = gpd.overlay(faults[["geometry"]], g[["cid", "geometry"]], how="intersection", keep_geom_type=True)
    g["fault_density"] = (
        xf.assign(l=xf.length / 1000).groupby("cid").l.sum().reindex(g.cid).fillna(0).values / g.area_km2.values
    )
    audit["n_faults"] = len(faults)


def add_landslide_sources(g: gpd.GeoDataFrame, audit: dict) -> None:
    """Share of each catchment covered by mapped landslide source polygons."""
    inv = gpd.read_file(raw("inventory")).to_crs(CRS)
    inv["code"] = inv.Name.astype(str).str.upper().str.replace(" ", "")
    inv["geometry"] = inv.geometry.make_valid().buffer(0)
    audit["inventory_classes"] = inv.code.value_counts().to_dict()
    sources = inv[inv.code.isin(SOURCE_CLASSES)]
    x = gpd.overlay(sources[["geometry"]], g[["cid", "geometry"]], how="intersection", keep_geom_type=True)
    g["slide_source_frac"] = (
        x.assign(a=x.area / 1e6).groupby("cid").a.sum().reindex(g.cid).fillna(0).values / g.area_km2.values
    )


def add_target(g: gpd.GeoDataFrame, audit: dict) -> None:
    """Fan type at the outlet, with the DFSAA/DFSA debris fans recoded, and the binary target."""
    fans = gpd.read_file(raw("fans"))
    g["fan_type_db"] = g.D_F_Type.fillna("none")
    g["fan_code_recoded"] = (g.rule_label == "debris_flow") & g.HydroID.isin(
        fans[fans.Name.isin(RECODED_FAN_CODES)].HydroID
    )
    g["fan_type"] = np.where(g.fan_code_recoded, "Debris Fan Active", g.fan_type_db)
    g["has_fan"] = g.D_F_Type.notna().astype(int)
    g["y_debris"] = np.where(g.fan_type.isin(DEBRIS_FANS), 1.0, np.where(g.fan_type == ALLUVIAL_FAN, 0.0, np.nan))
    audit["fan_type_counts"] = g.fan_type.value_counts().to_dict()
    audit["fan_type_counts_db"] = g.fan_type_db.value_counts().to_dict()
    audit["n_fan_code_recoded"] = int(g.fan_code_recoded.sum())
    audit["crosstab_rule_vs_fan"] = pd.crosstab(g.rule_label, g.fan_type).to_dict()
    audit["n_labelled"] = int(g.y_debris.notna().sum())
    audit["n_debris"] = int((g.y_debris == 1).sum())


def check_fan_positions(g: gpd.GeoDataFrame, audit: dict) -> None:
    """Does each digitised fan sit at the outlet of the catchment it was joined to?"""
    fan = gpd.read_file(raw("fans")).to_crs(CRS)
    fan = fan[fan.Name.notna()].copy()
    fan["geometry"] = fan.geometry.make_valid()
    sub = g[g.rule_label == "debris_flow"].set_index("HydroID")
    outlets = gpd.GeoSeries(
        gpd.points_from_xy(sub.outlet_x.reindex(fan.HydroID), sub.outlet_y.reindex(fan.HydroID)), crs=CRS
    )
    dist = gpd.GeoSeries(fan.geometry.values, crs=CRS).distance(outlets).values
    touching = (
        gpd.sjoin(
            gpd.GeoDataFrame(geometry=fan.geometry.buffer(1).values, crs=CRS).reset_index(),
            g[["cid", "geometry"]],
            predicate="intersects",
        )
        .groupby("index")
        .cid.nunique()
    )
    audit["fan_outlet_check"] = dict(
        n_fans=len(fan),
        median_dist_m=float(np.median(dist)),
        within_250m=int((dist <= 250).sum()),
        within_1km=int((dist <= 1000).sum()),
        max_dist_m=float(np.max(dist)),
        touching_one=int((touching == 1).sum()),
        touching_two_or_more=int((touching >= 2).sum()),
    )


def write_figure_layers() -> None:
    """District boundary and a 125 m DEM, so that the map figures need no raw data."""
    gpd.read_file(raw("boundary")).to_crs(CRS)[["geometry"]].to_file(DATA / "chitral_boundary.gpkg", driver="GPKG")
    with rasterio.open(raw("dem")) as src:
        f = DEM_FIGURE_FACTOR
        h, w = src.height // f, src.width // f
        z = src.read(1, out_shape=(h, w)).astype("float32")
        z[(z == src.nodata) | (z <= 0)] = np.nan
        transform = src.transform * Affine.scale(src.width / w, src.height / h)
        profile = dict(
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype="float32",
            crs=src.crs,
            transform=transform,
            nodata=np.nan,
            compress="deflate",
            predictor=3,
            tiled=True,
        )
    with rasterio.open(DATA / "dem_125m.tif", "w", **profile) as dst:
        dst.write(z, 1)


def main() -> None:
    audit: dict = {}
    g = load_catchments()
    audit["n_catchments"] = len(g)
    add_terrain(g, audit)
    add_morphometry(g, audit)
    grid = Grid(g)
    add_rainfall(g, grid, audit)
    add_sentinel2(g, grid, audit)
    add_geology(g, audit)
    add_landslide_sources(g, audit)
    add_target(g, audit)
    audit["total_area_km2"] = float(g.area_km2.sum())
    audit["district_km2"] = float(gpd.read_file(raw("boundary")).to_crs(CRS).area.sum() / 1e6)
    centroid = g.geometry.centroid
    g["cx"], g["cy"] = centroid.x, centroid.y
    check_fan_positions(g, audit)
    g["melton_stored_abs_diff"] = (g.melton - g.Melton_rat).abs()
    audit["stored_melton_diff_gt_0p1"] = int((g.melton_stored_abs_diff > 0.1).sum())
    audit["rule_label_melton_split"] = {
        k: dict(
            stored_min=float(v.Melton_rat.min()),
            stored_max=float(v.Melton_rat.max()),
            dem_min=float(v.melton.min()),
            dem_max=float(v.melton.max()),
        )
        for k, v in g.groupby("rule_label")
    }
    missing = g.drop(columns="geometry").isna().sum()
    audit["missing"] = {k: int(v) for k, v in missing[missing > 0].items()}
    audit["crystalline_missing"] = int(g.crystalline_missing.sum())

    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    g.to_file(DATA / "catchments.gpkg", driver="GPKG")
    g.drop(columns="geometry").to_csv(DATA / "catchments.csv", index=False)
    write_figure_layers()
    with open(RESULTS / "data_audit.json", "w") as f:
        json.dump(audit, f, indent=1, default=str)
    print(f"{len(g)} catchments, {audit['n_labelled']} with a fan type ({audit['n_debris']} debris-dominated)")


if __name__ == "__main__":
    main()
