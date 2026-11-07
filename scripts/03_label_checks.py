"""Step 03. Independent checks of the fan labels and of stored attributes.

Requires CHITRAL_RAW_DATA (see README).

1. Gradient of the mapped fan polygons, by fan-map code: are the DFSAA/DFSA fans,
   stored as alluvial, as steep as the other debris fans?
2. Overlap of the DFSAA/DFSA polygons with the landslide inventories.
3. A DEM proxy of fan gradient for all 104 labelled outlets (mapped polygons exist
   only for the debris catchments): the ground just below the outlet, over a
   half-disc with the stored fan area. The proxy is validated against the mapped
   polygons and then compared between alluvial and debris-dominated fans.
4. Stored basin length against the polygon, and stored house counts against
   building footprints.

Outputs (results/)
    label_checks.json        all summary values
    fan_gradient_proxy.csv   proxy and polygon gradient per labelled catchment
"""

from __future__ import annotations

import json
import re
import warnings
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.windows import from_bounds
from scipy.stats import mannwhitneyu, spearmanr
from shapely.geometry import Polygon
from sklearn.metrics import roc_auc_score

from fanscreen.config import DATA, RESULTS, raw
from fanscreen.features import RECODED_FAN_CODES

warnings.filterwarnings("ignore")

FAN_INVENTORY_CODES = ["AF", "DF", "DFA", "AF,DF", "AF+DF"]
STEEP_ALLUVIAL_DEG = 6.0


def polygon_gradient(dem, geom) -> float:
    """Fan gradient (degrees): 2nd-98th percentile elevation range over the long axis of the polygon."""
    a, _ = rio_mask(dem, [geom], crop=True, filled=True, nodata=-9999)
    a = a[0].astype("float32")
    a[(a == -9999) | (a == dem.nodata) | (a <= 0)] = np.nan
    xx, yy = geom.minimum_rotated_rectangle.exterior.coords.xy
    length = max(np.hypot(xx[1] - xx[0], yy[1] - yy[0]), np.hypot(xx[2] - xx[1], yy[2] - yy[1]))
    return float(np.degrees(np.arctan((np.nanpercentile(a, 98) - np.nanpercentile(a, 2)) / length)))


def dem_window(dem, x: float, y: float, radius: float):
    """DEM values and cell-centre coordinates in a square window around (x, y)."""
    w = from_bounds(x - radius, y - radius, x + radius, y + radius, dem.transform)
    z = dem.read(1, window=w, boundless=True, fill_value=dem.nodata).astype("float32")
    z[(z == dem.nodata) | (z <= 0)] = np.nan
    t = dem.window_transform(w)
    cx = t.c + (np.arange(z.shape[1]) + 0.5) * t.a
    cy = t.f + (np.arange(z.shape[0]) + 0.5) * t.e
    return z, *np.meshgrid(cx, cy)


def proxy_gradient(dem, c, crs) -> tuple[float, float]:
    """Gradient (degrees) of the ground below the outlet, outside the catchment, within a half-disc of the stored fan area."""
    res = dem.res[0]
    radius = max(np.sqrt(2 * c.Fan_Area * 1e6 / np.pi), 3 * res)
    z, X, Y = dem_window(dem, c.outlet_x, c.outlet_y, radius + 2 * res)
    dist = np.hypot(X - c.outlet_x, Y - c.outlet_y)
    apex = np.nanmin(z[dist <= 2 * res])
    inside = gpd.GeoSeries(gpd.points_from_xy(X.ravel(), Y.ravel()), crs=crs).within(c.geometry).values.reshape(z.shape)
    below = (dist <= radius) & ~inside & (z <= apex + 1) & np.isfinite(z)
    grad = np.degrees(np.arctan((apex - np.nanpercentile(z[below], 5)) / radius)) if below.sum() >= 5 else np.nan
    return grad, radius


def inventory_polygons_from_kmz(path, crs) -> gpd.GeoDataFrame:
    """Polygons of the Google Earth inventory, with their class code."""
    kml = zipfile.ZipFile(path).read("doc.kml").decode("utf8", "ignore")
    found = []
    for placemark in re.findall(r"<Placemark.*?</Placemark>", kml, re.S):
        name, coords = re.search(r"<name>([^<]*)", placemark), re.search(r"<coordinates>\s*([^<]*)", placemark)
        if name and coords and "Polygon" in placemark:
            pts = [tuple(map(float, q.split(",")[:2])) for q in coords.group(1).split()]
            if len(pts) >= 4:
                found.append((name.group(1).strip().upper().replace(" ", ""), Polygon(pts).buffer(0)))
    kz = gpd.GeoDataFrame({"code": [c for c, _ in found]}, geometry=[p for _, p in found], crs=4326).to_crs(crs)
    return kz[~kz.geometry.apply(lambda s: s.wkb_hex).duplicated()]


def farthest_boundary_km(row) -> float:
    pts = np.vstack([np.asarray(p.exterior.coords) for p in getattr(row.geometry, "geoms", [row.geometry])])
    return float(np.hypot(pts[:, 0] - row.outlet_x, pts[:, 1] - row.outlet_y).max() / 1000)


def main() -> None:
    g = gpd.read_file(DATA / "catchments.gpkg")
    labelled = g[g.y_debris.notna()].copy()
    R: dict = {}

    fan = gpd.read_file(raw("fans")).to_crs(g.crs)
    fan = fan[fan.Name.notna()].copy()
    fan["geometry"] = fan.geometry.make_valid()

    with rasterio.open(raw("dem")) as dem:
        assert str(dem.crs) == str(g.crs), (dem.crs, g.crs)

        # 1. Mapped polygon gradients by code
        fan["grad"] = [polygon_gradient(dem, geom) for geom in fan.geometry]
        regular = fan[fan.Name == "DFAA"].grad
        coded = fan[fan.Name.isin(RECODED_FAN_CODES)].grad
        u, p = mannwhitneyu(regular, coded)
        R["fan_gradient"] = dict(
            dfaa_median=float(regular.median()),
            dfsaa_median=float(coded.median()),
            n_dfaa=int(len(regular)),
            n_dfsaa=int(len(coded)),
            auc=float(u / (len(regular) * len(coded))),
            p=float(p),
            by_code=fan.groupby("Name").grad.median().round(2).to_dict(),
        )
        R["n_code_polygons"] = int(fan.Name.isin(RECODED_FAN_CODES).sum())

        # 3. DEM proxy for every labelled outlet
        rows = []
        for _, c in labelled.iterrows():
            grad, radius = proxy_gradient(dem, c, g.crs)
            rows.append(
                dict(
                    cid=c.cid,
                    HydroID=c.HydroID,
                    y=int(c.y_debris),
                    recoded=bool(c.fan_code_recoded),
                    fan_km2=c.Fan_Area,
                    rad_m=radius,
                    proxy_grad=grad,
                    melton=c.melton,
                )
            )
    P = pd.DataFrame(rows).merge(
        fan[["HydroID", "Name", "grad"]].rename(columns={"grad": "poly_grad"}).drop_duplicates("HydroID"),
        on="HydroID",
        how="left",
    )

    # 2. Overlap of the recoded fans with the landslide inventories
    inv = gpd.read_file(raw("inventory")).to_crs(g.crs)
    inv["code"] = inv.Name.astype(str).str.upper().str.replace(" ", "")
    inv["geometry"] = inv.geometry.make_valid().buffer(0)
    R["fan_overlap"] = {}
    for name, polygons in [("merged", inv), ("kmz", inventory_polygons_from_kmz(raw("inventory_kmz"), g.crs))]:
        polygons = polygons[polygons.code.isin(FAN_INVENTORY_CODES)]
        x = gpd.overlay(
            fan[fan.Name.isin(RECODED_FAN_CODES)][["geometry"]].reset_index(),
            polygons[["code", "geometry"]],
            how="intersection",
            keep_geom_type=True,
        )
        x["a"] = x.area
        dominant = x.groupby(["index", "code"]).a.sum().reset_index().sort_values("a").groupby("index").tail(1)
        R["fan_overlap"][name] = dominant.code.value_counts().to_dict()

    # 3 (continued). Validate the proxy and compare the fan types
    valid = P.dropna(subset=["poly_grad", "proxy_grad"])
    rho, p_rho = spearmanr(valid.poly_grad, valid.proxy_grad)
    alluvial, debris = P[P.y == 0].proxy_grad.dropna(), P[P.y == 1].proxy_grad.dropna()
    not_recoded = P[(P.y == 1) & ~P.recoded].proxy_grad.dropna()
    recoded = P[P.recoded].proxy_grad.dropna()
    u, p = mannwhitneyu(debris, alluvial)
    u2, p2 = mannwhitneyu(not_recoded, alluvial)
    u3, p3 = mannwhitneyu(recoded, alluvial)
    steep = (P.y == 0) & (P.proxy_grad > STEEP_ALLUVIAL_DEG)

    def describe(s: pd.Series) -> dict:
        return dict(
            n=int(len(s)),
            median=round(s.median(), 2),
            q25=round(s.quantile(0.25), 2),
            q75=round(s.quantile(0.75), 2),
            share_lt4=round((s < 4).mean(), 3),
        )

    R["proxy"] = dict(
        n_valid=int(len(valid)),
        vs_polygon_spearman=round(rho, 3),
        p_rho=float(p_rho),
        median_bias_deg=round(float((valid.proxy_grad - valid.poly_grad).median()), 2),
        alluvial=dict(**describe(alluvial), share_lt6=round((alluvial < 6).mean(), 3)),
        debris=describe(debris),
        debris_not_recoded_median=round(not_recoded.median(), 2),
        auc_debris_gt_alluvial=round(u / (len(debris) * len(alluvial)), 3),
        p=float(p),
        auc_unrecoded_vs_alluvial=round(u2 / (len(not_recoded) * len(alluvial)), 3),
        p2=float(p2),
        recoded=dict(
            n=int(P.recoded.sum()),
            median=round(recoded.median(), 2),
            auc_vs_alluvial=round(u3 / (P.recoded.sum() * len(alluvial)), 3),
            p=float(p3),
        ),
        alluvial_steeper_than_debris_median=int((alluvial > debris.median()).sum()),
        fan_km2_median=dict(
            alluvial=round(P[P.y == 0].fan_km2.median(), 3), debris=round(P[P.y == 1].fan_km2.median(), 3)
        ),
        rho_proxy_melton_alluvial=round(spearmanr(P[P.y == 0].proxy_grad, P[P.y == 0].melton, nan_policy="omit")[0], 3),
        steep_alluvial=dict(
            n=int(steep.sum()),
            melton_auc_all=round(roc_auc_score(P.y, P.melton), 3),
            melton_auc_steep_as_debris=round(roc_auc_score(P.y.where(~steep, 1), P.melton), 3),
        ),
    )

    # 4. Stored basin length and house counts
    length_poly = labelled.apply(farthest_boundary_km, axis=1)
    R["length_check"] = dict(
        n=int(len(labelled)),
        spearman=round(spearmanr(labelled.length_km, length_poly)[0], 3),
        median_abs_diff_km=round(float((labelled.length_km - length_poly).abs().median()), 2),
        auc_stored=round(roc_auc_score(labelled.y_debris, labelled.length_km), 3),
        auc_polygon=round(roc_auc_score(labelled.y_debris, length_poly), 3),
    )
    houses = gpd.read_file(raw("house_footprints")).to_crs(g.crs)
    houses["geometry"] = houses.geometry.centroid
    counts = (
        gpd.sjoin(houses, g[["cid", "geometry"]], predicate="within").groupby("cid").size().reindex(g.cid).fillna(0)
    )
    R["house_check"] = dict(
        n_footprints=int(len(houses)),
        spearman=float(pd.Series(counts.values).corr(pd.Series(g.Houses.values), method="spearman")),
        stored_total=int(g.Houses.sum()),
        footprint_total=int(counts.sum()),
    )

    P.round(3).to_csv(RESULTS / "fan_gradient_proxy.csv", index=False)
    with open(RESULTS / "label_checks.json", "w") as f:
        json.dump(R, f, indent=1, default=float)
    print(json.dumps(R, indent=1, default=float))


if __name__ == "__main__":
    main()
