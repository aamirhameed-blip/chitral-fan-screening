# Data

Analysis-ready data for the 169 catchments of the Chitral district, Hindu Kush, Pakistan. Licence: [CC BY 4.0](LICENSE).
All files are written by `scripts/01_build_catchments.py`.

| File | Content |
|---|---|
| `catchments.gpkg` | One polygon per catchment with all attributes below. EPSG:32643 (WGS 84 / UTM zone 43N). |
| `catchments.csv` | The same table without geometry. |
| `lithology_area_km2_raw.csv` | Area (km²) of each mapped rock unit in each catchment (`cid` by unit). |
| `chitral_boundary.gpkg` | District boundary, used in the maps. |
| `dem_125m.tif` | The 12.5 m ALOS PALSAR DEM resampled to 125 m, used for the map hillshade only. |

## Columns of `catchments.gpkg` / `catchments.csv`

### Identifiers and target

| Column | Description |
|---|---|
| `cid` | Catchment id used throughout the analysis (1-169). |
| `HydroID`, `GridID` | Identifiers in the 2021 hazard database. |
| `rule_label` | Legacy 2021 label, `debris_flow` or `flood`. It was derived from the Melton ratio and basin length, so it is **not** an independent reference. |
| `fan_type_db` | Fan type at the outlet as stored in the database (`none` if no fan type is recorded). |
| `fan_code_recoded` | True where the fan-map code is DFSAA or DFSA (a debris fan) while the stored type is alluvial. |
| `fan_type` | Fan type used in the analysis, with the recoding applied. |
| `has_fan` | 1 if a fan type is recorded. |
| `y_debris` | Target: 1 = debris-dominated fan (debris fan, glacial debris fan, debris cone), 0 = alluvial fan, empty = no fan type (not used in fitting). |

### Morphometry (recomputed from the 12.5 m DEM)

| Column | Unit | Description |
|---|---|---|
| `area_km2` | km² | Catchment area from the polygon. |
| `elev_min`, `elev_max`, `elev_mean`, `elev_std` | m | Elevation statistics. |
| `relief_km` | km | `elev_max - elev_min`. |
| `melton` | – | Melton ratio, `relief_km / sqrt(area_km2)`. |
| `length_km` | km | Basin length (`WS_length` / 1000). |
| `relief_ratio` | – | `relief_km / length_km`. |
| `hypsometric` | – | Hypsometric integral, `(elev_mean - elev_min) / (elev_max - elev_min)`. |
| `elongation` | – | Elongation ratio, diameter of the circle of equal area / basin length. |
| `circularity` | – | `4π area / perimeter²`. |
| `slope_mean` | degrees | Mean slope. |
| `slope_gt30`, `slope_gt45` | fraction | Share of cells steeper than 30° and 45°. |
| `stream_km` | km | Length of mapped streams inside the catchment. |
| `drainage_density` | km⁻¹ | `stream_km / area_km2`. |
| `outlet_x`, `outlet_y` | m | Lowest DEM cell of the catchment (fan apex). |
| `cx`, `cy` | m | Polygon centroid, used to form the spatial blocks. |
| `cov_dem` | fraction | Share of the polygon covered by valid DEM cells. |

### Climate, cryosphere, geology and land surface

| Column | Unit | Description |
|---|---|---|
| `rain_trmm` | as stored | Mean of the TRMM 2010-2019 rainfall field, interpolated linearly between the grid-cell values of the source layer; used as a relative field. |
| `snow_ice_frac` | fraction | Share of the catchment with median NDSI > 0.4 in the 2016-2025 post-monsoon Sentinel-2 composites. |
| `ndvi` | – | Median NDVI over the same composites. |
| `cov_s2` | fraction | Share of the catchment with valid Sentinel-2 data. |
| `crystalline_frac` | fraction | Share of the mapped bedrock that is crystalline (granite, granodiorite, gneiss, pegmatite, amphibolite); ice, snow and alluvium excluded. |
| `crystalline_missing` | bool | True where no bedrock is mapped; `crystalline_frac` is then the median. |
| `fault_density` | km⁻¹ | Mapped fault length per catchment area. |
| `slide_source_frac` | fraction | Share covered by mapped landslide source polygons (fan classes excluded). |

### Attributes stored in the 2021 hazard database (kept for the audit)

| Column | Description |
|---|---|
| `area_km` | Area, km². |
| `WS_length`, `WS_width` | Basin length and width, m. |
| `Melton_rat`, `Relief_rat` | Stored Melton ratio and relief ratio. |
| `melton_stored_abs_diff` | Absolute difference between `melton` and `Melton_rat`. |
| `Str_Length` | Stored stream length. |
| `Glacr_area` | Stored glacier area, m². `glacier_frac_raw` = `Glacr_area` / catchment area. |
| `Fan_Area` | Stored fan area, km². |
| `D_F_Type` | Stored fan type. |
| `streams_na` | Stream regime (ephemeral or perennial). |
| `Houses`, `Schools`, `hospitals`, `road_lengt` | Stored exposure counts and road length per catchment. |
