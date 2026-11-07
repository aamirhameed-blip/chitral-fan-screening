"""Project paths and analysis settings.

The analysis-ready data in ``data/`` are versioned with the code, so the benchmark
(step 02) and the figures (step 04) run without any other input. The raw GIS layers
are needed only to rebuild the catchment table (step 01) and for the label checks
(step 03). Point the ``CHITRAL_RAW_DATA`` environment variable at the folder that
holds them, and ``CHITRAL_S2_DIR`` at the Sentinel-2 composites from step 00.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

CRS = "EPSG:32643"  # WGS 84 / UTM zone 43N

# Validation settings used in the paper
BLOCK_KM = 30  # side of the square spatial blocks
N_FOLDS = 5  # folds per partition
N_REPEATS = 30  # random block-to-fold partitions
N_BOOT = 2000  # block-bootstrap resamples
N_PERM = 500  # label permutations for the null distribution
N_JOBS = int(os.environ.get("FANSCREEN_JOBS", "8"))

# Raw layers, relative to CHITRAL_RAW_DATA
RAW_LAYERS = {
    "debris_catchments": "Flood_flow_finalrisk/Flood_flow_finalrisk/debris_flows.shp",
    "flood_catchments": "Flood_flow_finalrisk/Flood_flow_finalrisk/flash_flood.shp",
    "dem": "ChitralDEM/Chitral.tif",
    "dem_clipped": "Other_Data/DEM.tif",
    "streams": "Other_Data/Streams.shp",
    "trmm": "TRMM_Rainafall_2010_2019/TRMM_2010_19.shp",
    "geology": "Other_Data/Geology.shp",
    "faults": "Other_Data/Faults.shp",
    "boundary": "Other_Data/Chitral_BD.shp",
    "inventory": "Chitral_Inventries_Updated/All_LS_Merged_UTM.shp",
    "inventory_kmz": "Other_Data/All Invenotries, Updated.kmz",
    "fans": "Debris Flow Analysis2/Fan_2_Join.shp",
    "house_footprints": "Chitral_Houses_Footprint_Completed/House_footprints_Chitral.shp",
}


def _env_dir(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Set the {name} environment variable first (see README, 'Rebuilding from the raw data').")
    path = Path(value)
    if not path.is_dir():
        raise SystemExit(f"{name} points to a folder that does not exist: {path}")
    return path


def raw(layer: str) -> Path:
    """Path of a raw layer inside ``CHITRAL_RAW_DATA``."""
    return _env_dir("CHITRAL_RAW_DATA") / RAW_LAYERS[layer]


def sentinel2(year: int) -> Path:
    """Path of the annual Sentinel-2 composite written by step 00."""
    return _env_dir("CHITRAL_S2_DIR") / f"S2_{year}.tif"
