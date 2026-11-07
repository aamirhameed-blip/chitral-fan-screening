"""Target and predictor definitions."""

from __future__ import annotations

# Fan types at the catchment outlet. Debris-dominated fans form the positive class.
DEBRIS_FANS = ("Debris Fan Active", "Glacial Debris Fan Active", "Debris Cone Active")
ALLUVIAL_FAN = "Alluvial Fan Active"

# Fan-map codes. DFSAA and DFSA are debris fans (confirmed by the author of the
# fan map) although the stored catchment attribute records them as alluvial.
DEBRIS_FAN_CODES = ("DFAA", "GDFAA", "GDFSAA", "GDFCAA")
RECODED_FAN_CODES = ("DFSAA", "DFSA")

# Inventory classes that mark landslide source areas (fan classes are excluded so
# that no fan information leaks into the predictors).
SOURCE_CLASSES = ("SS", "RS", "RF", "CS", "DS", "USS")

# Geological formations counted as crystalline bedrock
CRYSTALLINE_UNITS = ("Granodiorite", "Granite", "Gneiss", "Pegmatite", "miraphibolite", "Amphibolite")
NON_BEDROCK_UNITS = ("Glacier and snow", "Quaternary Alluvium", "Unknown")

MORPHOMETRIC = ["melton", "length_km", "slope_mean", "hypsometric", "elongation", "drainage_density"]
ENVIRONMENTAL = ["elev_mean", "snow_ice_frac", "rain_trmm", "crystalline_frac", "fault_density"]
FULL = MORPHOMETRIC + ENVIRONMENTAL

PREDICTOR_SETS = {
    "Melton": ["melton"],
    "Melton+length": ["melton", "length_km"],
    "Morphometry": MORPHOMETRIC,
    "Environment": ENVIRONMENTAL,
    "Full": FULL,
    "Full without Melton": [f for f in FULL if f != "melton"],
}

# Predictor groups permuted together in the importance analysis
IMPORTANCE_GROUPS = {
    "Melton ratio": ["melton"],
    "Basin length": ["length_km"],
    "Slope": ["slope_mean"],
    "Hypsometry": ["hypsometric"],
    "Elongation": ["elongation"],
    "Drainage density": ["drainage_density"],
    "Elevation + snow/ice": ["elev_mean", "snow_ice_frac"],
    "Rainfall": ["rain_trmm"],
    "Lithology": ["crystalline_frac"],
    "Fault density": ["fault_density"],
    "All morphometry": MORPHOMETRIC,
    "All environment": ENVIRONMENTAL,
}
