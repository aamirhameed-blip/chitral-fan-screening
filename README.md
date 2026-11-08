# chitral-fan-screening

[![tests](https://github.com/aamirhameed-blip/chitral-fan-screening/actions/workflows/tests.yml/badge.svg)](https://github.com/aamirhameed-blip/chitral-fan-screening/actions/workflows/tests.yml)

Data and code for the paper

> Hameed, A., Confuorto, P. *Does Machine Learning Improve on the Melton Ratio? A Fan-Based Evaluation of Debris-Flow and Flood Process Screening in Glacierised Catchments of the Hindu Kush, Chitral, Pakistan.* Submitted to *Geomorphology*.

The Melton ratio (basin relief divided by the square root of basin area) is widely used to separate catchments prone to debris flows from those prone to floods. This repository tests whether machine learning improves on it, using the fan type mapped at the outlet of 104 catchments in Chitral as an independent reference, and validating every model on spatial blocks that it has not seen.

**Main result.** A logistic model on the Melton ratio alone reaches a ROC AUC of 0.86 under 30 km spatial-block cross-validation. Random Forest, gradient boosting and multi-predictor logistic models give no resolved gain. Correcting 25 debris fans that were stored as alluvial changes the Melton AUC from 0.60 to 0.86: label quality matters more than model complexity.

## Repository layout

```
chitral-fan-screening/
├── data/                     analysis-ready data (CC BY 4.0), described in data/README.md
├── src/fanscreen/            reusable code: settings, predictors, models, validation, statistics, plotting, workflow diagram
├── scripts/                  the analysis, one numbered step per script
│   ├── 00_sentinel2_composites.py   optional: annual Sentinel-2 composites   [downloads imagery]
│   ├── 01_build_catchments.py       raw GIS layers -> data/                   [needs raw data]
│   ├── 02_benchmark.py              data/ -> results/ (all tables of the paper)
│   ├── 03_label_checks.py           checks of the fan labels                  [needs raw data]
│   └── 04_figures.py                data/ + results/ -> figures/
├── results/                  outputs of steps 01 to 03, as used in the paper
├── figures/                  Figures 1 to 8 of the paper (PNG and PDF)
├── tests/                    unit tests of the validation and statistics code
├── environment.yml           conda environment with the exact package versions used
└── pyproject.toml            package metadata
```

## Installation

Install [Miniforge](https://github.com/conda-forge/miniforge) or Anaconda, then:

```bash
git clone https://github.com/aamirhameed-blip/chitral-fan-screening.git
cd chitral-fan-screening
conda env create -f environment.yml
conda activate fanscreen
```

This installs the `fanscreen` package in editable mode together with all dependencies. With pip instead: `pip install -e ".[dev]"` (rasterio and geopandas are easier to install with conda on Windows).

## Reproducing the results

Steps 02 and 04 need only the files in `data/`:

```bash
python scripts/02_benchmark.py      # about 25 minutes on 8 cores
python scripts/04_figures.py        # under a minute
pytest                              # unit tests, a few seconds
```

Step 02 overwrites the tables in `results/`. All random steps are seeded, so a run reproduces the published values. The number of parallel workers defaults to 8 and can be changed with the `FANSCREEN_JOBS` environment variable.

### Rebuilding from the raw data

Steps 01 and 03 read the raw GIS layers of the 2021 Chitral hazard database (catchments, 12.5 m ALOS PALSAR DEM, streams, TRMM rainfall, geology and faults, landslide and fan inventories, building footprints). These layers are not redistributed here; they are available from the corresponding author on reasonable request. The expected folder structure is listed in `src/fanscreen/config.py` (`RAW_LAYERS`). Point two environment variables at the data, then run the steps:

```bash
export CHITRAL_RAW_DATA=/path/to/ChitralProjectData     # Windows PowerShell: $env:CHITRAL_RAW_DATA = "D:\path\to\ChitralProjectData"
export CHITRAL_S2_DIR=/path/to/sentinel2
python scripts/00_sentinel2_composites.py               # optional: only if the composites are not yet available
python scripts/01_build_catchments.py
python scripts/03_label_checks.py
```

## Method in brief

| Item | Setting |
|---|---|
| Units | 169 catchments; 104 with a mapped fan type at the outlet (68 debris-dominated, 36 alluvial) |
| Target | Debris fan, glacial debris fan or debris cone = 1; alluvial fan = 0 |
| Predictors | 6 morphometric (Melton ratio, basin length, mean slope, hypsometric integral, elongation, drainage density) and 5 environmental (mean elevation, snow and ice fraction, TRMM rainfall, crystalline bedrock fraction, fault density) |
| Models | Logistic regression, Random Forest, histogram gradient boosting, with fixed, untuned settings (`src/fanscreen/models.py`) |
| Baselines | Melton thresholds 0.30, 0.46 and 0.60; Wilford et al. (2004) debris-flow field; legacy 2021 labels |
| Validation | 5-fold cross-validation on 30 km spatial blocks, 30 random block partitions |
| Uncertainty | Block bootstrap (2000 draws) for 95% intervals and for the paired AUC gain over the Melton model; label-permutation null |
| Robustness | 20 and 50 km blocks, alternative label codings, input substitutions, geographic transfer, Moran's I of residuals |

## Data

`data/catchments.gpkg` (EPSG:32643, WGS 84 / UTM zone 43N) holds one polygon per catchment with all predictors, the fan type and the target; `data/catchments.csv` is the same table without geometry. Every column is described in [data/README.md](data/README.md).

## Licence

Code: [MIT](LICENSE). Data in `data/`: [CC BY 4.0](data/LICENSE).

## Citation

Please cite the paper and the archived release of this repository. Citation metadata are in [CITATION.cff](CITATION.cff); GitHub shows them under *Cite this repository*.

## Contact

Aamir Hameed, Department of Earth and Geoenvironmental Sciences, University of Bari Aldo Moro, Italy — aamir.hameed@uniba.it
