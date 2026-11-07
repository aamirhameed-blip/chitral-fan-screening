"""Step 02. Benchmark Melton-ratio screening against machine learning under spatial-block validation.

Target: debris-dominated fan (debris fan, glacial debris fan or debris cone) = 1,
alluvial fan = 0, for the 104 catchments with a mapped fan type. Reads only
data/catchments.gpkg. All random steps are seeded, so the outputs are reproducible.

Outputs (results/)
    fixed_rules.csv               published and legacy rules, no fitting
    benchmark_spatial30km.csv     all model x predictor-set combinations
    oof_probabilities.csv         out-of-fold probabilities, mean over repeats
    sensitivity.csv               robustness to blocks, labels and inputs
    random_vs_spatial.csv         random versus spatial cross-validation
    geographic_transfer.csv       train on one half of the district, test on the other
    group_contrasts.csv           debris versus alluvial medians, single-predictor AUC
    lr_coefficients.csv           standardised logistic coefficients with bootstrap intervals
    permutation_importance.csv    held-out AUC drop per predictor group
    partial_dependence.csv        partial dependence for LR and RF
    label_coding_scenarios.csv    effect of the DFSAA/DFSA fan-code correction
    catchment_predictions.csv/.gpkg
    model_results.json            summary values quoted in the paper
"""

from __future__ import annotations

import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
from sklearn.base import clone
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fanscreen.config import DATA, N_BOOT, N_JOBS, N_PERM, N_REPEATS, RESULTS
from fanscreen.features import ENVIRONMENTAL, FULL, IMPORTANCE_GROUPS, MORPHOMETRIC, PREDICTOR_SETS
from fanscreen.models import build_models
from fanscreen.stats import benjamini_hochberg, morans_i_test
from fanscreen.validation import block_ids, bootstrap_index, out_of_fold, partition, spatial_cv, summarise

warnings.filterwarnings("ignore")

MODELS = build_models()
g = gpd.read_file(DATA / "catchments.gpkg")
d = g[g.y_debris.notna()].reset_index(drop=True).copy()
y = d.y_debris.astype(int).values
groups30 = block_ids(d, 30)
RES: dict = {}


def save_csv(df: pd.DataFrame, name: str, decimals: int | None = None) -> None:
    (df.round(decimals) if decimals is not None else df).to_csv(RESULTS / name, index=False)


def log(*args) -> None:
    print(*args, flush=True)


# 0. Sample ----------------------------------------------------------------------------
RES["n"] = dict(
    n=int(len(d)), pos=int(y.sum()), **{f"blocks{km}": int(len(np.unique(block_ids(d, km)))) for km in (20, 30, 50)}
)
log("Sample:", RES["n"])

# 1. Fixed published and legacy rules ----------------------------------------------------
RULES = {
    "Legacy 2021 rule label": d.rule_label == "debris_flow",
    "Melton > 0.30 (DEM)": d.melton > 0.30,
    "Melton > 0.46 (DEM)": d.melton > 0.46,
    "Melton > 0.60 (DEM)": d.melton > 0.60,
    "Melton > 0.46 (stored 2021 values)": d.Melton_rat > 0.46,
    "Wilford debris-flow field (Melton > 0.6, L < 2.7 km)": (d.melton > 0.6) & (d.length_km < 2.7),
}


def rule_skill(yy: np.ndarray, hit: np.ndarray) -> tuple:
    sens = ((hit == 1) & (yy == 1)).sum() / max(yy.sum(), 1)
    spec = ((hit == 0) & (yy == 0)).sum() / max((1 - yy).sum(), 1)
    return (hit == yy).mean(), (sens + spec) / 2, sens, spec


rng = np.random.default_rng(11)
rows = []
for name, rule in RULES.items():
    hit = rule.values.astype(int)
    boot = np.array([rule_skill(y[i], hit[i]) for i in (bootstrap_index(groups30, rng) for _ in range(N_BOOT))])
    acc, bal, sens, spec = rule_skill(y, hit)
    rows.append(
        dict(
            rule=name,
            n_pred_debris=int(hit.sum()),
            accuracy=acc,
            bal_acc=bal,
            sens=sens,
            spec=spec,
            bal_lo=np.percentile(boot[:, 1], 2.5),
            bal_hi=np.percentile(boot[:, 1], 97.5),
            mcc=matthews_corrcoef(y, hit) if hit.sum() else 0,
        )
    )
save_csv(pd.DataFrame(rows), "fixed_rules.csv", 3)
RES["raw_rank_auc"] = dict(
    melton_dem=roc_auc_score(y, d.melton),
    melton_stored=roc_auc_score(y, d.Melton_rat),
    length=roc_auc_score(y, -d.length_km),
    area=roc_auc_score(y, -d.area_km2),
)
log(pd.DataFrame(rows).round(3).to_string())

# 2. Main benchmark ------------------------------------------------------------------------
PRED = {}
P_melton, _, _ = spatial_cv(MODELS["LR"], d, ["melton"], y)
PRED["LR|Melton"] = P_melton
rows = []
for set_name, feats in PREDICTOR_SETS.items():
    for model_name, model in MODELS.items():
        if len(feats) == 1 and model_name != "LR":
            continue  # tree models add nothing on a single monotone predictor
        P, T, gr = spatial_cv(model, d, feats, y)
        key = f"{model_name}|{set_name}"
        PRED[key] = P
        rows.append(summarise(key, P, T, y, gr, baseline=None if key == "LR|Melton" else P_melton))
        log(key, {k: rows[-1].get(k) for k in ("auc", "auc_lo", "auc_hi", "dauc", "dauc_lo", "dauc_hi")})
bench = pd.DataFrame(rows)
save_csv(bench, "benchmark_spatial30km.csv")
RES["best_by_mean_auc"] = bench.sort_values("auc", ascending=False).name.iloc[0]
oof_mean = pd.DataFrame({"cid": d.cid, "y": y, **{k: v.mean(axis=0) for k, v in PRED.items()}})
save_csv(oof_mean, "oof_probabilities.csv")


# 3. Label-permutation null for the headline models ----------------------------------------
def permuted_auc(model, X: np.ndarray, seed: int) -> float:
    y_perm = np.random.default_rng(seed).permutation(y)
    prob, _ = out_of_fold(model, X, y_perm, partition(groups30, seed))
    return roc_auc_score(y_perm, prob)


for model_name, set_name, n in [("LR", "Full", N_PERM), ("RF", "Full", 200), ("LR", "Melton", N_PERM)]:
    X = d[PREDICTOR_SETS[set_name]].values
    null = np.array(Parallel(n_jobs=N_JOBS)(delayed(permuted_auc)(MODELS[model_name], X, s) for s in range(n)))
    observed = bench.loc[bench.name == f"{model_name}|{set_name}", "auc"].iloc[0]
    RES[f"null_{model_name}_{set_name}"] = dict(
        n=n,
        mean=float(null.mean()),
        q95=float(np.percentile(null, 95)),
        observed=float(observed),
        p=float((1 + (null >= observed).sum()) / (n + 1)),
    )
    log("Null", model_name, set_name, RES[f"null_{model_name}_{set_name}"])

# 4. Sensitivity to validation design, labels and inputs -----------------------------------
rows = []


def sensitivity(
    label: str,
    model_name: str,
    df: pd.DataFrame,
    feats: list,
    yy: np.ndarray,
    km: int = 30,
    baseline_feats: tuple = ("melton",),
) -> None:
    P, T, gr = spatial_cv(MODELS[model_name], df, feats, yy, km=km)
    B, _, _ = spatial_cv(MODELS["LR"], df, list(baseline_feats), yy, km=km)
    s = summarise(model_name, P, T, yy, gr, baseline=B)
    s.update(test=label, n=len(yy), pos=int(yy.sum()))
    rows.append(s)
    log(label, model_name, s["auc"], s["auc_lo"], s["auc_hi"], s.get("dauc"))


def subset(mask: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    sub = d[mask].reset_index(drop=True)
    return sub, sub.y_debris.astype(int).values


for mn in ["LR", "RF"]:
    for km in (20, 50):
        sensitivity(f"Block size {km} km", mn, d, FULL, y, km=km)
    sub, yy = subset(d.fan_type != "Glacial Debris Fan Active")
    sensitivity("Glacial debris fans excluded", mn, sub, FULL, yy)
    sub, yy = subset(d.fan_type != "Debris Cone Active")
    sensitivity("Debris cone excluded", mn, sub, FULL, yy)
    sensitivity(
        "Glacial debris fans coded as non-debris",
        mn,
        d,
        FULL,
        d.fan_type.isin(["Debris Fan Active", "Debris Cone Active"]).astype(int).values,
    )
    sub, yy = subset(d.area_km2 <= d.area_km2.quantile(0.9))
    sensitivity("Largest 10% of catchments excluded", mn, sub, FULL, yy)
    sensitivity(
        "Stored 2021 Melton instead of DEM Melton",
        mn,
        d,
        [("Melton_rat" if f == "melton" else f) for f in FULL],
        y,
        baseline_feats=("Melton_rat",),
    )
    sensitivity(
        "Stored 2021 glacier area instead of NDSI",
        mn,
        d,
        [("glacier_frac_raw" if f == "snow_ice_frac" else f) for f in FULL],
        y,
    )
    sensitivity("NDVI added", mn, d, FULL + ["ndvi"], y)
    sensitivity("Mapped landslide source fraction added", mn, d, FULL + ["slide_source_frac"], y)
    sensitivity("Rainfall removed", mn, d, [f for f in FULL if f != "rain_trmm"], y)
    sensitivity("Centroid coordinates added", mn, d, FULL + ["cx", "cy"], y)
    sensitivity(
        "Elevation and snow-ice removed", mn, d, [f for f in FULL if f not in ("elev_mean", "snow_ice_frac")], y
    )
save_csv(pd.DataFrame(rows), "sensitivity.csv")

# Random (non-spatial) versus spatial cross-validation
rows = []
for model_name, set_name in [("LR", "Melton"), ("LR", "Full"), ("RF", "Full"), ("GB", "Full")]:
    X = d[PREDICTOR_SETS[set_name]].values

    def random_cv(seed: int, model=MODELS[model_name], X=X) -> np.ndarray:
        folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
        return out_of_fold(model, X, y, folds)[0]

    aucs = [roc_auc_score(y, p) for p in Parallel(n_jobs=N_JOBS)(delayed(random_cv)(s) for s in range(N_REPEATS))]
    rows.append(
        dict(
            model=f"{model_name}|{set_name}",
            random_auc=np.mean(aucs),
            spatial_auc=bench.loc[bench.name == f"{model_name}|{set_name}", "auc"].iloc[0],
        )
    )
save_csv(pd.DataFrame(rows), "random_vs_spatial.csv", 3)

# Geographic transfer: fit on one side of the median coordinate, test on the other
rows = []
for axis in ["cy", "cx"]:
    cut = d[axis].median()
    for direction, train, test in [
        ("low to high", d[axis] <= cut, d[axis] > cut),
        ("high to low", d[axis] > cut, d[axis] <= cut),
    ]:
        for model_name, set_name in [("LR", "Melton"), ("LR", "Full"), ("RF", "Full")]:
            X = d[PREDICTOR_SETS[set_name]].values
            fitted = clone(MODELS[model_name]).fit(X[train.values], y[train.values])
            rows.append(
                dict(
                    axis={"cy": "northing", "cx": "easting"}[axis],
                    direction=direction,
                    model=f"{model_name}|{set_name}",
                    n_train=int(train.sum()),
                    n_test=int(test.sum()),
                    auc=roc_auc_score(y[test.values], fitted.predict_proba(X[test.values])[:, 1]),
                )
            )
save_csv(pd.DataFrame(rows), "geographic_transfer.csv", 3)

# 5. Spatial autocorrelation of the target and of the residuals ---------------------------
xy = d[["cx", "cy"]].values
rng = np.random.default_rng(3)
for label, values in [
    ("target", y.astype(float)),
    ("LR_Melton_residual", y - PRED["LR|Melton"].mean(axis=0)),
    ("LR_Full_residual", y - PRED["LR|Full"].mean(axis=0)),
    ("RF_Full_residual", y - PRED["RF|Full"].mean(axis=0)),
]:
    RES[f"moran_{label}"] = morans_i_test(values, xy, band=30000, rng=rng)
log({k: v for k, v in RES.items() if k.startswith("moran")})

# 6. Predictor effects ---------------------------------------------------------------------
rows = []
for f in FULL + ["area_km2", "relief_km", "relief_ratio", "ndvi"]:
    debris, alluvial = d.loc[y == 1, f], d.loc[y == 0, f]
    rows.append(
        dict(
            feature=f,
            med_debris=debris.median(),
            med_alluvial=alluvial.median(),
            auc=roc_auc_score(y, d[f]),
            p=mannwhitneyu(debris, alluvial).pvalue,
        )
    )
contrasts = pd.DataFrame(rows)
contrasts["p_bh"] = benjamini_hochberg(contrasts.p.values)
save_csv(contrasts, "group_contrasts.csv", 4)

# standardised logistic coefficients with block-bootstrap intervals
X_full = d[FULL].values
lr_full = clone(MODELS["LR"]).fit(X_full, y)
rng = np.random.default_rng(5)
boot = []
for _ in range(N_BOOT):
    i = bootstrap_index(groups30, rng)
    if len(np.unique(y[i])) == 2:
        boot.append(clone(MODELS["LR"]).fit(X_full[i], y[i])[-1].coef_[0])
boot = np.array(boot)
coefs = pd.DataFrame(
    dict(
        feature=FULL,
        coef_per_sd=lr_full[-1].coef_[0],
        lo=np.percentile(boot, 2.5, axis=0),
        hi=np.percentile(boot, 97.5, axis=0),
        sign_stability=np.maximum((boot > 0).mean(axis=0), (boot < 0).mean(axis=0)),
    )
)
save_csv(coefs, "lr_coefficients.csv", 3)


def importance_repeat(model_name: str, seed: int) -> dict:
    """Held-out AUC drop when each predictor group is permuted within the test fold."""
    rng_ = np.random.default_rng(seed)
    base = np.full(len(y), np.nan)
    permuted = {k: np.full(len(y), np.nan) for k in IMPORTANCE_GROUPS}
    for train, test in partition(groups30, seed):
        fitted = clone(MODELS[model_name]).fit(X_full[train], y[train])
        base[test] = fitted.predict_proba(X_full[test])[:, 1]
        for k, feats in IMPORTANCE_GROUPS.items():
            Xp = X_full[test].copy()
            cols = [FULL.index(f) for f in feats]
            Xp[:, cols] = Xp[rng_.permutation(len(test))][:, cols]
            permuted[k][test] = fitted.predict_proba(Xp)[:, 1]
    auc = roc_auc_score(y, base)
    return {k: auc - roc_auc_score(y, v) for k, v in permuted.items()}


rows = []
for mn in ["LR", "RF"]:
    drops = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(importance_repeat)(mn, s) for s in range(N_REPEATS * 2)))
    rows += [
        dict(model=mn, group=k, drop=drops[k].mean(), lo=drops[k].quantile(0.025), hi=drops[k].quantile(0.975))
        for k in IMPORTANCE_GROUPS
    ]
save_csv(pd.DataFrame(rows), "permutation_importance.csv", 4)

# partial dependence of the models fitted on all labelled catchments
rf_full = clone(MODELS["RF"]).fit(X_full, y)
rows = []
for f in FULL:
    for v in np.unique(np.quantile(d[f], np.linspace(0.05, 0.95, 19))):
        X = X_full.copy()
        X[:, FULL.index(f)] = v
        rows.append(
            dict(feature=f, value=v, rf=rf_full.predict_proba(X)[:, 1].mean(), lr=lr_full.predict_proba(X)[:, 1].mean())
        )
save_csv(pd.DataFrame(rows), "partial_dependence.csv", 4)

# 7. Screening consequences and domain of applicability ---------------------------------------
d["p_full"] = oof_mean["LR|Full"].values
d["model_debris"] = (d.p_full >= y.mean()).astype(int)
d["rule_debris"] = (d.rule_label == "debris_flow").astype(int)


def exposure(mask) -> dict:
    return dict(
        n=int(mask.sum()),
        houses=int(d.loc[mask, "Houses"].sum()),
        schools=int(d.loc[mask, "Schools"].sum()),
        hospitals=int(d.loc[mask, "hospitals"].sum()),
        road_km=float(d.loc[mask, "road_lengt"].sum()),
    )


RES["consequence"] = dict(
    rule_flood_fan_debris=exposure((d.rule_debris == 0) & (y == 1)),
    rule_debris_fan_alluvial=exposure((d.rule_debris == 1) & (y == 0)),
    rule_agree=exposure(d.rule_debris.values == y),
    model_flood_fan_debris=exposure((d.model_debris == 0) & (y == 1)),
    model_debris_fan_alluvial=exposure((d.model_debris == 1) & (y == 0)),
    model_agree=exposure(d.model_debris.values == y),
    all=exposure(np.ones(len(d), bool)),
)
lo, hi = d[FULL].quantile(0.05), d[FULL].quantile(0.95)
g["domain_share"] = ((g[FULL] >= lo) & (g[FULL] <= hi)).mean(axis=1)
labelled = g.y_debris.notna()
RES["domain"] = dict(
    labelled_mean=float(g.loc[labelled, "domain_share"].mean()),
    unlabelled_mean=float(g.loc[~labelled, "domain_share"].mean()),
    unlabelled_ge80=int((g.loc[~labelled, "domain_share"] >= 0.8).sum()),
    n_unlabelled=int((~labelled).sum()),
)
RES["labelled_vs_unlabelled"] = (
    g.groupby(labelled)[
        ["area_km2", "melton", "elev_min", "elev_mean", "snow_ice_frac", "rain_trmm", "Houses", "road_lengt"]
    ]
    .median()
    .round(3)
    .to_dict()
)
g["p_debris_lr"] = lr_full.predict_proba(g[FULL].values)[:, 1]
g.loc[labelled, "p_debris_lr"] = (
    d.set_index("cid")["p_full"].reindex(g.loc[labelled, "cid"]).values
)  # out-of-fold where labelled
g[
    [
        "cid",
        "HydroID",
        "rule_label",
        "fan_type",
        "y_debris",
        "p_debris_lr",
        "domain_share",
        "melton",
        "Melton_rat",
        "area_km2",
        "Houses",
        "Schools",
        "cx",
        "cy",
        "outlet_x",
        "outlet_y",
        "cov_s2",
    ]
].round(4).to_csv(RESULTS / "catchment_predictions.csv", index=False)
g[["cid", "p_debris_lr", "domain_share", "geometry"]].to_file(RESULTS / "catchment_predictions.gpkg", driver="GPKG")

# 8. Effect of the fan-code correction ----------------------------------------------------------
recoded = d.fan_code_recoded.astype(bool)
kept = d[~recoded].reset_index(drop=True)
rows = []
for scenario, D_, Y_ in [
    ("Debris fan codes", d, y),
    ("Stored attribute", d, np.where(recoded, 0, y)),
    ("Coded fans excluded", kept, kept.y_debris.astype(int).values),
]:
    legacy = (D_.rule_label == "debris_flow").astype(int).values
    B_, _, _ = spatial_cv(MODELS["LR"], D_, ["melton"], Y_)
    row = dict(scenario=scenario, n=int(len(Y_)), pos=int(Y_.sum()), legacy_disagree=int((legacy != Y_).sum()))
    for model_name, set_name in [
        ("LR", "Melton"),
        ("LR", "Morphometry"),
        ("LR", "Full"),
        ("RF", "Full"),
        ("GB", "Full"),
    ]:
        P_, T_, gr_ = spatial_cv(MODELS[model_name], D_, PREDICTOR_SETS[set_name], Y_)
        s_ = summarise(
            f"{model_name}|{set_name}",
            P_,
            T_,
            Y_,
            gr_,
            baseline=None if (model_name, set_name) == ("LR", "Melton") else B_,
        )
        row.update(
            {
                f"{model_name}_{set_name}_{k}": s_.get(k)
                for k in ["auc", "auc_lo", "auc_hi", "dauc", "dauc_lo", "dauc_hi"]
            }
        )
    rows.append(row)
    log(scenario, row["LR_Melton_auc"])
save_csv(pd.DataFrame(rows), "label_coding_scenarios.csv")
RES["n_code_uncertain"] = int(recoded.sum())

RES["desc"] = d[MORPHOMETRIC + ENVIRONMENTAL].describe().round(3).to_dict()
RES["crystalline_missing"] = int(g.crystalline_missing.sum())
with open(RESULTS / "model_results.json", "w") as f:
    json.dump(RES, f, indent=1, default=float)
log("Best model by mean AUC:", RES["best_by_mean_auc"])
log("Done.")
