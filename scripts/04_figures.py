"""Step 04. Figures of the paper, from data/ and results/ only.

Outputs: figures/Fig01_study_area ... Fig08_maps, each as PNG (300 dpi) and PDF.
"""

from __future__ import annotations

import json
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from sklearn.metrics import roc_curve

from fanscreen.config import DATA, FIGURES, RESULTS
from fanscreen.models import MODEL_NAMES
from fanscreen.plotting import AQUA, BLUE, GRID, INK, MUTED, NAVY, ORANGE, VIOLET, panel_letter, save, use_style

warnings.filterwarnings("ignore")
use_style()

g = gpd.read_file(DATA / "catchments.gpkg")
table = pd.read_csv(DATA / "catchments.csv")
labelled = table[table.y_debris.notna()].copy()
boundary = gpd.read_file(DATA / "chitral_boundary.gpkg").to_crs(g.crs)
audit = json.loads((RESULTS / "data_audit.json").read_text())
res = json.loads((RESULTS / "model_results.json").read_text())
bench = pd.read_csv(RESULTS / "benchmark_spatial30km.csv")
bench[["model", "set"]] = bench.name.str.split("|", expand=True)

# Hillshade background for the maps
with rasterio.open(DATA / "dem_125m.tif") as src:
    z = src.read(1)
    extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    cell = src.res[0]
shade = LightSource(azdeg=315, altdeg=40).hillshade(
    np.where(np.isnan(z), np.nanmean(z), z), vert_exag=1.2, dx=cell, dy=cell
)
shade = np.where(np.isnan(z), np.nan, shade)
XLIM = (g.total_bounds[0] - 6000, g.total_bounds[2] + 6000)
YLIM = (g.total_bounds[1] - 6000, g.total_bounds[3] + 6000)


def basemap(ax) -> None:
    ax.imshow(shade, extent=extent, cmap="Greys_r", vmin=0, vmax=1.3, alpha=0.55, zorder=0)
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}")
    ax.set_xlabel("UTM 43N easting (km)")
    ax.set_ylabel("UTM 43N northing (km)")
    ax.spines[["top", "right"]].set_visible(True)
    boundary.boundary.plot(ax=ax, color=INK, lw=0.8, zorder=4)


def scale_bar_and_north(ax, km: int = 50) -> None:
    x0, y0 = XLIM[1] - 62000, YLIM[0] + 7000
    ax.plot([x0, x0 + km * 1000], [y0, y0], color=INK, lw=2.2, solid_capstyle="butt", zorder=6)
    ax.text(x0 + km * 500, y0 + 2500, f"{km} km", ha="center", va="bottom", zorder=6)
    ax.annotate(
        "N",
        xy=(XLIM[1] - 12000, YLIM[0] + 55000),
        xytext=(XLIM[1] - 12000, YLIM[0] + 34000),
        ha="center",
        va="center",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1),
        zorder=6,
    )


def fig01_study_area() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    basemap(ax)
    g[g.y_debris.isna()].plot(ax=ax, facecolor="#ffffff", alpha=0.35, edgecolor=MUTED, lw=0.35, zorder=1)
    g[g.y_debris == 0].plot(ax=ax, facecolor=BLUE, alpha=0.72, edgecolor="white", lw=0.45, zorder=2)
    g[g.y_debris == 1].plot(ax=ax, facecolor=ORANGE, alpha=0.78, edgecolor="white", lw=0.45, zorder=3)
    scale_bar_and_north(ax)
    ax.legend(
        handles=[
            Patch(fc=BLUE, alpha=0.72, label=f"Alluvial fan at outlet (n = {(g.y_debris == 0).sum()})"),
            Patch(fc=ORANGE, alpha=0.78, label=f"Debris-dominated fan at outlet (n = {(g.y_debris == 1).sum()})"),
            Patch(fc="#ffffff", ec=MUTED, label=f"No mapped fan type (n = {g.y_debris.isna().sum()})"),
            Line2D([], [], color=INK, lw=0.8, label="Chitral boundary"),
        ],
        loc="center right",
        bbox_to_anchor=(1.0, 0.42),
    )
    save(fig, FIGURES, "Fig01_study_area")


def fig02_workflow() -> None:
    """Six-stage workflow: each box gives a title, its purpose (italic) and its content."""
    n_pos, n_neg = int((labelled.y_debris == 1).sum()), int((labelled.y_debris == 0).sum())
    stages = [  # (label, edge colour, purpose colour, box height, arrows between boxes in the row, boxes)
        (
            "1  Data",
            "#1e88e5",
            INK,
            13.2,
            False,
            [
                (
                    "Catchment database",
                    "Units of analysis",
                    f"{audit['n_catchments']} catchments (ArcGIS, 2021)\nstored morphometry and\nlegacy hazard labels",
                ),
                (
                    "Terrain",
                    "Relief, slope and drainage",
                    f"12.5 m ALOS PALSAR DEM\n{audit['streams_n']:,} stream lines",
                ),
                (
                    "Imagery and climate",
                    "Cryosphere and rainfall",
                    "Sentinel-2 L2A, 2016–2025\nTRMM rainfall, 2010–2019",
                ),
                (
                    "Geology and fans",
                    "Lithology and reference fans",
                    "GSP geology and faults\nfan map digitised from\nGoogle Earth imagery",
                ),
            ],
        ),
        (
            "2  Audit",
            "#f4511e",
            INK,
            13.2,
            True,
            [
                (
                    "Audit stored attributes",
                    "Can the stored values be trusted?",
                    f"{audit['stored_melton_diff_gt_0p1']} of {audit['n_catchments']} Melton ratios differ\nfrom the DEM by more than 0.1;\nlegacy labels are formula-based",
                ),
                (
                    "Correct fan codes",
                    "Fix debris fans stored as alluvial",
                    f"DFSAA and DFSA recoded as\ndebris fans in {res['n_code_uncertain']} catchments,\nconfirmed by the map author",
                ),
                (
                    "Define independent target",
                    "Reference not built from predictors",
                    f"Debris-dominated fan = 1 (n = {n_pos})\nAlluvial fan = 0 (n = {n_neg})\n{n_pos + n_neg} labelled catchments",
                ),
            ],
        ),
        (
            "3  Predictors",
            "#2e9b3f",
            INK,
            13.2,
            False,
            [
                (
                    "Morphometric (6)",
                    "Basin shape and steepness",
                    "Melton ratio, basin length,\nmean slope, hypsometric integral,\nelongation, drainage density",
                ),
                (
                    "Environmental (5)",
                    "Climate, cryosphere and geology",
                    "Mean elevation, snow and ice\nfraction, rainfall field, crystalline\nfraction, fault density",
                ),
                (
                    "Six predictor sets",
                    "Isolate what adds skill",
                    "Melton; Melton + length;\nmorphometric; environmental;\nfull 11; full without Melton",
                ),
            ],
        ),
        (
            "4  Benchmark",
            "#1e88e5",
            NAVY,
            13.2,
            False,
            [
                (
                    "Fixed rules (no fitting)",
                    "Baselines used in practice",
                    "Melton > 0.30, 0.46 and 0.60\nWilford-type debris-flow field\nlegacy 2021 rule label",
                ),
                (
                    "Statistical models",
                    "Does learning add skill?",
                    "Logistic regression (LR), Random\nForest (RF), gradient boosting (GB)\nwith fixed, untuned settings",
                ),
            ],
        ),
        (
            "5  Validation",
            "#7b1fa2",
            NAVY,
            11.4,
            False,
            [
                (
                    "Spatially blocked cross-validation",
                    "Estimate skill in unseen areas, not in neighbouring catchments",
                    "Five folds of 30 km blocks · 30 repartitions · AUC with block-bootstrap 95% intervals\npaired AUC gain over the Melton ratio · label-permutation null",
                )
            ],
        ),
        (
            "6  Analysis",
            "#e0a800",
            INK,
            13.2,
            False,
            [
                (
                    "Robustness",
                    "Do the results hold?",
                    "Block size, labels and inputs;\ngeographic transfer;\nMoran's I of residuals",
                ),
                (
                    "Process interpretation",
                    "What separates the fan types?",
                    "Group contrasts, coefficients,\nheld-out permutation importance,\npartial dependence",
                ),
                (
                    "Screening consequences",
                    "What changes in practice?",
                    "Disagreement with mapped fans,\nexposed assets,\ndomain of applicability",
                ),
            ],
        ),
    ]
    gap_rows, x0, x1 = 4.6, 11.7, 99.2
    height = sum(s[3] for s in stages) + gap_rows * len(stages) + 8.2
    fig, ax = plt.subplots(figsize=(7.4, 7.4 * height / 100))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, height)
    fig.subplots_adjust(0, 0, 1, 1)

    def arrow(a, b, colour=INK):
        ax.add_patch(
            FancyArrowPatch(
                a, b, arrowstyle="-|>", mutation_scale=8, color=colour, lw=0.9, shrinkA=0, shrinkB=0, zorder=3
            )
        )

    top, prev = height - 1.2, None
    for label, edge, purpose_colour, h, chain, boxes in stages:
        y = top - h
        ax.add_patch(
            FancyBboxPatch(
                (0, y - 1.1),
                100,
                h + 2.2,
                boxstyle="round,pad=0,rounding_size=1.4",
                fc="none",
                ec=NAVY,
                lw=0.8,
                ls=(0, (4, 2.5)),
                zorder=0,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.8, y), 9.4, h, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=edge, lw=1.3, zorder=1
            )
        )
        ax.text(
            5.5,
            y + h / 2,
            label,
            rotation=90,
            ha="center",
            va="center",
            color=INK,
            fontweight="bold",
            fontsize=7.4,
            zorder=2,
        )
        gap = 3.4 if chain else 2.2
        w = (x1 - x0 - gap * (len(boxes) - 1)) / len(boxes)
        xs = [x0 + k * (w + gap) for k in range(len(boxes))]
        for x, (title, purpose, body) in zip(xs, boxes):
            ax.add_patch(
                FancyBboxPatch(
                    (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc="white", ec=edge, lw=1.3, zorder=1
                )
            )
            ax.text(
                x + w / 2,
                y + h - 1.3,
                title,
                ha="center",
                va="top",
                fontweight="bold",
                fontsize=7.6,
                color=INK,
                zorder=2,
            )
            ax.text(
                x + w / 2,
                y + h - 4.1,
                purpose,
                ha="center",
                va="top",
                style="italic",
                fontsize=6.6,
                color=purpose_colour,
                zorder=2,
            )
            ax.text(
                x + w / 2, y + h - 6.9, body, ha="center", va="top", fontsize=6.6, color=INK, linespacing=1.35, zorder=2
            )
        if chain:
            for x in xs[1:]:
                arrow((x - gap + 0.3, y + h / 2), (x - 0.3, y + h / 2), "#e53935")
        if prev:
            pxs, pw, py = prev
            if len(pxs) == 1 or len(xs) == 1:  # fan out from, or into, a single wide box
                sources = [px + pw / 2 for px in pxs] if len(pxs) > 1 else [x + w / 2 for x in xs]
                targets = [x + w / 2 for x in xs] if len(xs) > 1 else sources
                for s, t in zip(sources, targets):
                    arrow((s, py - 0.2), (t, top + 0.2))
            else:
                arrow(((x0 + x1) / 2, py - 0.2), ((x0 + x1) / 2, top + 0.2))
        prev = (xs, w, y)
        top = y - gap_rows

    # key-result strip
    y = 0.4
    h = top - y
    ax.add_patch(
        FancyBboxPatch((0, y), 100, h, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=INK, lw=1.3, zorder=1)
    )
    b = bench.set_index("name")
    m = b.loc["LR|Melton"]
    ax.text(
        50,
        y + h / 2,
        f"Key result:  Melton ratio alone (LR) reaches AUC {m.auc:.2f} ({m.auc_lo:.2f}–{m.auc_hi:.2f}); "
        f"RF ({b.loc['RF|Full', 'auc']:.2f}) and GB ({b.loc['GB|Full', 'auc']:.2f}) give no resolved gain",
        ha="center",
        va="center",
        color=INK,
        fontsize=7.2,
        fontweight="bold",
        zorder=2,
    )
    pxs, pw, py = prev
    for px in pxs:
        arrow((px + pw / 2, py - 0.2), (px + pw / 2, top + 0.2))
    save(fig, FIGURES, "Fig02_workflow")


def fig03_audit_thresholds() -> None:
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.75), gridspec_kw=dict(wspace=0.48, width_ratios=[1, 0.9, 1.12]))
    ax = axs[0]
    for rule, colour, label in [("flood", BLUE, "Legacy flood"), ("debris_flow", ORANGE, "Legacy debris flow")]:
        s = table[table.rule_label == rule]
        ax.scatter(s.melton, s.Melton_rat, s=13, c=colour, edgecolor="white", lw=0.35, alpha=0.9, label=label)
    ax.plot([0, 1.5], [0, 1.5], color=MUTED, lw=0.7, ls="--")
    ax.axhline(0.46, color=MUTED, lw=0.6, ls=":")
    ax.text(1.49, 0.475, "0.46", color=MUTED, ha="right", va="bottom", fontsize=7.5)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.5)
    ax.set_xticks([0, 0.5, 1, 1.5])
    ax.set_yticks([0, 0.5, 1, 1.5])
    ax.set_xlabel("Melton ratio from DEM")
    ax.set_ylabel("Stored 2021 Melton ratio")
    ax.legend(loc="upper left", handletextpad=0.1, borderaxespad=0, fontsize=7.3)
    panel_letter(ax, "a")

    ax = axs[1]
    rng = np.random.default_rng(2)  # horizontal jitter only
    for i, (value, colour) in enumerate([(0, BLUE), (1, ORANGE)]):
        s = labelled[labelled.y_debris == value].melton.values
        ax.scatter(
            i + rng.uniform(-0.17, 0.17, len(s)), s, s=12, c=colour, edgecolor="white", lw=0.35, alpha=0.9, zorder=3
        )
        ax.plot([i - 0.28, i + 0.28], [np.median(s)] * 2, color=INK, lw=1.6, solid_capstyle="round", zorder=4)
    for t in [0.3, 0.46, 0.6]:
        ax.axhline(t, color=MUTED, lw=0.6, ls=":", zorder=1)
        ax.text(-0.78, t + 0.012, f"{t:.2f}", color=MUTED, va="bottom", ha="left", fontsize=7.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Alluvial", "Debris-\ndominated"])
    ax.set_xlim(-0.8, 1.5)
    ax.set_ylabel("Melton ratio from DEM")
    ax.set_ylim(0, 1.05)
    panel_letter(ax, "b")

    ax = axs[2]
    for value, colour, label in [(0, BLUE, "Alluvial"), (1, ORANGE, "Debris-dominated")]:
        s = labelled[labelled.y_debris == value]
        ax.scatter(
            s.length_km,
            s.melton,
            s=14,
            c=colour,
            marker="o" if value == 0 else "^",
            edgecolor="white",
            lw=0.35,
            alpha=0.9,
            label=label,
        )
    ax.set_xscale("log")
    ax.axhline(0.6, color=MUTED, lw=0.6, ls=":")
    ax.axvline(2.7, color=MUTED, lw=0.6, ls=":")
    ax.fill_between([1.5, 2.7], 0.6, 1.05, color=GRID, alpha=0.7, lw=0)
    ax.annotate(
        "Wilford-type debris-\nflow field: no catchments",
        xy=(2.6, 0.82),
        xytext=(3.4, 0.88),
        ha="left",
        va="center",
        fontsize=6.8,
        color=MUTED,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5),
    )
    ax.set_xlim(1.5, 60)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.set_xticklabels(["2", "5", "10", "20", "50"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Basin length (km)")
    ax.set_ylabel("Melton ratio from DEM")
    ax.legend(loc="lower left", handletextpad=0.1, borderaxespad=0, fontsize=7.3)
    panel_letter(ax, "c")
    save(fig, FIGURES, "Fig03_audit_thresholds")


def fig04_benchmark() -> None:
    order = ["Melton", "Melton+length", "Morphometry", "Environment", "Full", "Full without Melton"]
    colours = {"LR": BLUE, "RF": ORANGE, "GB": AQUA}
    markers = {"LR": "o", "RF": "s", "GB": "^"}
    offsets = {"LR": 0.24, "RF": 0, "GB": -0.24}
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    reference = bench[bench.name == "LR|Melton"].auc.iloc[0]
    ax.axvline(0.5, color=MUTED, lw=0.7, ls="--")
    ax.axvline(reference, color=BLUE, lw=0.7, ls=":")
    ax.text(0.5, 5.62, "No skill", ha="center", va="bottom", color=MUTED, fontsize=7.5)
    ax.text(reference, 5.62, "Melton LR", ha="center", va="bottom", color=MUTED, fontsize=7.5)
    for i, set_name in enumerate(order[::-1]):
        for m in ["LR", "RF", "GB"]:
            row = bench[(bench.set == set_name) & (bench.model == m)]
            if row.empty:
                continue
            row = row.iloc[0]
            y = i + offsets[m] if set_name != "Melton" else i
            ax.plot([row.auc_lo, row.auc_hi], [y, y], color=colours[m], lw=1.3, solid_capstyle="round")
            ax.scatter(row.auc, y, s=26, c=colours[m], marker=markers[m], edgecolor="white", lw=0.5, zorder=3)
    ax.set_yticks(range(6))
    ax.set_yticklabels(order[::-1])
    ax.set_ylim(-0.55, 5.55)
    ax.set_xlim(0.35, 1.0)
    ax.set_xlabel("ROC AUC under 30 km spatial-block validation (mean, 95% block-bootstrap interval)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Line2D([], [], color=colours[m], marker=markers[m], lw=1.3, ms=5, label=MODEL_NAMES[m]) for m in colours
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    save(fig, FIGURES, "Fig04_benchmark")


def fig05_roc_calibration() -> None:
    oof = pd.read_csv(RESULTS / "oof_probabilities.csv")
    shown = [
        ("LR|Melton", "Melton (LR)", BLUE),
        ("LR|Melton+length", "Melton + length (LR)", ORANGE),
        ("LR|Morphometry", "Morphometry (LR)", AQUA),
        ("RF|Full", "Full (RF)", VIOLET),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.15), gridspec_kw=dict(wspace=0.32))
    ax = axs[0]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls="--")
    for key, label, colour in shown:
        fpr, tpr, _ = roc_curve(oof.y, oof[key])
        ax.plot(fpr, tpr, color=colour, lw=1.6, label=label)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", handlelength=1.4)
    panel_letter(ax, "a")
    ax = axs[1]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls="--")
    for key, label, colour in shown:
        bins = pd.qcut(oof[key], 5, duplicates="drop")
        t = oof.groupby(bins, observed=True).agg(p=(key, "mean"), y=("y", "mean"))
        ax.plot(t.p, t.y, color=colour, lw=1.3, marker="o", ms=4.5, mec="white", mew=0.5, label=label)
    ax.set_xlabel("Mean predicted probability, quintile")
    ax.set_ylabel("Observed debris-dominated share")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_letter(ax, "b")
    save(fig, FIGURES, "Fig05_roc_calibration")


def fig06_importance() -> None:
    imp = pd.read_csv(RESULTS / "permutation_importance.csv")
    groups = [
        "All morphometry",
        "All environment",
        "Basin length",
        "Elevation + snow/ice",
        "Melton ratio",
        "Slope",
        "Hypsometry",
        "Elongation",
        "Rainfall",
        "Drainage density",
        "Lithology",
        "Fault density",
    ]
    fig = plt.figure(figsize=(7.3, 3.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], wspace=0.33, hspace=0.62)
    ax = fig.add_subplot(gs[:, 0])
    for m, colour, marker, dy in [("LR", BLUE, "o", 0.17), ("RF", ORANGE, "s", -0.17)]:
        for i, k in enumerate(groups[::-1]):
            r = imp[(imp.model == m) & (imp.group == k)].iloc[0]
            ax.plot([r.lo, r.hi], [i + dy] * 2, color=colour, lw=1.2, solid_capstyle="round")
            ax.scatter(r["drop"], i + dy, c=colour, s=20, marker=marker, edgecolor="white", lw=0.4, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups[::-1])
    ax.set_xlabel("Drop in held-out ROC AUC when permuted")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Line2D([], [], color=BLUE, marker="o", ms=4.5, lw=1.2, label="LR, Full"),
            Line2D([], [], color=ORANGE, marker="s", ms=4.5, lw=1.2, label="RF, Full"),
        ],
        loc="lower right",
    )
    panel_letter(ax, "a")
    pdp = pd.read_csv(RESULTS / "partial_dependence.csv")
    for pos, feature, label, letter in [
        (gs[0, 1], "length_km", "Basin length (km)", "b"),
        (gs[1, 1], "snow_ice_frac", "Snow and ice fraction", "c"),
    ]:
        ax = fig.add_subplot(pos)
        s = pdp[pdp.feature == feature]
        ax.plot(s.value, s.lr, color=BLUE, lw=1.6, label="LR")
        ax.plot(s.value, s.rf, color=ORANGE, lw=1.6, label="RF")
        ax.set_xlabel(label)
        ax.set_ylabel("P(debris)")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        panel_letter(ax, letter)
        values = labelled[feature].values
        ax.plot(
            values, np.full(len(values), 0.115), ls="", marker="|", ms=3.5, color=MUTED, mew=0.45
        )  # rug of observed values
        if letter == "b":
            ax.legend(loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.2, borderaxespad=0)
    save(fig, FIGURES, "Fig06_importance")


def fig07_sensitivity() -> None:
    sens = pd.read_csv(RESULTS / "sensitivity.csv")
    tests = sens[sens.name == "LR"].test.tolist()
    primary = {m: bench[(bench.set == "Full") & (bench.model == m)].iloc[0] for m in ["LR", "RF"]}
    rows = ["Primary analysis (30 km blocks)"] + tests
    fig, ax = plt.subplots(figsize=(6.9, 4.7))
    for m, colour, marker, dy in [("LR", BLUE, "o", 0.17), ("RF", ORANGE, "s", -0.17)]:
        for i, test in enumerate(rows[::-1]):
            r = primary[m] if test == rows[0] else sens[(sens.name == m) & (sens.test == test)].iloc[0]
            ax.plot([r.auc_lo, r.auc_hi], [i + dy] * 2, color=colour, lw=1.2, solid_capstyle="round")
            ax.scatter(r.auc, i + dy, c=colour, s=20, marker=marker, edgecolor="white", lw=0.4, zorder=3)
    ax.axvline(0.5, color=MUTED, lw=0.7, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows[::-1])
    ax.set_xlim(0.42, 1.0)
    ax.set_xlabel("ROC AUC, Full predictor set (mean, 95% block-bootstrap interval)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.legend(
        handles=[
            Line2D([], [], color=BLUE, marker="o", ms=4.5, lw=1.2, label=MODEL_NAMES["LR"]),
            Line2D([], [], color=ORANGE, marker="s", ms=4.5, lw=1.2, label=MODEL_NAMES["RF"]),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=2,
    )
    save(fig, FIGURES, "Fig07_sensitivity")


def fig08_maps() -> None:
    pred = pd.read_csv(RESULTS / "catchment_predictions.csv")
    m = g.merge(pred[["cid", "p_debris_lr"]], on="cid")
    lab = m[m.y_debris.notna()].copy()
    lab["rule"] = (lab.rule_label == "debris_flow").astype(int)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.25), gridspec_kw=dict(wspace=0.17))

    ax = axs[0]
    basemap(ax)
    cmap = LinearSegmentedColormap.from_list("seq", ["#eaf2fb", "#a6c8ef", "#5a9be3", "#2a78d6", "#14498a"])
    m[m.y_debris.isna()].plot(ax=ax, facecolor="#ffffff", alpha=0.35, edgecolor=MUTED, lw=0.25, zorder=1)
    lab.plot(ax=ax, column="p_debris_lr", cmap=cmap, vmin=0, vmax=1, edgecolor=MUTED, lw=0.25, zorder=2)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(mappable, cax=ax.inset_axes([0.02, -0.31, 0.96, 0.045]), orientation="horizontal")
    cb.set_label("Out-of-fold P(debris-dominated), Full LR")
    cb.outline.set_linewidth(0.4)
    panel_letter(ax, "a")

    ax = axs[1]
    basemap(ax)
    ax.tick_params(labelleft=False)
    m[m.y_debris.isna()].plot(ax=ax, facecolor="#ffffff", alpha=0.35, edgecolor=MUTED, lw=0.25, zorder=1)
    agree = lab[lab.rule == lab.y_debris]
    missed = lab[(lab.rule == 0) & (lab.y_debris == 1)]
    false_alarm = lab[(lab.rule == 1) & (lab.y_debris == 0)]
    agree.plot(ax=ax, facecolor="#c9c8c1", edgecolor="white", lw=0.25, zorder=2)
    missed.plot(ax=ax, facecolor=ORANGE, edgecolor="white", lw=0.25, zorder=3)
    false_alarm.plot(ax=ax, facecolor=BLUE, edgecolor="white", lw=0.25, zorder=3)
    ax.legend(
        handles=[
            Patch(fc="#c9c8c1", label=f"Rule agrees with fan (n = {len(agree)})"),
            Patch(fc=ORANGE, label=f"Rule: flood, fan: debris (n = {len(missed)})"),
            Patch(fc=BLUE, label=f"Rule: debris flow, fan: alluvial (n = {len(false_alarm)})"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=1,
        fontsize=7.3,
        borderaxespad=0,
        labelspacing=0.3,
    )
    ax.set_ylabel("")
    panel_letter(ax, "b")
    save(fig, FIGURES, "Fig08_maps")


if __name__ == "__main__":
    for make in (
        fig01_study_area,
        fig02_workflow,
        fig03_audit_thresholds,
        fig04_benchmark,
        fig05_roc_calibration,
        fig06_importance,
        fig07_sensitivity,
        fig08_maps,
    ):
        make()
    print(f"Figures written to {FIGURES}")
