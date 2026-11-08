"""Study workflow diagram (Figure 2): dashed stage groups, data inputs as parallelograms, elbow connectors.

Shaded boxes mark the tests of label quality and of the gain over the Melton ratio.
"""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

INK, SHADE = "#000000", "#d9d9d9"
HEAD = "-|>,head_length=0.35,head_width=0.18"
WIDTH, HEIGHT = 100.0, 113.0  # drawing units; give the figure the same aspect ratio


class _Canvas:
    """Small drawing helpers in data units."""

    def __init__(self, ax):
        self.ax = ax

    def box(self, x, y, w, h, title, body, shade=False, data=False) -> float:
        """Process box (or data parallelogram) with a bold title; returns its centre x."""
        if data:
            k = 1.6
            corners = [(x + k, y + h), (x + w, y + h), (x + w - k, y), (x, y)]
            self.ax.add_patch(Polygon(corners, fc="white", ec=INK, lw=0.7, zorder=2))
        else:
            self.ax.add_patch(Rectangle((x, y), w, h, fc=SHADE if shade else "white", ec=INK, lw=0.7, zorder=2))
        self.ax.text(x + w / 2, y + h - 1.1, title, ha="center", va="top", fontsize=5.6, fontweight="bold", zorder=3)
        self.ax.text(x + w / 2, y + h - 3.3, body, ha="center", va="top", fontsize=4.9, linespacing=1.3, zorder=3)
        return x + w / 2

    def group(self, x, y, w, h, label) -> None:
        """Dashed stage outline with its numbered title centred on the top edge."""
        outline = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="none", ec=INK, lw=0.6, ls=(0, (3, 2)), zorder=1
        )
        self.ax.add_patch(outline)
        tw = 0.92 * len(label) + 3
        self.ax.add_patch(Rectangle((x + w / 2 - tw / 2, y + h - 1.4), tw, 2.8, fc="white", ec=INK, lw=0.7, zorder=2))
        self.ax.text(x + w / 2, y + h, label, ha="center", va="center", fontsize=5.8, fontweight="bold", zorder=3)

    def line(self, xs, ys) -> None:
        self.ax.plot(xs, ys, color=INK, lw=0.6, zorder=1.5, solid_capstyle="butt")

    def arrow(self, start, end) -> None:
        props = dict(arrowstyle=HEAD, color=INK, lw=0.6)
        self.ax.annotate("", xy=end, xytext=start, arrowprops=props, zorder=4)

    def elbow(self, x1, y1, x2, y2, y_mid) -> None:
        """Down from (x1, y1) to y_mid, across to x2, then down to an arrowhead at (x2, y2)."""
        self.line([x1, x1, x2, x2], [y1, y_mid, y_mid, y2 + 1.0])
        self.arrow((x2, y2 + 1.0), (x2, y2))


def draw_workflow(
    ax,
    n_catchments: int,
    n_melton_diff: int,
    n_recoded: int,
    n_debris: int,
    n_alluvial: int,
    melton_auc: tuple[float, float, float],
    rf_auc: float,
    gb_auc: float,
) -> None:
    """Draw the seven-stage workflow on ``ax``.

    ``melton_auc`` is (mean, lower, upper) of the Melton-only logistic model.
    """
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")
    c = _Canvas(ax)

    # 1. Data compilation
    gy = 96.0
    c.group(2, gy, 96, 15.5, "1. Data compilation")
    by = gy + 2.4
    inputs = [
        (
            5.0,
            "Catchment database",
            f"{n_catchments} catchments (ArcGIS, 2021);\nstored morphometry and\nlegacy hazard labels",
        ),
        (28.5, "Terrain", "12.5 m ALOS PALSAR DEM;\nstream network"),
        (52.0, "Imagery and climate", "Sentinel-2 L2A, 2016-2025;\nTRMM rainfall, 2010-2019"),
        (75.5, "Geology and fan map", "GSP geology and faults;\nfans digitised from\nGoogle Earth imagery"),
    ]
    centres = [c.box(x, by, 21.5, 9.5, title, body, data=True) for x, title, body in inputs]
    bus = gy - 1.2
    for cx in centres:
        c.line([cx, cx], [by, bus])
    c.line([centres[0], centres[-1]], [bus, bus])

    # 2. Label audit and reference target
    gy2, gh2, bh2 = 76.0, 14.5, 9.0
    c.group(10, gy2, 80, gh2, "2. Label audit and reference target")
    c.line([50, 50], [bus, gy2 + gh2 + 2.4])
    c.arrow((50, gy2 + gh2 + 2.4), (50, gy2 + gh2 + 1.4))
    by2 = gy2 + 2.2
    c.box(
        14,
        by2,
        22,
        bh2,
        "Audit stored attributes",
        f"{n_melton_diff} of {n_catchments} Melton ratios differ\nfrom the DEM by > 0.1; legacy\nlabels are formula-based",
    )
    c.box(
        39,
        by2,
        22,
        bh2,
        "Correct fan codes",
        f"DFSAA and DFSA recoded\nas debris fans in {n_recoded}\ncatchments (map author)",
        shade=True,
    )
    target = c.box(
        64,
        by2,
        22,
        bh2,
        "Independent target",
        f"Debris-dominated fan = 1 (n = {n_debris});\nalluvial fan = 0 (n = {n_alluvial});\n"
        f"{n_debris + n_alluvial} labelled catchments",
    )
    c.arrow((36, by2 + bh2 / 2), (39, by2 + bh2 / 2))
    c.arrow((61, by2 + bh2 / 2), (64, by2 + bh2 / 2))

    # 3. Predictor construction and 4. Screening methods
    gy3, gh3, bh3 = 55.0, 15.5, 10.2
    c.group(2, gy3, 56, gh3, "3. Predictor construction")
    c.group(61, gy3, 37, gh3, "4. Screening methods")
    c.elbow(target, by2, 30, gy3 + gh3 + 1.4, gy2 - 1.6)
    c.elbow(target, by2, 79.5, gy3 + gh3 + 1.4, gy2 - 1.6)
    by3 = gy3 + 2.2
    c.box(
        4.5,
        by3,
        16.5,
        bh3,
        "Morphometric (6)",
        "Melton ratio, basin length,\nmean slope, hypsometric\nintegral, elongation,\ndrainage density",
    )
    c.box(
        22.5,
        by3,
        16.5,
        bh3,
        "Environmental (5)",
        "Mean elevation, snow and\nice fraction, rainfall,\ncrystalline fraction,\nfault density",
    )
    c.box(
        40.5,
        by3,
        15.5,
        bh3,
        "Six predictor sets",
        "Melton; Melton + length;\nmorphometric;\nenvironmental; full;\nfull without Melton",
    )
    c.box(
        63.5,
        by3,
        15.5,
        bh3,
        "Fixed rules",
        "Melton > 0.30, 0.46\nand 0.60; Wilford-type\nfield; legacy 2021\nlabel (no fitting)",
    )
    c.box(
        80.5,
        by3,
        15.5,
        bh3,
        "Statistical models",
        "Logistic regression,\nRandom Forest,\ngradient boosting\n(fixed settings)",
    )

    # 5. Model validation
    gy5, gh5, bh5 = 36.0, 14.5, 9.0
    c.group(2, gy5, 96, gh5, "5. Model validation")
    c.elbow(30, gy3, 50, gy5 + gh5 + 1.4, gy3 - 1.6)
    c.elbow(79.5, gy3, 50, gy5 + gh5 + 1.4, gy3 - 1.6)
    by5 = gy5 + 2.2
    c.box(
        6,
        by5,
        28,
        bh5,
        "Spatial-block cross-validation",
        "30 km blocks, 5 folds,\n30 random block partitions;\nout-of-fold probabilities",
    )
    bootstrap = c.box(
        36,
        by5,
        28,
        bh5,
        "Block bootstrap",
        "2000 draws: AUC with 95% intervals;\npaired AUC gain over the\nMelton-only model",
        shade=True,
    )
    c.box(66, by5, 28, bh5, "Label-permutation null", "500 permutations of the\nfan labels; skill expected\nby chance")
    c.arrow((34, by5 + bh5 / 2), (36, by5 + bh5 / 2))
    c.arrow((64, by5 + bh5 / 2), (66, by5 + bh5 / 2))

    # 6. Robustness tests and 7. Interpretation
    gy6, gh6, bh6 = 15.5, 15.5, 10.2
    c.group(2, gy6, 56, gh6, "6. Robustness tests")
    c.group(61, gy6, 37, gh6, "7. Interpretation")
    c.elbow(bootstrap, by5, 30, gy6 + gh6 + 1.4, gy5 - 1.6)
    c.elbow(bootstrap, by5, 79.5, gy6 + gh6 + 1.4, gy5 - 1.6)
    by6 = gy6 + 2.2
    c.box(
        4.5,
        by6,
        16.5,
        bh6,
        "Fan-code scenarios",
        "DFSAA and DFSA fans\ncorrected, left as\nstored, or excluded",
        shade=True,
    )
    c.box(
        22.5,
        by6,
        16.5,
        bh6,
        "Blind image review",
        f"Independent review of\nall {n_catchments} outlets; disputed\nfans relabelled or\nexcluded",
        shade=True,
    )
    c.box(
        40.5,
        by6,
        15.5,
        bh6,
        "Design and inputs",
        "20 and 50 km blocks;\ngeographic transfer;\ninput substitutions;\nMoran's I of residuals",
    )
    c.box(
        63.5,
        by6,
        15.5,
        bh6,
        "Process controls",
        "Group contrasts,\ncoefficients, permutation\nimportance, partial\ndependence",
    )
    c.box(
        80.5,
        by6,
        15.5,
        bh6,
        "Consequences",
        "Disagreement with\nmapped fans; exposed\nassets; domain of\napplicability",
    )

    # Key result and legend
    ky, kh = 1.5, 9.0
    outline = FancyBboxPatch(
        (14, ky), 64, kh, boxstyle="round,pad=0,rounding_size=1.2", fc="none", ec=INK, lw=0.6, ls=(0, (3, 2)), zorder=1
    )
    ax.add_patch(outline)
    ax.add_patch(Rectangle((18, ky + 1.4), 56, kh - 2.8, fc="white", ec=INK, lw=0.7, zorder=2))
    c.elbow(30, gy6, 46, ky + kh, gy6 - 1.6)
    c.elbow(79.5, gy6, 46, ky + kh, gy6 - 1.6)
    auc, lo, hi = melton_auc
    ax.text(46, ky + kh / 2 + 1.2, "Key result", ha="center", va="center", fontsize=5.6, fontweight="bold", zorder=3)
    ax.text(
        46,
        ky + kh / 2 - 1.3,
        f"Melton ratio alone: AUC {auc:.2f} ({lo:.2f}-{hi:.2f}); Random Forest ({rf_auc:.2f}) and\n"
        f"gradient boosting ({gb_auc:.2f}) give no resolved gain under any label treatment",
        ha="center",
        va="center",
        fontsize=4.9,
        linespacing=1.3,
        zorder=3,
    )
    ax.add_patch(Rectangle((81.5, 4.9), 3.2, 2.2, fc=SHADE, ec=INK, lw=0.6))
    ax.text(
        85.5,
        6.0,
        "Tests of label quality\nand of the gain over\nthe Melton ratio",
        ha="left",
        va="center",
        fontsize=4.6,
        linespacing=1.2,
    )
