"""Draw main-figure panels from released source-data tables."""

from __future__ import annotations

import csv
import html
import json
import math
import os
import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox


REPO_ROOT = Path(__file__).resolve().parents[3]
DRAFT = Path(
    os.getenv("RNA_PORTRAIT_FIGURE_WORK_ROOT", REPO_ROOT / "outputs")
).resolve()
FIG_DIR = Path(
    os.getenv("RNA_PORTRAIT_FIGURE_OUTPUT_ROOT", DRAFT / "figures")
).resolve()
SOURCE_DIR = Path(
    os.getenv("RNA_PORTRAIT_SOURCE_DATA_DIR", DRAFT / "source_data")
).resolve()
RESULTS = Path(
    os.getenv(
        "RNA_PORTRAIT_ANALYSIS_OUTPUT_ROOT",
        DRAFT / "manuscript_analysis_tables",
    )
).resolve()
CODE_DIR = REPO_ROOT / "workflows"


PALETTE = {
    "blue": "#1F5A9D",
    "blue_light": "#9DB9D8",
    "blue_soft": "#DCE8F3",
    "red": "#B64A4A",
    "red_light": "#E7B1AA",
    "green": "#5B8C5A",
    "green_light": "#B8D3B4",
    "violet": "#7E6AAE",
    "orange": "#C9822B",
    "orange_light": "#E1B66D",
    "grey": "#8F8F8F",
    "grey_light": "#D9D9D9",
    "grey_soft": "#EFEFEF",
    "black": "#222222",
    "paper": "#FFFFFF",
}

STATE_ORDER = [
    "stable_consensus",
    "hematologic_override",
    "epithelial_override",
    "clean_anchor_override",
    "generic_context_override",
    "unsupported_semantics",
    "family_conflict",
    "other",
]

STATE_LABELS = {
    "stable_consensus": "single clear signal",
    "hematologic_override": "blood/immune",
    "epithelial_override": "epithelial-like context",
    "clean_anchor_override": "cleaner-anchor context",
    "generic_context_override": "broad-context",
    "unsupported_semantics": "weak-evidence",
    "family_conflict": "conflict",
    "other": "other",
}

STATUS_LABELS = {
    "stable": "single clear signal",
    "mixed": "mixed-signal",
    "unsupported": "weak-evidence",
}

CLAIM_LABELS = {
    "epithelial_or_tumor_like_signal": "epithelial or tumour-like signal",
    "clean_or_non_malignant_context": "cleaner or non-malignant context",
    "mixed_or_unstable_disease_reading": "mixed or inconsistent label reading",
    "immune_or_blood_signal": "immune or blood signal",
    "context_or_stromal_signal": "stromal or broad-context signal",
}

CLAIM_TYPE_LABELS = {
    "epithelial_or_tumor_like_signal": "epithelial_or_tumour_like_signal",
    "clean_or_non_malignant_context": "cleaner_or_non_malignant_context",
    "mixed_or_unstable_disease_reading": "mixed_or_inconsistent_label_reading",
    "immune_or_blood_signal": "immune_or_blood_signal",
    "context_or_stromal_signal": "stromal_or_broad_context_signal",
}

STATE_COLORS = {
    "stable_consensus": "#303030",
    "hematologic_override": "#1F5A9D",
    "epithelial_override": "#B64A4A",
    "clean_anchor_override": "#5B8C5A",
    "generic_context_override": "#7E6AAE",
    "unsupported_semantics": "#A9A9A9",
    "family_conflict": "#C9822B",
    "other": "#6B6B6B",
}

SOURCE_MANIFEST: list[dict[str, str]] = []


def ensure_dirs() -> None:
    for path in [FIG_DIR, SOURCE_DIR, CODE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.4,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def read_csv(rel: str) -> pd.DataFrame:
    path = RESULTS / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def apply_reader_claim_labels(claims: pd.DataFrame) -> pd.DataFrame:
    claims = claims.copy()
    claims["claim_label"] = claims["claim_type"].map(CLAIM_LABELS).fillna(claims["claim_label"])
    claims["claim_type"] = claims["claim_type"].map(CLAIM_TYPE_LABELS).fillna(claims["claim_type"])
    return claims


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def add_source(name: str, source: str, description: str, df: pd.DataFrame | None = None) -> None:
    out = ""
    if df is not None:
        out_path = SOURCE_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)
        out = out_path.relative_to(DRAFT).as_posix()
    SOURCE_MANIFEST.append(
        {
            "name": name,
            "source": source,
            "description": description,
            "draft_copy": out,
        }
    )


def display_label_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Return a reader-facing copy with legacy label names clarified."""
    out = df.copy()
    replacements = {
        "fixed disease label": "predefined disease label",
        "fixed disease": "predefined disease",
        "external pool": "evaluation pool",
        "project/source": "project",
        "expected tissue/site": "tissue/site",
        "RNA portrait group": "RNA portrait category",
    }
    for col in ["factor_label", "control_label", "target_label"]:
        if col in out.columns:
            out[col] = out[col].replace(replacements)
    return out


def predefined_label_portrait_diversity(df: pd.DataFrame) -> pd.DataFrame:
    state_cols = [c for c in STATE_ORDER if c in df.columns]
    rows = []
    for _, row in df.iterrows():
        counts = row[state_cols].astype(float)
        total = float(counts.sum())
        if total <= 0:
            continue
        probs = counts / total
        nonzero = probs[probs > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum())
        entropy_norm = entropy / math.log(len(state_cols)) if len(state_cols) > 1 else 0.0
        dominant_state = str(counts.idxmax())
        rows.append(
            {
                "predefined_disease_label": str(row["closed_set_disease_family"]),
                "display_label": str(row["closed_set_disease_family"]).replace("_", " "),
                "n_samples": int(total),
                "n_portrait_families": int((counts > 0).sum()),
                "portrait_diversity": entropy_norm,
                "dominant_portrait_family": STATE_LABELS.get(dominant_state, dominant_state),
                "dominant_portrait_fraction": float(probs.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_samples", "portrait_diversity"], ascending=[False, False]).reset_index(drop=True)


def save_figure(fig: plt.Figure, name: str) -> dict[str, str]:
    paths = {}
    for ext in ["svg", "pdf", "png", "tiff"]:
        path = FIG_DIR / f"{name}.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if ext == "png":
            kwargs["dpi"] = 320
        if ext == "tiff":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        paths[ext] = path.relative_to(DRAFT).as_posix()
    plt.close(fig)
    return paths


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=8.5, fontweight="bold")


def tidy_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#E8E8E8", lw=0.55, zorder=0)
    ax.set_axisbelow(True)


def wrap_label(text: str, width: int = 16) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def weighted_mean_table(df: pd.DataFrame, value_cols: list[str], group_col: str = "semantic_state_family") -> pd.DataFrame:
    rows = []
    for state, g in df.groupby(group_col, sort=False):
        row = {group_col: state}
        weights = pd.to_numeric(g["n"], errors="coerce").fillna(0).to_numpy(dtype=float)
        for col in value_cols:
            values = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            row[col] = float(np.average(values[mask], weights=weights[mask])) if mask.any() else np.nan
        row["n_total"] = int(g["n"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def heatmap(ax: plt.Axes, matrix: pd.DataFrame, vmin: float, vmax: float, cbar: bool = False, label: str = "") -> None:
    cmap = LinearSegmentedColormap.from_list("blue_white_red", ["#2F5D8C", "#F8F8F8", "#B64A4A"])
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(matrix.shape[1]), [wrap_label(c, 12) for c in matrix.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), [wrap_label(STATE_LABELS.get(i, i), 18) for i in matrix.index])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
        cb.ax.set_ylabel(label, rotation=90)


def figure_1_architecture() -> dict[str, str]:
    set_style()
    fig = plt.figure(figsize=(7.2, 4.65))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, title, body, color, lw=1.0):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            fc="white",
            ec=color,
            lw=lw,
        )
        ax.add_patch(patch)
        ax.text(x + 0.018, y + h - 0.035, title, ha="left", va="top", fontsize=8.2, fontweight="bold", color=color)
        ax.text(x + 0.018, y + h - 0.073, body, ha="left", va="top", fontsize=6.8, color="#333333", linespacing=1.25)

    def arrow(x1, y1, x2, y2, color="#444444", lw=1.1):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=lw, color=color))

    ax.text(0.035, 0.94, "a", fontsize=9, fontweight="bold")
    ax.text(0.065, 0.94, "Image-language analogy", fontsize=9, fontweight="bold")
    ax.text(0.535, 0.94, "b", fontsize=9, fontweight="bold")
    ax.text(0.565, 0.94, "RNA-language system", fontsize=9, fontweight="bold")

    rng = np.random.default_rng(3)
    image_panel = fig.add_axes([0.055, 0.61, 0.16, 0.23])
    image_panel.imshow(rng.normal(size=(22, 22)), cmap="Greys", interpolation="nearest")
    image_panel.add_patch(Rectangle((2, 3), 8, 9, fill=False, ec=PALETTE["blue"], lw=2))
    image_panel.add_patch(Rectangle((11, 12), 8, 6, fill=False, ec=PALETTE["orange"], lw=2))
    image_panel.set_xticks([])
    image_panel.set_yticks([])
    for s in image_panel.spines.values():
        s.set_visible(False)
    ax.text(0.055, 0.57, "many pixels and objects", fontsize=6.7, color="#444444")
    arrow(0.225, 0.725, 0.33, 0.725, PALETTE["grey"])
    box(0.335, 0.64, 0.18, 0.17, "vision-language model", "learns a relation between\nvisual patterns and words", PALETTE["grey"])
    arrow(0.515, 0.725, 0.62, 0.725, PALETTE["grey"])
    box(0.625, 0.64, 0.23, 0.17, "natural-language description", "whole-scene description\nwith object relationships", PALETTE["grey"])

    genes = np.linspace(0, 1, 60)
    expr = np.sin(genes * 13) + rng.normal(0, 0.23, len(genes))
    expr_panel = fig.add_axes([0.055, 0.26, 0.19, 0.23])
    expr_panel.plot(expr, color=PALETTE["blue"], lw=1.2)
    expr_panel.fill_between(np.arange(len(expr)), expr, expr.min() - 0.2, color=PALETTE["blue_light"], alpha=0.55)
    expr_panel.set_xticks([])
    expr_panel.set_yticks([])
    expr_panel.set_xlabel("genes", labelpad=0)
    expr_panel.set_ylabel("RNA", labelpad=0)
    for s in expr_panel.spines.values():
        s.set_linewidth(0.55)
    ax.text(0.055, 0.22, "thousands of coordinated transcripts", fontsize=6.7, color="#444444")
    arrow(0.255, 0.38, 0.345, 0.38, PALETTE["blue"])
    box(0.35, 0.29, 0.17, 0.18, "RNA encoder", "log-expression MLP,\ngene gate and normalization", PALETTE["blue"])
    arrow(0.52, 0.38, 0.61, 0.38, PALETTE["blue"])
    box(0.615, 0.29, 0.17, 0.18, "shared space", "256-dimensional RNA–text\ncoordinates trained by contrast", PALETTE["violet"])
    arrow(0.785, 0.38, 0.875, 0.38, PALETTE["violet"])
    box(0.88, 0.29, 0.10, 0.18, "portrait", "state signals\nin language", PALETTE["green"])

    box(0.35, 0.09, 0.18, 0.13, "text encoder", "metadata-derived text\nencoded by MiniLM + MLP", PALETTE["orange"])
    arrow(0.53, 0.155, 0.615, 0.31, PALETTE["orange"])
    box(0.615, 0.09, 0.26, 0.13, "prototype attention", "organizes RNA–text coordinates into\nstate components and interpretation notes", PALETTE["green"])
    arrow(0.745, 0.22, 0.765, 0.29, PALETTE["green"])

    ax.text(
        0.055,
        0.055,
        "The analogy is conceptual: RNA profiles are not images, but both are high-dimensional patterns whose meaning is distributed across many measured elements.",
        fontsize=7.0,
        color="#333333",
    )
    add_source(
        "figure_1_schematic_note",
        "Conceptual schematic, no quantitative source table",
        "Architecture summary based on the reproduction package and trained model summaries.",
        pd.DataFrame(
            [
                {"module": "RNA encoder", "role": "maps expression vector to shared embedding"},
                {"module": "text encoder", "role": "maps metadata-derived text to shared embedding"},
                {"module": "prototype attention", "role": "organizes aligned coordinates into portrait components"},
            ]
        ),
    )
    return save_figure(fig, "Figure_1_architecture")

def figure_2_alignment() -> dict[str, str]:
    set_style()
    coords = read_csv("rich_figures/rich_expression_umap_coordinates.csv")
    cos = read_csv("T1_RNA_language_alignment/t1_cosine_summary.csv")
    exact = read_csv("T1_RNA_language_alignment/t1_exact_retrieval_metrics.csv")
    broad = read_csv("T1_RNA_language_alignment/t1_broad_semantic_retrieval.csv")
    add_source("figure_2_expression_coordinates", "rich_figures/rich_expression_umap_coordinates.csv", "Expression-space coordinates for 630 external profiles.", coords)
    add_source("figure_2_cosine_summary", "T1_RNA_language_alignment/t1_cosine_summary.csv", "Paired and shuffled RNA–text cosine summary.", cos)
    add_source("figure_2_exact_retrieval", "T1_RNA_language_alignment/t1_exact_retrieval_metrics.csv", "Exact RNA–text retrieval metrics.", exact)
    add_source("figure_2_broad_retrieval", "T1_RNA_language_alignment/t1_broad_semantic_retrieval.csv", "Broad semantic retrieval metrics.", broad)

    fig = plt.figure(figsize=(7.2, 4.45))
    gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1.22, 1.22, 0.92, 0.92], height_ratios=[1, 1], wspace=0.62, hspace=0.64)
    ax0 = fig.add_subplot(gs[:, :2])
    low_confidence_states = {"unsupported_semantics", "family_conflict", "other"}
    for state in STATE_ORDER:
        sub = coords.loc[coords["semantic_state_family"].eq(state)]
        if len(sub) == 0:
            continue
        alpha = 0.48 if state in low_confidence_states else 0.78
        size = 9 if state in low_confidence_states else 11
        ax0.scatter(sub["expr_x"], sub["expr_y"], s=size, alpha=alpha, lw=0, color=STATE_COLORS[state], label=STATE_LABELS[state])
    ax0.set_xlabel("RNA coordinate 1")
    ax0.set_ylabel("RNA coordinate 2")
    ax0.set_title("external RNA-coordinate map", fontsize=8, pad=6)
    tidy_axis(ax0, None)
    panel_label(ax0, "a")
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(
        handles,
        labels,
        title="portrait category",
        loc="upper center",
        bbox_to_anchor=(0.50, -0.105),
        ncol=4,
        handletextpad=0.25,
        columnspacing=0.75,
        markerscale=0.78,
        borderaxespad=0,
        fontsize=5.8,
        title_fontsize=5.8,
    )

    ax1 = fig.add_subplot(gs[0, 2])
    sub = cos.set_index("group").loc[["shuffled", "paired"]].reset_index()
    y = sub["mean"].to_numpy()
    yerr = np.vstack([y - sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy() - y])
    ax1.bar([0, 1], y, color=[PALETTE["grey_light"], PALETTE["blue"]], width=0.58, zorder=3)
    ax1.errorbar([0, 1], y, yerr=yerr, fmt="none", ecolor=PALETTE["black"], elinewidth=0.65, capsize=2, zorder=4)
    ax1.set_xticks([0, 1], ["shuffled", "paired"])
    ax1.set_ylabel("RNA–text cosine")
    ax1.set_ylim(0, 0.78)
    ax1.set_title("cosine similarity", fontsize=8, pad=7)
    tidy_axis(ax1)
    panel_label(ax1, "b", x=-0.30, y=1.15)

    ax2 = fig.add_subplot(gs[0, 3])
    e = exact.loc[exact["direction"].eq("rna_to_text")].copy()
    e["lift_ci_low"] = e["ci_low"] / e["random_baseline"]
    e["lift_ci_high"] = e["ci_high"] / e["random_baseline"]
    e_y = e["lift_over_random"].to_numpy(dtype=float)
    e_yerr = np.vstack([e_y - e["lift_ci_low"].to_numpy(dtype=float), e["lift_ci_high"].to_numpy(dtype=float) - e_y])
    ax2.bar(np.arange(len(e)), e["lift_over_random"], width=0.62, color=PALETTE["blue_light"], zorder=3)
    ax2.errorbar(
        np.arange(len(e)),
        e_y,
        yerr=e_yerr,
        fmt="none",
        ecolor=PALETTE["black"],
        elinewidth=0.65,
        capsize=2,
        zorder=4,
    )
    ax2.axhline(1.0, color=PALETTE["grey"], lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax2.text(2.45, 1.85, "1× random", color=PALETTE["grey"], fontsize=5.4, ha="right", va="bottom")
    for xi, ci_high, retrieval_rate in zip(np.arange(len(e)), e["lift_ci_high"].to_numpy(dtype=float), e["value"].to_numpy(dtype=float)):
        rate_label = f"{retrieval_rate * 100:.2f}%" if retrieval_rate < 0.02 else f"{retrieval_rate * 100:.1f}%"
        ax2.text(
            xi,
            ci_high + 1.25,
            rate_label,
            ha="center",
            va="bottom",
            fontsize=5.6,
            color=PALETTE["black"],
            zorder=5,
        )
    ax2.set_xticks(np.arange(len(e)), e["metric"])
    ax2.set_ylabel("retrieval lift over random (fold)")
    ax2.set_ylim(0, max(54, float(e["lift_ci_high"].max()) * 1.12))
    ax2.set_title("exact retrieval", fontsize=8, pad=7)
    tidy_axis(ax2)
    panel_label(ax2, "c", x=-0.18, y=1.15)

    ax3 = fig.add_subplot(gs[1, 2:])
    b = broad.loc[broad["k"].eq(1)].set_index("label").loc[["site", "tumor", "disease"]].reset_index()
    x = np.arange(len(b))
    w = 0.34
    ax3.bar(x - w / 2, b["random_baseline"], width=w, color=PALETTE["grey_light"], label="random", zorder=3)
    ax3.bar(x + w / 2, b["value"], width=w, color=PALETTE["blue"], label="observed", zorder=3)
    ax3.errorbar(
        x + w / 2,
        b["value"],
        yerr=np.vstack([b["value"] - b["ci_low"], b["ci_high"] - b["value"]]),
        fmt="none",
        ecolor=PALETTE["black"],
        elinewidth=0.65,
        capsize=2,
        zorder=4,
    )
    ax3.set_xticks(x, ["tissue/site", "tumour status", "disease family"])
    ax3.set_ylabel("top-1 attribute match")
    ax3.set_ylim(0, 1.03)
    ax3.legend(loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.8, bbox_to_anchor=(1.0, 1.01))
    ax3.set_title("biological attribute recovery", fontsize=8, pad=8)
    tidy_axis(ax3)
    panel_label(ax3, "d", x=-0.12, y=1.15)
    return save_figure(fig, "Figure_2_RNA_language_alignment")


def figure_4_portrait_vs_fixed_labels() -> dict[str, str]:
    set_style()
    ext = read_csv("T2_calibrated_closed_set/t2b_external_thresholds.csv")
    pool_summary = read_csv("T2_calibrated_closed_set/t2b_external_pool_summary.csv")
    transitions = read_csv("T3_A3_sample_level_transition/a3_sample_level_transitions.csv")
    fixed_x_portrait = read_csv("T8_shortcut_exclusion_controls/t8_closed_set_disease_family_x_portrait.csv")
    label_diversity = predefined_label_portrait_diversity(fixed_x_portrait)
    add_source("figure_4_thresholds", "T2_calibrated_closed_set/t2b_external_thresholds.csv", "Confidence-threshold behaviour of predefined disease labels.", ext)
    add_source("figure_4_external_pool_summary", "T2_calibrated_closed_set/t2b_external_pool_summary.csv", "External predefined-label coverage and uncertain-but-labelled rates.", pool_summary)
    add_source("figure_4_sample_transitions", "T3_A3_sample_level_transition/a3_sample_level_transitions.csv", "Sample-level transition from raw labels to portrait status.", transitions)
    add_source("figure_4_predefined_label_portraits", "T8_shortcut_exclusion_controls/t8_closed_set_disease_family_x_portrait.csv", "Predefined disease labels crossed with portrait categories.", fixed_x_portrait)
    add_source("figure_4_predefined_label_diversity", "computed from predefined disease labels crossed with portrait categories", "Portrait-category diversity within each predefined disease label.", label_diversity)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.45),
        gridspec_kw={"width_ratios": [0.95, 2.18], "height_ratios": [1.00, 1.00], "wspace": 0.42, "hspace": 0.70},
    )
    ax0 = axes[0, 0]
    ax1 = axes[1, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 1]

    sub = ext.loc[(ext["model"].eq("temperature_scaled_sgd")) & (ext["confidence_threshold"].eq(0.7))]
    sub = sub.set_index("pool").loc[["External-180", "MultiSource-450"]].reset_index()
    summ = pool_summary.loc[pool_summary["model"].eq("temperature_scaled_sgd")].set_index("pool").loc[["External-180", "MultiSource-450"]].reset_index()
    pool_labels = {
        "External-180": "180-profile\nrecount3",
        "MultiSource-450": "450-profile\nmulti-source",
    }
    x = np.arange(len(sub))
    w = 0.34
    retained_counts = sub["n_covered"].astype(int).to_numpy()
    retained_mixed_counts = np.rint(summ["highconf_overcall_70"].to_numpy(dtype=float) * summ["n"].to_numpy(dtype=float)).astype(int)
    ax0.bar(x - w / 2, sub["coverage"], width=w, color=PALETTE["blue_light"], label="retained label (all profiles)", zorder=3)
    ax0.bar(x + w / 2, summ["highconf_overcall_70"], width=w, color=PALETTE["red"], label="retained + mixed/weak (all profiles)", zorder=3)
    for xi, yval, count in zip(x - w / 2, sub["coverage"], retained_counts):
        ax0.text(xi, yval + 0.009, f"n={count}", ha="center", va="bottom", fontsize=5.8)
    for xi, yval, mixed_count, retained_count in zip(x + w / 2, summ["highconf_overcall_70"], retained_mixed_counts, retained_counts):
        ax0.text(xi, yval + 0.009, f"{mixed_count}/{retained_count}", ha="center", va="bottom", fontsize=5.8)
    ax0.set_xticks(x, [pool_labels[pool] for pool in sub["pool"]])
    ymax = max(0.235, float(max(sub["coverage"].max(), summ["highconf_overcall_70"].max())) * 1.45)
    ax0.set_ylim(0, ymax)
    ax0.set_ylabel("fraction of all profiles")
    ax0.set_title("disease-label comparator at ≥0.70")
    ax0.legend(loc="upper left", bbox_to_anchor=(-0.03, -0.24), ncol=1, handlelength=0.9, handletextpad=0.35, borderaxespad=0, fontsize=5.4)
    tidy_axis(ax0)
    panel_label(ax0, "a")

    status = transitions.groupby(["pool", "resolved_status"]).size().reset_index(name="n")
    pools = ["External-180", "MultiSource-450"]
    x_status = np.arange(len(pools))
    bottom = np.zeros(len(pools))
    status_colors = {"stable": PALETTE["green"], "mixed": PALETTE["orange_light"], "unsupported": PALETTE["grey"]}
    status_summary = []
    for status_key in ["stable", "mixed", "unsupported"]:
        vals = []
        for pool in pools:
            n = status.loc[(status["pool"].eq(pool)) & (status["resolved_status"].eq(status_key)), "n"].sum()
            total = status.loc[status["pool"].eq(pool), "n"].sum()
            vals.append(n / total if total else 0)
            status_summary.append({"pool": pool, "portrait_status": status_key, "fraction": vals[-1], "n": int(n), "total": int(total)})
        ax1.bar(x_status, vals, bottom=bottom, color=status_colors[status_key], label=STATUS_LABELS[status_key], zorder=3)
        bottom += np.array(vals)
    ax1.set_xticks(x_status, [pool_labels[pool] for pool in pools])
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("fraction of profiles")
    ax1.set_title("portrait status", pad=7)
    ax1.legend(loc="upper left", bbox_to_anchor=(-0.03, -0.24), ncol=1, handlelength=0.9, handletextpad=0.35, borderaxespad=0, fontsize=5.8)
    tidy_axis(ax1, "y")
    panel_label(ax1, "b", x=-0.14, y=1.16)
    add_source("figure_4_status_summary", "computed from T3_A3_sample_level_transition/a3_sample_level_transitions.csv", "Portrait-status fractions by external pool.", pd.DataFrame(status_summary))

    keep = label_diversity.loc[label_diversity["n_samples"].ge(10)].sort_values("n_samples", ascending=False).head(8)
    y = np.arange(len(keep))
    ax2.barh(y, keep["portrait_diversity"], color=PALETTE["violet"], zorder=3)
    ax2.set_yticks(y, [wrap_label(label, 16) for label in keep["display_label"]])
    for yi, (_, row) in enumerate(keep.iterrows()):
        ax2.text(row["portrait_diversity"] + 0.025, yi, f"n={int(row['n_samples'])}", va="center", ha="left", fontsize=5.5)
    ax2.set_xlim(0, 1.08)
    ax2.set_xlabel("normalized Shannon diversity")
    ax2.set_title("within-label portrait diversity", pad=9)
    ax2.invert_yaxis()
    tidy_axis(ax2, "x")
    panel_label(ax2, "c", x=-0.08, y=1.16)

    fx = fixed_x_portrait.copy()
    label_col = "closed_set_disease_family"
    fx["total"] = fx.drop(columns=[label_col]).sum(axis=1)
    ordered_labels = keep["predefined_disease_label"].tolist()
    fx = fx.set_index(label_col).loc[ordered_labels].reset_index()
    mat = fx.set_index(label_col)[[c for c in STATE_ORDER if c in fx.columns]]
    mat = mat.div(mat.sum(axis=1), axis=0).fillna(0)
    cmap = LinearSegmentedColormap.from_list("white_blue", ["#FFFFFF", "#D7E3F1", PALETTE["blue"]])
    im = ax3.imshow(mat.to_numpy(), aspect="equal", cmap=cmap, vmin=0, vmax=float(mat.max().max()))
    ax3.set_anchor("W")
    ax3.set_yticks(np.arange(mat.shape[0]), [wrap_label(x.replace("_", " "), 18) for x in mat.index])
    heatmap_state_labels = {
        "stable_consensus": "single\nclear",
        "hematologic_override": "blood/\nimmune",
        "epithelial_override": "epithelial",
        "clean_anchor_override": "cleaner",
        "generic_context_override": "broad",
        "unsupported_semantics": "weak",
        "family_conflict": "conflict",
        "other": "other",
    }
    ax3.set_xticks(np.arange(mat.shape[1]), [heatmap_state_labels.get(c, STATE_LABELS.get(c, c)) for c in mat.columns], rotation=55, ha="right")
    ax3.tick_params(length=0)
    ax3.tick_params(axis="x", labelsize=5.4, pad=1)
    for s in ax3.spines.values():
        s.set_visible(False)
    cax = ax3.inset_axes([1.025, 0.0, 0.030, 1.0], transform=ax3.transAxes)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_ylabel("row-normalized fraction", rotation=90)
    ax3.set_xlabel("portrait category")
    ax3.set_ylabel("")
    ax3.set_title("within-label portrait composition", pad=9)
    panel_label(ax3, "d", x=-0.08, y=1.20)
    fig.canvas.draw()
    heatmap_bbox = ax3.transData.transform_bbox(Bbox.from_extents(-0.5, -0.5, mat.shape[1] - 0.5, mat.shape[0] - 0.5))
    fig_bbox = fig.bbox
    pos2 = ax2.get_position()
    ax2.set_position([heatmap_bbox.x0 / fig_bbox.width, pos2.y0, heatmap_bbox.width / fig_bbox.width, pos2.height])
    return save_figure(fig, "Figure_4_portraits_not_single_labels")


def figure_3_biology_grounding() -> dict[str, str]:
    set_style()
    marker = read_csv("T4_marker_program_state_validation/t4_marker_scores_by_state.csv")
    pathway = read_csv("T4b_pathway_state_validation/t4b_pathway_scores_by_state.csv")
    epic_eff = read_csv("T4d_official_EPIC_deconvolution/t4d_epic_state_vs_rest_effects.csv")
    mcp_eff = read_csv("T4e_official_MCPcounter_deconvolution/t4e_mcpcounter_state_vs_rest_effects.csv")
    claims = apply_reader_claim_labels(read_csv("T7_portrait_claim_grounding/t7_claim_support_summary.csv"))
    add_source("figure_3_marker_scores", "T4_marker_program_state_validation/t4_marker_scores_by_state.csv", "Marker programme scores by portrait category.", marker)
    add_source("figure_3_pathway_scores", "T4b_pathway_state_validation/t4b_pathway_scores_by_state.csv", "Pathway scores by portrait category.", pathway)
    add_source("figure_3_epic_effects", "T4d_official_EPIC_deconvolution/t4d_epic_state_vs_rest_effects.csv", "EPIC state-versus-rest effect sizes.", epic_eff)
    add_source("figure_3_mcp_effects", "T4e_official_MCPcounter_deconvolution/t4e_mcpcounter_state_vs_rest_effects.csv", "MCP-counter state-versus-rest effect sizes.", mcp_eff)
    add_source("figure_3_statement_support", "T7_portrait_claim_grounding/t7_claim_support_summary.csv", "Statement support summary across marker, pathway and deconvolution evidence.", claims)

    marker_cols = [
        "immune_core_z",
        "t_cell_nk_z",
        "myeloid_inflammation_z",
        "hematologic_lineage_z",
        "epithelial_z",
        "stromal_ecm_z",
        "proliferation_z",
        "interferon_z",
    ]
    pathway_cols = [
        "ifn_gamma_score",
        "inflammatory_response_score",
        "t_cell_cytotoxic_score",
        "emt_stromal_score",
        "epithelial_identity_score",
        "g2m_checkpoint_score",
        "angiogenesis_score",
        "oxidative_phosphorylation_score",
    ]
    m_state = weighted_mean_table(marker, marker_cols).set_index("semantic_state_family").reindex(STATE_ORDER[:6])
    p_state = weighted_mean_table(pathway, pathway_cols).set_index("semantic_state_family").reindex(STATE_ORDER[:6])
    m_state = m_state.rename(
        columns={
            "immune_core_z": "immune",
            "t_cell_nk_z": "T/NK",
            "myeloid_inflammation_z": "myeloid",
            "hematologic_lineage_z": "blood",
            "epithelial_z": "epithelial",
            "stromal_ecm_z": "stroma",
            "proliferation_z": "prolif.",
            "interferon_z": "IFN",
        }
    ).drop(columns=["n_total"])
    p_state = p_state.rename(
        columns={
            "ifn_gamma_score": "IFN-γ",
            "inflammatory_response_score": "inflam.",
            "t_cell_cytotoxic_score": "cytotoxic",
            "emt_stromal_score": "EMT",
            "epithelial_identity_score": "epith.",
            "g2m_checkpoint_score": "G2M",
            "angiogenesis_score": "angio.",
            "oxidative_phosphorylation_score": "OXPHOS",
        }
    ).drop(columns=["n_total"])
    marker_source = m_state.reset_index()
    pathway_source = p_state.reset_index()
    add_source("figure_3_marker_heatmap_matrix", "computed weighted mean from marker scores", "Weighted marker heatmap matrix used in panel a.", marker_source)
    add_source("figure_3_pathway_heatmap_matrix", "computed weighted mean from pathway scores", "Weighted pathway heatmap matrix used in panel b.", pathway_source)

    fig = plt.figure(figsize=(7.35, 5.15))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.08, 1.03], height_ratios=[1, 1], wspace=0.86, hspace=0.74)
    ax0 = fig.add_subplot(gs[0, 0])
    heatmap(ax0, m_state, -1.1, 1.1, cbar=True, label="z score")
    ax0.set_title("marker programmes")
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    heatmap(ax1, p_state, -1.1, 1.1, cbar=True, label="module z score")
    ax1.set_title("pathway programmes")
    panel_label(ax1, "b")

    ax2 = fig.add_subplot(gs[1, 0])
    combined = pd.concat(
        [
            epic_eff.assign(source="EPIC"),
            mcp_eff.assign(source="MCP-counter"),
        ],
        ignore_index=True,
    )
    selected_scores = [
        "epic_immune_fraction_z",
        "epic_caf_fraction_z",
        "epic_endothelial_fraction_z",
        "mcp_t_cells_z",
        "mcp_cytotoxic_lymphocytes_z",
        "mcp_nk_cells_z",
        "mcp_endothelial_cells_z",
        "mcp_fibroblasts_z",
    ]
    score_labels = {
        "epic_immune_fraction_z": "EPIC\nImm.",
        "epic_caf_fraction_z": "EPIC\nCAF",
        "epic_endothelial_fraction_z": "EPIC\nEndo.",
        "mcp_t_cells_z": "MCP\nT",
        "mcp_cytotoxic_lymphocytes_z": "MCP\nCyto.",
        "mcp_nk_cells_z": "MCP\nNK",
        "mcp_endothelial_cells_z": "MCP\nEndo.",
        "mcp_fibroblasts_z": "MCP\nFib.",
    }
    dot = combined.loc[
        combined["semantic_state_family"].isin(STATE_ORDER[:5]) & combined["score"].isin(selected_scores)
    ].copy()
    dot["score_label"] = dot["score"].map(score_labels)
    dot["state_label"] = dot["semantic_state_family"].map(STATE_LABELS)
    dot["size"] = np.clip(-np.log10(dot["mannwhitney_p"].clip(lower=1e-12)), 0, 8)
    pivot_states = [STATE_LABELS[s] for s in STATE_ORDER[:5]]
    pivot_scores = [score_labels[s] for s in selected_scores]
    effect_cmap = LinearSegmentedColormap.from_list("d", ["#2F5D8C", "#F8F8F8", "#B64A4A"])
    x = dot["score_label"].map({label: idx for idx, label in enumerate(pivot_scores)}).to_numpy()
    y_dot = dot["state_label"].map({label: idx for idx, label in enumerate(pivot_states)}).to_numpy()
    sc = ax2.scatter(
        x,
        y_dot,
        s=14 + 10 * dot["size"].to_numpy(dtype=float),
        c=dot["cohen_d"].to_numpy(dtype=float),
        cmap=effect_cmap,
        vmin=-1.2,
        vmax=1.2,
        edgecolor="#333333",
        linewidth=0.25,
        zorder=3,
    )
    ax2.set_xticks(np.arange(len(pivot_scores)), pivot_scores, rotation=0, ha="center")
    ax2.set_yticks(np.arange(len(pivot_states)), pivot_states)
    ax2.set_title("deconvolution checks")
    ax2.set_xlim(-0.6, len(pivot_scores) - 0.4)
    ax2.set_ylim(len(pivot_states) - 0.5, -0.5)
    cb = plt.colorbar(sc, ax=ax2, fraction=0.034, pad=0.02)
    cb.ax.set_ylabel("Cohen's d", rotation=90)
    size_handles = [
        ax2.scatter([], [], s=14 + 10 * value, facecolor="#F8F8F8", edgecolor="#333333", linewidth=0.25)
        for value in [1, 3, 6]
    ]
    ax2.legend(
        size_handles,
        ["1", "3", "6"],
        title="−log10(P)",
        loc="upper left",
        bbox_to_anchor=(1.18, 1.02),
        borderaxespad=0,
        handletextpad=0.8,
    )
    tidy_axis(ax2, None)
    panel_label(ax2, "c")

    ax3 = fig.add_subplot(gs[1, 1])
    claims = claims.sort_values("partial_or_strong_rate")
    y = np.arange(len(claims))
    claims["partial_only_rate"] = (claims["partial_or_strong_rate"] - claims["strong_rate"]).clip(lower=0)
    claim_short_labels = {
        "epithelial or tumour-like signal": "epithelial/\ntumour-like",
        "cleaner or non-malignant context": "cleaner/\nnon-malignant",
        "mixed or inconsistent label reading": "mixed/\ninconsistent",
        "immune or blood signal": "immune/\nblood",
        "stromal or broad-context signal": "stromal/\nbroad-context",
    }
    ax3.barh(y, claims["partial_only_rate"], color=PALETTE["green_light"], label="partial support", zorder=3)
    ax3.barh(y, claims["strong_rate"], left=claims["partial_only_rate"], color=PALETTE["green"], label="strong support", zorder=3)
    for yi, (_, row) in zip(y, claims.iterrows()):
        ax3.text(
            min(float(row["partial_or_strong_rate"]) + 0.025, 0.96),
            yi,
            f"n={int(row['n_claim_rows'])}",
            va="center",
            ha="left",
            fontsize=5.9,
            color="#444444",
        )
    ax3.set_yticks(y, [claim_short_labels.get(label, wrap_label(label, 16)) for label in claims["claim_label"]])
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("fraction with partial/strong support")
    ax3.set_title("portrait-word statements")
    ax3.legend(loc="lower right", bbox_to_anchor=(1.0, -0.03), ncol=1)
    tidy_axis(ax3, "x")
    panel_label(ax3, "d")
    return save_figure(fig, "Figure_3_biological_grounding")


def figure_5_stress_tests() -> dict[str, str]:
    set_style()
    mixing = read_csv("T5c_expanded_calibrated_mixing_bootstrap/t5c_fraction_summary.csv")
    partial = read_csv("T9_strict_shortcut_residual_controls/t9_partial_r2_after_controls.csv")
    shortcut = read_csv("T8_shortcut_exclusion_controls/t8_shortcut_association.csv")
    robust = read_csv("T10_quality_heterogeneity_robustness/t10_subset_robustness_summary.csv")
    boundary = read_csv("T11_failure_mode_reliability_boundaries/t11_boundary_flag_summary.csv")
    boundary_display = boundary.copy()
    if "boundary_label" in boundary_display:
        boundary_display["boundary_label"] = boundary_display["boundary_label"].map(normalize_visible_terms)
    add_source("figure_5_mixing", "T5c_expanded_calibrated_mixing_bootstrap/t5c_fraction_summary.csv", "Controlled RNA-signal mixing summary with bootstrap intervals.", mixing)
    add_source("figure_5_partial_r2", "T9_strict_shortcut_residual_controls/t9_partial_r2_after_controls.csv", "Portrait incremental R² after controlling source and metadata factors.", display_label_copy(partial))
    add_source("figure_5_metadata_association", "T8_shortcut_exclusion_controls/t8_shortcut_association.csv", "Association between metadata factors and portrait categories.", display_label_copy(shortcut))
    add_source("figure_5_robustness", "T10_quality_heterogeneity_robustness/t10_subset_robustness_summary.csv", "Robustness across data-quality subsets.", robust)
    add_source("figure_5_reliability_flags", "T11_failure_mode_reliability_boundaries/t11_boundary_flag_summary.csv", "Reliability flags.", boundary_display)

    # Keep all robustness subsets legible in the fixed manuscript layout.
    fig = plt.figure(figsize=(7.2, 6.30))
    gs = gridspec.GridSpec(
        2,
        12,
        figure=fig,
        width_ratios=[1] * 12,
        height_ratios=[1, 1.85],
        wspace=0.58,
        hspace=0.42,
    )

    ax0 = fig.add_subplot(gs[0, 0:3])
    mix = mixing.loc[mixing["model"].eq("temperature_scaled_sgd")].copy()
    design_colors = {"normal_immune": PALETTE["green"], "tumor_immune": PALETTE["orange"], "tumor_normal": PALETTE["red"]}
    # fraction_b runs from the pure normal endpoint to the pure tumour endpoint.
    design_labels = {"normal_immune": "normal + immune", "tumor_immune": "tumour + immune", "tumor_normal": "normal + tumour"}
    for design, g in mix.groupby("design"):
        g = g.sort_values("fraction_b")
        color = design_colors.get(design, PALETTE["grey"])
        ax0.plot(g["fraction_b"], g["openworld_mixed_or_unsupported_rate"], marker="o", ms=3.2, lw=1.15, color=color, label=design_labels.get(design, design))
        ax0.fill_between(
            g["fraction_b"].to_numpy(dtype=float),
            g["openworld_mixed_or_unsupported_rate_lo"].to_numpy(dtype=float),
            g["openworld_mixed_or_unsupported_rate_hi"].to_numpy(dtype=float),
            color=color,
            alpha=0.16,
            lw=0,
        )
    ax0.set_xlabel("mixing fraction")
    ax0.set_ylabel("mixed-signal or weak-evidence rate")
    ax0.set_ylim(0, 1.05)
    ax0.set_xlim(-0.03, 1.12)
    ax0.set_title("controlled profile mixing")
    ax0.legend(loc="lower right", bbox_to_anchor=(0.99, 0.04), ncol=1, handlelength=1.0, handletextpad=0.35, borderaxespad=0, fontsize=5.4)
    tidy_axis(ax0)
    panel_label(ax0, "a", x=-0.22, y=1.18)

    ax1 = fig.add_subplot(gs[0, 4:7])
    nmi = shortcut.loc[shortcut["target"].eq("semantic_state_family")].copy()
    nmi = nmi.loc[nmi["factor"].isin(["pool", "project_prefix", "project", "expected_site_family", "expected_disease_family", "closed_set_disease_family"])]
    nmi["label"] = nmi["factor_label"].replace(
        {
            "external pool": "evaluation pool",
            "source prefix": "source prefix",
            "project/source": "project",
            "expected tissue/site": "tissue/site",
            "expected disease label": "expected disease",
            "fixed disease label": "predefined disease",
        }
    )
    nmi = nmi.sort_values("nmi")
    y = np.arange(len(nmi))
    colors = [PALETTE["grey"] if x != "project" else PALETTE["red_light"] for x in nmi["label"]]
    ax1.barh(y, nmi["nmi"], color=colors, zorder=3)
    direct_labels = {
        "source prefix": "source\nprefix",
        "expected disease": "expected\ndisease",
        "predefined disease": "predefined\ndisease",
        "evaluation pool": "evaluation\npool",
    }
    ax1.set_yticks(y)
    ax1.set_yticklabels([direct_labels.get(label, label) for label in nmi["label"]])
    ax1.tick_params(axis="y", length=3.0, width=0.8, direction="out", pad=2, labelsize=5.6)
    ax1.set_xlim(0, max(0.34, float(nmi["nmi"].max()) * 1.10))
    ax1.set_xlabel("NMI with portraits")
    ax1.set_title("metadata links")
    tidy_axis(ax1, "x")
    panel_label(ax1, "b", x=-0.22, y=1.18)

    ax2 = fig.add_subplot(gs[0, 8:11])
    controls = ["source_prefix", "expected_site", "expected_disease", "fixed_disease", "source_site_disease"]
    axes = ["immune_support", "context_support", "tumor_like_support", "mixed_evidence_support", "clean_context_support"]
    mat = (
        partial.loc[partial["control_id"].isin(controls) & partial["evidence_axis"].isin(axes)]
        .pivot(index="control_label", columns="evidence_axis_label", values="portrait_incremental_r2")
        .rename(index={"fixed disease": "predefined disease"})
        .reindex(["source + prefix", "tissue/site", "expected disease", "predefined disease", "source + site + disease"])
    )
    mat = mat.rename(columns=lambda c: normalize_visible_terms(str(c)).replace("clean context", "clean-context").replace("mixed signal", "mixed-signal"))
    im = ax2.imshow(mat.to_numpy(), aspect="auto", cmap=LinearSegmentedColormap.from_list("white_blue", ["#FFFFFF", PALETTE["blue_soft"], PALETTE["blue"]]), vmin=0, vmax=0.22)
    ax2.set_xticks(np.arange(mat.shape[1]), [wrap_label(c, 10) for c in mat.columns], rotation=45, ha="right")
    ax2.set_yticks(np.arange(mat.shape[0]), [wrap_label(i, 15) for i in mat.index])
    ax2.tick_params(length=0)
    for s in ax2.spines.values():
        s.set_visible(False)
    cax = ax2.inset_axes([1.05, 0.0, 0.045, 1.0], transform=ax2.transAxes)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_ylabel("incremental R²", rotation=90)
    joint_control = partial.loc[partial["control_id"].eq("source_site_disease") & partial["evidence_axis"].isin(axes)]
    joint_mean = float(joint_control["portrait_incremental_r2"].mean())
    joint_p = float(joint_control["empirical_p_greater"].max())
    ax2.text(
        0.98,
        0.08,
        f"mean ΔR² = {joint_mean:.3f}\nempirical P = {joint_p:.3f}",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=PALETTE["black"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D8D8D8", linewidth=0.45, alpha=0.92),
        zorder=5,
    )
    ax2.set_title("after source controls")
    panel_label(ax2, "c", x=-0.12, y=1.18)
    add_source("figure_5_partial_r2_matrix", "computed from source and metadata controls", "Matrix used in panel c.", mat.reset_index())

    ax3 = fig.add_subplot(gs[1, 0:4])
    # Report every prespecified robustness subset, including negative controls.
    keep = robust.sort_values("mean_incremental_r2").copy()
    def display_robustness_subset(value: str) -> str:
        return (
            str(value)
            .replace("External-180", "180-profile recount3")
            .replace("MultiSource-450", "450-profile multi-source")
            .replace("External 180", "180-profile recount3")
            .replace("MultiSource 450", "450-profile multi-source")
        )

    ax3.barh(keep["subset_label"].map(lambda x: wrap_label(display_robustness_subset(x), 26)), keep["mean_incremental_r2"], color=PALETTE["blue"], zorder=3)
    ax3.set_xlabel("mean incremental R²")
    ax3.set_title("robustness subsets")
    ax3.tick_params(axis="y", labelsize=5.8)
    tidy_axis(ax3, "x")
    panel_label(ax3, "d", x=-0.18, y=1.18)

    ax4 = fig.add_subplot(gs[1, 6:12])
    b = boundary_display.sort_values("fraction_samples", ascending=True)
    labels = b["boundary_label"].map(lambda x: wrap_label(x, 15))
    frequent_flags = set(boundary_display.nlargest(2, "fraction_samples")["boundary_label"])
    b_colors = [PALETTE["red_light"] if label in frequent_flags else PALETTE["grey"] for label in b["boundary_label"]]
    ax4.barh(labels, b["fraction_samples"], color=b_colors, zorder=3)
    ax4.set_xlim(0, max(0.62, float(b["fraction_samples"].max()) * 1.15))
    ax4.set_xlabel("fraction of external profiles")
    ax4.set_title("reliability flags")
    ax4.tick_params(axis="y", labelsize=5.8)
    tidy_axis(ax4, "x")
    panel_label(ax4, "e", x=-0.12, y=1.18)
    return save_figure(fig, "Figure_5_stress_tests_and_reliability")


def extended_data_1_local_parts() -> dict[str, str]:
    set_style()
    cv = read_csv("T12_whole_profile_vs_local_parts/t12_cv_model_comparison.csv")
    reduc = read_csv("T12_whole_profile_vs_local_parts/t12_portrait_reducibility.csv")
    add_source("extended_data_1_cv_comparison", "T12_whole_profile_vs_local_parts/t12_cv_model_comparison.csv", "Cross-validated derived-feature and portrait comparisons.", cv)
    add_source("extended_data_1_portrait_reducibility", "T12_whole_profile_vs_local_parts/t12_portrait_reducibility.csv", "Prediction of portrait families from labels and expression-derived features.", reduc)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), gridspec_kw={"width_ratios": [1.35, 0.95]})
    ax0 = axes[0]
    outcomes = ["immune_support", "context_support", "tumor_like_support", "mixed_evidence_support", "boundary_flag_count"]
    labels = ["immune", "context", "tumour-like", "mixed-signal", "reliability-flag count"]
    pred_order = ["labels_only", "local_parts_only", "labels_plus_local_parts", "labels_local_plus_portrait"]
    legend_labels = {
        "labels_only": "Disease labels",
        "local_parts_only": "Molecular features",
        "labels_plus_local_parts": "Labels + molecular features",
        "labels_local_plus_portrait": "Labels + molecular features + portrait",
    }
    colors = [PALETTE["grey_light"], PALETTE["blue_light"], PALETTE["blue"], PALETTE["green"]]
    x = np.arange(len(outcomes))
    width = 0.18
    for i, pred in enumerate(pred_order):
        vals = [cv.loc[(cv["outcome"].eq(o)) & (cv["predictor_set"].eq(pred)), "cv_r2"].mean() for o in outcomes]
        ax0.bar(x + (i - 1.5) * width, vals, width=width, color=colors[i], label=legend_labels[pred], zorder=3)
    ax0.set_xticks(x, labels, rotation=35, ha="right")
    ax0.set_ylabel("cross-validated R²")
    ax0.set_xlim(-0.6, len(outcomes) + 3.4)
    ax0.set_title("support-measure prediction", pad=8)
    ax0.legend(loc="upper right", bbox_to_anchor=(0.99, 0.98), ncol=1, labelspacing=0.25, handlelength=1.1, handletextpad=0.35, borderaxespad=0, fontsize=5.0)
    tidy_axis(ax0)
    panel_label(ax0, "a")

    ax1 = axes[1]
    reduc = reduc.set_index("predictor_set").loc[["majority_baseline", "labels_only", "local_parts_only", "labels_plus_local_parts"]].reset_index()
    reduc["display_label"] = reduc["predictor_set"].replace(
        {
            "majority_baseline": "Majority baseline",
            "labels_only": "Disease labels",
            "local_parts_only": "Molecular features",
            "labels_plus_local_parts": "Labels + molecular features",
        }
    )
    reduc_colors = {
        "majority_baseline": PALETTE["grey"],
        "labels_only": PALETTE["grey_light"],
        "local_parts_only": PALETTE["blue_light"],
        "labels_plus_local_parts": PALETTE["blue"],
    }
    ax1.barh(reduc["display_label"], reduc["macro_f1"], color=[reduc_colors[p] for p in reduc["predictor_set"]], zorder=3)
    for yi, value in enumerate(reduc["macro_f1"]):
        ax1.text(value + 0.012, yi, f"{value:.3f}", va="center", ha="left", fontsize=5.8, color=PALETTE["black"])
    ax1.set_xlim(0, 0.55)
    ax1.set_xlabel("macro F1")
    ax1.set_title("portrait-family reconstruction", pad=8)
    tidy_axis(ax1, "x")
    panel_label(ax1, "b")
    fig.tight_layout(w_pad=1.2)
    return save_figure(fig, "Extended_Data_1_local_parts_reliability")


def normalize_visible_terms(text: str) -> str:
    replacements = [
        ("RNA-text", "RNA–text"),
        ("tumor-like", "tumour-like"),
        ("Tumor-like", "Tumour-like"),
        ("several-signal or weak-evidence", "mixed-signal or weak-evidence"),
        ("Several-signal or weak-evidence", "Mixed-signal or weak-evidence"),
        ("several-signal or weak-evidence", "mixed-signal or weak-evidence"),
        ("Several-signal or weak-evidence", "Mixed-signal or weak-evidence"),
        ("several signals", "mixed signal"),
        ("several-signal", "mixed-signal"),
        ("Several-signal", "Mixed-signal"),
        ("multi-signal flags", "mixed-signal flags"),
        ("multi-signal flag", "mixed-signal flag"),
        ("multi-signal boundaries", "mixed-signal flags"),
        ("multi-signal boundary", "mixed-signal flag"),
        ("multi-signal", "mixed-signal"),
        ("scored claim rows", "scored statements"),
        ("portrait-word claims", "portrait-word statements"),
        ("textual portrait claims", "textual portrait-word statements"),
        ("portrait claims", "portrait-word statements"),
        ("immune or blood claims", "immune or blood statements"),
        ("mixed-signal claims", "mixed-signal statements"),
        ("stromal or broader-context claims", "stromal or broad-context statements"),
        ("stromal or broader-context statements", "stromal or broad-context statements"),
        ("stromal or broader-context signal", "stromal or broad-context signal"),
        ("low claim support", "low statement support"),
        ("Low claim support", "Low statement support"),
        ("low claim", "low statement support"),
        ("Low claim", "Low statement support"),
        ("claim support", "statement support"),
        ("Claim support", "Statement support"),
        ("cleaner anchor context", "cleaner-anchor context"),
        ("cleaner anchor evidence", "cleaner-anchor evidence"),
        ("cleaner anchor", "cleaner-anchor"),
        ("weak evidence readings", "weak-evidence readings"),
        ("weak evidence portrait", "weak-evidence portrait"),
        ("single-clear-signal", "single clear signal"),
        ("broader-context statements", "broad-context statements"),
        ("broader context", "broad-context"),
        ("broad context", "broad-context"),
        ("caution flags", "interpretation notes"),
        ("state components and cautions", "state components and interpretation notes"),
        ("interpretation cautions", "interpretation notes"),
        ("caution boundary", "cautionary tier"),
        ("Boundary flag fractions", "Reliability-flag fractions"),
        ("boundary flag fractions", "reliability-flag fractions"),
        ("Boundary flags", "Reliability flags"),
        ("boundary flags", "reliability flags"),
        ("boundary flag", "reliability flag"),
        ("Boundary statement", "Scope note"),
        ("Interpretation boundary", "Interpretation note"),
        ("interpretation boundary", "interpretation note"),
        ("Reliability boundaries", "Reliability flags"),
        ("reliability boundaries", "reliability flags"),
        ("data-quality boundaries", "data-quality limits"),
        ("cautious-use boundaries", "cautious-use limits"),
        ("bounded statement", "scoped statement"),
        ("boundary count", "reliability-flag count"),
        ("Prior work boundary", "Prior work scope"),
        ("Discussion boundary", "Discussion scope"),
        ("quality boundary", "quality limit"),
        ("180-profile recount3 set", "180-profile recount3-derived validation set"),
        ("180-profile recount3 validation set", "180-profile recount3-derived validation set"),
        ("recount3-only 180-profile subset", "180-profile recount3-derived validation subset"),
        ("450-profile multi-source set", "450-profile multi-source public benchmark"),
        ("450-profile public benchmark", "450-profile multi-source public benchmark"),
        ("450-profile multi-source benchmark", "450-profile multi-source public benchmark"),
        ("broader external sets", "external evaluation sets"),
        ("combined external set", "combined external evaluation set"),
        ("portrait groups", "portrait families"),
        ("portrait group", "portrait family"),
        ("Portrait groups", "Portrait families"),
        ("Portrait group", "Portrait family"),
        ("portrait outputs", "portrait labels"),
        ("portrait output", "portrait label"),
        ("portrait target", "portrait label"),
    ]
    for source, target in replacements:
        text = text.replace(source, target)
    text = re.sub(r"(?<![A-Za-z0-9])R2(?![A-Za-z0-9])", "R²", text)
    return text
