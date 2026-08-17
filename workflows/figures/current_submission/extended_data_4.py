#!/usr/bin/env python3
"""Rebuild Extended Data Fig. 4 at its final print size.

The script reads the released heatmap panel source, redraws axes and labels at
the final 180-mm size, and places the two bar panels in the remaining space.
"""

from __future__ import annotations

import base64
import io
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


FIGS = Path(os.environ["RNA_PORTRAIT_FIGURE_OUTPUT_ROOT"]).resolve()
PANEL_SOURCES = Path(os.environ["RNA_PORTRAIT_PANEL_SOURCE_DIR"]).resolve()
ARCHIVE = PANEL_SOURCES / "Extended_Data_Fig_4_pre_x_label_wrap_20260812.svg"
OUT = FIGS / "Extended_Data_Fig_4"
SOURCE_DATA = Path(os.environ["RNA_PORTRAIT_SOURCE_DATA_DIR"]).resolve()
FIGS.mkdir(parents=True, exist_ok=True)

STATE_LABELS = [
    "single clear signal", "blood/immune", "epithelial-like context",
    "cleaner-anchor context", "broad-context", "weak-evidence", "conflict", "other",
]
MARKER_LABELS = [
    "immune core", "T cell NK", "myeloid inflammation", "haematologic lineage",
    "epithelial", "stromal ECM", "proliferation", "interferon", "liver metabolic", "neural",
]
PATHWAY_LABELS = [
    "IFN α", "IFN γ", "TNFα NFκB", "inflammatory response",
    "complement", "T cell cytotoxic", "myeloid activation", "EMT stromal",
    "epithelial identity", "G2M checkpoint", "E2F targets", "hypoxia",
    "angiogenesis", "oxidative phosphorylation", "fatty acid metabolism",
    "xenobiotic metabolism",
]
DECONV_LABELS = [
    "EPIC CAF fraction", "EPIC endothelial fraction", "EPIC immune fraction",
    "MCP cytotoxic lymphocytes", "MCP endothelial cells", "MCP fibroblasts",
    "MCP NK cells", "MCP T cells",
]


def increase_multiline_label_spacing(fig, increment=0.12):
    """Apply a small, uniform leading increase to wrapped figure labels."""
    seen = set()

    def adjust(text):
        if id(text) in seen:
            return
        seen.add(id(text))
        if "\n" in text.get_text():
            text.set_linespacing(getattr(text, "_linespacing", 1.2) + increment)

    for text in fig.texts:
        adjust(text)
    for ax in fig.axes:
        for text in [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label, ax.yaxis.label,
                     ax.title, ax._left_title, ax._right_title]:
            adjust(text)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                adjust(text)
    for legend in fig.legends:
        for text in legend.get_texts():
            adjust(text)


def embedded_heatmaps() -> list[np.ndarray]:
    """Return the three archived heatmap rasters in panel order (a, b, c)."""
    root = ET.fromstring(ARCHIVE.read_bytes())
    href_key = "{http://www.w3.org/1999/xlink}href"
    images = []
    for node in root.iter("{http://www.w3.org/2000/svg}image"):
        width = float(node.attrib.get("width", 0))
        height = float(node.attrib.get("height", 0))
        href = node.attrib.get(href_key, "")
        if width < 100 or height < 100 or not href.startswith("data:image/png;base64,"):
            continue
        raw = base64.b64decode(re.sub(r"\s+", "", href.split(",", 1)[1]))
        images.append(np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")))
    if len(images) != 3:
        raise RuntimeError(f"Expected three heatmap rasters in {ARCHIVE.name}; found {len(images)}")
    return images


def add_heatmap(fig, rect, image, xlabels, ylabels, title, cbar_label, label_rotation, label_size):
    ax = fig.add_axes(rect)
    # The archived image stores one coloured block as many raster pixels.
    # Giving it categorical data-space bounds makes the tick locations align
    # with the block centres rather than with the first 8–16 image pixels.
    ax.imshow(image, interpolation="nearest", aspect="auto",
              extent=(-0.5, len(xlabels) - 0.5, len(ylabels) - 0.5, -0.5))
    ax.set_title(title, fontsize=7.4, pad=5)
    ax.set_xticks(np.arange(len(xlabels)))
    labels = ax.set_xticklabels(xlabels, rotation=label_rotation, ha="right", va="top", fontsize=label_size)
    for label in labels:
        label.set_linespacing(0.72)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=5.0)
    ax.tick_params(axis="x", length=2.5, width=0.45, pad=1.2)
    ax.tick_params(axis="y", length=2.5, width=0.45, pad=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)

    x, y, w, h = rect
    cax = fig.add_axes([x + w + 0.012, y, 0.012, h])
    cmap = LinearSegmentedColormap.from_list("blue_white_red", ["#2F5D8C", "#F8F8F8", "#B64A4A"])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-1.1, 1.1), cmap=cmap), cax=cax)
    cb.set_label(cbar_label, fontsize=5.6, labelpad=3)
    cb.ax.tick_params(labelsize=5.2, width=0.45, length=2.5)
    cb.outline.set_linewidth(0.55)


def add_cv_bar_panel(fig, rect):
    """Panel d, rebuilt from its exact source table at its final size."""
    ax = fig.add_axes(rect)
    data = pd.read_csv(SOURCE_DATA / "extended_data_1_cv_comparison.csv")
    outcomes = ["immune_support", "context_support", "tumor_like_support", "mixed_evidence_support", "boundary_flag_count"]
    labels = ["immune", "context", "tumour-like", "mixed-signal", "boundary count"]
    predictors = ["labels_only", "local_parts_only", "labels_plus_local_parts", "labels_local_plus_portrait"]
    colors = ["#D9D9D9", "#9DB9D8", "#1F5A9D", "#5B8C5A"]
    x = np.arange(len(outcomes))
    for i, (predictor, color) in enumerate(zip(predictors, colors)):
        values = [data.loc[(data.outcome == outcome) & (data.predictor_set == predictor), "cv_r2"].iloc[0]
                  for outcome in outcomes]
        label = data.loc[data.predictor_set == predictor, "predictor_label"].iloc[0]
        ax.bar(x + (i - 1.5) * 0.18, values, width=0.18, color=color, label=label, zorder=3)
    ax.set_title("Local evidence and portrait information", fontsize=6.2, pad=3)
    ax.set_ylabel("cross-validated R2", fontsize=5.0)
    ax.set_xticks(x, labels, rotation=28, ha="right", fontsize=5.0)
    ax.tick_params(axis="y", labelsize=5.0, width=0.45, length=2.5)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.55)
    # Return the key so it can sit in the reserved whitespace below panel d,
    # rather than covering the tallest grouped bars.
    return ax.get_legend_handles_labels()


def add_reducibility_bar_panel(fig, rect):
    """Panel e, rebuilt from its exact source table without raster stretching."""
    ax = fig.add_axes(rect)
    data = pd.read_csv(SOURCE_DATA / "extended_data_1_portrait_reducibility.csv")
    order = ["majority_baseline", "labels_only", "local_parts_only", "labels_plus_local_parts"]
    plot = data.set_index("predictor_set").loc[order].reset_index()
    ax.barh(plot["predictor_label"], plot["macro_f1"],
            color=["#D9D9D9", "#9DB9D8", "#1F5A9D", "#5B8C5A"], zorder=3)
    ax.set_xlim(0, 0.55)
    ax.set_xlabel("macro F1", fontsize=5.2)
    ax.set_title("Portrait-group reducibility", fontsize=6.5, pad=3)
    ax.tick_params(axis="both", labelsize=5.0, width=0.45, length=2.5)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.55)


def main() -> None:
    heat_a, heat_b, heat_c = embedded_heatmaps()

    # Direct final-size layout: 180 mm wide x 166 mm high. Both top heatmaps
    # use single-line 45-degree x-axis labels; panel b retains Greek symbols.
    fig = plt.figure(figsize=(180 / 25.4, 166 / 25.4), facecolor="white")
    add_heatmap(fig, [0.14, 0.725, 0.27, 0.195], heat_a, MARKER_LABELS, STATE_LABELS,
                "Full marker-programme grounding", "weighted mean z score", 45, 5.2)
    add_heatmap(fig, [0.605, 0.725, 0.315, 0.195], heat_b, PATHWAY_LABELS, STATE_LABELS,
                "Full pathway-programme grounding", "weighted mean score", 45, 5.15)
    add_heatmap(fig, [0.14, 0.365, 0.27, 0.185], heat_c, DECONV_LABELS, STATE_LABELS[:5],
                "Deconvolution effect sizes", "Cohen d", 45, 5.2)
    handles, labels = add_cv_bar_panel(fig, [0.575, 0.39, 0.35, 0.15])
    add_reducibility_bar_panel(fig, [0.14, 0.075, 0.76, 0.135])

    # Dedicated legend strip in the empty band between panels d and e.
    legend_ax = fig.add_axes([0.55, 0.285, 0.38, 0.052])
    legend_ax.set_axis_off()
    legend_ax.legend(handles, labels, loc="center", ncol=2, fontsize=5.0, frameon=False,
                     handlelength=1.1, handletextpad=.35, columnspacing=.8, labelspacing=.3)

    for letter, x, y in [("a", 0.02, 0.96), ("b", 0.53, 0.96), ("c", 0.02, 0.59),
                         ("d", 0.53, 0.59), ("e", 0.02, 0.275)]:
        fig.text(x, y, letter, fontsize=9, fontweight="bold", va="top", ha="left")

    increase_multiline_label_spacing(fig)
    fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, facecolor="white")
    raster = io.BytesIO()
    fig.savefig(raster, format="png", dpi=600, facecolor="white")
    plt.close(fig)
    raster.seek(0)
    Image.open(raster).convert("RGB").save(OUT.with_suffix(".tiff"), compression="tiff_lzw", dpi=(600, 600))
    print("redrew Extended_Data_Fig_4 with Matplotlib: 180.0 x 166.0 mm")


if __name__ == "__main__":
    main()
