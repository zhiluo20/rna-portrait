#!/usr/bin/env python3
"""Rebuild Extended Data Figures 1–3 and 5–7 as Matplotlib plates.

Panels use released tabular source data. Values or raster marks that are only
represented in released vector-panel sources are read from those files, while
labels and composite layout are drawn as editable text.
"""

from __future__ import annotations

import base64
import io
import os
import re
import textwrap
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import pandas as pd
from PIL import Image


DATA = Path(os.environ["RNA_PORTRAIT_SOURCE_DATA_DIR"]).resolve()
PANELS = Path(os.environ["RNA_PORTRAIT_PANEL_SOURCE_DIR"]).resolve()
FIGURES = Path(os.environ["RNA_PORTRAIT_FIGURE_OUTPUT_ROOT"]).resolve()
FIGURES.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7.4,
        "axes.labelsize": 6.8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
    }
)

BLUE = "#1f5a9d"
GREEN = "#5b8c5a"
# Fixed colours shared by the released panel sources.
ORANGE = "#c9822b"
RED = "#b64a4a"
PURPLE = "#7e6aae"
LIGHT_BLUE = "#9db9d8"
GRID = "#e8e8e8"


def source(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def wrapped(text: object, width: int = 16) -> str:
    text = str(text).replace("_", " ").replace("-", "-")
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


DISPLAY = {
    "stable_consensus": "single clear signal",
    "hematologic_override": "blood/immune",
    "epithelial_override": "epithelial-like context",
    "clean_anchor_override": "clean-anchor context",
    "generic_context_override": "context",
    "generic_context": "broad context",
    "unsupported_semantics": "unsupported",
    "family_conflict": "conflict",
    "unsupported": "unsupported",
    "conflict": "conflict",
    "other": "other",
    "normal_immune": "normal–immune",
    "tumor_immune": "tumour–immune",
    "normal_tumor": "normal–tumour",
    "tumor_normal": "tumour–normal",
    "temperature_scaled_sgd": "temperature-scaled",
    "sigmoid_calibrated": "sigmoid-calibrated",
    "uncalibrated_sgd": "uncalibrated",
    "isotonic_calibrated": "isotonic-calibrated",
}


def label(value: object, width: int | None = None) -> str:
    out = DISPLAY.get(str(value), str(value).replace("_", " "))
    return wrapped(out, width) if width else out


def style_axes(ax, grid: str = "y") -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def panel_letter(ax, letter: str, y: float = 1.12, x: float = -0.09) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left", clip_on=False)


def increase_multiline_label_spacing(fig: plt.Figure, increment: float = .12) -> None:
    """Apply a small, uniform leading increase to wrapped figure labels."""
    seen: set[int] = set()

    def adjust(text) -> None:
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


def finish(fig: plt.Figure, number: int) -> None:
    """Export each plate in editable-vector and submission-raster forms."""
    increase_multiline_label_spacing(fig)
    base = FIGURES / f"Extended_Data_Fig_{number}"
    fig.savefig(base.with_suffix(".svg"), format="svg", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), format="pdf", facecolor="white")
    fig.savefig(base.with_suffix(".png"), format="png", dpi=320, facecolor="white")
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=600, facecolor="white")
    png.seek(0)
    Image.open(png).convert("RGB").save(
        base.with_suffix(".tiff"), compression="tiff_lzw", dpi=(600, 600)
    )
    plt.close(fig)


def svg_paths(panel_name: str, color: str) -> list[np.ndarray]:
    root = ET.parse(PANELS / panel_name).getroot()
    ns = "{http://www.w3.org/2000/svg}"
    found: list[np.ndarray] = []
    for node in root.iter(f"{ns}path"):
        style = node.get("style", "")
        if f"stroke: {color}" not in style or "fill: none" not in style:
            continue
        pairs = re.findall(r"[ML]\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)", node.get("d", ""))
        if len(pairs) > 3:
            found.append(np.asarray(pairs, dtype=float))
    # The long first line is the data series; short lines are legend keys.
    return sorted(found, key=len, reverse=True)


def svg_series(panel_name: str, y_max: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Recover vector points from the original two calibration panels."""
    x0, x1 = (32.31625, 218.31625) if panel_name.startswith("S4a") else (32.41, 218.41)
    y0, y1 = 181.642031, 15.322031
    out = {}
    for color, model in [("#8f8f8f", "uncalibrated_sgd"), (BLUE, "temperature_scaled_sgd"), (GREEN, "sigmoid_calibrated")]:
        points = svg_paths(panel_name, color)[0]
        x = (points[:, 0] - x0) / (x1 - x0)
        y = (y0 - points[:, 1]) / (y0 - y1) * y_max
        out[model] = (x, y)
    return out


def svg_bars(panel_name: str, colors: list[str], x0: float, x1: float) -> dict[str, list[tuple[float, float]]]:
    """Extract horizontal bar centres and widths from a vector source panel."""
    root = ET.parse(PANELS / panel_name).getroot()
    ns = "{http://www.w3.org/2000/svg}"
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    data_max = 6 if panel_name.startswith("S8c") else 18 if panel_name.startswith("S9b") else 42
    for parent in root.iter(f"{ns}g"):
        match = re.fullmatch(r"patch_(\d+)", parent.get("id", ""))
        # Matplotlib emits data rectangles first.  Later patches are legend keys;
        # patch_3 through patch_42 covers the largest stacked-bar source panel.
        if not match or not 3 <= int(match.group(1)) <= data_max:
            continue
        node = next((child for child in parent if child.tag == f"{ns}path"), None)
        if node is None:
            continue
        style = node.get("style", "")
        color = next((value for value in colors if f"fill: {value}" in style), None)
        if color is None:
            continue
        nums = np.asarray(re.findall(r"[-+0-9.eE]+", node.get("d", "")), dtype=float)
        if len(nums) < 8 or len(nums) % 2:
            continue
        xs, ys = nums[::2], nums[1::2]
        if np.ptp(ys) < 0.01:
            continue
        grouped[color].append((float(np.mean(ys)), float(np.ptp(xs) / (x1 - x0))))
    return {key: sorted(value, reverse=True) for key, value in grouped.items()}


def embedded_image(panel_name: str) -> Image.Image:
    root = ET.parse(PANELS / panel_name).getroot()
    ns = "{http://www.w3.org/2000/svg}"
    candidates: list[tuple[float, str]] = []
    for node in root.iter(f"{ns}image"):
        href = node.get("{http://www.w3.org/1999/xlink}href") or node.get("href") or ""
        if href.startswith("data:image"):
            candidates.append((float(node.get("width", 0)) * float(node.get("height", 0)), href))
    encoded = max(candidates, key=lambda x: x[0])[1].split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def svg_scatter_source(panel_name: str) -> tuple[list[dict[str, object]], list[tuple[str, str]], np.ndarray, np.ndarray]:
    """Read the original vector points, palette and axis ticks from one PCA panel."""
    root = ET.parse(PANELS / panel_name).getroot()
    ns = "{http://www.w3.org/2000/svg}"
    tick_text = []
    for node in root.iter(f"{ns}text"):
        text = "".join(node.itertext()).strip().replace("−", "-")
        try:
            value = float(text)
        except ValueError:
            continue
        style = node.get("style", "")
        x, y = node.get("x"), node.get("y")
        if x is None or y is None:
            continue
        tick_text.append((float(x), float(y), value, style))
    xt = [(x, value) for x, y, value, style in tick_text if "text-anchor: middle" in style and y > 200]
    yt = [(y, value) for x, y, value, style in tick_text if "text-anchor: end" in style and x < 45]
    if len(xt) < 2 or len(yt) < 2:
        raise RuntimeError(f"Could not recover PCA ticks from {panel_name}")
    x_positions, x_values = np.asarray(xt, float).T
    y_positions, y_values = np.asarray(yt, float).T
    x_fit = np.polyfit(x_positions, x_values, 1)
    y_fit = np.polyfit(y_positions, y_values, 1)

    groups = []
    for node in root.iter(f"{ns}g"):
        if not node.get("id", "").startswith("PathCollection"):
            continue
        uses = list(node.iter(f"{ns}use"))
        # One-use collections are legend samples; smaller real categories such
        # as adjacent-normal profiles must remain in the PCA panel.
        if len(uses) < 2:
            continue
        xy = np.asarray([(float(use.get("x")), float(use.get("y"))) for use in uses])
        colours, alpha = [], []
        for use in uses:
            style = use.get("style", "")
            fill = re.search(r"fill: (#[0-9a-fA-F]{6})", style)
            opacity = re.search(r"fill-opacity: ([0-9.]+)", style)
            if fill is None:
                raise RuntimeError(f"Missing point colour in {panel_name}")
            colours.append(to_rgba(fill.group(1), float(opacity.group(1)) if opacity else 1.0))
            alpha.append(float(opacity.group(1)) if opacity else 1.0)
        groups.append({
            "x": np.polyval(x_fit, xy[:, 0]),
            "y": np.polyval(y_fit, xy[:, 1]),
            "colours": np.asarray(colours),
            "colour": colours[0],
            "alpha": alpha[0],
        })

    entries: list[tuple[str, str]] = []
    legend = next((node for node in root.iter(f"{ns}g") if node.get("id") == "legend_1"), None)
    if legend is not None:
        pending = None
        for child in legend:
            uses = list(child.iter(f"{ns}use"))
            texts = [" ".join("".join(node.itertext()).split()) for node in child.iter(f"{ns}text")]
            if uses:
                match = re.search(r"fill: (#[0-9a-fA-F]{6})", uses[0].get("style", ""))
                pending = match.group(1) if match else None
            if texts and pending:
                entries.append((pending, " ".join(texts)))
                pending = None
    return groups, entries, x_values, y_values


def scatter_legend(legend_ax, entries: list[tuple[str, str]], compact: bool = False) -> None:
    legend_ax.set_axis_off()
    handles = [Line2D([], [], marker="o", linestyle="", markersize=3.0 if compact else 3.6, color=colour)
               for colour, _ in entries]
    labels = [str(text).replace("\n", " ") if compact else wrapped(text, 24) for _, text in entries]
    legend_ax.legend(handles, labels, loc="center left", frameon=False, fontsize=5.0,
                     handletextpad=.28 if compact else .35, handlelength=.55 if compact else .7,
                     labelspacing=.12 if compact else .24, borderaxespad=0)


def source_colourbar(cax, panel_name: str) -> None:
    root = ET.parse(PANELS / panel_name).getroot()
    ns = "{http://www.w3.org/2000/svg}"
    candidates = []
    for node in root.iter(f"{ns}image"):
        href = node.get("{http://www.w3.org/1999/xlink}href") or node.get("href") or ""
        if href.startswith("data:image"):
            candidates.append((float(node.get("height", 0)) / max(float(node.get("width", 1)), 1), href))
    encoded = max(candidates, key=lambda item: item[0])[1].split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    cax.imshow(image, origin="lower", extent=(0, 1, .1, .7), aspect="auto")
    cax.set(xticks=[], yticks=np.arange(.1, .71, .1), ylim=(.1, .7))
    cax.yaxis.tick_right()
    cax.tick_params(labelsize=5.0, width=.45, length=2)
    cax.set_ylabel("RNA–text gap", fontsize=5.2, rotation=90, labelpad=4)
    cax.yaxis.set_label_position("right")
    for spine in cax.spines.values():
        spine.set_linewidth(.5)


def scatter_source_panel(ax, legend_ax, panel_name: str, title: str, continuous: bool = False,
                         compact_legend: bool = False) -> None:
    groups, entries, xticks, yticks = svg_scatter_source(panel_name)
    for group in groups:
        colours = group["colours"]
        single_colour = np.allclose(colours, colours[0])
        ax.scatter(group["x"], group["y"], s=2.4, linewidths=0,
                   color=group["colour"] if single_colour else colours, rasterized=False)
    ax.set(xticks=xticks, yticks=yticks, xlabel="RNA embedding PC1", ylabel="RNA embedding PC2")
    # Reserve the left edge for the panel letter; centred long titles can run
    # into that letter even though neither lies over the plotted coordinates.
    ax.set_title(title, loc="left", x=.06, pad=5)
    ax.tick_params(labelsize=5.2)
    style_axes(ax, grid="")
    if continuous:
        source_colourbar(legend_ax, panel_name)
    else:
        scatter_legend(legend_ax, entries, compact=compact_legend)


def source_cell_colours(panel_name: str, n_rows: int, n_cols: int) -> np.ndarray:
    """Sample the original source heatmap at the centre of each colour cell."""
    image = np.asarray(embedded_image(panel_name), dtype=float)
    height, width = image.shape[:2]
    return np.asarray([
        [image[int((row + .5) * height / n_rows), int((col + .5) * width / n_cols)] / 255
         for col in range(n_cols)]
        for row in range(n_rows)
    ])


def source_violin_panel(ax, panel_name: str) -> None:
    """Rebuild an approved two-group violin directly from its vector paths."""
    root = ET.parse(PANELS / panel_name).getroot()
    ns = "{http://www.w3.org/2000/svg}"
    violin_paths: list[tuple[str, float, np.ndarray]] = []
    mean_lines: list[np.ndarray] = []
    for group in root.iter(f"{ns}g"):
        group_id = group.get("id", "")
        if group_id.startswith("PolyCollection"):
            path = next(group.iter(f"{ns}path"), None)
            if path is None:
                continue
            style = path.get("style", "")
            match = re.search(r"fill: (#[0-9a-fA-F]{6})", style)
            alpha = re.search(r"fill-opacity: ([0-9.]+)", style)
            pairs = re.findall(r"[ML]\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)", path.get("d", ""))
            if match and len(pairs) > 8:
                violin_paths.append((match.group(1), float(alpha.group(1)) if alpha else 1.0,
                                     np.asarray(pairs, dtype=float)))
        elif group_id.startswith("LineCollection"):
            for path in group.iter(f"{ns}path"):
                pairs = re.findall(r"[ML]\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)", path.get("d", ""))
                if len(pairs) == 2:
                    mean_lines.append(np.asarray(pairs, dtype=float))
    if len(violin_paths) != 2 or len(mean_lines) != 2:
        raise RuntimeError(f"Could not recover the two approved violin shapes from {panel_name}")

    # The two vector mean marks define categorical x=1 and x=2.  Numeric y
    # labels define the original data-coordinate transform.
    centers = np.asarray([line[:, 0].mean() for line in mean_lines])
    x_fit = np.polyfit(centers, [1.0, 2.0], 1)
    y_ticks: list[tuple[float, float]] = []
    for node in root.iter(f"{ns}text"):
        text = "".join(node.itertext()).strip().replace("−", "-")
        try:
            value = float(text)
        except ValueError:
            continue
        x, y = node.get("x"), node.get("y")
        if x is not None and y is not None and float(x) < 40:
            y_ticks.append((float(y), value))
    if len(y_ticks) < 2:
        raise RuntimeError(f"Could not recover cosine y-axis ticks from {panel_name}")
    y_source, y_data = np.asarray(y_ticks, dtype=float).T
    y_fit = np.polyfit(y_source, y_data, 1)

    for colour, alpha, points in violin_paths:
        points[:, 0] = np.polyval(x_fit, points[:, 0])
        points[:, 1] = np.polyval(y_fit, points[:, 1])
        ax.add_patch(Polygon(points, closed=True, facecolor=colour, alpha=alpha, edgecolor="none", zorder=2))
    for line in mean_lines:
        line[:, 0] = np.polyval(x_fit, line[:, 0])
        line[:, 1] = np.polyval(y_fit, line[:, 1])
        ax.plot(line[:, 0], line[:, 1], color="#222222", linewidth=1.5, solid_capstyle="butt", zorder=3)
    ax.set(xlim=(.5, 2.5), ylim=(-.82, 1.0), xticks=[1, 2], xticklabels=["paired", "shuffled"],
           yticks=np.arange(-.75, .76, .25), ylabel="RNA–text cosine")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_title("RNA–text cosine distributions", loc="left", x=.06, pad=5)
    style_axes(ax)


def external_composition(ax, frame: pd.DataFrame, variable: str, title: str, y_limit: float) -> None:
    counts = frame.groupby(["pool", variable]).size().unstack(fill_value=0)
    counts = counts.loc[:, counts.sum().sort_values(ascending=False).index]
    x = np.arange(len(counts.columns))
    for offset, colour, pool in [(-.20, "#9db9d8", "External-180"), (.20, "#1f5a9d", "MultiSource-450")]:
        ax.bar(x + offset, counts.loc[pool].to_numpy(), width=.36, color=colour, label=pool)
    ax.set(xticks=x, xticklabels=[label(value) for value in counts.columns], ylim=(0, y_limit),
           ylabel="number of external profiles")
    ax.set_title(title, loc="left", x=.065, pad=5)
    # With the dominant categories at the left, the upper-right range is unused.
    ax.legend(loc="upper right", frameon=False, fontsize=5.0, handlelength=1.0,
              labelspacing=.25, borderaxespad=.25)
    ax.tick_params(axis="y", labelsize=5.0)
    ax.tick_params(axis="x", labelsize=5.0, pad=1)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", va="top", rotation_mode="anchor")
    style_axes(ax)


def heatmap(ax, matrix: pd.DataFrame, title: str, cmap: str = "Blues", vmin: float | None = None,
            vmax: float | None = None, fmt: str | None = None, xwidth: int = 13, ywidth: int = 17):
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(matrix.shape[1]), [wrapped(x, xwidth) for x in matrix.columns], rotation=40, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), [wrapped(x, ywidth) for x in matrix.index])
    ax.tick_params(labelsize=5.2, length=0)
    ax.set_title(title, pad=3)
    if fmt:
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                value = matrix.iat[r, c]
                if np.isfinite(value):
                    ax.text(c, r, format(value, fmt), ha="center", va="center", fontsize=5.0,
                            color="white" if value > (np.nanmin(matrix.to_numpy()) + np.nanmax(matrix.to_numpy())) / 2 else "black")
    return im


def draw_ed1() -> None:
    external = source("figure_2_expression_coordinates.csv")
    fig = plt.figure(figsize=(180 * MM, 245 * MM))
    gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1, 1.55], hspace=.56, wspace=.33,
                          left=.065, right=.965, top=.965, bottom=.085)
    specs = [
        ("a", "S1a_training_embedding_pca_by_split.svg", "Training RNA embedding space by split", False),
        ("b", "S1b_training_embedding_pca_by_site.svg", "Training RNA embedding space by tissue/site", False),
        ("c", "S1c_training_embedding_pca_by_tumour_status.svg", "Training RNA embedding space by tumour status", False),
        ("d", "S1d_training_embedding_pca_by_disease_label.svg", "Training RNA embedding space by disease label", False),
        ("e", "S1e_training_embedding_pca_by_source.svg", "Training RNA embedding space by source", False),
        ("f", "S1f_training_embedding_pca_by_rna_text_gap.svg", "RNA–text gap across training space", True),
    ]
    for idx, spec in enumerate(specs):
        start = (idx % 2) * 3
        cell = gs[idx // 2, start:start + 3].subgridspec(1, 2, width_ratios=[1.95, 1.7], wspace=.05)
        ax = fig.add_subplot(cell[0])
        legend_ax = fig.add_subplot(cell[1])
        scatter_source_panel(ax, legend_ax, spec[1], spec[2], spec[3], compact_legend=spec[0] in {"d", "e"})
        if spec[0] == "f":
            # A narrow vertical colour scale frees a visible boundary after e.
            ax_box = ax.get_position()
            ax.set_position([ax_box.x0 + .028, ax_box.y0, ax_box.width, ax_box.height])
            legend_box = legend_ax.get_position()
            legend_ax.set_position([legend_box.x0 + .041, legend_box.y0, .024, legend_box.height])
        panel_letter(ax, spec[0])
    g = fig.add_subplot(gs[3, 0:2])
    external_composition(g, external, "expected_site_family", "external tissue/site composition", y_limit=200)
    g_box = g.get_position()
    g.set_position([g_box.x0, g_box.y0, g_box.width * .88, g_box.height])
    panel_letter(g, "g")
    h = fig.add_subplot(gs[3, 2:4])
    external_composition(h, external, "expected_disease_family", "external disease composition", y_limit=130)
    panel_letter(h, "h")
    i = fig.add_subplot(gs[3, 4:6])
    groups = external.groupby("semantic_state_family")["matched_selected_genes"]
    stats = groups.agg(median="median", lower=lambda x: x.quantile(.25), upper=lambda x: x.quantile(.75))
    stats = stats.sort_values("median") / 4096
    axvals = np.arange(len(stats))
    i.bar(axvals, stats["median"], color="#9db9d8", edgecolor="white", linewidth=0.4)
    i.errorbar(axvals, stats["median"], yerr=[stats["median"] - stats["lower"], stats["upper"] - stats["median"]],
               fmt="none", ecolor="#333333", capsize=1.6, linewidth=.65)
    i.set(xticks=axvals, xticklabels=[label(x) for x in stats.index], ylim=(0, stats["upper"].max() * 1.12),
          ylabel="fraction of selected genes available")
    i.set_title("gene coverage by RNA portrait", loc="left", x=.03, pad=5, fontsize=5.6)
    i.tick_params(axis="y", labelsize=5.1)
    i.tick_params(axis="x", labelsize=5.1, pad=1)
    plt.setp(i.get_xticklabels(), rotation=45, ha="right", va="top", rotation_mode="anchor")
    style_axes(i)
    i_box = i.get_position()
    i_width = i_box.width * .88
    i.set_position([i_box.x1 - i_width, i_box.y0, i_width, i_box.height])
    panel_letter(i, "i")
    finish(fig, 1)


def draw_ed2() -> None:
    exact = source("figure_2_exact_retrieval.csv")
    broad = source("figure_2_broad_retrieval.csv")
    fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 67 * MM), gridspec_kw={"wspace": 0.63})
    ax = axs[0]
    source_violin_panel(ax, "S3a_cosine_control_distributions.svg")
    panel_letter(ax, "a")
    ax = axs[1]
    metrics = list(exact["metric"].drop_duplicates())
    x = np.arange(len(metrics))
    exact_colours = {"rna_to_text": "#1f5a9d", "text_to_rna": "#c9822b"}
    exact_names = {"rna_to_text": "RNA-to-text", "text_to_rna": "text-to-RNA"}
    for offset, (direction, grp) in zip([-.18, .18], exact.groupby("direction", sort=False)):
        grp = grp.set_index("metric").loc[metrics]
        values = grp["lift_over_random"]
        interval = [values - grp["ci_low"] / grp["random_baseline"], grp["ci_high"] / grp["random_baseline"] - values]
        ax.bar(x + offset, values, width=.36, color=exact_colours[direction], label=exact_names[direction],
               yerr=interval, error_kw={"capsize": 2, "elinewidth": .7})
    ax.axhline(1, color="#8f8f8f", linewidth=0.8, linestyle="--")
    ax.set(xticks=x, xticklabels=metrics, xlabel="retrieval metric", ylabel="lift over random", title="exact retrieval lift")
    # The 50–60 region is empty above the bars, so the vertical key remains
    # inside that whitespace without obscuring a result.
    ax.legend(frameon=False, fontsize=5.0, loc="upper right", bbox_to_anchor=(.985, .985),
              ncol=1, borderaxespad=0, labelspacing=.3)
    style_axes(ax)
    panel_letter(ax, "b")
    ax = axs[2]
    broad_colours = {"site": "#1f5a9d", "tumor": "#5b8c5a", "disease": "#b64a4a"}
    broad_names = {"site": "tissue/site", "tumor": "tumour status", "disease": "disease family"}
    for name, grp in broad.groupby("label", sort=False):
        ax.plot(grp["k"], grp["value"], marker="o", markersize=3.0, linewidth=1.3,
                color=broad_colours[name], label=broad_names[name])
        ax.fill_between(grp["k"], grp["ci_low"], grp["ci_high"], color=broad_colours[name], alpha=.13)
    ax.set(xlabel="rank cutoff k", ylabel="semantic match", ylim=(0, 1.03), title="broad semantic retrieval across k")
    # Lower-right values remain below 0.60; the legend therefore fits entirely
    # in the right-hand lower blank region.
    ax.legend(frameon=False, fontsize=5.0, loc="lower right", bbox_to_anchor=(.985, .02),
              ncol=1, borderaxespad=0, labelspacing=.3)
    style_axes(ax)
    panel_letter(ax, "c")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.21)
    finish(fig, 2)


def draw_ed3() -> None:
    thresholds = source("figure_4_thresholds.csv")
    operating = source("figure_4_operating_points.csv")
    transitions = source("figure_4_sample_transitions.csv")
    fig = plt.figure(figsize=(180 * MM, 180 * MM))
    gs = fig.add_gridspec(3, 3, hspace=1.02, wspace=0.62, left=0.08, right=0.98, top=0.96, bottom=0.21)
    # Raise only the panel letters uniformly; all axes, titles and data remain fixed.
    ed3_letter_y = 1.18
    colors = {"uncalibrated_sgd": "#8f8f8f", "temperature_scaled_sgd": BLUE,
              "sigmoid_calibrated": GREEN, "isotonic_calibrated": "#4d4d4d"}
    for letter_, stem, ymax, title_, ylab, cell in [
        ("a", "S4a_internal_reliability_curve.svg", 1.0, "internal calibration", "observed fraction", gs[0, 0]),
        ("b", "S4b_internal_risk_coverage_curve.svg", 0.8, "risk–coverage behaviour", "error among covered", gs[0, 1]),
    ]:
        ax = fig.add_subplot(cell)
        for model, (x, y) in svg_series(stem, ymax).items():
            ax.plot(x, y, marker="o", markersize=2.6, linewidth=1.2, color=colors[model], label=label(model))
        if letter_ == "a":
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.75, color="#bbbbbb", label="ideal")
            ax.set(xlabel="mean predicted probability", xlim=(0, 1), ylim=(0, 1))
        else:
            ax.set(xlabel="coverage", xlim=(0, 1), ylim=(0, ymax))
        ax.set(ylabel=ylab, title=title_)
        if letter_ == "a":
            # The upper-left region lies clear of the three calibration traces.
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles, ["uncal.", "temperature scaled", "sigmoid calibrated", "ideal"],
                      frameon=False, fontsize=5.0, loc="upper left", bbox_to_anchor=(.015, .985),
                      ncol=1, borderaxespad=0, labelspacing=0, handlelength=.8, handletextpad=.28)
        else:
            # Lower right is below all risk–coverage observations.
            ax.legend(frameon=False, fontsize=5.0, loc="lower right", bbox_to_anchor=(.985, .02),
                      ncol=1, borderaxespad=0, labelspacing=.28)
        style_axes(ax)
        panel_letter(ax, letter_, y=ed3_letter_y)
    ax = fig.add_subplot(gs[0, 2])
    calibrated = thresholds[thresholds["model"] == "temperature_scaled_sgd"]
    curve_colours = {"External-180": "#1f5a9d", "MultiSource-450": "#c9822b"}
    for pool, grp in calibrated.groupby("pool", sort=False):
        colour = curve_colours[pool]
        ax.plot(grp["confidence_threshold"], grp["coverage"], marker="o", linewidth=1.15,
                markersize=2.8, color=colour, label=f"{pool}: label retained")
        ax.plot(grp["confidence_threshold"], grp["overcall_among_covered_vs_openworld"], marker="s",
                linestyle="--", linewidth=1.0, markersize=2.8, color=colour, alpha=.76,
                label=f"{pool}: uncertain but labelled")
    ax.set(xlabel="confidence threshold", ylabel="fraction or rate", ylim=(0, 1.04), title="predefined-label threshold sweep")
    # The lower-left corner is free of the threshold curves and retains a
    # compact vertical key without covering a line, marker or interval.
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["External: retained", "External: uncertain", "MultiSource: retained", "MultiSource: uncertain"],
              frameon=False, fontsize=5.0, loc="lower left", bbox_to_anchor=(.015, .02),
              ncol=1, borderaxespad=0, labelspacing=0, handlelength=.8, handletextpad=.28)
    style_axes(ax)
    panel_letter(ax, "c", y=ed3_letter_y)
    ax = fig.add_subplot(gs[1, 0])
    ops = operating[operating["model"] == "temperature_scaled_sgd"]
    thresholds_order = [0.5, 0.7, 0.9, 0.95]
    pools = ["External-180", "MultiSource-450"]
    x = np.arange(len(thresholds_order))
    width = 0.36
    for offset, pool, color in [(-width / 2, pools[0], "#1f5a9d"), (width / 2, pools[1], "#c9822b")]:
        group = ops[ops["pool"] == pool].set_index("confidence_threshold").reindex(thresholds_order)
        bars = ax.bar(x + offset, group["coverage"], width=width, color=color, label=pool)
        for rect, n in zip(bars, group["n_covered"]):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + .025, f"n={int(n)}",
                    ha="center", va="bottom", fontsize=5)
    ax.set(xticks=x, xticklabels=[f"{v:.2f}" for v in thresholds_order], xlabel="confidence threshold",
           ylabel="fraction retained", ylim=(0, 1.12), title="operating-point coverage")
    style_axes(ax)
    panel_letter(ax, "d", y=ed3_letter_y)
    ax = fig.add_subplot(gs[1, 1])
    for offset, pool, color in [(-width / 2, pools[0], "#1f5a9d"), (width / 2, pools[1], "#c9822b")]:
        group = ops[ops["pool"] == pool].set_index("confidence_threshold").reindex(thresholds_order)
        ax.bar(x + offset, group["uncertain_among_covered"], width=width, color=color, label=pool)
    ax.set(xticks=x, xticklabels=[f"{v:.2f}" for v in thresholds_order], xlabel="confidence threshold",
           ylabel="uncertain among retained", ylim=(0, 1.12), title="operating-point uncertainty")
    style_axes(ax)
    panel_letter(ax, "e", y=ed3_letter_y)
    ax = fig.add_subplot(gs[1, 2])
    raw_top = transitions["raw_disease_family"].value_counts().index[:8]
    status_order = ["stable", "mixed", "unsupported"]
    status = pd.crosstab(transitions["raw_disease_family"], transitions["resolved_status"], normalize="index")
    status = status.reindex(index=raw_top, columns=status_order, fill_value=0)
    heatmap(ax, status, "raw disease to status", cmap="YlOrRd", vmin=0, vmax=1, fmt=".2f", xwidth=14, ywidth=14)
    ax.set_yticklabels([label(value) for value in status.index], fontsize=5.2)
    panel_letter(ax, "f", y=ed3_letter_y)
    # Bottom row: make room for the resolved-family scale and for a compact,
    # fully separate key beside panel h rather than below the plate.
    # A narrower matrix leaves a deliberately generous visual break before
    # panel h and its full, single-column key.
    ax = fig.add_axes([.12, .22, .26, .14])
    raw_top = transitions["raw_disease_family"].value_counts().index[:9]
    resolved_top = transitions["resolved_disease_family"].value_counts().index[:9]
    resolve = pd.crosstab(transitions["raw_disease_family"], transitions["resolved_disease_family"], normalize="index")
    resolve = resolve.reindex(index=raw_top, columns=resolved_top, fill_value=0)
    im = heatmap(ax, resolve, "", cmap="Blues", vmin=0, vmax=1, xwidth=12, ywidth=15)
    ax.set_title("raw disease to resolved family", loc="center", pad=4, fontsize=6.2)
    ax.set_xticklabels([label(value) for value in resolve.columns], rotation=45, ha="right", va="top",
                       rotation_mode="anchor", fontsize=5.2)
    ax.set_yticklabels([label(value) for value in resolve.index], fontsize=5.2)
    cax = fig.add_axes([.395, .22, .012, .14])
    cb = fig.colorbar(im, cax=cax, ticks=np.arange(0, 1.01, .2))
    cb.set_label("row fraction", fontsize=5.0, labelpad=3)
    cb.ax.tick_params(labelsize=5.0, width=.45, length=2)
    cb.outline.set_linewidth(.5)
    panel_letter(ax, "g", y=ed3_letter_y, x=-.26)
    # Preserve the wide boundary after panel g. Move panel h rightward and use
    # the otherwise unused left portion of its legend lane for the wider plot.
    ax = fig.add_axes([.62, .22, .195, .14])
    expected = source("supplement_s5c_expected_metadata_group_to_portrait.csv")
    state_order = ["single clear signal", "blood/immune", "epithelial-like context", "cleaner-anchor context",
                   "broad-context", "weak-evidence", "conflict", "other"]
    state_colours = ["#303030", "#1f5a9d", "#b64a4a", "#5b8c5a", "#7e6aae", "#a9a9a9", "#c9822b", "#6b6b6b"]
    grouped = expected.pivot(index="expected_metadata_group", columns="figure_label", values="fraction_within_expected_group")
    grouped = grouped.reindex(index=["healthy metadata", "non-tumour metadata", "tumour metadata"], columns=state_order, fill_value=0)
    left = np.zeros(len(grouped))
    for state, colour in zip(state_order, state_colours):
        ax.barh(np.arange(len(grouped)), grouped[state], left=left, height=.60, color=colour, label=state)
        left += grouped[state].to_numpy()
    group_labels = [f"{name}\n(n={int(expected.loc[expected.expected_metadata_group == name, 'group_total'].iloc[0])})" for name in grouped.index]
    ax.set(yticks=np.arange(len(grouped)), yticklabels=group_labels, xlim=(0, 1), xlabel="fraction within metadata group")
    ax.set_title("metadata groups to\nportrait families", loc="left", x=.02, pad=3, fontsize=5.2)
    ax.invert_yaxis()
    style_axes(ax, "x")
    panel_letter(ax, "h", y=ed3_letter_y)
    # Panels d and e use the same pool colours; one shared vertical key is
    # placed in panel d's unused upper-right region rather than below the grid.
    pool_handles = [plt.Rectangle((0, 0), 1, 1, color="#1f5a9d"), plt.Rectangle((0, 0), 1, 1, color="#c9822b")]
    d_legend_ax = fig.add_axes([.135, .575, .15, .08])
    d_legend_ax.set_axis_off()
    d_legend_ax.legend(pool_handles, pools, loc="upper right", frameon=False, fontsize=5.0,
                       ncol=1, handlelength=.9, handletextpad=.35, labelspacing=.28, borderaxespad=0)
    h_legend_ax = fig.add_axes([.775, .218, .205, .145])
    h_legend_ax.set_axis_off()
    state_labels = ["single clear signal", "blood/immune", "epithelial-like context", "cleaner-anchor context",
                    "broad-context", "weak-evidence", "conflict", "other"]
    h_legend_ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in state_colours], state_labels,
                       ncol=1, frameon=False, fontsize=5.0, loc="center right", bbox_to_anchor=(1, .5), borderaxespad=0,
                       handlelength=.8, handletextpad=.3, labelspacing=.25)
    finish(fig, 3)


def draw_ed5() -> None:
    mixing = source("figure_5_mixing.csv")
    assoc = source("figure_5_metadata_association.csv")
    # Four vector bars in the supplied panel use a 0–0.12 scale.
    bars = svg_bars("S8c_within_label_evidence_separation.svg", [PURPLE], 72.14125, 284.18125)[PURPLE]
    lifts = [width * 0.12 for _, width in bars]
    within = pd.DataFrame(
        {"factor": ["predefined disease label", "tissue/site", "project/source", "expected disease"], "lift": lifts}
    )
    # Panel b has seven full factor labels, so it receives the extra width
    # rather than abbreviating or clipping any of them.
    fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 66 * MM),
                            gridspec_kw={"wspace": 0.82, "width_ratios": [1, 1.38, 1]})
    ax = axs[0]
    colours = {"normal_immune": "#5b8c5a", "tumor_immune": "#c9822b", "normal_tumor": "#b64a4a", "tumor_normal": "#b64a4a"}
    use = mixing[mixing["model"] == "temperature_scaled_sgd"]
    for design, grp in use.groupby("design", sort=False):
        ax.plot(grp["fraction_b"], grp["mean_closed_set_confidence"], marker="o", markersize=2.7,
                linewidth=1.2, color=colours.get(design, "#555555"), label=label(design))
        ax.fill_between(grp["fraction_b"], grp["mean_closed_set_confidence_lo"], grp["mean_closed_set_confidence_hi"],
                        color=colours.get(design, "#555555"), alpha=0.12)
    ax.set(xlabel="mixture fraction", ylabel="mean closed-set confidence", title="predefined-label mixing")
    # Use the inter-panel whitespace for the key, leaving the data area and
    # the x-axis labels unobscured.
    ax.legend(frameon=False, fontsize=5.0, loc="center left", bbox_to_anchor=(1.02, .5), ncol=1,
              borderaxespad=0, handlelength=1.1)
    style_axes(ax)
    panel_letter(ax, "a")
    ax = axs[1]
    targets = ["RNA portrait category", "portrait disease status", "signal status"]
    factor_order = ["evaluation pool", "source prefix", "project", "source-quality flag", "tissue/site",
                    "expected disease label", "predefined disease label"]
    table = assoc.pivot(index="factor_label", columns="target_label", values="nmi").reindex(
        index=factor_order, columns=targets
    )
    # Explicit two-line breaks prevent the longest factor names from wrapping
    # into a third line at final print size.
    factor_ticklabels = [
        "evaluation\npool", "source\nprefix", "project", "source-quality\nflag", "tissue/site",
        "expected\ndisease label", "predefined\ndisease label",
    ]
    x = np.arange(len(table))
    metadata_colours = ["#1f5a9d", "#c9822b", "#5b8c5a"]
    for offset, target, colour in zip([-.25, 0, .25], targets, metadata_colours):
        ax.bar(x + offset, table[target], width=.25, color=colour, label=target)
    ax.set(xticks=x, xticklabels=factor_ticklabels, ylabel="normalized mutual information",
           ylim=(0, .365))
    # Preserve a clear inter-panel legend lane without allowing this label to
    # overlap it or the tick labels.
    ax.yaxis.set_label_coords(-.20, .5)
    ax.set_title("metadata association with portrait readouts", loc="left", x=.06, pad=5)
    ax.tick_params(axis="x", labelsize=5.0, pad=2)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", va="top", rotation_mode="anchor", linespacing=.75)
    # The upper-right portion of the panel is clear of the bars; keep the key
    # there rather than allocating a separate lower legend band.
    ax.legend(frameon=False, fontsize=5.0, loc="upper right", bbox_to_anchor=(.985, .985), ncol=1,
              borderaxespad=0, handlelength=1.1)
    style_axes(ax)
    panel_letter(ax, "b")
    ax = axs[2]
    y = np.arange(len(within))
    ax.barh(y, within["lift"], color=PURPLE)
    ax.set(yticks=y, yticklabels=[wrapped(v, 16) for v in within["factor"]], xlabel="weighted silhouette lift",
           xlim=(0, 0.12))
    ax.set_title("within-label evidence separation", loc="left", x=.08, pad=5, fontsize=6.2)
    style_axes(ax, "x")
    panel_letter(ax, "c")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.33)
    finish(fig, 5)


def draw_ed6() -> None:
    robustness = source("figure_5_robustness.csv").sort_values("mean_incremental_r2")
    quality = svg_bars("S9b_quality_metrics_by_portrait.svg", [LIGHT_BLUE, "#b8d3b4"], 91.2525, 325.6125)
    reliability = svg_bars("S10b_reliability_tiers_by_portrait.svg", [GREEN, LIGHT_BLUE, "#e1b66d", "#d9d9d9", "#e7b1aa"], 91.2525, 359.0925)
    fig = plt.figure(figsize=(180 * MM, 178 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.72, wspace=0.84, left=0.12, right=0.96, top=0.955, bottom=0.12)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(robustness))
    ax.barh(y, robustness["mean_incremental_r2"], color=BLUE)
    ax.errorbar(robustness["mean_incremental_r2"], y,
                xerr=[robustness["mean_incremental_r2"] - robustness["min_incremental_r2"],
                      robustness["max_incremental_r2"] - robustness["mean_incremental_r2"]],
                fmt="none", ecolor="#333333", capsize=1.5, linewidth=0.7)
    ax.set(yticks=y, yticklabels=robustness["subset_label"], xlabel="incremental R²",
           title="robustness across subsets")
    ax.tick_params(axis="y", labelsize=5.05)
    # Keep the subset descriptions intact on one line and make enough left
    # margin for the longest label without changing the panel's right edge.
    pos = ax.get_position()
    label_margin = .083
    ax.set_position([pos.x0 + label_margin, pos.y0, pos.width - label_margin, pos.height])
    style_axes(ax, "x")
    panel_letter(ax, "a", x=-.173)
    ax = fig.add_subplot(gs[0, 1])
    qlabels = ["weak evidence", "broad context", "blood/immune", "conflict", "single clear signal",
               "epithelial-like context", "other", "clean-anchor context"]
    qy = np.arange(len(qlabels))
    coverage = [v for _, v in quality[LIGHT_BLUE]]
    epic = [v for _, v in quality["#b8d3b4"]]
    ax.barh(qy - 0.18, coverage, height=0.34, color=LIGHT_BLUE, label="mean gene coverage")
    ax.barh(qy + 0.18, epic, height=0.34, color="#b8d3b4", label="EPIC converged")
    ax.set(yticks=qy, yticklabels=[wrapped(v, 16) for v in qlabels], xlim=(0, 1), xlabel="fraction or mean coverage",
           title="quality metrics by portrait family")
    # Panel keys are placed in the central whitespace between rows (below).
    ax.tick_params(axis="y", labelsize=5.1)
    style_axes(ax, "x")
    panel_letter(ax, "b")
    ax = fig.add_subplot(gs[1, 0])
    rlabels = ["clean-anchor context", "weak evidence", "other", "conflict", "epithelial-like context",
               "broad context", "blood/immune", "single clear signal"]
    ry = np.arange(len(rlabels))
    tier_names = ["high confidence", "auditable mixed", "cautionary tier", "quality limited", "weak/unsupported"]
    tier_colors = [GREEN, LIGHT_BLUE, "#e1b66d", "#d9d9d9", "#e7b1aa"]
    left = np.zeros(len(rlabels))
    for name, color in zip(tier_names, tier_colors):
        vals = np.asarray([v for _, v in reliability[color]])
        ax.barh(ry, vals, left=left, height=0.66, color=color, label=name)
        left += vals
    ax.set(yticks=ry, yticklabels=[wrapped(v, 16) for v in rlabels], xlim=(0, 1), xlabel="fraction of profiles",
           title="reliability tiers by portrait family")
    # Panel keys are placed in the central whitespace between rows (below).
    ax.tick_params(axis="y", labelsize=5.1)
    style_axes(ax, "x")
    panel_letter(ax, "c")
    ax = fig.add_subplot(gs[1, 1])
    flags = ["unsupported portrait", "low statement support", "low gene coverage", "EPIC non-converged",
             "fixed-label conflict", "mixed-signal flag", "low evidence margin"]
    rows = ["blood/immune", "epithelial-like context", "single clear signal", "clean-anchor context",
            "broad context", "weak evidence", "other", "conflict"]
    cell_colours = source_cell_colours("S10c_boundary_flags_by_portrait_heatmap.svg", len(rows), len(flags))
    for row, col in np.ndindex(cell_colours.shape[:2]):
        ax.add_patch(Rectangle((col - .5, row - .5), 1, 1, facecolor=cell_colours[row, col], edgecolor="none"))
    ax.set(xlim=(-.5, len(flags) - .5), ylim=(len(rows) - .5, -.5))
    ax.set(xticks=np.arange(len(flags)), xticklabels=flags, yticks=np.arange(len(rows)), yticklabels=rows,
           title="reliability flags by portrait family")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", va="top", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=5.0, pad=2)
    ax.tick_params(axis="y", labelsize=5.1, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    flag_cmap = LinearSegmentedColormap.from_list("flag_fraction", ["#ffffff", "#d9958e", "#b23f3f"])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=flag_cmap), ax=ax,
                      fraction=0.05, pad=0.04)
    cb.set_label("fraction", fontsize=5.3)
    cb.ax.tick_params(labelsize=5.0)
    panel_letter(ax, "d")
    # Both keys occupy the intentionally empty inter-row band; they do not
    # overlap the horizontal bars or their labels.
    quality_handles = [Rectangle((0, 0), 1, 1, color=LIGHT_BLUE), Rectangle((0, 0), 1, 1, color="#b8d3b4")]
    fig.legend(quality_handles, ["mean gene coverage", "EPIC converged"], ncol=2, frameon=False,
               fontsize=5.0, loc="center", bbox_to_anchor=(.73, .515), columnspacing=.9, handlelength=.9)
    fig.legend([Rectangle((0, 0), 1, 1, color=c) for c in tier_colors], tier_names, ncol=3, frameon=False,
               fontsize=5.0, loc="center", bbox_to_anchor=(.28, .515), columnspacing=.75, handlelength=.9)
    finish(fig, 6)


def draw_ed7() -> None:
    composition = source("supplement_s11_disease_portrait_composition_long.csv")
    performance = source("supplement_s11_model_performance_summary.csv")
    permutation = source("supplement_s11_within_tissue_source_permutation.csv")
    identity = source("supplement_s11_group_level_composition_identification_by_disease.csv")
    fig = plt.figure(figsize=(180 * MM, 165 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.8, wspace=0.75, left=0.11, right=0.98, top=0.95, bottom=0.12)
    ax = fig.add_subplot(gs[0, 0])
    matrix = composition.pivot(index="disease_label", columns="portrait_label", values="fraction").fillna(0)
    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]
    im = heatmap(ax, matrix, "disease portrait composition", cmap="Blues", vmin=0, vmax=0.75, xwidth=13, ywidth=16)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("fraction", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    ax.set_xticklabels([label(value) for value in matrix.columns], rotation=45, ha="right", va="top",
                       rotation_mode="anchor", fontsize=5.2)
    ax.set_yticklabels([label(value) for value in matrix.index], fontsize=5.2)
    # The matrix keeps its right edge and colour bar position while moving only
    # far enough right to show the longest single-line disease label.
    pos = ax.get_position()
    label_margin = .05
    ax.set_position([pos.x0 + label_margin, pos.y0, pos.width - label_margin, pos.height])
    panel_letter(ax, "a", x=-.14)
    ax = fig.add_subplot(gs[0, 1])
    perf = performance.sort_values("balanced_accuracy_mean")
    y = np.arange(len(perf))
    model_colours = {
        "stratified_dummy": "#8f8f8f",
        "portrait_composition_centroid": "#1f5a9d",
        "portrait_family": "#1f5a9d",
        "portrait_context": "#1f5a9d",
        "metadata_controls": "#c9822b",
        "controls_plus_portrait": "#1f5a9d",
        "marker_pathway": "#5b8c5a",
        "controls_plus_marker_pathway": "#5b8c5a",
        "controls_plus_portrait_marker_pathway": "#1f5a9d",
    }
    x = np.arange(len(perf))
    ax.bar(x, perf["balanced_accuracy_mean"], yerr=perf["balanced_accuracy_sem"],
           color=[model_colours[model] for model in perf["model"]], error_kw={"capsize": 1.8, "elinewidth": .7})
    ax.set(xticks=x, xticklabels=[label(v) for v in perf["model_label"]], ylim=(0, 1),
           ylabel="balanced accuracy", title="disease prediction models")
    ax.tick_params(axis="x", labelsize=5.0, pad=2)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", va="top", rotation_mode="anchor")
    style_axes(ax, "y")
    panel_letter(ax, "b")
    ax = fig.add_subplot(gs[1, 0])
    perm = permutation[permutation["test"] == "within_tissue_source_permutation"]["delta_balanced_accuracy"]
    real = float(permutation.loc[permutation["test"] == "real_increment_over_controls", "delta_balanced_accuracy"].iloc[0])
    pvalue = float(permutation["one_sided_p_value_for_real_delta"].dropna().iloc[0])
    ax.hist(perm, bins=14, color="#d9d9d9", edgecolor="white")
    ax.axvline(real, color=RED, linewidth=1.4)
    ax.text(real, ax.get_ylim()[1] * .94, f"observed = {real:.3f}\nP = {pvalue:.3f}", color=RED,
            ha="right", va="top", fontsize=6)
    ax.set(xlabel="increment in balanced accuracy", ylabel="permutations", title="portrait increment after controls")
    style_axes(ax)
    panel_letter(ax, "c")
    ax = fig.add_subplot(gs[1, 1])
    ident = identity.sort_values("composition_id_accuracy")
    y = np.arange(len(ident))
    ax.barh(y, ident["composition_id_accuracy"], color=PURPLE)
    ax.axvline(ident["composition_id_accuracy"].mean(), color="#333333", linestyle="--", linewidth=.8,
               label=f"mean = {ident['composition_id_accuracy'].mean():.3f}")
    ax.set(yticks=y, yticklabels=[label(v) for v in ident["disease_label"]], xlim=(0, 1),
           xlabel="composition identification accuracy", title="group-level composition identification")
    ax.tick_params(axis="y", labelsize=5.1)
    # The reference value is stated in the axis title and caption; omitting an
    # in-panel key leaves every bar unobscured.
    style_axes(ax, "x")
    panel_letter(ax, "d")
    finish(fig, 7)


def main() -> None:
    draw_ed1()
    draw_ed2()
    draw_ed3()
    draw_ed5()
    draw_ed6()
    draw_ed7()


if __name__ == "__main__":
    main()
