#!/usr/bin/env python3
"""Regenerate main Figures 2, 4 and 5 from released source data.

Figures 1 and 3 have dedicated modules for their schematic and worked-example
layouts.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
BASE = HERE / "_main_figure_base.py"
SOURCE_DIR = Path(os.environ["RNA_PORTRAIT_SOURCE_DATA_DIR"]).resolve()
FIG_DIR = Path(os.environ["RNA_PORTRAIT_FIGURE_OUTPUT_ROOT"]).resolve()
TARGET_W_MM = 180.0
MM_PER_IN = 25.4


def fit_width(fig, target_mm: float = TARGET_W_MM) -> None:
    """Uniformly shrink renders that exceed Nature's 180-mm width."""
    for _ in range(4):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        width_mm = (bb.width + 0.06) * MM_PER_IN
        if width_mm <= target_mm:
            return
        scale = target_mm / width_mm
        width, height = fig.get_size_inches()
        fig.set_size_inches(width * scale, height * scale)


def increase_multiline_label_spacing(fig, increment: float = 0.12) -> None:
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
        for text in [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label,
                     ax.yaxis.label, ax.title, ax._left_title, ax._right_title]:
            adjust(text)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                adjust(text)
    for legend in fig.legends:
        for text in legend.get_texts():
            adjust(text)


def save_figure(fig, name: str) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    increase_multiline_label_spacing(fig)
    fit_width(fig)
    for extension in ["svg", "pdf", "png"]:
        options = {"bbox_inches": "tight", "pad_inches": 0.03}
        if extension == "png":
            options["dpi"] = 320
        fig.savefig(FIG_DIR / f"{name}.{extension}", **options)
    with Image.open(FIG_DIR / f"{name}.png") as png:
        width = int(round(png.size[0] / 320 * 600))
        height = int(round(png.size[1] / 320 * 600))
        png.convert("RGB").resize((width, height), Image.Resampling.LANCZOS).save(
            FIG_DIR / f"{name}.tiff", compression="tiff_lzw", dpi=(600, 600)
        )
        print(f"  {name:42s} {png.size[0] / 320 * 25.4:5.1f} x {png.size[1] / 320 * 25.4:5.1f} mm")
    plt.close(fig)
    return {}


def load_module():
    source = BASE.read_text(encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_rna_portrait_submission_figures", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    mapping = dict(
        (match.group(2), match.group(1))
        for match in re.finditer(r'add_source\(\s*"([^"]+)"\s*,\s*"([^"]+)"', source)
    )

    def read_csv(relative_path: str) -> pd.DataFrame:
        name = mapping.get(relative_path)
        if not name:
            raise FileNotFoundError(f"No released source-data mapping for {relative_path}")
        path = SOURCE_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    module.DRAFT = FIG_DIR.parent
    module.FIG_DIR = FIG_DIR
    module.SOURCE_DIR = SOURCE_DIR
    module.read_csv = read_csv
    module.add_source = lambda *args, **kwargs: None
    module.save_figure = save_figure
    return module


def main() -> None:
    module = load_module()
    for function_name in ["figure_2_alignment", "figure_4_portrait_vs_fixed_labels", "figure_5_stress_tests"]:
        getattr(module, function_name)()


if __name__ == "__main__":
    main()
