# -*- coding: utf-8 -*-
"""Rebuild Figure 3 from released source data.

Panels e and f use `source_data/figure_3ef_worked_example.csv`; all outputs
share the same palette, axis limits and 180-mm maximum width.
"""

from __future__ import annotations

import pathlib
import sys
import textwrap
import warnings

warnings.filterwarnings("ignore")
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import main_figures as FX

SRC = FX.SOURCE_DIR / "figure_3ef_worked_example.csv"

CLAIM_ORDER = [
    ("immune_or_blood_signal", "immune/blood"),
    ("context_or_stromal_signal", "stromal/broad-context"),
    ("epithelial_or_tumor_like_signal", "epithelial/tumour-like"),
    ("clean_or_non_malignant_context", "cleaner/non-malignant"),
    ("mixed_or_unstable_disease_reading", "mixed/inconsistent"),
]
EVIDENCE_ROWS = [
    ("immune_core_z", "immune"), ("myeloid_inflammation_z", "myeloid"),
    ("hematologic_lineage_z", "blood"), ("stromal_ecm_z", "stroma"),
    ("epithelial_z", "epith."), ("ifn_gamma_score", "IFN-γ"),
    ("emt_stromal_score", "EMT"), ("epic_immune_fraction_z", "EPIC imm."),
    ("epic_stromal_fraction_z", "EPIC strom."), ("mcp_immune_z_mean", "MCP imm."),
    ("mcp_stromal_z_mean", "MCP strom."),
]
STRONG_C, PARTIAL_C, WEAK_C, POS_C, NEG_C = "#5B8C5A", "#B8D3B4", "#D9D9D9", "#1F5A9D", "#B64A4A"
ACCENT = "#B64A4A"
EXTRA_IN = 4.9


def num(v):
    try:
        if pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def load_worked_example():
    d = pd.read_csv(SRC)
    out = {}
    for panel in ("e", "f"):
        sub = d.loc[d["panel"].eq(panel)]
        first = sub.iloc[0]
        out[panel] = {
            "claims": {r["claim_type"]: r for _, r in sub.iterrows()},
            "row": first,
            "portrait": str(first["portrait_text"]).split(" | "),
        }
    return out


def draw_worked_panel(fig, rect, d, tag, title):
    x0, y0, w, h = rect
    ax = fig.add_axes([x0, y0, w, h])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    r = d["row"]
    cov = int(float(r["matched_selected_genes"]))

    ax.text(-0.045, 1.02, tag, fontsize=8.5, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(0.0, 1.02, title, fontsize=7.2, fontweight="bold", va="top", color=ACCENT)
    ax.text(0.0, 0.925,
            "%s   |   metadata: %s / %s   |   %s of 4,096 selected genes matched (coverage %.2f)"
            % (r["project"], str(r["expected_site_family"]).replace("_", " "),
               str(r["expected_disease_family"]).replace("_", " "), format(cov, ","), cov / 4096),
            fontsize=5.9, color="#555555", va="top")

    ax.add_patch(Rectangle((0.0, 0.545), 1.0, 0.325, transform=ax.transAxes,
                           facecolor="#F4F6F7", edgecolor="#DDDDDD", linewidth=0.6, zorder=0))
    ax.text(0.012, 0.845, "RNA portrait (model output)", fontsize=6.1,
            fontweight="bold", color="#333333", va="top")
    body = "\n".join("• " + textwrap.fill(seg, 104, subsequent_indent="   ")
                     for seg in d["portrait"][:3])
    ax.text(0.012, 0.795, body, fontsize=5.8, color="#222222", va="top", linespacing=1.3)

    ax.text(0.0, 0.492,
            "predefined disease-label comparator: %s (calibrated confidence %.2f)     portrait status: %s"
            % (str(r["closed_set_disease_family"]).replace("_", " "),
               float(r["closed_set_calibrated_confidence"]), r["openworld_status"]),
            fontsize=5.9, color="#555555", va="top")

    axb = fig.add_axes([x0 + 0.215 * w, y0 + 0.055 * h, 0.29 * w, 0.335 * h])
    labs, sc, cols = [], [], []
    for ct, lab in CLAIM_ORDER:
        c = d["claims"].get(ct)
        if c is None:
            continue
        labs.append(lab)
        sc.append(num(c["support_score"]) or 0.0)
        lv = c["support_level"]
        cols.append(STRONG_C if lv == "strong" else PARTIAL_C if lv == "partial" else WEAK_C)
    y = list(range(len(labs)))
    axb.barh(y, sc, color=cols, height=0.66, linewidth=0, zorder=3)
    axb.axvline(0, color="#888888", linewidth=0.6)
    axb.axvline(0.20, color=STRONG_C, linewidth=0.6, linestyle=":", zorder=2)
    axb.axvline(0.50, color=STRONG_C, linewidth=0.6, linestyle="--", zorder=2)
    axb.set_yticks(y, labs, fontsize=5.7)
    axb.invert_yaxis()
    axb.set_xlim(-3.4, 1.6)
    axb.set_xlabel("statement support score", fontsize=6.0)
    axb.tick_params(axis="x", labelsize=5.7)
    axb.set_title("scored statements", fontsize=6.4, pad=3, loc="left")
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)

    axe = fig.add_axes([x0 + 0.645 * w, y0 + 0.055 * h, 0.275 * w, 0.335 * h])
    vals = [num(r.get(k)) for k, _ in EVIDENCE_ROWS]
    names = [n for _, n in EVIDENCE_ROWS]
    y = list(range(len(vals)))
    axe.barh(y, [v if v is not None else 0 for v in vals],
             color=[POS_C if (v or 0) >= 0 else NEG_C for v in vals],
             height=0.66, linewidth=0, zorder=3)
    for i, v in enumerate(vals):
        if v is None:
            axe.text(0.08, i, "not measurable", fontsize=5.2, color="#999999", va="center")
    axe.axvline(0, color="#888888", linewidth=0.6)
    axe.set_yticks(y, names, fontsize=5.7)
    axe.invert_yaxis()
    axe.set_xlim(-4.0, 2.4)
    axe.set_xlabel("RNA-derived evidence (z)", fontsize=6.0)
    axe.tick_params(axis="x", labelsize=5.7)
    axe.set_title("RNA-derived molecular evidence", fontsize=6.4, pad=3, loc="left")
    for sp in ("top", "right"):
        axe.spines[sp].set_visible(False)


def main():
    mod = FX.load_module()
    captured = {}
    mod.save_figure = lambda fig, name: captured.update(fig=fig) or {}
    mod.figure_3_biology_grounding()
    fig = captured["fig"]

    w_in, h_in = fig.get_size_inches()
    new_h = h_in + EXTRA_IN
    scale = h_in / new_h
    shift = EXTRA_IN / new_h
    for ax in fig.axes:
        p = ax.get_position()
        ax.set_position([p.x0, p.y0 * scale + shift, p.width, p.height * scale])
    for t in fig.texts:
        x, y = t.get_position()
        t.set_position((x, y * scale + shift))
    fig.set_size_inches(w_in, new_h)

    d = load_worked_example()
    pad_x = 0.085
    pw = 1.0 - 2 * pad_x
    ph = (EXTRA_IN / new_h) * 0.435
    draw_worked_panel(fig, (pad_x, shift - ph - 0.012, pw, ph), d["e"], "e",
                      "Portrait supported by RNA-derived evidence")
    draw_worked_panel(fig, (pad_x, 0.020, pw, ph), d["f"], "f",
                      "Same portrait statement, evidence not recoverable")
    fig.add_artist(plt.Line2D([pad_x - 0.05, 1 - pad_x + 0.02],
                              [shift - ph - 0.038, shift - ph - 0.038],
                              color="#E2E2E2", linewidth=0.7, transform=fig.transFigure))

    name = "Figure_3_biological_grounding"
    FX.increase_multiline_label_spacing(fig)
    FX.fit_width(fig)
    for ext in ["svg", "pdf", "png"]:
        kw = {"bbox_inches": "tight", "pad_inches": 0.03}
        if ext == "png":
            kw["dpi"] = 320
        fig.savefig(mod.FIG_DIR / f"{name}.{ext}", **kw)
    png = Image.open(mod.FIG_DIR / f"{name}.png")
    png.convert("RGB").resize((int(round(png.size[0] / 320 * 600)),
                               int(round(png.size[1] / 320 * 600))), Image.LANCZOS) \
        .save(mod.FIG_DIR / f"{name}.tiff", compression="tiff_lzw", dpi=(600, 600))
    print("  %-42s %5.1f x %5.1f mm" % (name, png.size[0] / 320 * 25.4, png.size[1] / 320 * 25.4))
    png.close()
    plt.close(fig)


if __name__ == "__main__":
    main()
