# -*- coding: utf-8 -*-
"""Rebuild Figure 1 as an editable vector schematic.

The two panel rows use a fixed grid and show the image-language analogy and the
RNA/text-encoder-to-portrait workflow.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import main_figures as FX


EXTS = ["svg", "pdf", "png", "tiff"]


def main():
    mod = FX.load_module()
    P = mod.PALETTE
    mod.set_style()

    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.010",
                                    fc="white", ec=color, lw=1.0))
        ax.text(x + 0.016, y + h - 0.030, title, ha="left", va="top",
                fontsize=8.0, fontweight="bold", color=color)
        ax.text(x + 0.016, y + h - 0.066, body, ha="left", va="top",
                fontsize=6.5, color="#333333", linespacing=1.30)

    def arrow(x1, y1, x2, y2, color="#444444", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                     lw=1.1, color=color,
                                     connectionstyle="arc3,rad=%.2f" % rad))

    # ---------------- strict grid -------------------------------------
    # four columns, uniform width steps and gaps; every box the same height
    X = [0.055, 0.265, 0.515, 0.750]          # column left edges
    W = [0.160, 0.200, 0.185, 0.190]          # column widths
    BH = 0.165                                 # uniform box height
    ROW_A = 0.735                              # panel a lane (bottom edge)
    ROW_T = 0.355                              # panel b upper lane
    ROW_B = 0.145                              # panel b lower lane
    ROW_M = (ROW_T + ROW_B) / 2.0              # centred lane (shared space, portrait)
    ROW_P = 0.045                              # prototype attention lane
    cx = lambda i: X[i] + W[i] / 2.0
    right = lambda i: X[i] + W[i]

    # ================= panel a : image-language analogy =================
    ax.text(0.020, 0.960, "a", fontsize=9.5, fontweight="bold", va="top")
    ax.text(0.048, 0.958, "Image\u2013language analogy", fontsize=8.6, fontweight="bold", va="top")

    scene = fig.add_axes([X[0], ROW_A, W[0], BH])
    g = np.zeros((24, 24))
    g[:10, :] = 0.20
    g[10:, :] = 0.55
    scene.imshow(g, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    scene.add_patch(Circle((18.5, 4.0), 2.6, fc="#D8D8D8", ec="none"))
    scene.add_patch(Polygon([[3, 11], [8, 3], [13, 11]], fc="#9A9A9A", ec="none"))
    scene.add_patch(Rectangle((14, 12), 6, 8, fc="#7A7A7A", ec="none"))
    scene.add_patch(Rectangle((1.6, 9.6), 12.4, 11.0, fill=False, ec=P["blue"], lw=1.8))
    scene.add_patch(Rectangle((13.4, 11.4), 7.4, 9.2, fill=False, ec=P["orange"], lw=1.8))
    scene.set_xticks([])
    scene.set_yticks([])
    for sp in scene.spines.values():
        sp.set_visible(False)
    ax.text(X[0], ROW_A - 0.014, "many pixels and objects", fontsize=6.6, color="#444444", va="top")

    ya = ROW_A + BH / 2.0
    arrow(right(0) + 0.008, ya, X[1] - 0.008, ya, P["grey"])
    box(X[1], ROW_A, W[1] + 0.035, BH, "vision\u2013language model",
        "learns a relation between\nvisual patterns and words", P["grey"])
    arrow(X[1] + W[1] + 0.043, ya, X[2] + 0.027, ya, P["grey"])
    box(X[2] + 0.035, ROW_A, W[2] + W[3] - 0.010, BH, "natural-language description",
        "whole-scene description\nwith object relationships", P["grey"])

    ax.plot([0.020, 0.980], [0.665, 0.665], color="#E4E4E4", lw=0.8)

    # ================= panel b : RNA-language model =================
    ax.text(0.020, 0.625, "b", fontsize=9.5, fontweight="bold", va="top")
    ax.text(0.048, 0.623, "RNA\u2013language model", fontsize=8.6, fontweight="bold", va="top")

    rng = np.random.default_rng(3)
    expr = np.sin(np.linspace(0, 1, 60) * 13) + rng.normal(0, 0.23, 60)
    ep = fig.add_axes([X[0] + 0.026, ROW_T + 0.030, W[0] - 0.030, BH - 0.048])
    ep.plot(expr, color=P["blue"], lw=1.1)
    ep.fill_between(np.arange(len(expr)), expr, expr.min() - 0.2, color=P["blue_light"], alpha=0.55)
    ep.set_xticks([])
    ep.set_yticks([])
    ep.set_xlabel("genes", labelpad=1, fontsize=6.5)
    ep.set_ylabel("expression", labelpad=1, fontsize=6.5)
    for sp in ep.spines.values():
        sp.set_linewidth(0.55)
    ax.text(X[0], ROW_T - 0.012, "thousands of coordinated transcripts",
            fontsize=6.6, color="#444444", va="top")

    box(X[0], ROW_B, W[0], BH, "sample metadata",
        "site, tumour status,\ndisease label", P["orange"])

    yt = ROW_T + BH / 2.0
    yb = ROW_B + BH / 2.0
    ym = ROW_M + BH / 2.0
    arrow(right(0) + 0.008, yt, X[1] - 0.008, yt, P["blue"])
    box(X[1], ROW_T, W[1], BH, "RNA encoder",
        "log-expression MLP with\ngene gate and normalization", P["blue"])
    arrow(right(0) + 0.008, yb, X[1] - 0.008, yb, P["orange"])
    box(X[1], ROW_B, W[1], BH, "text encoder",
        "metadata-derived text\nencoded by MiniLM + MLP", P["orange"])

    box(X[2], ROW_M, W[2], BH, "shared space",
        "256-dimensional\nRNA\u2013text coordinates\ntrained by contrast", P["violet"])
    arrow(right(1) + 0.008, yt, X[2] - 0.008, ym + 0.030, P["blue"], rad=-0.22)
    arrow(right(1) + 0.008, yb, X[2] - 0.008, ym - 0.030, P["orange"], rad=0.22)

    box(X[2], ROW_P, W[2], BH, "prototype attention",
        "organizes the coordinates\ninto state components and\ninterpretation notes", P["green"])
    arrow(cx(2), ROW_M - 0.008, cx(2), ROW_P + BH + 0.008, P["violet"])

    box(X[3], ROW_M, W[3], BH, "RNA portrait",
        "sample-level state\nsignals expressed\nin language", P["green"])
    arrow(right(2) + 0.008, ym, X[3] - 0.008, ym, P["violet"])
    arrow(right(2) + 0.008, ROW_P + BH / 2.0, X[3] + 0.045, ROW_M - 0.008, P["green"], rad=-0.28)

    ax.text(0.020, -0.005,
            "The analogy is conceptual. RNA profiles are not images, but both are high-dimensional\n"
            "patterns whose meaning is distributed across many measured elements.",
            fontsize=6.8, color="#333333", va="bottom")

    FX.save_figure(fig, "Figure_1_architecture")



if __name__ == "__main__":
    main()
