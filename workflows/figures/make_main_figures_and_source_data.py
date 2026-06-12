from __future__ import annotations

import csv
import json
import math
import os
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.getenv("RNA_PORTRAIT_PROJECT_ROOT", Path.cwd())).resolve()
BUNDLE = Path(os.getenv("RNA_PORTRAIT_BUNDLE_ROOT", str(ROOT))).resolve()
TRAINED = Path(os.getenv("RNA_PORTRAIT_TRAINED_RESULTS_ROOT", str(BUNDLE / "trained_models_and_results"))).resolve()
RESULTS = Path(os.getenv("RNA_PORTRAIT_RESULTS_ROOT", str(TRAINED / "result_tables" / "analysis_results"))).resolve()
MODELS = Path(os.getenv("RNA_PORTRAIT_MODEL_ARTIFACT_ROOT", str(TRAINED / "trained_models" / "bulk_multimodal_embedding"))).resolve()
OUTPUT_ROOT = Path(os.getenv("RNA_PORTRAIT_FIGURE_OUTPUT_ROOT", str(ROOT / "figure_outputs"))).resolve()
FIG_DIR = OUTPUT_ROOT / "figures"
SOURCE_DIR = OUTPUT_ROOT / "source_data"


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
    "clean_anchor_override": "cleaner anchor context",
    "generic_context_override": "broad context",
    "unsupported_semantics": "weak evidence",
    "family_conflict": "conflict",
    "other": "other",
}

STATUS_LABELS = {
    "stable": "single clear signal",
    "mixed": "several signals",
    "unsupported": "weak evidence",
}

CLAIM_LABELS = {
    "epithelial_or_tumor_like_signal": "epithelial or tumour-like signal",
    "clean_or_non_malignant_context": "cleaner or non-malignant context",
    "mixed_or_unstable_disease_reading": "mixed or inconsistent label reading",
    "immune_or_blood_signal": "immune or blood signal",
    "context_or_stromal_signal": "stromal or broader-context signal",
}

CLAIM_TYPE_LABELS = {
    "epithelial_or_tumor_like_signal": "epithelial_or_tumour_like_signal",
    "clean_or_non_malignant_context": "cleaner_or_non_malignant_context",
    "mixed_or_unstable_disease_reading": "mixed_or_inconsistent_label_reading",
    "immune_or_blood_signal": "immune_or_blood_signal",
    "context_or_stromal_signal": "stromal_or_broader_context_signal",
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
    for path in [FIG_DIR, SOURCE_DIR]:
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
    if not path.exists() and rel == "rich_figures/rich_expression_umap_coordinates.csv":
        return compute_expression_coordinates()
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def sample_id_from_file(file_name: str) -> str:
    file_name = str(file_name)
    return file_name[:-4] if file_name.endswith(".txt") else file_name


def read_cpm_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "gene" not in df.columns:
        raise ValueError(f"missing gene column in {path}")
    return df.groupby("gene", as_index=True).mean(numeric_only=True)


def compute_expression_coordinates() -> pd.DataFrame:
    """Regenerate the external-profile PCA coordinates used in Fig. 2a."""
    out_dir = RESULTS / "rich_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rich_expression_umap_coordinates.csv"

    external_predictions = pd.read_csv(RESULTS / "T2_calibrated_closed_set" / "t2b_external_predictions.csv")
    external_predictions = external_predictions.loc[external_predictions["model"].eq("temperature_scaled_sgd")].copy()
    external_predictions["sample_id"] = external_predictions["file"].map(sample_id_from_file)
    closed = external_predictions[
        [
            "pool",
            "sample_id",
            "closed_set_disease_family",
            "closed_set_calibrated_confidence",
            "openworld_status",
            "closed_set_highconf_overcall_70",
        ]
    ].drop_duplicates(["pool", "sample_id"]).reset_index(drop=True)
    prediction_only_cols = [
        "model",
        "matched_selected_genes_baseline",
        "selected_gene_coverage_baseline",
        "closed_set_disease_family",
        "closed_set_calibrated_confidence",
        "expected_disease_family_mapped",
        "closed_set_expected_match_raw",
        "closed_set_expected_match_train_vocab",
        "openworld_status",
        "openworld_stable",
        "closed_set_over_openworld_mixed_or_unsupported",
        "closed_set_highconf_overcall_00",
        "closed_set_highconf_overcall_30",
        "closed_set_highconf_overcall_50",
        "closed_set_highconf_overcall_70",
        "closed_set_highconf_overcall_90",
    ]
    details = external_predictions.drop(columns=[c for c in prediction_only_cols if c in external_predictions.columns])
    details = details.drop_duplicates(["pool", "sample_id"]).reset_index(drop=True)

    external = read_cpm_matrix(RESULTS / "T4d_official_EPIC_deconvolution" / "external_180_cpm_matrix.tsv")
    multi = read_cpm_matrix(RESULTS / "T4d_official_EPIC_deconvolution" / "multisource_450_cpm_matrix.tsv")
    common_genes = external.index.intersection(multi.index)
    external = external.loc[common_genes].copy()
    multi = multi.loc[common_genes].copy()
    external.columns = [f"External-180||{col}" for col in external.columns]
    multi.columns = [f"MultiSource-450||{col}" for col in multi.columns]
    expr = pd.concat([external, multi], axis=1)
    expr = np.log1p(expr)

    records = []
    for coord_key in expr.columns:
        pool, sample_id = coord_key.split("||", 1)
        records.append({"coord_key": coord_key, "pool": pool, "sample_id": sample_id})
    sample_info = pd.DataFrame(records).merge(details, on=["pool", "sample_id"], how="left")
    sample_info = sample_info.merge(closed, on=["pool", "sample_id"], how="left")
    sample_keys = sample_info["coord_key"].tolist()
    expr = expr[sample_keys]

    variance = expr.var(axis=1).sort_values(ascending=False)
    top_genes = variance.head(min(2500, len(variance))).index
    x = expr.loc[top_genes].T.to_numpy(dtype=float)
    x = StandardScaler().fit_transform(x)
    n_pcs = min(30, x.shape[0] - 1, x.shape[1])
    pcs = PCA(n_components=n_pcs, random_state=13).fit_transform(x)
    coords = PCA(n_components=2, random_state=13).fit_transform(pcs)

    out = sample_info.copy()
    out["expr_x"] = coords[:, 0]
    out["expr_y"] = coords[:, 1]
    out["embedding_method"] = "PCA"
    out.to_csv(out_path, index=False)
    return out


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
        out = out_path.relative_to(OUTPUT_ROOT).as_posix()
    SOURCE_MANIFEST.append(
        {
            "name": name,
            "source": source,
            "description": description,
            "output_path": out,
        }
    )


def display_label_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Return a reader-facing copy with label names clarified."""
    out = df.copy()
    replacements = {
        "fixed disease label": "predefined disease label",
        "fixed disease": "predefined disease",
    }
    for col in ["factor_label", "control_label"]:
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
        paths[ext] = path.relative_to(OUTPUT_ROOT).as_posix()
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
    box(0.625, 0.64, 0.23, 0.17, "natural-language description", "a whole-scene description\nrather than one object label", PALETTE["grey"])

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
    box(0.615, 0.29, 0.17, 0.18, "shared space", "256-dimensional RNA-text\ncoordinates trained by contrast", PALETTE["violet"])
    arrow(0.785, 0.38, 0.875, 0.38, PALETTE["violet"])
    box(0.88, 0.29, 0.10, 0.18, "portrait", "state signals\nin language", PALETTE["green"])

    box(0.35, 0.09, 0.18, 0.13, "text encoder", "metadata-derived text\nencoded by MiniLM + MLP", PALETTE["orange"])
    arrow(0.53, 0.155, 0.615, 0.31, PALETTE["orange"])
    box(0.615, 0.09, 0.26, 0.13, "prototype attention", "organizes RNA-text coordinates into\nstate components and cautions", PALETTE["green"])
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


def methods_portrait_label_flowchart() -> dict[str, str]:
    set_style()
    fig = plt.figure(figsize=(7.2, 2.85))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.014",
            fc="white",
            ec=color,
            lw=0.95,
        )
        ax.add_patch(patch)
        ax.text(x + 0.014, y + h - 0.045, title, ha="left", va="top", fontsize=7.2, fontweight="bold", color=color)
        ax.text(x + 0.014, y + h - 0.105, body, ha="left", va="top", fontsize=5.7, color=PALETTE["black"], linespacing=1.22)

    def arrow(x0: float, y0: float, x1: float, y1: float, color: str = "#6E7D8E") -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=10,
                lw=0.9,
                color=color,
                shrinkA=3,
                shrinkB=3,
            )
        )

    steps = [
        (0.025, 0.63, 0.135, 0.23, "RNA profile", "standardized\n4,096-gene vector", PALETTE["black"]),
        (0.200, 0.63, 0.145, 0.23, "Shared space", "RNA encoder +\ntext projection\n256 dimensions", PALETTE["blue"]),
        (0.390, 0.63, 0.150, 0.23, "Semantic evidence", "nearest prototypes,\nnearest neighbours,\nsemantic majority", PALETTE["blue"]),
        (0.585, 0.63, 0.150, 0.23, "Context checks", "anchor context,\ncontext neighbours,\nsemantic tension", PALETTE["violet"]),
        (0.785, 0.63, 0.180, 0.23, "Portrait label", "reader-facing name\nfor an evidence\npattern", PALETTE["green"]),
    ]
    for args in steps:
        box(*args)
    for left, right in zip(steps[:-1], steps[1:]):
        x0, y0, w0, h0 = left[:4]
        x1 = right[0]
        arrow(x0 + w0, y0 + h0 / 2, x1, y0 + h0 / 2)

    ax.text(
        0.025,
        0.48,
        "The label is assigned from the semantic-state card after inference; it is not manually assigned for plotting.",
        ha="left",
        va="center",
        fontsize=6.4,
        color=PALETTE["black"],
    )

    mapping = [
        ("stable_consensus", "concordant evidence", "single clear signal", "semantic majority and prototype-neighbour context agree"),
        ("hematologic_override", "blood/immune evidence", "blood/immune", "blood or immune context changes the disease reading"),
        ("epithelial_override", "epithelial-like evidence", "epithelial-like context", "epithelial or solid-tumour-like signal appears within a broader context"),
        ("clean_anchor_override", "cleaner anchor evidence", "cleaner anchor context", "structured anchor looks cleaner than the tumour-like semantic frame"),
        ("generic_context_override", "broad-context evidence", "broad context", "context-heavy signal dominates a narrow disease-family call"),
        ("unsupported_semantics", "weak evidence", "weak evidence", "evidence is too weak for a disease-level reading"),
        ("family_conflict", "conflicting evidence", "conflict", "prototype and neighbours disagree at broad family level"),
    ]
    x0, y0, row_h = 0.035, 0.39, 0.044
    ax.text(x0, y0, "Evidence state", fontsize=6.2, fontweight="bold", color=PALETTE["black"], ha="left", va="top")
    ax.text(0.295, y0, "Figure label", fontsize=6.2, fontweight="bold", color=PALETTE["black"], ha="left", va="top")
    ax.text(0.510, y0, "Assignment meaning", fontsize=6.2, fontweight="bold", color=PALETTE["black"], ha="left", va="top")
    ax.plot([0.03, 0.97], [0.365, 0.365], color=PALETTE["grey_light"], lw=0.8)
    for i, (internal, evidence_state, label, meaning) in enumerate(mapping):
        y = 0.342 - i * row_h
        ax.text(x0, y, evidence_state, fontsize=5.6, color="#4D4D4D", ha="left", va="top")
        ax.text(0.295, y, label, fontsize=5.6, color=STATE_COLORS.get(internal, PALETTE["black"]), ha="left", va="top", fontweight="bold")
        ax.text(0.510, y, meaning, fontsize=5.55, color=PALETTE["black"], ha="left", va="top")
        if i < len(mapping) - 1:
            ax.plot([0.03, 0.97], [y - 0.026, y - 0.026], color="#F0F0F0", lw=0.55)

    add_source(
        "methods_portrait_label_flowchart",
        "bulk_multimodal_embedding/infer_unknown_rna_semantic_explainer.py and rna_portrait/external.py",
        "Schematic summary of molecular-portrait family assignment from semantic-state-card inference.",
        pd.DataFrame(
            [
                {"internal_state_family": internal, "evidence_state": evidence_state, "figure_label": label, "assignment_meaning": meaning}
                for internal, evidence_state, label, meaning in mapping
            ]
        ),
    )
    return save_figure(fig, "Methods_portrait_label_flowchart")


def figure_2_alignment() -> dict[str, str]:
    set_style()
    coords = read_csv("rich_figures/rich_expression_umap_coordinates.csv")
    cos = read_csv("T1_RNA_language_alignment/t1_cosine_summary.csv")
    exact = read_csv("T1_RNA_language_alignment/t1_exact_retrieval_metrics.csv")
    broad = read_csv("T1_RNA_language_alignment/t1_broad_semantic_retrieval.csv")
    add_source("figure_2_expression_coordinates", "rich_figures/rich_expression_umap_coordinates.csv", "Expression-space coordinates for 630 external profiles.", coords)
    add_source("figure_2_cosine_summary", "T1_RNA_language_alignment/t1_cosine_summary.csv", "Paired and shuffled RNA-text cosine summary.", cos)
    add_source("figure_2_exact_retrieval", "T1_RNA_language_alignment/t1_exact_retrieval_metrics.csv", "Exact RNA-text retrieval metrics.", exact)
    add_source("figure_2_broad_retrieval", "T1_RNA_language_alignment/t1_broad_semantic_retrieval.csv", "Broad semantic retrieval metrics.", broad)

    fig = plt.figure(figsize=(7.2, 4.45))
    gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1.22, 1.22, 0.92, 0.92], height_ratios=[1, 1], wspace=0.62, hspace=0.64)
    ax0 = fig.add_subplot(gs[:, :2])
    for state in STATE_ORDER:
        sub = coords.loc[coords["semantic_state_family"].eq(state)]
        if len(sub) == 0:
            continue
        ax0.scatter(sub["expr_x"], sub["expr_y"], s=11, alpha=0.74, lw=0, color=STATE_COLORS[state], label=STATE_LABELS[state])
    ax0.set_xlabel("RNA coordinate 1")
    ax0.set_ylabel("RNA coordinate 2")
    ax0.set_title("external RNA profiles")
    tidy_axis(ax0, None)
    panel_label(ax0, "a")
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.105),
        ncol=4,
        handletextpad=0.25,
        columnspacing=0.75,
        markerscale=0.78,
        borderaxespad=0,
    )

    ax1 = fig.add_subplot(gs[0, 2])
    sub = cos.set_index("group").loc[["shuffled", "paired"]].reset_index()
    y = sub["mean"].to_numpy()
    yerr = np.vstack([y - sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy() - y])
    ax1.bar([0, 1], y, color=[PALETTE["grey_light"], PALETTE["blue"]], width=0.58, zorder=3)
    ax1.errorbar([0, 1], y, yerr=yerr, fmt="none", ecolor=PALETTE["black"], elinewidth=0.65, capsize=2, zorder=4)
    ax1.set_xticks([0, 1], ["shuffled", "paired"])
    ax1.set_ylabel("RNA-text cosine")
    ax1.set_ylim(0, 0.78)
    ax1.set_title("paired text is closer", pad=7)
    tidy_axis(ax1)
    panel_label(ax1, "b", x=-0.18, y=1.15)

    ax2 = fig.add_subplot(gs[0, 3])
    e = exact.loc[exact["direction"].eq("rna_to_text")].copy()
    ax2.bar(np.arange(len(e)), e["lift_over_random"], width=0.62, color=PALETTE["blue_light"], zorder=3)
    ax2.set_xticks(np.arange(len(e)), e["metric"])
    ax2.set_ylabel("lift over random")
    ax2.set_title("exact pairing", pad=7)
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
    ax3.set_xticks(x, ["site", "tumour status", "disease family"])
    ax3.set_ylabel("top-1 semantic match")
    ax3.set_ylim(0, 1.03)
    ax3.legend(loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.8, bbox_to_anchor=(1.0, 1.01))
    ax3.set_title("broad semantic attributes", pad=8)
    tidy_axis(ax3)
    panel_label(ax3, "d", x=-0.12, y=1.15)
    return save_figure(fig, "Figure_2_RNA_language_alignment")


def figure_3_portrait_vs_fixed_labels() -> dict[str, str]:
    set_style()
    ext = read_csv("T2_calibrated_closed_set/t2b_external_thresholds.csv")
    pool_summary = read_csv("T2_calibrated_closed_set/t2b_external_pool_summary.csv")
    transitions = read_csv("sample_level_disease_transition/sample_level_transitions.csv")
    fixed_x_portrait = read_csv("T8_shortcut_exclusion_controls/t8_closed_set_disease_family_x_portrait.csv")
    label_diversity = predefined_label_portrait_diversity(fixed_x_portrait)
    add_source("figure_3_thresholds", "T2_calibrated_closed_set/t2b_external_thresholds.csv", "Confidence-threshold behaviour of predefined disease labels.", ext)
    add_source("figure_3_external_pool_summary", "T2_calibrated_closed_set/t2b_external_pool_summary.csv", "External predefined-label coverage and uncertain-but-labelled rates.", pool_summary)
    add_source("figure_3_sample_transitions", "sample_level_disease_transition/sample_level_transitions.csv", "Sample-level transition from raw labels to portrait status.", transitions)
    add_source("figure_3_predefined_label_portraits", "T8_shortcut_exclusion_controls/t8_closed_set_disease_family_x_portrait.csv", "Predefined disease labels crossed with portrait groups.", fixed_x_portrait)
    add_source("figure_3_predefined_label_diversity", "computed from predefined disease labels crossed with portrait groups", "Portrait diversity within each predefined disease label.", label_diversity)

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[0.95, 1.12, 1.38], height_ratios=[0.86, 1.16], wspace=0.55, hspace=0.64)

    ax0 = fig.add_subplot(gs[0, 0])
    sub = ext.loc[(ext["model"].eq("temperature_scaled_sgd")) & (ext["confidence_threshold"].eq(0.7))]
    sub = sub.set_index("pool").loc[["External-180", "MultiSource-450"]].reset_index()
    summ = pool_summary.loc[pool_summary["model"].eq("temperature_scaled_sgd")].set_index("pool").loc[["External-180", "MultiSource-450"]].reset_index()
    x = np.arange(len(sub))
    w = 0.34
    ax0.bar(x - w / 2, sub["coverage"], width=w, color=PALETTE["blue_light"], label="label >=0.70", zorder=3)
    ax0.bar(x + w / 2, summ["highconf_overcall_70"], width=w, color=PALETTE["red"], label="uncertain but labelled", zorder=3)
    ax0.set_xticks(x, ["Ext-180", "MS-450"])
    ymax = max(0.22, float(max(sub["coverage"].max(), summ["highconf_overcall_70"].max())) * 1.35)
    ax0.set_ylim(0, ymax)
    ax0.set_ylabel("fraction of profiles")
    ax0.set_title("predefined disease labels")
    ax0.legend(loc="upper right", handlelength=1.25, handletextpad=0.35)
    tidy_axis(ax0)
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1:])
    status = transitions.groupby(["pool", "resolved_status"]).size().reset_index(name="n")
    pools = ["External-180", "MultiSource-450"]
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
        ax1.bar(pools, vals, bottom=bottom, color=status_colors[status_key], label=STATUS_LABELS[status_key], zorder=3)
        bottom += np.array(vals)
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("fraction of profiles")
    ax1.set_title("portrait status keeps signal ambiguity", pad=7)
    ax1.legend(loc="upper right", ncol=1, handlelength=1.15, borderaxespad=0.25)
    tidy_axis(ax1)
    panel_label(ax1, "b", x=-0.08, y=1.14)
    add_source("figure_3_status_summary", "computed from sample_level_disease_transition/sample_level_transitions.csv", "Portrait-status fractions by external pool.", pd.DataFrame(status_summary))

    ax2 = fig.add_subplot(gs[1, 0])
    keep = label_diversity.loc[label_diversity["n_samples"].ge(10)].sort_values("n_samples", ascending=False).head(8)
    keep = keep.sort_values("portrait_diversity")
    y = np.arange(len(keep))
    ax2.barh(y, keep["portrait_diversity"], color=PALETTE["violet"], zorder=3)
    ax2.set_yticks(y, [wrap_label(label, 16) for label in keep["display_label"]])
    for yi, (_, row) in enumerate(keep.iterrows()):
        ax2.text(row["portrait_diversity"] + 0.025, yi, f"n={int(row['n_samples'])}", va="center", ha="left", fontsize=5.5)
    ax2.set_xlim(0, 1.08)
    ax2.set_xlabel("portrait diversity")
    ax2.set_title("diversity within disease names", pad=9)
    tidy_axis(ax2, "x")
    panel_label(ax2, "c", x=-0.14, y=1.22)

    ax3 = fig.add_subplot(gs[1, 1:])
    fx = fixed_x_portrait.copy()
    label_col = "closed_set_disease_family"
    fx["total"] = fx.drop(columns=[label_col]).sum(axis=1)
    fx = fx.sort_values("total", ascending=False).head(8)
    mat = fx.set_index(label_col)[[c for c in STATE_ORDER if c in fx.columns]]
    mat = mat.div(mat.sum(axis=1), axis=0).fillna(0)
    cmap = LinearSegmentedColormap.from_list("white_blue", ["#FFFFFF", "#D7E3F1", PALETTE["blue"]])
    im = ax3.imshow(mat.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=float(mat.max().max()))
    ax3.set_yticks(np.arange(mat.shape[0]), [wrap_label(x.replace("_", " "), 18) for x in mat.index])
    ax3.set_xticks(np.arange(mat.shape[1]), [wrap_label(STATE_LABELS.get(c, c), 10) for c in mat.columns], rotation=45, ha="right")
    ax3.tick_params(length=0)
    for s in ax3.spines.values():
        s.set_visible(False)
    cb = plt.colorbar(im, ax=ax3, fraction=0.030, pad=0.018)
    cb.ax.set_ylabel("fraction within disease name", rotation=90)
    ax3.set_xlabel("molecular portrait family")
    ax3.set_ylabel("predefined disease name")
    ax3.set_title("one disease name, several portraits", pad=9)
    panel_label(ax3, "d", x=-0.08, y=1.20)
    return save_figure(fig, "Figure_3_portraits_not_single_labels")


def figure_4_biology_grounding() -> dict[str, str]:
    set_style()
    marker = read_csv("T4_marker_program_state_validation/t4_marker_scores_by_state.csv")
    pathway = read_csv("T4b_pathway_state_validation/t4b_pathway_scores_by_state.csv")
    epic_eff = read_csv("T4d_official_EPIC_deconvolution/t4d_epic_state_vs_rest_effects.csv")
    mcp_eff = read_csv("T4e_official_MCPcounter_deconvolution/t4e_mcpcounter_state_vs_rest_effects.csv")
    claims = apply_reader_claim_labels(read_csv("T7_portrait_claim_grounding/t7_claim_support_summary.csv"))
    add_source("figure_4_marker_scores", "T4_marker_program_state_validation/t4_marker_scores_by_state.csv", "Marker programme scores by portrait group.", marker)
    add_source("figure_4_pathway_scores", "T4b_pathway_state_validation/t4b_pathway_scores_by_state.csv", "Pathway scores by portrait group.", pathway)
    add_source("figure_4_epic_effects", "T4d_official_EPIC_deconvolution/t4d_epic_state_vs_rest_effects.csv", "EPIC state-versus-rest effect sizes.", epic_eff)
    add_source("figure_4_mcp_effects", "T4e_official_MCPcounter_deconvolution/t4e_mcpcounter_state_vs_rest_effects.csv", "MCP-counter state-versus-rest effect sizes.", mcp_eff)
    add_source("figure_4_claim_support", "T7_portrait_claim_grounding/t7_claim_support_summary.csv", "Claim support summary across marker, pathway and deconvolution evidence.", claims)

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
            "ifn_gamma_score": "IFN-g",
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
    add_source("figure_4_marker_heatmap_matrix", "computed weighted mean from marker scores", "Weighted marker heatmap matrix used in panel a.", marker_source)
    add_source("figure_4_pathway_heatmap_matrix", "computed weighted mean from pathway scores", "Weighted pathway heatmap matrix used in panel b.", pathway_source)

    fig = plt.figure(figsize=(7.2, 4.9))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.10, 1.03], height_ratios=[1, 1], wspace=0.82, hspace=0.62)
    ax0 = fig.add_subplot(gs[0, 0])
    heatmap(ax0, m_state, -1.1, 1.1, cbar=True, label="z score")
    ax0.set_title("marker programmes")
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    heatmap(ax1, p_state, -1.1, 1.1, cbar=True, label="score")
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
    pivot_scores = list(dict.fromkeys(dot["score_label"].tolist()))
    for _, row in dot.iterrows():
        ax2.scatter(
            pivot_scores.index(row["score_label"]),
            pivot_states.index(row["state_label"]),
            s=12 + 10 * row["size"],
            c=row["cohen_d"],
            cmap=LinearSegmentedColormap.from_list("d", ["#2F5D8C", "#F8F8F8", "#B64A4A"]),
            vmin=-1.2,
            vmax=1.2,
            edgecolor="#333333",
            linewidth=0.25,
        )
    ax2.set_xticks(np.arange(len(pivot_scores)), pivot_scores, rotation=0, ha="center")
    ax2.set_yticks(np.arange(len(pivot_states)), pivot_states)
    ax2.set_title("deconvolution checks")
    ax2.set_xlim(-0.6, len(pivot_scores) - 0.4)
    ax2.set_ylim(len(pivot_states) - 0.5, -0.5)
    tidy_axis(ax2, None)
    panel_label(ax2, "c")

    ax3 = fig.add_subplot(gs[1, 1])
    claims = claims.sort_values("partial_or_strong_rate")
    y = np.arange(len(claims))
    ax3.barh(y, claims["partial_or_strong_rate"], color=PALETTE["green"], zorder=3)
    ax3.set_yticks(y, [wrap_label(label, 20) for label in claims["claim_label"]])
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("partial or strong support")
    ax3.set_title("portrait claims")
    tidy_axis(ax3, "x")
    panel_label(ax3, "d")
    return save_figure(fig, "Figure_4_biological_grounding")


def figure_5_stress_tests() -> dict[str, str]:
    set_style()
    mixing = read_csv("T5c_expanded_calibrated_mixing_bootstrap/t5c_fraction_summary.csv")
    partial = read_csv("T9_strict_shortcut_residual_controls/t9_partial_r2_after_controls.csv")
    shortcut = read_csv("T8_shortcut_exclusion_controls/t8_shortcut_association.csv")
    robust = read_csv("T10_quality_heterogeneity_robustness/t10_subset_robustness_summary.csv")
    boundary = read_csv("T11_failure_mode_reliability_boundaries/t11_boundary_flag_summary.csv")
    add_source("figure_5_mixing", "T5c_expanded_calibrated_mixing_bootstrap/t5c_fraction_summary.csv", "Controlled RNA-signal mixing summary with bootstrap intervals.", mixing)
    add_source("figure_5_partial_r2", "T9_strict_shortcut_residual_controls/t9_partial_r2_after_controls.csv", "Portrait incremental R2 after controlling source and metadata factors.", display_label_copy(partial))
    add_source("figure_5_metadata_association", "T8_shortcut_exclusion_controls/t8_shortcut_association.csv", "Association between metadata factors and portrait groups.", display_label_copy(shortcut))
    add_source("figure_5_robustness", "T10_quality_heterogeneity_robustness/t10_subset_robustness_summary.csv", "Robustness across data-quality subsets.", robust)
    add_source("figure_5_boundaries", "T11_failure_mode_reliability_boundaries/t11_boundary_flag_summary.csv", "Reliability boundary flags.", boundary)

    fig = plt.figure(figsize=(7.2, 5.15))
    gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1.22, 0.92, 1.08, 1.18], height_ratios=[1, 1], wspace=0.68, hspace=0.66)

    ax0 = fig.add_subplot(gs[:, 0])
    mix = mixing.loc[mixing["model"].eq("temperature_scaled_sgd")].copy()
    design_colors = {"normal_immune": PALETTE["green"], "tumor_immune": PALETTE["orange"], "tumor_normal": PALETTE["red"]}
    design_labels = {"normal_immune": "normal + immune", "tumor_immune": "tumour + immune", "tumor_normal": "tumour + normal"}
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
    ax0.set_ylabel("several-signal or weak-evidence rate")
    ax0.set_ylim(0, 1.05)
    ax0.set_xlim(-0.03, 1.12)
    ax0.set_title("controlled profile mixing")
    ax0.text(0.46, 1.005, "normal+immune", color=PALETTE["green"], fontsize=5.6, ha="left", va="center")
    ax0.text(0.46, 0.955, "tumour+immune", color=PALETTE["orange"], fontsize=5.6, ha="left", va="center")
    ax0.text(0.46, 0.905, "tumour+normal", color=PALETTE["red"], fontsize=5.6, ha="left", va="center")
    tidy_axis(ax0)
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    nmi = shortcut.loc[shortcut["target"].eq("semantic_state_family")].copy()
    nmi = nmi.loc[nmi["factor"].isin(["pool", "project_prefix", "project", "expected_site_family", "expected_disease_family", "closed_set_disease_family"])]
    nmi["label"] = nmi["factor_label"].replace(
        {
            "external pool": "pool",
            "source prefix": "source prefix",
            "project/source": "project",
            "expected tissue/site": "tissue",
            "expected disease label": "expected disease",
            "fixed disease label": "predefined disease",
        }
    )
    nmi = nmi.sort_values("nmi")
    y = np.arange(len(nmi))
    colors = [PALETTE["grey"] if x != "project" else PALETTE["red_light"] for x in nmi["label"]]
    ax1.barh(y, nmi["nmi"], color=colors, zorder=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels([])
    ax1.tick_params(axis="y", length=0)
    ax1.set_xlim(0, max(0.34, float(nmi["nmi"].max()) * 1.10))
    direct_labels = {
        "source prefix": "source\nprefix",
        "expected disease": "expected\ndisease",
        "predefined disease": "predefined\ndisease",
    }
    for yi, label in zip(y, nmi["label"]):
        ax1.text(0.007, yi, direct_labels.get(label, label), ha="left", va="center", fontsize=5.6, linespacing=0.86, color=PALETTE["black"])
    ax1.set_xlabel("NMI with portraits")
    ax1.set_title("simple metadata links")
    tidy_axis(ax1, "x")
    panel_label(ax1, "b")

    ax2 = fig.add_subplot(gs[0, 2:])
    controls = ["source_prefix", "expected_site", "expected_disease", "fixed_disease", "source_site_disease"]
    axes = ["immune_support", "context_support", "tumor_like_support", "mixed_evidence_support", "clean_context_support"]
    mat = (
        partial.loc[partial["control_id"].isin(controls) & partial["evidence_axis"].isin(axes)]
        .pivot(index="control_label", columns="evidence_axis_label", values="portrait_incremental_r2")
        .rename(index={"fixed disease": "predefined disease"})
        .reindex(["source + prefix", "tissue/site", "expected disease", "predefined disease", "source + site + disease"])
    )
    im = ax2.imshow(mat.to_numpy(), aspect="auto", cmap=LinearSegmentedColormap.from_list("white_blue", ["#FFFFFF", PALETTE["blue_soft"], PALETTE["blue"]]), vmin=0, vmax=0.22)
    ax2.set_xticks(np.arange(mat.shape[1]), [wrap_label(c, 10) for c in mat.columns], rotation=45, ha="right")
    ax2.set_yticks(np.arange(mat.shape[0]), [wrap_label(i, 15) for i in mat.index])
    ax2.tick_params(length=0)
    for s in ax2.spines.values():
        s.set_visible(False)
    cb = plt.colorbar(im, ax=ax2, fraction=0.047, pad=0.02)
    cb.ax.set_ylabel("incremental R2", rotation=90)
    ax2.set_title("after source controls")
    panel_label(ax2, "c")
    add_source("figure_5_partial_r2_matrix", "computed from source and metadata controls", "Matrix used in panel c.", mat.reset_index())

    ax3 = fig.add_subplot(gs[1, 1])
    keep = robust.loc[
        robust["subset_id"].isin(
            [
                "all_external",
                "exclude_lowest_10pct_coverage",
                "high_coverage_top50",
                "low_coverage_bottom50",
                "external_180_only",
                "multisource_450_only",
                "source_site_disease",
            ]
        )
    ].copy()
    if keep.empty:
        keep = robust.head(7).copy()
    keep = keep.head(7).iloc[::-1]
    ax3.barh(keep["subset_label"].map(lambda x: wrap_label(x, 22)), keep["mean_incremental_r2"], color=PALETTE["blue"], zorder=3)
    ax3.set_xlabel("mean incremental R2")
    ax3.set_title("robustness subsets")
    tidy_axis(ax3, "x")
    panel_label(ax3, "d")

    ax4 = fig.add_subplot(gs[1, 2:])
    b = boundary.sort_values("fraction_samples", ascending=True)
    labels = b["boundary_label"].map(lambda x: wrap_label(x, 18))
    b_colors = [PALETTE["red_light"] if v >= 0.28 else PALETTE["grey"] for v in b["fraction_samples"]]
    ax4.barh(labels, b["fraction_samples"], color=b_colors, zorder=3)
    ax4.set_xlim(0, max(0.62, float(b["fraction_samples"].max()) * 1.15))
    ax4.set_xlabel("fraction of external profiles")
    ax4.set_title("reliability boundaries")
    tidy_axis(ax4, "x")
    panel_label(ax4, "e")
    return save_figure(fig, "Figure_5_stress_tests_and_boundaries")


def extended_data_1_local_parts() -> dict[str, str]:
    set_style()
    cv = read_csv("T12_whole_profile_vs_local_parts/t12_cv_model_comparison.csv")
    reduc = read_csv("T12_whole_profile_vs_local_parts/t12_portrait_reducibility.csv")
    add_source("extended_data_1_cv_comparison", "T12_whole_profile_vs_local_parts/t12_cv_model_comparison.csv", "Cross-validated local-part and portrait comparisons.", cv)
    add_source("extended_data_1_portrait_reducibility", "T12_whole_profile_vs_local_parts/t12_portrait_reducibility.csv", "Prediction of portrait groups from labels and local evidence.", reduc)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.35, 0.95]})
    ax0 = axes[0]
    outcomes = ["immune_support", "context_support", "tumor_like_support", "mixed_evidence_support", "boundary_flag_count"]
    labels = ["immune", "context", "tumour-like", "several-signal", "boundary count"]
    pred_order = ["labels_only", "local_parts_only", "labels_plus_local_parts", "labels_local_plus_portrait"]
    colors = [PALETTE["grey_light"], PALETTE["blue_light"], PALETTE["blue"], PALETTE["green"]]
    x = np.arange(len(outcomes))
    width = 0.18
    for i, pred in enumerate(pred_order):
        vals = [cv.loc[(cv["outcome"].eq(o)) & (cv["predictor_set"].eq(pred)), "cv_r2"].mean() for o in outcomes]
        ax0.bar(x + (i - 1.5) * width, vals, width=width, color=colors[i], label=cv.loc[cv["predictor_set"].eq(pred), "predictor_label"].iloc[0], zorder=3)
    ax0.set_xticks(x, labels, rotation=35, ha="right")
    ax0.set_ylabel("cross-validated R2")
    ax0.set_title("local evidence explains many support axes")
    ax0.legend(loc="upper left", ncol=1)
    tidy_axis(ax0)
    panel_label(ax0, "a")

    ax1 = axes[1]
    reduc = reduc.set_index("predictor_set").loc[["majority_baseline", "labels_only", "local_parts_only", "labels_plus_local_parts"]].reset_index()
    ax1.barh(reduc["predictor_label"], reduc["macro_f1"], color=[PALETTE["grey_light"], PALETTE["blue_light"], PALETTE["blue"], PALETTE["green"]], zorder=3)
    ax1.set_xlim(0, 0.55)
    ax1.set_xlabel("macro F1")
    ax1.set_title("portrait groups are not reducible")
    tidy_axis(ax1, "x")
    panel_label(ax1, "b")
    fig.tight_layout(w_pad=1.2)
    return save_figure(fig, "Extended_Data_1_local_parts_boundary")



def write_source_manifest() -> None:
    """Write the source-data manifest for regenerated quantitative figure panels."""
    source_csv = SOURCE_DIR / "source_data_manifest.csv"
    source_csv.parent.mkdir(parents=True, exist_ok=True)
    with source_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "source", "description", "output_path"])
        writer.writeheader()
        writer.writerows(SOURCE_MANIFEST)


def main() -> None:
    ensure_dirs()
    figure_1_architecture()
    methods_portrait_label_flowchart()
    figure_2_alignment()
    figure_3_portrait_vs_fixed_labels()
    figure_4_biology_grounding()
    figure_5_stress_tests()
    extended_data_1_local_parts()
    write_source_manifest()


if __name__ == "__main__":
    main()
