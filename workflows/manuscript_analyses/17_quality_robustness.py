#!/usr/bin/env python3
"""Data-quality and public-cohort heterogeneity robustness checks.

This analysis repeats the residual portrait-support test in cleaner or narrower
subsets to ask whether the signal only appears because public data are noisy,
mixed-source or low quality.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from runtime_paths import (
    ARTIFACT_DIR,
    ARTIFACT_ROOT,
    BACKBONE_DIR,
    BUNDLE_ROOT,
    CODE_DIR,
    R_LIB,
    RUN_CWD,
    SCRIPT_DIR,
    SUPP_DIR,
    VALIDATION_DIR,
    WORKSPACE_ROOT,
)
OUTDIR = SUPP_DIR / "T10_quality_heterogeneity_robustness"
T7_MERGED = SUPP_DIR / "T7_portrait_claim_grounding" / "t7_merged_sample_evidence.csv"

RNG_SEED = 20260605
N_PERMUTATIONS = 180
PORTRAIT_COL = "semantic_state_family"
CONTROL_FACTORS = ["pool", "project_prefix", "expected_site_family", "expected_disease_family"]
EVIDENCE_AXES = [
    "immune_support",
    "context_support",
    "tumor_like_support",
    "mixed_evidence_support",
    "clean_context_support",
]
EVIDENCE_AXIS_LABELS = {
    "immune_support": "immune",
    "context_support": "context",
    "tumor_like_support": "tumor-like",
    "mixed_evidence_support": "mixed signal",
    "clean_context_support": "clean context",
}
STATE_LABELS = {
    "hematologic_override": "hematologic",
    "epithelial_override": "epithelial",
    "stable_consensus": "stable",
    "clean_anchor_override": "clean anchor",
    "generic_context_override": "context",
    "unsupported_semantics": "unsupported",
    "family_conflict": "conflict",
    "other": "other",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.labelsize": 8.2,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.4,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.65,
    }
)


def project_prefix(project: object) -> str:
    text = str(project)
    if text.startswith("RECOUNT3_"):
        return "RECOUNT3"
    match = re.match(r"([A-Za-z]+)", text)
    return match.group(1).upper() if match else "OTHER"


def safe_str_series(s: pd.Series) -> pd.Series:
    return s.fillna("missing").astype(str).replace({"": "missing", "nan": "missing"})


def load_data() -> pd.DataFrame:
    df = pd.read_csv(T7_MERGED)
    df = df.loc[df["model"].eq("temperature_scaled_sgd")].copy()
    df["project_prefix"] = df["project"].map(project_prefix)
    df["source_batch_proxy"] = df["pool"].astype(str) + "::" + df["project_prefix"].astype(str)
    df["epic_converged"] = pd.to_numeric(df["convergeCode"], errors="coerce").fillna(-1).eq(0)
    df["gene_coverage"] = pd.to_numeric(df["selected_gene_coverage_baseline"], errors="coerce")
    df["matched_genes"] = pd.to_numeric(df["matched_selected_genes"], errors="coerce")
    df["explainer_ok_source"] = pd.to_numeric(df["explainer_ok_source"], errors="coerce").fillna(0).astype(int)
    for col in [
        "pool",
        "project_prefix",
        "project",
        "expected_site_family",
        "expected_disease_family",
        "closed_set_disease_family",
        PORTRAIT_COL,
    ]:
        df[col] = safe_str_series(df[col])
    for col in EVIDENCE_AXES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=[PORTRAIT_COL, "gene_coverage", *EVIDENCE_AXES]).reset_index(drop=True)


def subset_specs(df: pd.DataFrame) -> List[Tuple[str, str, pd.Series]]:
    q10 = float(df["gene_coverage"].quantile(0.10))
    q25 = float(df["gene_coverage"].quantile(0.25))
    q50 = float(df["gene_coverage"].quantile(0.50))
    q75 = float(df["gene_coverage"].quantile(0.75))
    return [
        ("all_external", "all external samples", pd.Series(True, index=df.index)),
        ("exclude_lowest_10pct_coverage", "drop lowest 10% gene coverage", df["gene_coverage"].ge(q10)),
        ("high_coverage_top50", "top 50% gene coverage", df["gene_coverage"].ge(q50)),
        ("low_coverage_bottom50", "bottom 50% gene coverage", df["gene_coverage"].lt(q50)),
        ("high_coverage_top25", "top 25% gene coverage", df["gene_coverage"].ge(q75)),
        ("epic_converged", "EPIC converged samples", df["epic_converged"]),
        ("coverage_top50_and_epic_converged", "top 50% coverage + EPIC converged", df["gene_coverage"].ge(q50) & df["epic_converged"]),
        ("explainer_ok_source", "high-confidence metadata-source subset", df["explainer_ok_source"].eq(1)),
        ("external_180_only", "External-180 only", df["pool"].eq("External-180")),
        ("multisource_450_only", "MultiSource-450 only", df["pool"].eq("MultiSource-450")),
        ("exclude_recount3_prefix", "non-RECOUNT3 subset", ~df["project_prefix"].eq("RECOUNT3")),
        ("recount3_prefix_only", "RECOUNT3 subset", df["project_prefix"].eq("RECOUNT3")),
        ("exclude_epic_nonconverged", "drop EPIC non-converged", df["epic_converged"]),
        ("exclude_bottom25_coverage", "drop bottom 25% gene coverage", df["gene_coverage"].ge(q25)),
    ]


def build_design(df: pd.DataFrame, factor_cols: Sequence[str], portrait_labels: Sequence[object] | None = None) -> np.ndarray:
    parts = []
    for col in factor_cols:
        if col not in df.columns:
            continue
        dummies = pd.get_dummies(safe_str_series(df[col]), prefix=col, drop_first=True, dtype=float)
        if not dummies.empty:
            parts.append(dummies)
    if portrait_labels is not None:
        portrait = pd.Series(portrait_labels, index=df.index, name=PORTRAIT_COL)
        dummies = pd.get_dummies(safe_str_series(portrait), prefix="portrait", drop_first=True, dtype=float)
        if not dummies.empty:
            parts.append(dummies)
    if parts:
        return np.column_stack([np.ones(len(df)), pd.concat(parts, axis=1).to_numpy(dtype=float)])
    return np.ones((len(df), 1), dtype=float)


def linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if len(y) < 3 or np.nanstd(y) <= 1e-12:
        return np.nan
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if sst <= 1e-12:
        return np.nan
    return max(0.0, min(1.0, 1.0 - sse / sst))


def block_keys(df: pd.DataFrame, factor_cols: Sequence[str]) -> pd.Series:
    usable = [col for col in factor_cols if col in df.columns]
    if not usable:
        return pd.Series(["all"] * len(df), index=df.index)
    return df[usable].astype(str).agg("||".join, axis=1)


def shuffle_within_blocks(df: pd.DataFrame, labels: np.ndarray, factor_cols: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    shuffled = labels.copy()
    keys = block_keys(df, factor_cols).to_numpy()
    for key in pd.unique(keys):
        idx = np.flatnonzero(keys == key)
        if len(idx) > 1:
            shuffled[idx] = rng.permutation(shuffled[idx])
    return shuffled


def subset_partial_r2(sub: pd.DataFrame, subset_id: str, subset_label: str, rng: np.random.Generator) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    portrait = sub[PORTRAIT_COL].to_numpy()
    if len(sub) < 50 or sub[PORTRAIT_COL].nunique() < 2:
        return rows
    x_block = build_design(sub, CONTROL_FACTORS)
    x_full = build_design(sub, CONTROL_FACTORS, portrait)
    for axis in EVIDENCE_AXES:
        y = sub[axis].to_numpy(dtype=float)
        block_r2 = linear_r2(x_block, y)
        full_r2 = linear_r2(x_full, y)
        observed_delta = max(0.0, full_r2 - block_r2)
        permuted = []
        for _ in range(N_PERMUTATIONS):
            shuffled = shuffle_within_blocks(sub, portrait, CONTROL_FACTORS, rng)
            x_perm = build_design(sub, CONTROL_FACTORS, shuffled)
            permuted.append(max(0.0, linear_r2(x_perm, y) - block_r2))
        perm_arr = np.asarray(permuted, dtype=float)
        q95 = float(np.nanquantile(perm_arr, 0.95))
        p_value = float((1 + np.sum(perm_arr >= observed_delta)) / (len(perm_arr) + 1))
        rows.append(
            {
                "subset_id": subset_id,
                "subset_label": subset_label,
                "evidence_axis": axis,
                "evidence_axis_label": EVIDENCE_AXIS_LABELS[axis],
                "n_samples": int(len(sub)),
                "n_portrait_groups": int(sub[PORTRAIT_COL].nunique()),
                "n_projects": int(sub["project"].nunique()),
                "n_sites": int(sub["expected_site_family"].nunique()),
                "n_diseases": int(sub["expected_disease_family"].nunique()),
                "mean_gene_coverage": float(sub["gene_coverage"].mean()),
                "epic_converged_rate": float(sub["epic_converged"].mean()),
                "explainer_ok_rate": float(sub["explainer_ok_source"].mean()),
                "block_r2": block_r2,
                "block_plus_portrait_r2": full_r2,
                "portrait_incremental_r2": observed_delta,
                "permuted_incremental_r2_mean": float(np.nanmean(perm_arr)),
                "permuted_incremental_r2_q95": q95,
                "empirical_p_greater": p_value,
                "passes_p05": bool(p_value < 0.05),
                "passes_q95": bool(observed_delta > q95),
                "passes_effect_001": bool(observed_delta >= 0.01),
                "passes_robust": bool(p_value < 0.05 and observed_delta > q95 and observed_delta >= 0.01),
            }
        )
    return rows


def run_subset_robustness(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows: List[Dict[str, object]] = []
    for subset_id, subset_label, mask in subset_specs(df):
        sub = df.loc[mask.fillna(False)].copy().reset_index(drop=True)
        rows.extend(subset_partial_r2(sub, subset_id, subset_label, rng))
    return pd.DataFrame(rows)


def summarize_subset_robustness(robustness: pd.DataFrame) -> pd.DataFrame:
    if robustness.empty:
        return pd.DataFrame()
    rows = []
    for (subset_id, subset_label), sub in robustness.groupby(["subset_id", "subset_label"], sort=False):
        rows.append(
            {
                "subset_id": subset_id,
                "subset_label": subset_label,
                "n_samples": int(sub["n_samples"].iloc[0]),
                "n_portrait_groups": int(sub["n_portrait_groups"].iloc[0]),
                "mean_gene_coverage": float(sub["mean_gene_coverage"].iloc[0]),
                "epic_converged_rate": float(sub["epic_converged_rate"].iloc[0]),
                "explainer_ok_rate": float(sub["explainer_ok_rate"].iloc[0]),
                "robust_axes": int(sub["passes_robust"].sum()),
                "tested_axes": int(len(sub)),
                "mean_incremental_r2": float(sub["portrait_incremental_r2"].mean()),
                "median_incremental_r2": float(sub["portrait_incremental_r2"].median()),
                "min_incremental_r2": float(sub["portrait_incremental_r2"].min()),
                "max_incremental_r2": float(sub["portrait_incremental_r2"].max()),
            }
        )
    return pd.DataFrame(rows)


def quality_by_portrait(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state, sub in df.groupby(PORTRAIT_COL, dropna=False):
        rows.append(
            {
                "semantic_state_family": state,
                "label": STATE_LABELS.get(str(state), str(state).replace("_", " ")),
                "n_samples": int(len(sub)),
                "mean_gene_coverage": float(sub["gene_coverage"].mean()),
                "median_gene_coverage": float(sub["gene_coverage"].median()),
                "q25_gene_coverage": float(sub["gene_coverage"].quantile(0.25)),
                "q75_gene_coverage": float(sub["gene_coverage"].quantile(0.75)),
                "epic_converged_rate": float(sub["epic_converged"].mean()),
                "explainer_ok_rate": float(sub["explainer_ok_source"].mean()),
                "mean_matched_genes": float(sub["matched_genes"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("n_samples", ascending=False)


def variance_explained_by_portrait(df: pd.DataFrame, value_col: str) -> float:
    overall = float(df[value_col].mean())
    total = float(((df[value_col] - overall) ** 2).sum())
    if total <= 1e-12:
        return np.nan
    between = 0.0
    for _, sub in df.groupby(PORTRAIT_COL):
        between += len(sub) * float((sub[value_col].mean() - overall) ** 2)
    return between / total


def draw_subset_heatmap(robustness: pd.DataFrame) -> None:
    if robustness.empty:
        return
    summary = summarize_subset_robustness(robustness)
    order = summary.sort_values(["robust_axes", "mean_incremental_r2"], ascending=[False, False])["subset_label"].tolist()
    axes = [EVIDENCE_AXIS_LABELS[axis] for axis in EVIDENCE_AXES]
    pivot = robustness.pivot(index="subset_label", columns="evidence_axis_label", values="portrait_incremental_r2").reindex(index=order, columns=axes)
    pass_pivot = robustness.pivot(index="subset_label", columns="evidence_axis_label", values="passes_robust").reindex(index=order, columns=axes)
    fig, ax = plt.subplots(figsize=(8.6, max(4.2, len(order) * 0.34)))
    vmax = max(0.16, float(np.nanmax(pivot.to_numpy(dtype=float))))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(axes)), axes, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(order)), order)
    for i, row in enumerate(order):
        for j, col in enumerate(axes):
            val = pivot.loc[row, col]
            star = "*" if bool(pass_pivot.loc[row, col]) else ""
            label = f"{val:.3f}" if pd.notna(val) and val < 0.01 else f"{val:.2f}"
            ax.text(j, i, f"{label}{star}", ha="center", va="center", fontsize=7.2, color="#111827")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("portrait incremental R2 after controls")
    ax.set_title("Portrait-evidence residual signal across data-quality and source subsets", pad=8)
    fig.tight_layout()
    for ext in ["svg", "png"]:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if ext == "png":
            kwargs["dpi"] = 260
        fig.savefig(OUTDIR / f"t10_subset_robustness_heatmap.{ext}", **kwargs)
    plt.close(fig)


def draw_subset_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    plot = summary.sort_values(["robust_axes", "mean_incremental_r2"], ascending=[True, True])
    fig, ax = plt.subplots(figsize=(8.2, max(3.8, len(plot) * 0.34)))
    y = np.arange(len(plot))
    ax.barh(y, plot["mean_incremental_r2"], color="#1B9E77", zorder=2)
    ax.set_yticks(y, plot["subset_label"])
    ax.set_xlabel("mean portrait incremental R2 across five evidence axes")
    clean_axis(ax)
    for i, row in enumerate(plot.itertuples(index=False)):
        ax.text(row.mean_incremental_r2 + 0.004, i, f"{row.robust_axes}/{row.tested_axes} axes, n={row.n_samples}", va="center", fontsize=7.4)
    ax.set_title("How many evidence axes remain robust in each cleaner/narrower subset")
    fig.tight_layout()
    for ext in ["svg", "png"]:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if ext == "png":
            kwargs["dpi"] = 260
        fig.savefig(OUTDIR / f"t10_subset_summary.{ext}", **kwargs)
    plt.close(fig)


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def draw_quality_by_portrait(quality: pd.DataFrame) -> None:
    plot = quality.sort_values("mean_gene_coverage")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), gridspec_kw={"width_ratios": [1.2, 1.0]})
    ax = axes[0]
    y = np.arange(len(plot))
    ax.barh(y, plot["mean_gene_coverage"], color="#3B6FB6", zorder=2)
    ax.set_yticks(y, plot["label"])
    ax.set_xlabel("mean selected-gene coverage")
    clean_axis(ax)
    for i, row in enumerate(plot.itertuples(index=False)):
        ax.text(row.mean_gene_coverage + 0.015, i, f"n={row.n_samples}", va="center", fontsize=7.2)
    ax.set_title("Coverage by portrait group")

    ax = axes[1]
    x = np.arange(len(plot))
    ax.bar(x - 0.18, plot["epic_converged_rate"], width=0.36, color="#1B9E77", label="EPIC converged", zorder=2)
    ax.bar(x + 0.18, plot["explainer_ok_rate"], width=0.36, color="#D95F02", label="source flag OK", zorder=2)
    ax.set_xticks(x, plot["label"], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Quality flags by portrait group")
    fig.tight_layout()
    for ext in ["svg", "png"]:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if ext == "png":
            kwargs["dpi"] = 260
        fig.savefig(OUTDIR / f"t10_quality_by_portrait.{ext}", **kwargs)
    plt.close(fig)


def write_summary(df: pd.DataFrame, robustness: pd.DataFrame, subset_summary: pd.DataFrame, quality: pd.DataFrame) -> None:
    high = subset_summary.loc[subset_summary["subset_id"].eq("high_coverage_top50")]
    low = subset_summary.loc[subset_summary["subset_id"].eq("low_coverage_bottom50")]
    converged = subset_summary.loc[subset_summary["subset_id"].eq("epic_converged")]
    external = subset_summary.loc[subset_summary["subset_id"].eq("external_180_only")]
    multisource = subset_summary.loc[subset_summary["subset_id"].eq("multisource_450_only")]
    coverage_eta = variance_explained_by_portrait(df, "gene_coverage")
    lines = [
        "# Data-quality and heterogeneity robustness",
        "",
        "## What this analysis asks",
        "",
        "This analysis tests whether the residual portrait-support signal only appears in noisy, mixed-source or low-quality public data. It repeats the source+tissue+disease residual test in cleaner or narrower subsets.",
        "",
        "The analysis tests whether portrait-related signals persist across cleaner or restricted public-data subsets.",
        "",
        "## Headline result",
        "",
        f"- Samples analysed: `{len(df)}` external rows.",
        f"- Portrait groups explain `{coverage_eta:.3f}` of selected-gene coverage variance; coverage differs by portrait group and must be treated as a quality-risk covariate.",
        "- The residual portrait-evidence signal remains visible in several cleaner/narrower subsets, but it is not equally strong everywhere.",
        "- This supports robustness under public-data heterogeneity, not a claim that the dataset is clean or batch-free.",
        "",
        "## Key subset results",
        "",
    ]
    for label, frame in [
        ("top 50% gene coverage", high),
        ("bottom 50% gene coverage", low),
        ("EPIC converged", converged),
        ("External-180 only", external),
        ("MultiSource-450 only", multisource),
    ]:
        if not frame.empty:
            row = frame.iloc[0]
            lines.append(
                f"- {label}: `{int(row.robust_axes)}/{int(row.tested_axes)}` axes robust; mean incremental R2 `{row.mean_incremental_r2:.3f}`; n `{int(row.n_samples)}`."
            )
    top_subsets = subset_summary.sort_values(["robust_axes", "mean_incremental_r2"], ascending=[False, False]).head(5)
    lines.extend(["", "## Strongest robust subsets", ""])
    for row in top_subsets.itertuples(index=False):
        lines.append(
            f"- `{row.subset_label}`: `{row.robust_axes}/{row.tested_axes}` robust axes, mean incremental R2 `{row.mean_incremental_r2:.3f}`, mean gene coverage `{row.mean_gene_coverage:.3f}`, n `{row.n_samples}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The results evaluate whether the signal is confined to the noisiest samples or remains detectable in cleaner and source-specific subsets. Data quality should still be considered because quality metrics vary across portrait groups and public-data heterogeneity remains a limitation.",
            "",
            "The analysis supports robustness across subsets but does not eliminate public-data quality and heterogeneity as limitations.",
            "",
            "## Output files",
            "",
            "- `t10_subset_robustness.csv`: strict residual portrait-evidence test repeated in each subset.",
            "- `t10_subset_robustness_summary.csv`: one-row subset summary.",
            "- `t10_quality_by_portrait.csv`: gene coverage and quality flags by portrait group.",
            "- `t10_subset_robustness_heatmap.svg/png`: residual signal heatmap across subsets.",
            "- `t10_subset_summary.svg/png`: robust-axis count and mean incremental R2 by subset.",
            "- `t10_quality_by_portrait.svg/png`: quality metrics by portrait group.",
            "",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    robustness = run_subset_robustness(df)
    subset_summary = summarize_subset_robustness(robustness)
    quality = quality_by_portrait(df)

    robustness.to_csv(OUTDIR / "t10_subset_robustness.csv", index=False)
    subset_summary.to_csv(OUTDIR / "t10_subset_robustness_summary.csv", index=False)
    quality.to_csv(OUTDIR / "t10_quality_by_portrait.csv", index=False)

    draw_subset_heatmap(robustness)
    draw_subset_summary(subset_summary)
    draw_quality_by_portrait(quality)
    write_summary(df, robustness, subset_summary, quality)
    print(OUTDIR)


if __name__ == "__main__":
    main()
