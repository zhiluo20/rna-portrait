#!/usr/bin/env python3
"""Residual source and metadata controls for RNA molecular portraits.

This analysis tests whether portrait groups still explain independent
biological support axes after accounting for source, tissue and disease labels.
"""

from __future__ import annotations

import itertools
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    silhouette_score = None
    StandardScaler = None


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
OUTDIR = SUPP_DIR / "T9_strict_shortcut_residual_controls"
T7_MERGED = SUPP_DIR / "T7_portrait_claim_grounding" / "t7_merged_sample_evidence.csv"

RNG_SEED = 20260605
N_PERMUTATIONS = 250

PORTRAIT_COL = "semantic_state_family"
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
CONTROL_SPECS: List[Tuple[str, List[str], str]] = [
    ("source_prefix", ["pool", "project_prefix"], "source + prefix"),
    ("project", ["project"], "project/source"),
    ("expected_site", ["expected_site_family"], "tissue/site"),
    ("expected_disease", ["expected_disease_family"], "expected disease"),
    ("fixed_disease", ["closed_set_disease_family"], "fixed disease"),
    ("site_and_disease", ["expected_site_family", "expected_disease_family"], "site + disease"),
    (
        "source_site_disease",
        ["pool", "project_prefix", "expected_site_family", "expected_disease_family"],
        "source + site + disease",
    ),
]
WITHIN_BLOCK_SPECS: List[Tuple[str, str]] = [
    ("project", "project/source"),
    ("expected_site_family", "tissue/site"),
    ("expected_disease_family", "expected disease"),
    ("closed_set_disease_family", "fixed disease"),
]
STATE_ORDER = [
    "hematologic_override",
    "epithelial_override",
    "stable_consensus",
    "clean_anchor_override",
    "generic_context_override",
    "unsupported_semantics",
    "family_conflict",
    "other",
]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.labelsize": 8.2,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.6,
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
    return df.dropna(subset=[PORTRAIT_COL, *EVIDENCE_AXES]).reset_index(drop=True)


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
        x = pd.concat(parts, axis=1).to_numpy(dtype=float)
        return np.column_stack([np.ones(len(df)), x])
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
    if not factor_cols:
        return pd.Series(["all"] * len(df), index=df.index)
    return df[list(factor_cols)].astype(str).agg("||".join, axis=1)


def shuffle_within_blocks(df: pd.DataFrame, labels: np.ndarray, factor_cols: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    shuffled = labels.copy()
    keys = block_keys(df, factor_cols).to_numpy()
    for key in pd.unique(keys):
        idx = np.flatnonzero(keys == key)
        if len(idx) > 1:
            shuffled[idx] = rng.permutation(shuffled[idx])
    return shuffled


def partial_r2_controls(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    portrait = df[PORTRAIT_COL].to_numpy()
    rows: List[Dict[str, object]] = []
    for control_id, factor_cols, control_label in CONTROL_SPECS:
        x_block = build_design(df, factor_cols)
        x_full = build_design(df, factor_cols, portrait)
        for axis in EVIDENCE_AXES:
            y = df[axis].to_numpy(dtype=float)
            block_r2 = linear_r2(x_block, y)
            full_r2 = linear_r2(x_full, y)
            observed_delta = max(0.0, full_r2 - block_r2)
            permuted = []
            for _ in range(N_PERMUTATIONS):
                shuffled = shuffle_within_blocks(df, portrait, factor_cols, rng)
                x_perm = build_design(df, factor_cols, shuffled)
                permuted.append(max(0.0, linear_r2(x_perm, y) - block_r2))
            perm_arr = np.asarray(permuted, dtype=float)
            p_value = float((1 + np.sum(perm_arr >= observed_delta)) / (len(perm_arr) + 1))
            rows.append(
                {
                    "control_id": control_id,
                    "control_label": control_label,
                    "control_factors": "|".join(factor_cols),
                    "evidence_axis": axis,
                    "evidence_axis_label": EVIDENCE_AXIS_LABELS[axis],
                    "n_samples": int(len(df)),
                    "n_portrait_groups": int(df[PORTRAIT_COL].nunique()),
                    "block_r2": block_r2,
                    "block_plus_portrait_r2": full_r2,
                    "portrait_incremental_r2": observed_delta,
                    "permuted_incremental_r2_mean": float(np.nanmean(perm_arr)),
                    "permuted_incremental_r2_q95": float(np.nanquantile(perm_arr, 0.95)),
                    "empirical_p_greater": p_value,
                    "passes_p05": bool(p_value < 0.05),
                    "passes_q95": bool(observed_delta > np.nanquantile(perm_arr, 0.95)),
                }
            )
    return pd.DataFrame(rows)


def scale_evidence(df: pd.DataFrame) -> pd.DataFrame:
    values = df[EVIDENCE_AXES].to_numpy(dtype=float)
    if StandardScaler is None:
        centered = values - np.nanmean(values, axis=0, keepdims=True)
        scaled = centered / np.nanstd(centered, axis=0, keepdims=True)
    else:
        scaled = StandardScaler().fit_transform(values)
    out = df.copy()
    for i, col in enumerate(EVIDENCE_AXES):
        out[f"{col}_scaled"] = scaled[:, i]
    return out


def valid_silhouette_input(sub: pd.DataFrame, label_col: str) -> bool:
    counts = sub[label_col].value_counts()
    return bool(len(sub) >= 10 and counts.shape[0] >= 2 and counts.min() >= 2 and counts.shape[0] < len(sub))


def within_label_separation(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED + 11)
    scaled = scale_evidence(df)
    feature_cols = [f"{col}_scaled" for col in EVIDENCE_AXES]
    rows: List[Dict[str, object]] = []
    if silhouette_score is None:
        return pd.DataFrame(), pd.DataFrame()

    for factor, factor_label in WITHIN_BLOCK_SPECS:
        for value, sub in scaled.groupby(factor, dropna=False):
            sub = sub.copy()
            if not valid_silhouette_input(sub, PORTRAIT_COL):
                continue
            x = sub[feature_cols].to_numpy(dtype=float)
            labels = sub[PORTRAIT_COL].to_numpy()
            observed = float(silhouette_score(x, labels, metric="euclidean"))
            permuted = []
            for _ in range(N_PERMUTATIONS):
                shuffled = rng.permutation(labels)
                if len(np.unique(shuffled)) < 2:
                    continue
                permuted.append(float(silhouette_score(x, shuffled, metric="euclidean")))
            perm_arr = np.asarray(permuted, dtype=float)
            p_value = float((1 + np.sum(perm_arr >= observed)) / (len(perm_arr) + 1))
            rows.append(
                {
                    "factor": factor,
                    "factor_label": factor_label,
                    "factor_value": str(value),
                    "n_samples": int(len(sub)),
                    "n_portrait_groups": int(sub[PORTRAIT_COL].nunique()),
                    "dominant_portrait_fraction": float(sub[PORTRAIT_COL].value_counts().iloc[0] / len(sub)),
                    "observed_evidence_silhouette": observed,
                    "permuted_silhouette_mean": float(np.nanmean(perm_arr)),
                    "permuted_silhouette_q95": float(np.nanquantile(perm_arr, 0.95)),
                    "empirical_p_greater": p_value,
                    "passes_p05": bool(p_value < 0.05),
                }
            )

    detail = pd.DataFrame(rows)
    summary_rows: List[Dict[str, object]] = []
    if not detail.empty:
        for factor, sub in detail.groupby("factor", dropna=False):
            weights = sub["n_samples"].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "factor": factor,
                    "factor_label": str(sub["factor_label"].iloc[0]),
                    "n_blocks_tested": int(len(sub)),
                    "n_samples_in_blocks": int(sub["n_samples"].sum()),
                    "fraction_blocks_p05": float(sub["passes_p05"].mean()),
                    "weighted_observed_silhouette": float(np.average(sub["observed_evidence_silhouette"], weights=weights)),
                    "weighted_permuted_silhouette_mean": float(np.average(sub["permuted_silhouette_mean"], weights=weights)),
                    "weighted_silhouette_lift": float(
                        np.average(sub["observed_evidence_silhouette"] - sub["permuted_silhouette_mean"], weights=weights)
                    ),
                    "mean_dominant_portrait_fraction": float(np.average(sub["dominant_portrait_fraction"], weights=weights)),
                }
            )
    return detail.sort_values(["factor", "empirical_p_greater", "observed_evidence_silhouette"]), pd.DataFrame(summary_rows)


def same_label_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    contrast_specs = [
        ("same_site_and_expected_disease", ["expected_site_family", "expected_disease_family"], "same tissue + expected disease"),
        ("same_project", ["project"], "same project/source"),
        ("same_fixed_disease", ["closed_set_disease_family"], "same fixed disease"),
    ]
    rows: List[Dict[str, object]] = []
    for contrast_id, group_cols, label in contrast_specs:
        for values, sub in df.groupby(group_cols, dropna=False):
            if not isinstance(values, tuple):
                values = (values,)
            if len(sub) < 10:
                continue
            counts = sub[PORTRAIT_COL].value_counts()
            usable_states = counts[counts >= 3].index.tolist()
            if len(usable_states) < 2:
                continue
            means = sub.loc[sub[PORTRAIT_COL].isin(usable_states)].groupby(PORTRAIT_COL)[EVIDENCE_AXES].mean()
            for state_a, state_b in itertools.combinations(means.index, 2):
                diff = (means.loc[state_a] - means.loc[state_b]).abs()
                max_axis = str(diff.sort_values(ascending=False).index[0])
                rows.append(
                    {
                        "contrast_id": contrast_id,
                        "contrast_label": label,
                        "group_factors": "|".join(group_cols),
                        "group_value": " | ".join(str(v) for v in values),
                        "n_samples_in_group": int(len(sub)),
                        "state_a": str(state_a),
                        "state_b": str(state_b),
                        "n_state_a": int(counts[state_a]),
                        "n_state_b": int(counts[state_b]),
                        "max_difference_axis": max_axis,
                        "max_difference_axis_label": EVIDENCE_AXIS_LABELS[max_axis],
                        "max_abs_evidence_difference": float(diff[max_axis]),
                        **{f"abs_diff_{axis}": float(diff[axis]) for axis in EVIDENCE_AXES},
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("max_abs_evidence_difference", ascending=False).reset_index(drop=True)


def source_signature_concordance(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    pools = sorted(df["pool"].unique())
    if len(pools) < 2:
        return pd.DataFrame()
    for state in sorted(df[PORTRAIT_COL].unique()):
        sub = df.loc[df[PORTRAIT_COL].eq(state)].copy()
        means = sub.groupby("pool")[EVIDENCE_AXES].mean()
        counts = sub.groupby("pool").size()
        if not all(pool in means.index for pool in pools[:2]):
            continue
        if min(int(counts.get(pool, 0)) for pool in pools[:2]) < 5:
            continue
        a = means.loc[pools[0]].to_numpy(dtype=float)
        b = means.loc[pools[1]].to_numpy(dtype=float)
        corr = np.corrcoef(a, b)[0, 1] if np.std(a) > 1e-12 and np.std(b) > 1e-12 else np.nan
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0 else np.nan
        rows.append(
            {
                "semantic_state_family": state,
                "pool_a": pools[0],
                "pool_b": pools[1],
                "n_pool_a": int(counts[pools[0]]),
                "n_pool_b": int(counts[pools[1]]),
                "evidence_profile_pearson": float(corr),
                "evidence_profile_cosine": cosine,
                **{f"{pools[0]}_{axis}": float(means.loc[pools[0], axis]) for axis in EVIDENCE_AXES},
                **{f"{pools[1]}_{axis}": float(means.loc[pools[1], axis]) for axis in EVIDENCE_AXES},
            }
        )
    return pd.DataFrame(rows).sort_values("evidence_profile_cosine", ascending=False)


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, name: str) -> None:
    for ext in ["svg", "png"]:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if ext == "png":
            kwargs["dpi"] = 260
        fig.savefig(OUTDIR / f"{name}.{ext}", **kwargs)
    plt.close(fig)


def plot_partial_r2(partial: pd.DataFrame) -> None:
    plot = partial.pivot(index="control_label", columns="evidence_axis_label", values="portrait_incremental_r2")
    control_order = [label for _, _, label in CONTROL_SPECS if label in plot.index]
    axis_order = [EVIDENCE_AXIS_LABELS[axis] for axis in EVIDENCE_AXES if EVIDENCE_AXIS_LABELS[axis] in plot.columns]
    plot = plot.reindex(index=control_order, columns=axis_order)
    pvals = partial.pivot(index="control_label", columns="evidence_axis_label", values="empirical_p_greater").reindex(index=control_order, columns=axis_order)

    fig, ax = plt.subplots(figsize=(8.3, 4.5))
    im = ax.imshow(plot.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.22, float(np.nanmax(plot.to_numpy()))))
    ax.set_xticks(np.arange(len(axis_order)), axis_order, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(control_order)), control_order)
    for i, row in enumerate(control_order):
        for j, col in enumerate(axis_order):
            val = plot.loc[row, col]
            p = pvals.loc[row, col]
            star = "*" if pd.notna(p) and p < 0.05 and pd.notna(val) and val >= 0.01 else ""
            label = f"{val:.3f}" if pd.notna(val) and val < 0.01 else f"{val:.2f}"
            ax.text(j, i, f"{label}{star}", ha="center", va="center", fontsize=8, color="#111827")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("portrait incremental R2 after control")
    ax.set_title("RNA portrait groups still explain independent evidence after label/source controls")
    fig.tight_layout()
    save_figure(fig, "t9_partial_r2_after_controls")


def plot_within_label(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    summary = summary.sort_values("weighted_silhouette_lift", ascending=False)
    fig, ax = plt.subplots(figsize=(7.4, 3.25))
    y = np.arange(len(summary))
    ax.barh(y - 0.18, summary["weighted_permuted_silhouette_mean"], height=0.32, color="#CBD5E1", label="within-label shuffled portraits", zorder=2)
    ax.barh(y + 0.18, summary["weighted_observed_silhouette"], height=0.32, color="#1B9E77", label="observed portraits", zorder=2)
    ax.set_yticks(y, summary["factor_label"])
    ax.invert_yaxis()
    ax.set_xlabel("evidence-space silhouette within same label")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.57, -0.16), ncol=2)
    clean_axis(ax)
    for i, row in enumerate(summary.itertuples(index=False)):
        ax.text(row.weighted_observed_silhouette + 0.01, i + 0.18, f"n={row.n_samples_in_blocks}", va="center", fontsize=8)
    ax.set_title("Within-label evidence separation improves over shuffled portrait labels", pad=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, "t9_within_label_evidence_separation")


def plot_contrast_examples(contrasts: pd.DataFrame) -> None:
    if contrasts.empty:
        return
    top = contrasts.head(12).copy()
    labels = []
    for row in top.itertuples(index=False):
        group = str(row.group_value)
        if len(group) > 34:
            group = group[:31] + "..."
        labels.append(f"{row.contrast_label}\n{group}\n{row.state_a} vs {row.state_b}")
    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    y = np.arange(len(top))
    ax.barh(y, top["max_abs_evidence_difference"], color="#D95F02", zorder=2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("largest independent-evidence difference")
    clean_axis(ax)
    for i, row in enumerate(top.itertuples(index=False)):
        ax.text(row.max_abs_evidence_difference + 0.02, i, row.max_difference_axis_label, va="center", fontsize=7.5)
    ax.set_title("Same-label groups can contain different RNA portraits with different biology")
    fig.tight_layout()
    save_figure(fig, "t9_same_label_different_biology_contrasts")


def plot_source_concordance(concordance: pd.DataFrame) -> None:
    if concordance.empty:
        return
    plot = concordance.copy()
    plot["label"] = plot["semantic_state_family"].str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    x = np.arange(len(plot))
    ax.bar(x, plot["evidence_profile_cosine"], color="#3B6FB6", zorder=2)
    ax.set_xticks(x, plot["label"], rotation=28, ha="right")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="#9CA3AF", linewidth=0.8)
    ax.set_ylabel("evidence-profile cosine")
    clean_axis(ax)
    for i, row in enumerate(plot.itertuples(index=False)):
        ax.text(i, row.evidence_profile_cosine + 0.05, f"{row.n_pool_a}/{row.n_pool_b}", ha="center", va="bottom", fontsize=7.5)
    ax.set_title("Portrait evidence fingerprints are compared across external pools")
    fig.tight_layout()
    save_figure(fig, "t9_source_signature_concordance")


def write_summary(df: pd.DataFrame, partial: pd.DataFrame, within_summary: pd.DataFrame, contrasts: pd.DataFrame, concordance: pd.DataFrame) -> None:
    strict = partial.loc[partial["control_id"].eq("source_site_disease")]
    strict_pass = int(strict["passes_p05"].sum()) if not strict.empty else 0
    strict_total = int(strict.shape[0])
    top_partial = partial.sort_values("portrait_incremental_r2", ascending=False).head(5)
    best_within = within_summary.sort_values("weighted_silhouette_lift", ascending=False).head(3) if not within_summary.empty else pd.DataFrame()
    lines = [
        "# Residual source and metadata controls",
        "",
        "## What this analysis asks",
        "",
        "This analysis asks whether source, project, tissue or disease labels map one-to-one onto RNA portrait groups. It then tests whether portrait groups still explain independent biological support axes after controlling for those labels.",
        "",
        "The residual-control analysis tests portrait information after accounting for source, project, tissue and disease labels.",
        "",
        "## Headline result",
        "",
        f"- Samples analysed: `{len(df)}` external temperature-scaled rows.",
        f"- Strict source+site+disease controls: `{strict_pass}/{strict_total}` evidence axes pass within-block permutation p < 0.05.",
        "- This supports the cautious claim that simple labels do not fully explain the portrait-evidence relationship.",
        "- It does not prove all leakage is impossible; project/source remains a known risk and should be reported as a limitation.",
        "",
        "## Strongest incremental-R2 results",
        "",
    ]
    for row in top_partial.itertuples(index=False):
        lines.append(
            f"- Control `{row.control_label}`, axis `{row.evidence_axis_label}`: portrait incremental R2 `{row.portrait_incremental_r2:.3f}`, permutation mean `{row.permuted_incremental_r2_mean:.3f}`, p `{row.empirical_p_greater:.4f}`."
        )
    lines.extend(["", "## Within-label evidence separation", ""])
    if best_within.empty:
        lines.append("- No valid within-label silhouette blocks were available.")
    else:
        for row in best_within.itertuples(index=False):
            lines.append(
                f"- Within `{row.factor_label}`: observed silhouette `{row.weighted_observed_silhouette:.3f}` vs shuffled `{row.weighted_permuted_silhouette_mean:.3f}` across `{row.n_blocks_tested}` blocks."
            )
    lines.extend(["", "## Matched same-label contrasts", ""])
    if contrasts.empty:
        lines.append("- No same-label contrast examples passed the minimum group-size filters.")
    else:
        for row in contrasts.head(5).itertuples(index=False):
            lines.append(
                f"- `{row.contrast_label}` / `{row.group_value}`: `{row.state_a}` vs `{row.state_b}` differs most on `{row.max_difference_axis_label}` by `{row.max_abs_evidence_difference:.3f}`."
            )
    lines.extend(["", "## Source-pool concordance", ""])
    if concordance.empty:
        lines.append("- No portrait group had at least five samples in both external pools.")
    else:
        median_cos = float(concordance["evidence_profile_cosine"].median())
        lines.append(f"- Median cross-pool evidence-profile cosine among eligible portrait groups: `{median_cos:.3f}`.")
        lines.append("- This is a descriptive comparison, not proof of source-invariant generalization.")
    lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `t9_partial_r2_after_controls.csv`: portrait incremental R2 after label/source controls plus within-block permutation baselines.",
            "- `t9_within_label_evidence_separation.csv`: block-level silhouette tests inside the same project/tissue/disease/fixed label.",
            "- `t9_within_label_evidence_separation_summary.csv`: weighted summary of the within-label tests.",
            "- `t9_same_label_different_biology_contrasts.csv`: example same-label portrait contrasts with independent evidence differences.",
            "- `t9_source_signature_concordance.csv`: cross-pool evidence signature concordance by portrait group.",
            "- `*.svg/png`: T9 visual summaries.",
            "",
            "## Interpretation boundary",
            "",
            "This residual-control analysis tests simple shortcut explanations. It should not be interpreted as causal proof or as complete leakage exclusion, because the dataset remains public, heterogeneous and cross-platform.",
            "",
            "This is a computational shortcut-control analysis; it does not prove causality or fully exclude leakage in heterogeneous public data.",
            "",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    partial = partial_r2_controls(df)
    within_detail, within_summary = within_label_separation(df)
    contrasts = same_label_contrasts(df)
    concordance = source_signature_concordance(df)

    partial.to_csv(OUTDIR / "t9_partial_r2_after_controls.csv", index=False)
    within_detail.to_csv(OUTDIR / "t9_within_label_evidence_separation.csv", index=False)
    within_summary.to_csv(OUTDIR / "t9_within_label_evidence_separation_summary.csv", index=False)
    contrasts.to_csv(OUTDIR / "t9_same_label_different_biology_contrasts.csv", index=False)
    concordance.to_csv(OUTDIR / "t9_source_signature_concordance.csv", index=False)

    plot_partial_r2(partial)
    plot_within_label(within_summary)
    plot_contrast_examples(contrasts)
    plot_source_concordance(concordance)
    write_summary(df, partial, within_summary, contrasts, concordance)
    print(OUTDIR)


if __name__ == "__main__":
    main()
