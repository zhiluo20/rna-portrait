#!/usr/bin/env python3
"""Reliability boundaries and failure-mode diagnostics for RNA portraits.

T11 does not claim to know a gold-standard human interpretation for each
sample. Instead, it asks when an RNA portrait should be written as reliable,
cautious, or weak/unsupported based on auditable evidence, data quality and
fixed-label conflicts.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors


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
OUTDIR = SUPP_DIR / "T11_failure_mode_reliability_boundaries"
T7_MERGED = SUPP_DIR / "T7_portrait_claim_grounding" / "t7_merged_sample_evidence.csv"
T7_CLAIMS = SUPP_DIR / "T7_portrait_claim_grounding" / "t7_portrait_claim_grounding_by_claim.csv"

EVIDENCE_AXES = [
    "immune_support",
    "context_support",
    "tumor_like_support",
    "mixed_evidence_support",
    "clean_context_support",
]
BOUNDARY_FLAGS = [
    "unsupported_portrait",
    "low_claim_support",
    "low_gene_coverage",
    "epic_nonconverged",
    "fixed_label_conflict",
    "multi_signal_boundary",
    "low_evidence_margin",
]
FLAG_LABELS = {
    "unsupported_portrait": "unsupported portrait",
    "low_claim_support": "low claim support",
    "low_gene_coverage": "low gene coverage",
    "epic_nonconverged": "EPIC non-converged",
    "fixed_label_conflict": "fixed-label conflict",
    "multi_signal_boundary": "multi-signal boundary",
    "low_evidence_margin": "low evidence margin",
}
TIER_ORDER = [
    "A_high_confidence",
    "B_mixed_but_auditable",
    "B_caution_boundary",
    "C_quality_limited",
    "C_weak_or_unsupported",
]
TIER_LABELS = {
    "A_high_confidence": "A high confidence",
    "B_mixed_but_auditable": "B mixed but auditable",
    "B_caution_boundary": "B caution boundary",
    "C_quality_limited": "C quality limited",
    "C_weak_or_unsupported": "C weak/unsupported",
}
TIER_COLORS = {
    "A_high_confidence": "#1B9E77",
    "B_mixed_but_auditable": "#3B6FB6",
    "B_caution_boundary": "#D95F02",
    "C_quality_limited": "#7B3294",
    "C_weak_or_unsupported": "#B2182B",
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
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.3,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.65,
    }
)


def sample_id_from_file(file_name: object) -> str:
    text = str(file_name)
    return text[:-4] if text.endswith(".txt") else text


def load_sample_data() -> pd.DataFrame:
    df = pd.read_csv(T7_MERGED)
    df = df.loc[df["model"].eq("temperature_scaled_sgd")].copy()
    df["sample_id"] = df["file"].map(sample_id_from_file)
    for col in [
        "semantic_state_family",
        "openworld_status",
        "semantic_disease_semantic_status",
        "project",
        "pool",
        "expected_site_family",
        "expected_disease_family",
        "closed_set_disease_family",
    ]:
        df[col] = df[col].fillna("missing").astype(str)
    numeric_cols = [
        "selected_gene_coverage_baseline",
        "matched_selected_genes",
        "convergeCode",
        "explainer_ok_source",
        "closed_set_calibrated_confidence",
        "semantic_state_top_evidence_score",
        "semantic_state_second_evidence_score",
        "semantic_state_top_evidence_strength",
        "semantic_state_second_evidence_strength",
        "positive_evidence_axis_count",
        *EVIDENCE_AXES,
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["epic_converged"] = df["convergeCode"].fillna(-1).eq(0)
    df["gene_coverage"] = df["selected_gene_coverage_baseline"]
    df["evidence_margin"] = df["semantic_state_top_evidence_score"] - df["semantic_state_second_evidence_score"]
    df["max_axis_support"] = df[EVIDENCE_AXES].max(axis=1)
    return df.reset_index(drop=True)


def claim_support_by_sample() -> pd.DataFrame:
    claims = pd.read_csv(T7_CLAIMS)
    claims["sample_id"] = claims["file"].map(sample_id_from_file)
    claims["support_score"] = pd.to_numeric(claims["support_score"], errors="coerce")
    levels = ["strong", "partial", "weak", "missing"]
    rows = []
    for (pool, sample_id), sub in claims.groupby(["pool", "sample_id"], dropna=False):
        counts = sub["support_level"].value_counts()
        n = int(len(sub))
        strong = int(counts.get("strong", 0))
        partial = int(counts.get("partial", 0))
        weak = int(counts.get("weak", 0))
        missing = int(counts.get("missing", 0))
        rows.append(
            {
                "pool": pool,
                "sample_id": sample_id,
                "claim_rows": n,
                "strong_claim_rows": strong,
                "partial_claim_rows": partial,
                "weak_claim_rows": weak,
                "missing_claim_rows": missing,
                "partial_or_strong_claim_rows": strong + partial,
                "partial_or_strong_claim_rate": (strong + partial) / n if n else np.nan,
                "weak_or_missing_claim_rate": (weak + missing) / n if n else np.nan,
                "mean_claim_support_score": float(sub["support_score"].mean(skipna=True)),
                "min_claim_support_score": float(sub["support_score"].min(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def classify_samples(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    q25_coverage = float(out["gene_coverage"].quantile(0.25))
    q25_margin = float(out["evidence_margin"].quantile(0.25))
    out["unsupported_portrait"] = out["semantic_state_family"].eq("unsupported_semantics") | out["openworld_status"].eq("unsupported")
    out["low_claim_support"] = out["partial_or_strong_claim_rate"].fillna(0).lt(0.40)
    out["low_gene_coverage"] = out["gene_coverage"].lt(q25_coverage)
    out["epic_nonconverged"] = ~out["epic_converged"]
    out["fixed_label_conflict"] = out["closed_set_highconf_overcall_70"].astype(bool)
    out["multi_signal_boundary"] = out["positive_evidence_axis_count"].fillna(0).ge(2)
    out["low_evidence_margin"] = out["evidence_margin"].lt(q25_margin)
    out["boundary_flag_count"] = out[BOUNDARY_FLAGS].sum(axis=1).astype(int)

    tiers = []
    reasons = []
    for row in out.itertuples(index=False):
        flags = {flag: bool(getattr(row, flag)) for flag in BOUNDARY_FLAGS}
        support_rate = float(getattr(row, "partial_or_strong_claim_rate") or 0)
        if flags["unsupported_portrait"] or support_rate < 0.25:
            tier = "C_weak_or_unsupported"
            reason = "unsupported portrait or very low claim support"
        elif flags["low_gene_coverage"] and flags["epic_nonconverged"]:
            tier = "C_quality_limited"
            reason = "low gene coverage and non-converged deconvolution"
        elif flags["multi_signal_boundary"] and support_rate >= 0.50 and not flags["unsupported_portrait"]:
            tier = "B_mixed_but_auditable"
            reason = "multiple evidence axes but claims remain auditable"
        elif flags["low_claim_support"] or flags["low_gene_coverage"] or flags["epic_nonconverged"] or flags["fixed_label_conflict"] or flags["low_evidence_margin"]:
            tier = "B_caution_boundary"
            reason = "one or more caution flags"
        else:
            tier = "A_high_confidence"
            reason = "evidence-backed with no major boundary flag"
        tiers.append(tier)
        reasons.append(reason)
    out["reliability_tier"] = tiers
    out["reliability_reason"] = reasons
    out["boundary_flags"] = out[BOUNDARY_FLAGS].apply(
        lambda row: "|".join(flag for flag, value in row.items() if bool(value)) or "none",
        axis=1,
    )
    return out


def build_reliability_table() -> pd.DataFrame:
    samples = load_sample_data()
    support = claim_support_by_sample()
    df = samples.merge(support, on=["pool", "sample_id"], how="left")
    for col in ["claim_rows", "strong_claim_rows", "partial_claim_rows", "weak_claim_rows", "missing_claim_rows", "partial_or_strong_claim_rows"]:
        df[col] = df[col].fillna(0).astype(int)
    for col in ["partial_or_strong_claim_rate", "weak_or_missing_claim_rate", "mean_claim_support_score", "min_claim_support_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return classify_samples(df)


def summarize_flags(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for flag in BOUNDARY_FLAGS:
        rows.append(
            {
                "boundary_flag": flag,
                "boundary_label": FLAG_LABELS[flag],
                "n_samples": int(df[flag].sum()),
                "fraction_samples": float(df[flag].mean()),
                "median_claim_support_rate": float(df.loc[df[flag], "partial_or_strong_claim_rate"].median(skipna=True)),
                "median_gene_coverage": float(df.loc[df[flag], "gene_coverage"].median(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_portrait(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state, sub in df.groupby("semantic_state_family", dropna=False):
        row: Dict[str, object] = {
            "semantic_state_family": state,
            "portrait_label": STATE_LABELS.get(str(state), str(state).replace("_", " ")),
            "n_samples": int(len(sub)),
            "median_claim_support_rate": float(sub["partial_or_strong_claim_rate"].median(skipna=True)),
            "mean_claim_support_rate": float(sub["partial_or_strong_claim_rate"].mean(skipna=True)),
            "mean_gene_coverage": float(sub["gene_coverage"].mean(skipna=True)),
            "epic_converged_rate": float(sub["epic_converged"].mean()),
            "median_evidence_margin": float(sub["evidence_margin"].median(skipna=True)),
            "mean_boundary_flag_count": float(sub["boundary_flag_count"].mean()),
        }
        for tier in TIER_ORDER:
            row[f"tier_{tier}_n"] = int(sub["reliability_tier"].eq(tier).sum())
            row[f"tier_{tier}_fraction"] = float(sub["reliability_tier"].eq(tier).mean())
        for flag in BOUNDARY_FLAGS:
            row[f"flag_{flag}_fraction"] = float(sub[flag].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_samples", ascending=False)


def representative_cases(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("weak_unsupported", "C_weak_or_unsupported", "partial_or_strong_claim_rate", True),
        ("quality_limited", "C_quality_limited", "gene_coverage", True),
        ("fixed_label_conflict", None, "closed_set_calibrated_confidence", False),
        ("low_margin", None, "evidence_margin", True),
        ("mixed_but_auditable", "B_mixed_but_auditable", "partial_or_strong_claim_rate", False),
        ("high_confidence", "A_high_confidence", "partial_or_strong_claim_rate", False),
    ]
    rows = []
    used = set()
    for case_type, tier, sort_col, ascending in specs:
        sub = df.copy()
        if tier is not None:
            sub = sub.loc[sub["reliability_tier"].eq(tier)].copy()
        if case_type == "fixed_label_conflict":
            sub = sub.loc[sub["fixed_label_conflict"]].copy()
        if case_type == "low_margin":
            sub = sub.loc[sub["low_evidence_margin"]].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(sort_col, ascending=ascending)
        picked = 0
        for row in sub.itertuples(index=False):
            key = (row.pool, row.sample_id)
            if key in used:
                continue
            used.add(key)
            rows.append(
                {
                    "case_type": case_type,
                    "pool": row.pool,
                    "sample_id": row.sample_id,
                    "file": row.file,
                    "project": row.project,
                    "expected_site_family": row.expected_site_family,
                    "expected_disease_family": row.expected_disease_family,
                    "closed_set_disease_family": row.closed_set_disease_family,
                    "closed_set_calibrated_confidence": row.closed_set_calibrated_confidence,
                    "semantic_state_family": row.semantic_state_family,
                    "openworld_status": row.openworld_status,
                    "reliability_tier": row.reliability_tier,
                    "reliability_reason": row.reliability_reason,
                    "boundary_flags": row.boundary_flags,
                    "partial_or_strong_claim_rate": row.partial_or_strong_claim_rate,
                    "mean_claim_support_score": row.mean_claim_support_score,
                    "gene_coverage": row.gene_coverage,
                    "epic_converged": row.epic_converged,
                    "evidence_margin": row.evidence_margin,
                    "positive_evidence_axis_count": row.positive_evidence_axis_count,
                    "portrait_text_for_grounding": row.portrait_text_for_grounding,
                }
            )
            picked += 1
            if picked >= 3:
                break
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, name: str) -> None:
    for ext in ["svg", "png"]:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if ext == "png":
            kwargs["dpi"] = 260
        fig.savefig(OUTDIR / f"{name}.{ext}", **kwargs)
    plt.close(fig)


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#E5E7EB", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def draw_flag_heatmap(by_portrait: pd.DataFrame) -> None:
    states = by_portrait.sort_values("n_samples", ascending=False)["semantic_state_family"].tolist()
    labels = by_portrait.set_index("semantic_state_family").loc[states, "portrait_label"].tolist()
    columns = [f"flag_{flag}_fraction" for flag in BOUNDARY_FLAGS]
    values = by_portrait.set_index("semantic_state_family").loc[states, columns].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, max(3.8, len(states) * 0.42)))
    im = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(BOUNDARY_FLAGS)), [FLAG_LABELS[flag] for flag in BOUNDARY_FLAGS], rotation=32, ha="right")
    ax.set_yticks(np.arange(len(states)), labels)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7.0, color="#111827")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("fraction of samples")
    ax.set_title("Boundary flags are visible and differ by RNA portrait group")
    fig.tight_layout()
    save_figure(fig, "t11_boundary_flag_heatmap")


def draw_tier_by_portrait(by_portrait: pd.DataFrame) -> None:
    plot = by_portrait.sort_values("n_samples", ascending=True)
    y = np.arange(len(plot))
    left = np.zeros(len(plot))
    fig, ax = plt.subplots(figsize=(8.0, max(3.8, len(plot) * 0.42)))
    for tier in TIER_ORDER:
        vals = plot[f"tier_{tier}_fraction"].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, color=TIER_COLORS[tier], label=TIER_LABELS[tier], height=0.66, zorder=2)
        left += vals
    ax.set_yticks(y, plot["portrait_label"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of samples")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    clean_axis(ax, grid_axis="x")
    for i, row in enumerate(plot.itertuples(index=False)):
        ax.text(1.01, i, f"n={row.n_samples}", va="center", fontsize=7.2)
    ax.set_title("Reliability tiers by RNA portrait group")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save_figure(fig, "t11_reliability_tier_by_portrait")


def draw_support_quality_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for tier in TIER_ORDER:
        sub = df.loc[df["reliability_tier"].eq(tier)]
        if sub.empty:
            continue
        ax.scatter(
            sub["gene_coverage"],
            sub["partial_or_strong_claim_rate"],
            s=20,
            color=TIER_COLORS[tier],
            alpha=0.72,
            linewidth=0,
            label=TIER_LABELS[tier],
            rasterized=True,
        )
    ax.set_xlabel("selected-gene coverage")
    ax.set_ylabel("partial/strong claim support rate")
    ax.set_xlim(-0.02, 0.84)
    ax.set_ylim(-0.04, 1.04)
    ax.legend(frameon=False, loc="lower right", ncol=1)
    clean_axis(ax)
    ax.set_title("Reliability boundaries combine evidence support and data quality")
    fig.tight_layout()
    save_figure(fig, "t11_support_quality_scatter")


def draw_representative_cases(cases: pd.DataFrame) -> None:
    if cases.empty:
        return
    plot = cases.head(14).copy()
    fig, ax = plt.subplots(figsize=(8.6, max(4.2, len(plot) * 0.34)))
    ax.axis("off")
    y_positions = np.linspace(0.96, 0.06, len(plot))
    for y, row in zip(y_positions, plot.itertuples(index=False)):
        color = TIER_COLORS.get(row.reliability_tier, "#6B7280")
        ax.add_patch(plt.Rectangle((0.01, y - 0.025), 0.98, 0.045, transform=ax.transAxes, color=colors.to_rgba(color, 0.08), ec="#E5E7EB", lw=0.5))
        left = f"{row.case_type} | {TIER_LABELS.get(row.reliability_tier, row.reliability_tier)}"
        middle = f"{row.semantic_state_family} | {row.openworld_status} | support {row.partial_or_strong_claim_rate:.2f} | cov {row.gene_coverage:.2f}"
        right = str(row.boundary_flags).replace("|", ", ")
        ax.text(0.02, y, left, transform=ax.transAxes, fontsize=7.2, fontweight="bold", va="center", color=color)
        ax.text(0.33, y, middle, transform=ax.transAxes, fontsize=7.0, va="center", color="#111827")
        ax.text(0.69, y, textwrap.shorten(right, width=56, placeholder="..."), transform=ax.transAxes, fontsize=6.8, va="center", color="#374151")
    ax.text(0.02, 1.02, "Representative reliability-boundary cases", transform=ax.transAxes, fontsize=9.0, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "t11_representative_boundary_cases")


def write_summary(df: pd.DataFrame, flag_summary: pd.DataFrame, by_portrait: pd.DataFrame, cases: pd.DataFrame) -> None:
    tier_counts = df["reliability_tier"].value_counts().reindex(TIER_ORDER).fillna(0).astype(int)
    tier_fracs = (tier_counts / len(df)).fillna(0)
    top_flags = flag_summary.sort_values("fraction_samples", ascending=False).head(4)
    lines = [
        "# Reliability boundaries and failure-mode diagnostics",
        "",
        "## What this analysis asks",
        "",
        "This analysis does not assume a gold-standard human interpretation for every sample. It asks when the generated RNA portrait should be treated as reliable, cautious, or weak based on auditable claim support, data quality, predefined-label conflicts and multi-signal evidence.",
        "",
        "The workflow defines reliability tiers for portrait interpretation rather than asserting a unique ground truth for every profile.",
        "",
        "## Headline result",
        "",
        f"- Samples analysed: `{len(df)}` external rows.",
    ]
    for tier in TIER_ORDER:
        lines.append(f"- {TIER_LABELS[tier]}: `{tier_counts[tier]}` samples ({tier_fracs[tier] * 100:.1f}%).")
    lines.extend(["", "## Most common boundary flags", ""])
    for row in top_flags.itertuples(index=False):
        lines.append(
            f"- `{row.boundary_label}`: `{row.n_samples}` samples ({row.fraction_samples * 100:.1f}%), median claim-support rate `{row.median_claim_support_rate:.2f}`, median gene coverage `{row.median_gene_coverage:.2f}`."
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "Strong portrait interpretations should be reserved for high-confidence or auditable mixed portraits, while unsupported, low-coverage, non-converged or low-claim-support samples are flagged as boundary cases."
    )
    lines.append("")
    lines.append(
        "This provides a cautious interpretation framework in which low-coverage, unsupported or poorly grounded profiles are treated as boundary cases."
    )
    lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `t11_sample_reliability_table.csv`: sample-level reliability tier and boundary flags.",
            "- `t11_boundary_flag_summary.csv`: one-row summary for each boundary flag.",
            "- `t11_reliability_by_portrait.csv`: portrait-group summary of reliability tiers and flags.",
            "- `t11_representative_boundary_cases.csv`: selected cases for manual review and figure annotation.",
            "- `t11_boundary_flag_heatmap.svg/png`: boundary flags by portrait group.",
            "- `t11_reliability_tier_by_portrait.svg/png`: reliability tiers by portrait group.",
            "- `t11_support_quality_scatter.svg/png`: claim support versus gene coverage.",
            "- `t11_representative_boundary_cases.svg/png`: compact case list.",
            "",
            "## Interpretation boundary",
            "",
            "The reliability tiers are audit flags, not labels of biological correctness. Low-tier profiles require additional support before strong interpretation.",
            "",
            "The reliability framework marks profiles that require additional support before strong interpretation.",
            "",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = build_reliability_table()
    flag_summary = summarize_flags(df)
    by_portrait = summarize_by_portrait(df)
    cases = representative_cases(df)

    df.to_csv(OUTDIR / "t11_sample_reliability_table.csv", index=False)
    flag_summary.to_csv(OUTDIR / "t11_boundary_flag_summary.csv", index=False)
    by_portrait.to_csv(OUTDIR / "t11_reliability_by_portrait.csv", index=False)
    cases.to_csv(OUTDIR / "t11_representative_boundary_cases.csv", index=False)

    draw_flag_heatmap(by_portrait)
    draw_tier_by_portrait(by_portrait)
    draw_support_quality_scatter(df)
    draw_representative_cases(cases)
    write_summary(df, flag_summary, by_portrait, cases)
    print(OUTDIR)


if __name__ == "__main__":
    main()
