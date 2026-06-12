#!/usr/bin/env python3
"""First-pass shortcut-exclusion controls for RNA molecular portraits."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
except Exception:  # pragma: no cover - fallback for minimal environments
    adjusted_mutual_info_score = None
    normalized_mutual_info_score = None


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
OUTDIR = SUPP_DIR / "T8_shortcut_exclusion_controls"

T7_MERGED = SUPP_DIR / "T7_portrait_claim_grounding" / "t7_merged_sample_evidence.csv"

FACTOR_SPECS: List[Tuple[str, str]] = [
    ("pool", "external pool"),
    ("project_prefix", "source prefix"),
    ("project", "project/source"),
    ("explainer_ok_source", "source-quality flag"),
    ("expected_site_family", "expected tissue/site"),
    ("expected_disease_family", "expected disease label"),
    ("closed_set_disease_family", "fixed disease label"),
]

TARGET_SPECS: List[Tuple[str, str]] = [
    ("semantic_state_family", "RNA portrait group"),
    ("semantic_disease_semantic_status", "portrait disease status"),
    ("openworld_status", "signal status"),
]

EVIDENCE_AXES = [
    "immune_support",
    "context_support",
    "tumor_like_support",
    "mixed_evidence_support",
    "clean_context_support",
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


def safe_str_series(s: pd.Series) -> pd.Series:
    return s.fillna("missing").astype(str).replace({"": "missing"})


def project_prefix(project: object) -> str:
    text = str(project)
    if text.startswith("RECOUNT3_"):
        return "RECOUNT3"
    match = re.match(r"([A-Za-z]+)", text)
    return match.group(1).upper() if match else "OTHER"


def entropy(values: Iterable[object]) -> float:
    counts = pd.Series(list(values)).value_counts()
    if counts.empty:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def normalized_entropy(values: Iterable[object], global_k: int) -> float:
    if global_k <= 1:
        return 0.0
    return entropy(values) / math.log2(global_k)


def nmi_ami(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    x = safe_str_series(x)
    y = safe_str_series(y)
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan
    if normalized_mutual_info_score is None or adjusted_mutual_info_score is None:
        return np.nan, np.nan
    return (
        float(normalized_mutual_info_score(x, y)),
        float(adjusted_mutual_info_score(x, y)),
    )


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(safe_str_series(x), safe_str_series(y))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan
    observed = table.to_numpy(dtype=float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / expected)
    n = observed.sum()
    phi2 = chi2 / n
    r, k = observed.shape
    return float(np.sqrt(phi2 / max(1, min(k - 1, r - 1))))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(T7_MERGED)
    df = df.loc[df["model"].eq("temperature_scaled_sgd")].copy()
    df["project_prefix"] = df["project"].map(project_prefix)
    df["source_batch_proxy"] = (
        df["pool"].astype(str) + "::" + df["project_prefix"].astype(str)
    )
    for col, _ in FACTOR_SPECS + TARGET_SPECS:
        if col in df.columns:
            df[col] = safe_str_series(df[col])
    for col in EVIDENCE_AXES:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def association_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for factor, factor_label in FACTOR_SPECS:
        if factor not in df.columns:
            continue
        for target, target_label in TARGET_SPECS:
            if target not in df.columns:
                continue
            nmi, ami = nmi_ami(df[factor], df[target])
            rows.append(
                {
                    "factor": factor,
                    "factor_label": factor_label,
                    "target": target,
                    "target_label": target_label,
                    "n_samples": int(df[[factor, target]].dropna().shape[0]),
                    "factor_levels": int(df[factor].nunique(dropna=False)),
                    "target_levels": int(df[target].nunique(dropna=False)),
                    "nmi": nmi,
                    "ami": ami,
                    "cramers_v": cramers_v(df[factor], df[target]),
                }
            )
    return pd.DataFrame(rows)


def diversity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    global_k = df["semantic_state_family"].nunique(dropna=False)
    for factor, factor_label in FACTOR_SPECS:
        if factor not in df.columns:
            continue
        group_rows = []
        for value, sub in df.groupby(factor, dropna=False):
            if len(sub) < 3:
                continue
            state_counts = sub["semantic_state_family"].value_counts()
            max_fraction = float(state_counts.iloc[0] / len(sub))
            evidence_spread = float(
                sub[EVIDENCE_AXES].std(numeric_only=True).mean(skipna=True)
            )
            group_rows.append(
                {
                    "factor": factor,
                    "factor_label": factor_label,
                    "factor_value": value,
                    "n_samples": int(len(sub)),
                    "unique_portrait_groups": int(state_counts.shape[0]),
                    "max_portrait_fraction": max_fraction,
                    "portrait_entropy": entropy(sub["semantic_state_family"]),
                    "portrait_entropy_norm": normalized_entropy(
                        sub["semantic_state_family"], global_k
                    ),
                    "evidence_axis_mean_sd": evidence_spread,
                }
            )
        group_df = pd.DataFrame(group_rows)
        if group_df.empty:
            continue
        weights = group_df["n_samples"].astype(float)
        rows.append(
            {
                "factor": factor,
                "factor_label": factor_label,
                "n_groups_ge3": int(group_df.shape[0]),
                "n_samples_in_groups": int(group_df["n_samples"].sum()),
                "fraction_multi_portrait_groups": float(
                    (group_df["unique_portrait_groups"] >= 2).mean()
                ),
                "weighted_unique_portrait_groups": float(
                    np.average(group_df["unique_portrait_groups"], weights=weights)
                ),
                "weighted_portrait_entropy_norm": float(
                    np.average(group_df["portrait_entropy_norm"], weights=weights)
                ),
                "weighted_max_portrait_fraction": float(
                    np.average(group_df["max_portrait_fraction"], weights=weights)
                ),
                "weighted_evidence_axis_mean_sd": float(
                    np.average(
                        group_df["evidence_axis_mean_sd"].fillna(0.0), weights=weights
                    )
                ),
            }
        )
        group_df.to_csv(OUTDIR / f"t8_within_{factor}_group_details.csv", index=False)
    return pd.DataFrame(rows)


def crosstab_table(
    df: pd.DataFrame, row_col: str, col_col: str = "semantic_state_family"
) -> pd.DataFrame:
    table = pd.crosstab(df[row_col], df[col_col])
    ordered_cols = [c for c in STATE_ORDER if c in table.columns] + [
        c for c in table.columns if c not in STATE_ORDER
    ]
    table = table[ordered_cols]
    return table.sort_index()


def top_same_label_examples(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for label_col in ["expected_disease_family", "closed_set_disease_family", "expected_site_family", "project"]:
        if label_col not in df.columns:
            continue
        for label, sub in df.groupby(label_col, dropna=False):
            if len(sub) < 6:
                continue
            state_counts = sub["semantic_state_family"].value_counts()
            if state_counts.shape[0] < 2:
                continue
            axis_sd = sub[EVIDENCE_AXES].std(numeric_only=True).sort_values(ascending=False)
            representative = []
            for state in state_counts.head(4).index:
                state_sub = sub.loc[sub["semantic_state_family"].eq(state)].copy()
                support_cols = [c for c in EVIDENCE_AXES if c in state_sub.columns]
                state_sub["max_axis_support"] = state_sub[support_cols].max(axis=1)
                pick = state_sub.sort_values("max_axis_support", ascending=False).iloc[0]
                representative.append(
                    f"{state}:{pick.get('file')}:{pick.get('expected_disease_family')}"
                )
            rows.append(
                {
                    "label_column": label_col,
                    "label_value": label,
                    "n_samples": int(len(sub)),
                    "n_portrait_groups": int(state_counts.shape[0]),
                    "top_portrait_groups": "; ".join(
                        f"{k}={v}" for k, v in state_counts.head(5).items()
                    ),
                    "largest_evidence_axis_sd": axis_sd.index[0] if not axis_sd.empty else "",
                    "largest_evidence_axis_sd_value": float(axis_sd.iloc[0]) if not axis_sd.empty else np.nan,
                    "representative_samples": " | ".join(representative),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["n_portrait_groups", "n_samples", "largest_evidence_axis_sd_value"],
            ascending=[False, False, False],
        )
    return out


def top_same_portrait_examples(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for state, sub in df.groupby("semantic_state_family", dropna=False):
        if len(sub) < 6:
            continue
        disease_counts = sub["expected_disease_family"].value_counts()
        site_counts = sub["expected_site_family"].value_counts()
        project_counts = sub["project"].value_counts()
        rows.append(
            {
                "portrait_group": state,
                "n_samples": int(len(sub)),
                "n_expected_disease_labels": int(disease_counts.shape[0]),
                "top_expected_disease_labels": "; ".join(
                    f"{k}={v}" for k, v in disease_counts.head(6).items()
                ),
                "n_expected_site_labels": int(site_counts.shape[0]),
                "top_expected_site_labels": "; ".join(
                    f"{k}={v}" for k, v in site_counts.head(6).items()
                ),
                "n_projects": int(project_counts.shape[0]),
                "top_projects": "; ".join(f"{k}={v}" for k, v in project_counts.head(6).items()),
                "mean_immune_support": float(sub["immune_support"].mean()),
                "mean_context_support": float(sub["context_support"].mean()),
                "mean_tumor_like_support": float(sub["tumor_like_support"].mean()),
                "mean_mixed_evidence_support": float(sub["mixed_evidence_support"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["n_expected_disease_labels", "n_expected_site_labels", "n_samples"],
            ascending=[False, False, False],
        )
    return out


def draw_association_plot(assoc: pd.DataFrame) -> None:
    focus = assoc.loc[assoc["target"].eq("semantic_state_family")].copy()
    focus = focus.sort_values("nmi", ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = ["#315C73" if v < 0.35 else "#B85C38" for v in focus["nmi"]]
    ax.barh(focus["factor_label"], focus["nmi"], color=colors)
    ax.set_xlabel("NMI with RNA portrait group")
    ax.set_title("Can one shortcut variable explain the portrait groups?")
    ax.axvline(0.35, color="#B85C38", lw=1, ls="--")
    ax.text(0.355, len(focus) - 0.35, "strong shortcut warning", color="#8D3F2B", fontsize=8)
    ax.set_xlim(0, max(0.55, float(focus["nmi"].max()) * 1.15))
    for i, val in enumerate(focus["nmi"]):
        ax.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "t8_shortcut_association_nmi.svg")
    fig.savefig(OUTDIR / "t8_shortcut_association_nmi.png", dpi=220)
    plt.close(fig)


def draw_diversity_plot(diversity: pd.DataFrame) -> None:
    data = diversity.sort_values("weighted_unique_portrait_groups", ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    y = np.arange(len(data))
    ax.barh(
        y - 0.18,
        data["weighted_unique_portrait_groups"],
        height=0.34,
        label="weighted unique portrait groups",
        color="#4F7A62",
    )
    ax.barh(
        y + 0.18,
        data["fraction_multi_portrait_groups"] * data["weighted_unique_portrait_groups"].max(),
        height=0.34,
        label="multi-portrait group fraction (scaled)",
        color="#B99C4A",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(data["factor_label"])
    ax.set_xlabel("Within-factor portrait diversity")
    ax.set_title("Do labels/sources still contain multiple RNA portraits?")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTDIR / "t8_within_factor_portrait_diversity.svg")
    fig.savefig(OUTDIR / "t8_within_factor_portrait_diversity.png", dpi=220)
    plt.close(fig)


def draw_heatmap(table: pd.DataFrame, title: str, path_stem: str, top_n: int = 12) -> None:
    top_rows = table.sum(axis=1).sort_values(ascending=False).head(top_n).index
    plot = table.loc[top_rows].copy()
    row_sums = plot.sum(axis=1).replace(0, np.nan)
    frac = plot.div(row_sums, axis=0).fillna(0.0)

    fig_w = max(8.0, 0.75 * frac.shape[1] + 3.0)
    fig_h = max(4.5, 0.36 * frac.shape[0] + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(frac.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(frac.shape[1]))
    ax.set_xticklabels(frac.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(frac.shape[0]))
    ax.set_yticklabels(frac.index, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("RNA portrait group")
    ax.set_ylabel("Label")
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            count = int(plot.iloc[i, j])
            if count > 0:
                ax.text(j, i, str(count), ha="center", va="center", fontsize=7, color="#17202A")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("row fraction")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{path_stem}.svg")
    fig.savefig(OUTDIR / f"{path_stem}.png", dpi=220)
    plt.close(fig)


def write_summary(
    df: pd.DataFrame,
    assoc: pd.DataFrame,
    diversity: pd.DataFrame,
    same_label: pd.DataFrame,
    same_portrait: pd.DataFrame,
) -> None:
    state_assoc = assoc.loc[assoc["target"].eq("semantic_state_family")].sort_values(
        "nmi", ascending=False
    )
    top_assoc = state_assoc.head(4)
    top_div = diversity.sort_values("weighted_unique_portrait_groups", ascending=False).head(4)

    disease_div = diversity.loc[diversity["factor"].eq("expected_disease_family")]
    site_div = diversity.loc[diversity["factor"].eq("expected_site_family")]
    project_div = diversity.loc[diversity["factor"].eq("project")]

    def metric(df_: pd.DataFrame, col: str) -> str:
        if df_.empty:
            return "NA"
        return f"{float(df_.iloc[0][col]):.3f}"

    lines = [
        "# Source and metadata controls",
        "",
        "## Purpose",
        "",
        "This first-pass analysis asks whether RNA molecular portraits can be fully explained by simple shortcut variables: external pool, project/source, source prefix, expected tissue/site label, expected disease label, fixed-choice disease label or a source-quality proxy.",
        "",
        "## Data",
        "",
        f"- External temperature-scaled sample rows: `{len(df)}`",
        f"- Projects: `{df['project'].nunique()}`",
        f"- Expected site labels: `{df['expected_site_family'].nunique()}`",
        f"- Expected disease labels: `{df['expected_disease_family'].nunique()}`",
        f"- RNA portrait groups: `{df['semantic_state_family'].nunique()}`",
        "",
        "## Main results",
        "",
        "- Strong shortcut variables should have high association with the RNA portrait groups and low within-label portrait diversity.",
        "- The observed pattern is many-to-many: disease/site/source labels explain some structure, but they do not collapse the portraits to one label.",
        f"- Expected disease labels: weighted unique portrait groups within disease label = `{metric(disease_div, 'weighted_unique_portrait_groups')}`, multi-portrait label fraction = `{metric(disease_div, 'fraction_multi_portrait_groups')}`.",
        f"- Expected tissue/site labels: weighted unique portrait groups within site label = `{metric(site_div, 'weighted_unique_portrait_groups')}`, multi-portrait label fraction = `{metric(site_div, 'fraction_multi_portrait_groups')}`.",
        f"- Project/source groups: weighted unique portrait groups within project = `{metric(project_div, 'weighted_unique_portrait_groups')}`, multi-portrait project fraction = `{metric(project_div, 'fraction_multi_portrait_groups')}`.",
        "",
        "Top factor associations with RNA portrait group:",
        "",
    ]
    for _, row in top_assoc.iterrows():
        lines.append(
            f"- `{row['factor']}` ({row['factor_label']}): NMI `{row['nmi']:.3f}`, AMI `{row['ami']:.3f}`, Cramer's V `{row['cramers_v']:.3f}`."
        )
    lines.extend(
        [
            "",
            "Top within-factor diversity signals:",
            "",
        ]
    )
    for _, row in top_div.iterrows():
        lines.append(
            f"- `{row['factor']}` ({row['factor_label']}): weighted unique portrait groups `{row['weighted_unique_portrait_groups']:.2f}`, multi-portrait fraction `{row['fraction_multi_portrait_groups']:.2f}`, weighted max-label fraction `{row['weighted_max_portrait_fraction']:.2f}`."
        )

    if not same_label.empty:
        lines.extend(
            [
                "",
                "Representative same-label/different-portrait examples:",
                "",
            ]
        )
        for _, row in same_label.head(5).iterrows():
            lines.append(
                f"- `{row['label_column']}={row['label_value']}`: n `{row['n_samples']}`, portrait groups `{row['top_portrait_groups']}`."
            )

    if not same_portrait.empty:
        lines.extend(
            [
                "",
                "Representative same-portrait/different-label examples:",
                "",
            ]
        )
        for _, row in same_portrait.head(5).iterrows():
            lines.append(
                f"- `{row['portrait_group']}`: n `{row['n_samples']}`, expected disease labels `{row['n_expected_disease_labels']}`, site labels `{row['n_expected_site_labels']}`."
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This analysis does not exclude all possible shortcuts. It tests whether a single source, tissue or disease label is sufficient to explain the portrait structure. Multiple portraits within the same label, and the same portrait across multiple labels, indicate that additional residual-control analyses are required.",
            "",
            "## Output files",
            "",
            "- `t8_shortcut_association.csv`",
            "- `t8_within_factor_diversity.csv`",
            "- `t8_expected_disease_family_x_portrait.csv`",
            "- `t8_expected_site_family_x_portrait.csv`",
            "- `t8_same_label_different_portrait_examples.csv`",
            "- `t8_same_portrait_different_label_examples.csv`",
            "- `t8_shortcut_association_nmi.svg`",
            "- `t8_within_factor_portrait_diversity.svg`",
            "- `t8_disease_label_portrait_heatmap.svg`",
            "- `t8_site_label_portrait_heatmap.svg`",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    df.to_csv(OUTDIR / "t8_merged_shortcut_input.csv", index=False)

    assoc = association_table(df)
    assoc.to_csv(OUTDIR / "t8_shortcut_association.csv", index=False)

    diversity = diversity_table(df)
    diversity.to_csv(OUTDIR / "t8_within_factor_diversity.csv", index=False)

    disease_ct = crosstab_table(df, "expected_disease_family")
    disease_ct.to_csv(OUTDIR / "t8_expected_disease_family_x_portrait.csv")

    site_ct = crosstab_table(df, "expected_site_family")
    site_ct.to_csv(OUTDIR / "t8_expected_site_family_x_portrait.csv")

    closed_ct = crosstab_table(df, "closed_set_disease_family")
    closed_ct.to_csv(OUTDIR / "t8_closed_set_disease_family_x_portrait.csv")

    same_label = top_same_label_examples(df)
    same_label.to_csv(OUTDIR / "t8_same_label_different_portrait_examples.csv", index=False)

    same_portrait = top_same_portrait_examples(df)
    same_portrait.to_csv(
        OUTDIR / "t8_same_portrait_different_label_examples.csv", index=False
    )

    draw_association_plot(assoc)
    draw_diversity_plot(diversity)
    draw_heatmap(
        disease_ct,
        "Same disease label can contain different RNA portraits",
        "t8_disease_label_portrait_heatmap",
    )
    draw_heatmap(
        site_ct,
        "Same tissue/site label can contain different RNA portraits",
        "t8_site_label_portrait_heatmap",
        top_n=8,
    )

    write_summary(df, assoc, diversity, same_label, same_portrait)
    print(f"Wrote {OUTDIR}")


if __name__ == "__main__":
    main()
