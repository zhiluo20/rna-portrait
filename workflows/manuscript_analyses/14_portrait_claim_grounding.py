#!/usr/bin/env python3
"""Ground natural-language molecular-portrait claims in independent evidence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List

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
OUTDIR = SUPP_DIR / "T7_portrait_claim_grounding"

T2B = SUPP_DIR / "T2_calibrated_closed_set" / "t2b_external_predictions.csv"
T4_MARKER = SUPP_DIR / "T4_marker_program_state_validation" / "t4_marker_scores_by_sample.csv"
T4_PATHWAY = SUPP_DIR / "T4b_pathway_state_validation" / "t4b_pathway_scores_by_sample.csv"
T4D_EPIC = SUPP_DIR / "T4d_official_EPIC_deconvolution" / "t4d_epic_scores_by_sample.csv"
T4E_MCP = SUPP_DIR / "T4e_official_MCPcounter_deconvolution" / "t4e_mcpcounter_scores_by_sample.csv"


CLAIM_SPECS: List[Dict[str, object]] = [
    {
        "claim_type": "immune_or_blood_signal",
        "label": "immune / blood signal",
        "patterns": [
            r"\bimmune\b",
            r"\bblood\b",
            r"\bhematologic\b",
            r"\bmarrow\b",
            r"\bleukemia\b",
            r"\blymphoma\b",
            r"\binfiltrat",
            r"\bcytotoxic\b",
        ],
        "support": "immune_support",
        "marker": "marker_immune_support",
        "pathway": "pathway_immune_support",
        "epic": "epic_immune_support",
        "mcp": "mcp_immune_support",
    },
    {
        "claim_type": "context_or_stromal_signal",
        "label": "context / stromal signal",
        "patterns": [
            r"\bcontext[- ]heavy\b",
            r"\bcontext[- ]dominant\b",
            r"\bclean_context\b",
            r"\bstromal\b",
            r"\bfibroblast\b",
            r"\bendothelial\b",
            r"\bsurrounding [^|.;]*context\b",
            r"\bbroader [^|.;]*context\b",
            r"\btissue[- ]background\b",
            r"\bmixed biology\b",
            r"\bactivated non[- ]malignant\b",
        ],
        "support": "context_support",
        "marker": "marker_context_support",
        "pathway": "pathway_context_support",
        "epic": "epic_context_support",
        "mcp": "mcp_context_support",
    },
    {
        "claim_type": "epithelial_or_tumor_like_signal",
        "label": "epithelial / tumor-like signal",
        "patterns": [
            r"\bepithelial\b",
            r"\bsolid[- ]tumou?r[- ]like\b",
            r"\btumou?r[- ]like\b",
            r"\bmalignan",
            r"\bdisease[- ]like\b",
            r"\bcarcinoma\b",
        ],
        "support": "tumor_like_support",
        "marker": "marker_tumor_like_support",
        "pathway": "pathway_tumor_like_support",
        "epic": "epic_context_support",
        "mcp": "mcp_context_support",
    },
    {
        "claim_type": "clean_or_non_malignant_context",
        "label": "clean / non-malignant context",
        "patterns": [
            r"\bcleaner\b",
            r"\bclean anchor\b",
            r"\bstructured anchor\b",
            r"\bnon[- ]malignant\b",
            r"\bbaseline\b",
            r"\bhealthy[- ]control[- ]like\b",
            r"\bless malignant\b",
        ],
        "support": "clean_context_support",
        "marker": "marker_context_support",
        "pathway": "pathway_context_support",
        "epic": "epic_context_support",
        "mcp": "mcp_context_support",
    },
    {
        "claim_type": "mixed_or_unstable_disease_reading",
        "label": "mixed / unstable disease reading",
        "patterns": [
            r"\bmixed\b",
            r"\btoo mixed\b",
            r"\bprevents a stable\b",
            r"\bdo not overcall\b",
            r"\bwithout context\b",
            r"\bnot a pure\b",
            r"\bunstable\b",
            r"\bunsupported\b",
            r"\bnot be read as a stable\b",
            r"\boverlap\b",
            r"\btoo unsupported\b",
        ],
        "support": "mixed_evidence_support",
        "marker": "mixed_marker_component",
        "pathway": "mixed_pathway_component",
        "epic": "mixed_epic_component",
        "mcp": "mixed_mcp_component",
    },
]


def stem(file_name: str) -> str:
    return Path(str(file_name)).stem


def mean_existing(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def max_existing(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].apply(pd.to_numeric, errors="coerce").max(axis=1)


def support_level(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value >= 0.50:
        return "strong"
    if value >= 0.20:
        return "partial"
    if value >= 0.00:
        return "weak"
    return "unsupported"


def load_merged() -> pd.DataFrame:
    base = pd.read_csv(T2B)
    base = base.loc[base["model"].eq("temperature_scaled_sgd")].copy()
    base["sample_id"] = base["file"].map(stem)
    base["sample_key"] = base["pool"].astype(str) + "::" + base["file"].astype(str)

    marker = pd.read_csv(T4_MARKER)
    marker_cols = [
        "pool",
        "file",
        "immune_core_z",
        "t_cell_nk_z",
        "myeloid_inflammation_z",
        "hematologic_lineage_z",
        "epithelial_z",
        "stromal_ecm_z",
        "proliferation_z",
        "interferon_z",
    ]
    marker = marker[[c for c in marker_cols if c in marker.columns]]

    pathway = pd.read_csv(T4_PATHWAY)
    pathway_cols = [
        "pool",
        "file",
        "ifn_alpha_score",
        "ifn_gamma_score",
        "tnfa_nfkb_score",
        "inflammatory_response_score",
        "complement_score",
        "t_cell_cytotoxic_score",
        "myeloid_activation_score",
        "emt_stromal_score",
        "epithelial_identity_score",
        "g2m_checkpoint_score",
        "e2f_targets_score",
        "hypoxia_score",
        "angiogenesis_score",
    ]
    pathway = pathway[[c for c in pathway_cols if c in pathway.columns]]

    epic = pd.read_csv(T4D_EPIC)
    epic_cols = [
        "pool",
        "file",
        "convergeCode",
        "epic_immune_fraction_z",
        "epic_tcell_fraction_z",
        "epic_macrophage_fraction_z",
        "epic_caf_fraction_z",
        "epic_endothelial_fraction_z",
        "epic_stromal_fraction_z",
        "epic_other_fraction_z",
    ]
    epic = epic[[c for c in epic_cols if c in epic.columns]]

    mcp = pd.read_csv(T4E_MCP)
    mcp_cols = [
        "pool",
        "file",
        "mcp_immune_z_mean",
        "mcp_t_nk_cytotoxic_z_mean",
        "mcp_myeloid_z_mean",
        "mcp_stromal_z_mean",
        "mcp_endothelial_z",
        "mcp_fibroblast_z",
    ]
    mcp = mcp[[c for c in mcp_cols if c in mcp.columns]]

    out = base.merge(marker, on=["pool", "file"], how="left")
    out = out.merge(pathway, on=["pool", "file"], how="left")
    out = out.merge(epic, on=["pool", "file"], how="left")
    out = out.merge(mcp, on=["pool", "file"], how="left")

    out["marker_immune_support"] = mean_existing(
        out,
        [
            "immune_core_z",
            "t_cell_nk_z",
            "myeloid_inflammation_z",
            "hematologic_lineage_z",
            "interferon_z",
        ],
    )
    out["pathway_immune_support"] = mean_existing(
        out,
        [
            "ifn_alpha_score",
            "ifn_gamma_score",
            "tnfa_nfkb_score",
            "inflammatory_response_score",
            "t_cell_cytotoxic_score",
            "myeloid_activation_score",
        ],
    )
    out["epic_immune_support"] = mean_existing(
        out, ["epic_immune_fraction_z", "epic_tcell_fraction_z", "epic_macrophage_fraction_z"]
    )
    out["mcp_immune_support"] = mean_existing(
        out, ["mcp_immune_z_mean", "mcp_t_nk_cytotoxic_z_mean", "mcp_myeloid_z_mean"]
    )
    out["immune_support"] = mean_existing(
        out, ["marker_immune_support", "pathway_immune_support", "epic_immune_support", "mcp_immune_support"]
    )

    out["marker_context_support"] = mean_existing(out, ["stromal_ecm_z", "epithelial_z", "proliferation_z"])
    out["pathway_context_support"] = mean_existing(
        out,
        [
            "emt_stromal_score",
            "epithelial_identity_score",
            "g2m_checkpoint_score",
            "e2f_targets_score",
            "hypoxia_score",
            "angiogenesis_score",
        ],
    )
    out["epic_context_support"] = mean_existing(
        out,
        [
            "epic_caf_fraction_z",
            "epic_endothelial_fraction_z",
            "epic_stromal_fraction_z",
            "epic_other_fraction_z",
        ],
    )
    out["mcp_context_support"] = mean_existing(
        out, ["mcp_stromal_z_mean", "mcp_endothelial_z", "mcp_fibroblast_z"]
    )
    out["context_support"] = mean_existing(
        out, ["marker_context_support", "pathway_context_support", "epic_context_support", "mcp_context_support"]
    )

    out["marker_tumor_like_support"] = mean_existing(out, ["epithelial_z", "proliferation_z"])
    out["pathway_tumor_like_support"] = mean_existing(
        out,
        [
            "epithelial_identity_score",
            "g2m_checkpoint_score",
            "e2f_targets_score",
            "hypoxia_score",
            "angiogenesis_score",
        ],
    )
    out["tumor_like_support"] = mean_existing(
        out,
        [
            "marker_tumor_like_support",
            "pathway_tumor_like_support",
            "epic_context_support",
            "mcp_context_support",
        ],
    )

    out["positive_evidence_axis_count"] = (
        out[["immune_support", "context_support", "tumor_like_support"]].apply(pd.to_numeric, errors="coerce") >= 0.20
    ).sum(axis=1)
    out["mixed_signal_max_support"] = max_existing(out, ["immune_support", "context_support", "tumor_like_support"])
    out["mixed_evidence_support"] = out["mixed_signal_max_support"] * np.where(
        out["positive_evidence_axis_count"] >= 2,
        1.0,
        np.where(out["positive_evidence_axis_count"].eq(1), 0.45, 0.0),
    )
    out["clean_context_support"] = out["context_support"] - np.maximum(
        out["immune_support"].apply(pd.to_numeric, errors="coerce").fillna(0), 0
    ) * 0.20

    out["mixed_marker_component"] = max_existing(out, ["marker_immune_support", "marker_context_support", "marker_tumor_like_support"])
    out["mixed_pathway_component"] = max_existing(
        out, ["pathway_immune_support", "pathway_context_support", "pathway_tumor_like_support"]
    )
    out["mixed_epic_component"] = max_existing(out, ["epic_immune_support", "epic_context_support"])
    out["mixed_mcp_component"] = max_existing(out, ["mcp_immune_support", "mcp_context_support"])

    text_cols = [
        "semantic_profile_summary",
        "semantic_disease_biological_readout",
        "semantic_disease_what_to_trust",
        "semantic_state_subprofile",
        "semantic_disease_semantic_subprofile",
    ]
    out["portrait_text_for_grounding"] = (
        out[[c for c in text_cols if c in out.columns]].fillna("").astype(str).agg(" | ".join, axis=1)
    )
    return out


def matched_phrases(text: str, patterns: Iterable[str]) -> str:
    lowered = text.lower()
    hits: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            token = match.group(0)
            if token not in hits:
                hits.append(token)
    return "; ".join(hits[:10])


def extract_claim_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        text = str(row.get("portrait_text_for_grounding", ""))
        for spec in CLAIM_SPECS:
            patterns = spec["patterns"]
            hits = matched_phrases(text, patterns)  # type: ignore[arg-type]
            if not hits:
                continue
            support_col = str(spec["support"])
            marker_col = str(spec["marker"])
            pathway_col = str(spec["pathway"])
            epic_col = str(spec["epic"])
            mcp_col = str(spec["mcp"])
            support = pd.to_numeric(pd.Series([row.get(support_col, np.nan)]), errors="coerce").iloc[0]
            evidence_values = [
                pd.to_numeric(pd.Series([row.get(marker_col, np.nan)]), errors="coerce").iloc[0],
                pd.to_numeric(pd.Series([row.get(pathway_col, np.nan)]), errors="coerce").iloc[0],
                pd.to_numeric(pd.Series([row.get(epic_col, np.nan)]), errors="coerce").iloc[0],
                pd.to_numeric(pd.Series([row.get(mcp_col, np.nan)]), errors="coerce").iloc[0],
            ]
            rows.append(
                {
                    "pool": row.get("pool"),
                    "file": row.get("file"),
                    "sample_id": row.get("sample_id"),
                    "sample_key": row.get("sample_key"),
                    "project": row.get("project"),
                    "expected_site_family": row.get("expected_site_family"),
                    "expected_disease_family": row.get("expected_disease_family"),
                    "closed_set_disease_family": row.get("closed_set_disease_family"),
                    "closed_set_calibrated_confidence": row.get("closed_set_calibrated_confidence"),
                    "openworld_status": row.get("openworld_status"),
                    "semantic_state_family": row.get("semantic_state_family"),
                    "semantic_state_subprofile": row.get("semantic_state_subprofile"),
                    "claim_type": spec["claim_type"],
                    "claim_label": spec["label"],
                    "matched_terms": hits,
                    "support_score": support,
                    "support_level": support_level(float(support)) if np.isfinite(support) else "missing",
                    "marker_component": evidence_values[0],
                    "pathway_component": evidence_values[1],
                    "epic_component": evidence_values[2],
                    "mcp_component": evidence_values[3],
                    "positive_evidence_axis_count": row.get("positive_evidence_axis_count"),
                    "immune_support": row.get("immune_support"),
                    "context_support": row.get("context_support"),
                    "tumor_like_support": row.get("tumor_like_support"),
                    "mixed_evidence_support": row.get("mixed_evidence_support"),
                    "portrait_text_for_grounding": text,
                }
            )
    return pd.DataFrame(rows)


def summarize_claims(claims: pd.DataFrame) -> pd.DataFrame:
    level_order = ["strong", "partial", "weak", "unsupported", "missing"]
    grouped = []
    for claim_type, sub in claims.groupby("claim_type", sort=False):
        levels = sub["support_level"].value_counts(normalize=True).reindex(level_order).fillna(0.0)
        grouped.append(
            {
                "claim_type": claim_type,
                "claim_label": sub["claim_label"].iloc[0],
                "n_claim_rows": len(sub),
                "n_samples": sub["sample_key"].nunique(),
                "mean_support_score": sub["support_score"].mean(),
                "median_support_score": sub["support_score"].median(),
                "strong_rate": levels["strong"],
                "partial_or_strong_rate": levels["strong"] + levels["partial"],
                "weak_or_unsupported_rate": levels["weak"] + levels["unsupported"],
                "marker_component_mean": sub["marker_component"].mean(),
                "pathway_component_mean": sub["pathway_component"].mean(),
                "epic_component_mean": sub["epic_component"].mean(),
                "mcp_component_mean": sub["mcp_component"].mean(),
            }
        )
    return pd.DataFrame(grouped)


def summarize_by_state(claims: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (state, claim_type), sub in claims.groupby(["semantic_state_family", "claim_type"], dropna=False):
        rows.append(
            {
                "semantic_state_family": state,
                "claim_type": claim_type,
                "claim_label": sub["claim_label"].iloc[0],
                "n_claim_rows": len(sub),
                "n_samples": sub["sample_key"].nunique(),
                "mean_support_score": sub["support_score"].mean(),
                "partial_or_strong_rate": sub["support_level"].isin(["partial", "strong"]).mean(),
                "strong_rate": sub["support_level"].eq("strong").mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["semantic_state_family", "claim_type"])


def plot_support_heatmap(summary: pd.DataFrame, path: Path) -> None:
    components = ["marker_component_mean", "pathway_component_mean", "epic_component_mean", "mcp_component_mean"]
    labels = ["Marker", "Pathway", "EPIC", "MCP-counter"]
    matrix = summary.set_index("claim_label")[components].fillna(0.0)
    fig, ax = plt.subplots(figsize=(7.2, max(2.7, 0.42 * len(matrix) + 1.4)))
    im = ax.imshow(matrix.values, aspect="auto", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.tick_params(labelsize=7)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color="black")
    ax.set_title("Independent evidence attached to portrait claims", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("mean claim-linked score", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_support_rates(summary: pd.DataFrame, path: Path) -> None:
    data = summary.sort_values("partial_or_strong_rate", ascending=True)
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7.2, max(2.7, 0.44 * len(data) + 1.3)))
    ax.barh(y, data["partial_or_strong_rate"], color="#2f6f9f", label="partial or strong")
    ax.barh(y, data["strong_rate"], color="#0b2e4a", label="strong")
    ax.set_yticks(y, labels=data["claim_label"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("support rate", fontsize=7)
    ax.set_title("How often portrait claims receive independent support", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.tick_params(labelsize=7)
    for yi, value in zip(y, data["partial_or_strong_rate"]):
        ax.text(value + 0.015, yi, f"{value:.2f}", va="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary(
    input_sample_count: int,
    claims: pd.DataFrame,
    summary: pd.DataFrame,
    state_summary: pd.DataFrame,
) -> None:
    total_samples = claims["sample_key"].nunique()
    total_claims = len(claims)
    claim_lines = []
    for _, row in summary.sort_values("partial_or_strong_rate", ascending=False).iterrows():
        claim_lines.append(
            "- `{claim_type}`: n={n}, mean support `{mean:.3f}`, partial/strong `{rate:.3f}`, strong `{strong:.3f}`".format(
                claim_type=row["claim_type"],
                n=int(row["n_claim_rows"]),
                mean=float(row["mean_support_score"]),
                rate=float(row["partial_or_strong_rate"]),
                strong=float(row["strong_rate"]),
            )
        )
    weak = summary.sort_values("partial_or_strong_rate", ascending=True).head(2)
    weak_lines = [
        f"- `{row.claim_type}` needs cautious wording: partial/strong `{row.partial_or_strong_rate:.3f}`, mean support `{row.mean_support_score:.3f}`"
        for row in weak.itertuples()
    ]
    text = "\n".join(
        [
            "# Portrait claim grounding",
            "",
            "## Purpose",
            "",
            "This analysis checks whether the natural-language molecular portraits can be traced to independent marker, pathway, EPIC and MCP-counter evidence. It is a reproducible alternative to asking readers to trust fluent model-generated text.",
            "",
            "## Data",
            "",
            f"- External samples with temperature-scaled predictions: `{input_sample_count}`",
            f"- Samples with at least one extracted auditable claim: `{total_samples}`",
            f"- Extracted claim rows: `{total_claims}`",
            "- Claim source fields: `semantic_profile_summary`, `semantic_disease_biological_readout`, `semantic_disease_what_to_trust`, `semantic_state_subprofile`, `semantic_disease_semantic_subprofile`",
            "- Independent evidence: marker programs, pathway modules, official EPIC deconvolution and official MCP-counter.",
            "",
            "## Claim-level support",
            "",
            *claim_lines,
            "",
            "## Main caution flags",
            "",
            *weak_lines,
            "",
            "## Interpretation",
            "",
            "This analysis links portrait terms to independent marker, pathway and deconvolution-derived signals. It tests whether the molecular descriptors used by the portrait readout are supported by post hoc biological measurements.",
            "",
            "## Output files",
            "",
            "- `t7_portrait_claim_grounding_by_claim.csv`",
            "- `t7_claim_support_summary.csv`",
            "- `t7_claim_support_by_state.csv`",
            "- `t7_claim_evidence_heatmap.svg`",
            "- `t7_claim_support_rates.svg`",
        ]
    )
    (OUTDIR / "summary.md").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    merged = load_merged()
    claims = extract_claim_rows(merged)
    if claims.empty:
        raise RuntimeError("No portrait claims were extracted; check claim patterns and source fields.")
    summary = summarize_claims(claims)
    state_summary = summarize_by_state(claims)

    merged.to_csv(OUTDIR / "t7_merged_sample_evidence.csv", index=False)
    claims.to_csv(OUTDIR / "t7_portrait_claim_grounding_by_claim.csv", index=False)
    summary.to_csv(OUTDIR / "t7_claim_support_summary.csv", index=False)
    state_summary.to_csv(OUTDIR / "t7_claim_support_by_state.csv", index=False)

    plot_support_heatmap(summary, OUTDIR / "t7_claim_evidence_heatmap.svg")
    plot_support_rates(summary, OUTDIR / "t7_claim_support_rates.svg")
    write_summary(merged["sample_key"].nunique(), claims, summary, state_summary)

    print(f"Wrote {len(claims)} claim rows for {claims['sample_key'].nunique()} sample rows to {OUTDIR}")


if __name__ == "__main__":
    main()
