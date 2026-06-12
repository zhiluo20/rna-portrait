#!/usr/bin/env python3
"""Structured case audit for open-world RNA molecular portraits."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, Iterable, List

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
OUTDIR = SUPP_DIR / "T6_structured_case_audit"

T2B = SUPP_DIR / "T2_calibrated_closed_set" / "t2b_external_predictions.csv"
T4_MARKER = SUPP_DIR / "T4_marker_program_state_validation" / "t4_marker_scores_by_sample.csv"
T4_PATHWAY = SUPP_DIR / "T4b_pathway_state_validation" / "t4b_pathway_scores_by_sample.csv"
T4D_EPIC = SUPP_DIR / "T4d_official_EPIC_deconvolution" / "t4d_epic_scores_by_sample.csv"
T4E_MCP = SUPP_DIR / "T4e_official_MCPcounter_deconvolution" / "t4e_mcpcounter_scores_by_sample.csv"


CASE_QUOTAS = [
    ("highconf_closed_set_overcall", 8),
    ("hematologic_immune_override", 4),
    ("clean_context_override", 4),
    ("generic_context_audit", 4),
    ("epithelial_mixed_audit", 4),
    ("stable_control", 4),
    ("unsupported_disease_semantics", 4),
]


STATE_CLAIMS = {
    "hematologic_override": "Blood/immune activation dominates; disease-family semantics should not be read as a stable hematologic malignancy without context.",
    "clean_anchor_override": "The cleaner anchor/context is stronger than the disease-like label; read this as a context-dominant profile.",
    "generic_context_override": "The sample carries broader context signal rather than a clean disease-family identity; this claim must remain cautious.",
    "epithelial_override": "The profile is mixed epithelial/solid-tumor-like plus immune/context signal, not a pure epithelial state.",
    "stable_consensus": "The broad disease semantic frame is relatively stable and can be used as a control case.",
    "unsupported_semantics": "The disease layer is too unstable to interpret directly; trust only site/context/state-level information.",
    "other": "No special override profile; keep the molecular portrait descriptive and cautious.",
}


def claim_for_row(row: pd.Series) -> str:
    state = str(row.get("semantic_state_family", "other"))
    status = str(row.get("openworld_status", ""))
    if status == "unsupported" or state == "unsupported_semantics":
        return STATE_CLAIMS["unsupported_semantics"]
    if state == "stable_consensus" and status != "stable":
        return "The state family is stable-like, but the disease card resolves the disease layer as mixed; do not overcall a stable disease identity."
    return STATE_CLAIMS.get(state, STATE_CLAIMS["other"])


def stem(file_name: str) -> str:
    return Path(str(file_name)).stem


def mean_existing(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].apply(pd.to_numeric, errors="coerce").mean(axis=1)


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
    base = base.loc[base["model"] == "temperature_scaled_sgd"].copy()
    base["sample_id"] = base["file"].map(stem)

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
        out, ["immune_core_z", "t_cell_nk_z", "myeloid_inflammation_z", "hematologic_lineage_z", "interferon_z"]
    )
    out["pathway_immune_support"] = mean_existing(
        out, ["ifn_alpha_score", "ifn_gamma_score", "tnfa_nfkb_score", "inflammatory_response_score", "t_cell_cytotoxic_score", "myeloid_activation_score"]
    )
    out["epic_immune_support"] = mean_existing(out, ["epic_immune_fraction_z", "epic_tcell_fraction_z", "epic_macrophage_fraction_z"])
    out["mcp_immune_support"] = mean_existing(out, ["mcp_immune_z_mean", "mcp_t_nk_cytotoxic_z_mean", "mcp_myeloid_z_mean"])
    out["immune_support"] = mean_existing(out, ["marker_immune_support", "pathway_immune_support", "epic_immune_support", "mcp_immune_support"])

    out["marker_context_support"] = mean_existing(out, ["stromal_ecm_z", "epithelial_z", "proliferation_z"])
    out["pathway_context_support"] = mean_existing(
        out, ["emt_stromal_score", "epithelial_identity_score", "g2m_checkpoint_score", "e2f_targets_score", "hypoxia_score", "angiogenesis_score"]
    )
    out["epic_context_support"] = mean_existing(
        out, ["epic_caf_fraction_z", "epic_endothelial_fraction_z", "epic_stromal_fraction_z", "epic_other_fraction_z"]
    )
    out["mcp_context_support"] = mean_existing(out, ["mcp_stromal_z_mean", "mcp_endothelial_z", "mcp_fibroblast_z"])
    out["context_support"] = mean_existing(out, ["marker_context_support", "pathway_context_support", "epic_context_support", "mcp_context_support"])

    out["epithelial_proliferation_support"] = mean_existing(
        out, ["epithelial_z", "proliferation_z", "epithelial_identity_score", "g2m_checkpoint_score", "e2f_targets_score"]
    )
    out["mixed_signal_support"] = out[["immune_support", "context_support", "epithelial_proliferation_support"]].apply(
        pd.to_numeric, errors="coerce"
    ).max(axis=1)

    out["state_claim"] = out.apply(claim_for_row, axis=1)
    out["expected_claim_support"] = np.nan
    state = out["semantic_state_family"].fillna("")
    out.loc[state.eq("hematologic_override"), "expected_claim_support"] = out.loc[state.eq("hematologic_override"), "immune_support"]
    out.loc[state.eq("clean_anchor_override"), "expected_claim_support"] = out.loc[state.eq("clean_anchor_override"), "context_support"]
    out.loc[state.eq("generic_context_override"), "expected_claim_support"] = out.loc[state.eq("generic_context_override"), "context_support"]
    out.loc[state.eq("epithelial_override"), "expected_claim_support"] = out.loc[state.eq("epithelial_override"), "mixed_signal_support"]
    stable = state.eq("stable_consensus") & out["openworld_status"].eq("stable")
    stable_mixed = state.eq("stable_consensus") & ~out["openworld_status"].eq("stable")
    out.loc[stable, "expected_claim_support"] = 1.0 - out.loc[stable, ["immune_support", "context_support"]].abs().mean(axis=1).clip(upper=1.0)
    out.loc[stable_mixed, "expected_claim_support"] = out.loc[stable_mixed, "mixed_signal_support"]
    out.loc[state.eq("unsupported_semantics"), "expected_claim_support"] = np.where(out.loc[state.eq("unsupported_semantics"), "openworld_status"].eq("unsupported"), 0.5, 0.0)
    out["support_level"] = out["expected_claim_support"].map(support_level)

    out["closed_set_conflict"] = (
        (pd.to_numeric(out["closed_set_calibrated_confidence"], errors="coerce") >= 0.70)
        & out["openworld_status"].isin(["mixed", "unsupported"])
    )
    out["retrieval_only_text"] = (
        "prototype="
        + out[["prototype_site", "prototype_tumor", "prototype_disease"]].fillna("NA").astype(str).agg(" / ".join, axis=1)
        + "; nearest_neighbor="
        + out[["neighbor1_site", "neighbor1_tumor", "neighbor1_disease"]].fillna("NA").astype(str).agg(" / ".join, axis=1)
    )
    out["open_world_portrait"] = (
        out["semantic_profile_summary"].fillna("no profile summary").astype(str)
        + " "
        + out["semantic_disease_biological_readout"].fillna("").astype(str)
    ).str.strip()
    return out


def select_cases(df: pd.DataFrame) -> pd.DataFrame:
    selected: List[pd.DataFrame] = []
    used: set[str] = set()

    def add(label: str, sub: pd.DataFrame, n: int, sort_cols: List[str]) -> None:
        nonlocal selected, used
        if sub.empty:
            return
        s = sub.copy()
        for col in sort_cols:
            s[col] = pd.to_numeric(s[col], errors="coerce")
        s = s.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        rows = []
        for _, row in s.iterrows():
            key = str(row["file"])
            if key in used:
                continue
            used.add(key)
            r = row.copy()
            r["case_reason"] = label
            rows.append(r)
            if len(rows) >= n:
                break
        if rows:
            selected.append(pd.DataFrame(rows))

    for label, n in CASE_QUOTAS:
        if label == "highconf_closed_set_overcall":
            sub = df.loc[df["closed_set_conflict"]]
            add(label, sub, n, ["closed_set_calibrated_confidence", "mixed_signal_support"])
        elif label == "hematologic_immune_override":
            sub = df.loc[df["semantic_state_family"].eq("hematologic_override")]
            add(label, sub, n, ["immune_support", "closed_set_calibrated_confidence"])
        elif label == "clean_context_override":
            sub = df.loc[df["semantic_state_family"].eq("clean_anchor_override")]
            add(label, sub, n, ["context_support", "closed_set_calibrated_confidence"])
        elif label == "generic_context_audit":
            sub = df.loc[df["semantic_state_family"].eq("generic_context_override")]
            add(label, sub, n, ["context_support", "closed_set_calibrated_confidence"])
        elif label == "epithelial_mixed_audit":
            sub = df.loc[df["semantic_state_family"].eq("epithelial_override")]
            add(label, sub, n, ["mixed_signal_support", "closed_set_calibrated_confidence"])
        elif label == "stable_control":
            sub = df.loc[df["semantic_state_family"].eq("stable_consensus") & df["openworld_status"].eq("stable")]
            add(label, sub, n, ["closed_set_calibrated_confidence", "expected_claim_support"])
        elif label == "unsupported_disease_semantics":
            sub = df.loc[df["semantic_state_family"].eq("unsupported_semantics")]
            add(label, sub, n, ["closed_set_calibrated_confidence", "mixed_signal_support"])

    cases = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    cases.insert(0, "case_id", [f"case_{i:02d}" for i in range(1, len(cases) + 1)])
    return cases


def write_heatmap(cases: pd.DataFrame, path: Path) -> None:
    cols = [
        ("closed_set_calibrated_confidence", "closed-set conf"),
        ("marker_immune_support", "marker immune"),
        ("pathway_immune_support", "pathway immune"),
        ("epic_immune_support", "EPIC immune"),
        ("mcp_immune_support", "MCP immune"),
        ("marker_context_support", "marker context"),
        ("pathway_context_support", "pathway context"),
        ("epic_context_support", "EPIC context"),
        ("mcp_context_support", "MCP context"),
        ("expected_claim_support", "claim support"),
    ]
    cell_w, cell_h = 95, 25
    left, top = 360, 112
    width = left + cell_w * len(cols) + 30
    height = top + cell_h * len(cases) + 60

    def color(v: float, conf: bool = False) -> str:
        if not np.isfinite(v):
            return "#f2f2f2"
        if conf:
            v = max(0, min(1, v))
            r = int(230 - 75 * v)
            g = int(240 - 130 * v)
            b = int(255 - 120 * v)
            return f"#{r:02x}{g:02x}{b:02x}"
        v = max(-1.5, min(1.5, v)) / 1.5
        if v >= 0:
            return f"#ff{int(246 - 122*v):02x}{int(230 - 164*v):02x}"
        v = abs(v)
        return f"#{int(225 - 146*v):02x}{int(238 - 94*v):02x}ff"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">Structured case audit evidence matrix</text>',
        '<text x="24" y="56" font-family="Arial" font-size="12" fill="#666">Rows are representative samples; evidence scores are z-like support values except closed-set confidence.</text>',
    ]
    for j, (_, label) in enumerate(cols):
        x = left + j * cell_w + cell_w / 2
        words = label.split(" ", 1)
        if len(words) == 1:
            parts.append(f'<text x="{x}" y="{top - 18}" font-family="Arial" font-size="10" text-anchor="middle">{html.escape(label)}</text>')
        else:
            parts.append(
                f'<text x="{x}" y="{top - 30}" font-family="Arial" font-size="10" text-anchor="middle">'
                f'<tspan x="{x}" dy="0">{html.escape(words[0])}</tspan>'
                f'<tspan x="{x}" dy="12">{html.escape(words[1])}</tspan>'
                "</text>"
            )
    for i, row in cases.iterrows():
        y = top + i * cell_h
        label = f"{row['case_id']} {row['semantic_state_family']} ({row['openworld_status']})"
        parts.append(f'<text x="20" y="{y + 16}" font-family="Arial" font-size="10">{html.escape(label[:58])}</text>')
        for j, (col, _) in enumerate(cols):
            x = left + j * cell_w
            val = float(row[col]) if pd.notna(row[col]) else np.nan
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{color(val, col == "closed_set_calibrated_confidence")}" stroke="#fff"/>')
            text = f"{val:.2f}" if np.isfinite(val) else "NA"
            parts.append(f'<text x="{x + cell_w/2}" y="{y + 16}" font-family="Arial" font-size="9" text-anchor="middle" fill="#222">{text}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_markdown(cases: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Structured case audit",
        "",
        "## Purpose",
        "",
        "This audit replaces an external expert blind review with a transparent, reproducible case panel. Each selected sample is linked to the calibrated closed-set call, open-world molecular portrait, and independent marker/pathway/EPIC/MCP-counter evidence.",
        "",
        "## Selection",
        "",
        f"- Representative cases: `{len(cases)}`",
        "- Source: temperature-scaled external predictions joined with marker, pathway and deconvolution sample-level evidence.",
        "- Case reasons: high-confidence closed-set overcall, hematologic immune override, clean context override, generic context audit, epithelial mixed audit, stable control, unsupported disease semantics.",
        "",
        "## Summary",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"- `{row.case_reason}`: n={int(row.n)}, closed-set conflicts={int(row.closed_set_conflicts)}, strong/partial support={int(row.strong_or_partial_support)}")
    lines.extend(["", "## Representative Cases", ""])
    for row in cases.itertuples(index=False):
        lines.extend(
            [
                f"### {row.case_id}: {row.pool} / {row.sample_id}",
                "",
                f"- reason: `{row.case_reason}`",
                f"- source label: `{row.expected_site_family}` / `{row.expected_disease_family}`",
                f"- closed-set label: `{row.closed_set_disease_family}` at confidence `{float(row.closed_set_calibrated_confidence):.3f}`",
                f"- open-world state: `{row.semantic_state_family}` / `{row.semantic_state_subprofile}`; disease status `{row.openworld_status}`",
                f"- retrieval-only text: {row.retrieval_only_text}",
                f"- portrait: {row.open_world_portrait}",
                f"- auditable claim: {row.state_claim}",
                f"- support level: `{row.support_level}`; claim support `{float(row.expected_claim_support):.3f}`",
                f"- evidence: immune marker `{float(row.marker_immune_support):.2f}`, immune pathway `{float(row.pathway_immune_support):.2f}`, EPIC immune `{float(row.epic_immune_support):.2f}`, MCP immune `{float(row.mcp_immune_support):.2f}`, context marker `{float(row.marker_context_support):.2f}`, context pathway `{float(row.pathway_context_support):.2f}`, EPIC context `{float(row.epic_context_support):.2f}`, MCP context `{float(row.mcp_context_support):.2f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Output files",
            "",
            "- `t6_case_audit_table.csv`",
            "- `t6_case_audit_summary.csv`",
            "- `t6_case_evidence_matrix.svg`",
            "",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    merged = load_merged()
    cases = select_cases(merged)
    summary = (
        cases.assign(
            strong_or_partial_support=cases["support_level"].isin(["strong", "partial"]),
            closed_set_conflict_flag=cases["closed_set_conflict"].astype(bool),
        )
        .groupby("case_reason")
        .agg(
            n=("case_id", "count"),
            closed_set_conflicts=("closed_set_conflict_flag", "sum"),
            strong_or_partial_support=("strong_or_partial_support", "sum"),
            mean_claim_support=("expected_claim_support", "mean"),
            mean_closed_set_confidence=("closed_set_calibrated_confidence", "mean"),
        )
        .reset_index()
    )
    keep_cols = [
        "case_id",
        "case_reason",
        "pool",
        "sample_id",
        "file",
        "project",
        "expected_site_family",
        "expected_disease_family",
        "closed_set_disease_family",
        "closed_set_calibrated_confidence",
        "closed_set_conflict",
        "openworld_status",
        "semantic_state_family",
        "semantic_state_subprofile",
        "semantic_disease_semantic_status",
        "retrieval_only_text",
        "open_world_portrait",
        "state_claim",
        "support_level",
        "expected_claim_support",
        "marker_immune_support",
        "pathway_immune_support",
        "epic_immune_support",
        "mcp_immune_support",
        "immune_support",
        "marker_context_support",
        "pathway_context_support",
        "epic_context_support",
        "mcp_context_support",
        "context_support",
        "epithelial_proliferation_support",
        "mixed_signal_support",
        "convergeCode",
    ]
    cases[keep_cols].to_csv(OUTDIR / "t6_case_audit_table.csv", index=False)
    summary.to_csv(OUTDIR / "t6_case_audit_summary.csv", index=False)
    write_heatmap(cases, OUTDIR / "t6_case_evidence_matrix.svg")
    write_markdown(cases, summary)


if __name__ == "__main__":
    main()
