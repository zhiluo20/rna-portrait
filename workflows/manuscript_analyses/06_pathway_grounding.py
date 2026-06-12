#!/usr/bin/env python3
"""Pathway-level validation of semantic state families with Hallmark-style modules."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List, Tuple

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
OUTDIR = SUPP_DIR / "T4b_pathway_state_validation"

POOLS = {
    "External-180": {
        "details": ARTIFACT_DIR / "external_180_trimmed_benchmark" / "details.csv",
        "inputs": VALIDATION_DIR / "external_180",
    },
    "MultiSource-450": {
        "details": ARTIFACT_DIR / "multisource_450_trimmed_benchmark" / "details.csv",
        "inputs": VALIDATION_DIR / "multisource_450",
    },
}

# Compact Hallmark-style modules. These are not full MSigDB gene sets; they are
# transparent pathway probes chosen to test broad biological direction.
PATHWAYS: Dict[str, List[str]] = {
    "ifn_alpha": ["ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1", "OAS1", "OAS2", "OASL", "IRF7", "STAT1", "IFI6", "IFI27"],
    "ifn_gamma": ["CXCL9", "CXCL10", "CXCL11", "STAT1", "IRF1", "GBP1", "GBP2", "IDO1", "HLA-DRA", "CD74", "B2M", "TAP1"],
    "tnfa_nfkb": ["NFKBIA", "TNFAIP3", "BIRC3", "RELA", "ICAM1", "JUNB", "CXCL2", "CXCL3", "IL6", "CCL2", "NFKB1", "TRAF1"],
    "inflammatory_response": ["IL1B", "IL6", "CXCL8", "CCL2", "CCL3", "CCL4", "NLRP3", "PTGS2", "TNF", "TLR2", "TLR4", "S100A8"],
    "complement": ["C1QA", "C1QB", "C1QC", "C3", "C4A", "C4B", "CFB", "SERPING1", "FCN1", "VSIG4", "ITGAM", "TYROBP"],
    "t_cell_cytotoxic": ["CD3D", "CD3E", "TRAC", "CD8A", "CD8B", "NKG7", "GNLY", "GZMB", "PRF1", "KLRD1", "CCL5", "LCK"],
    "myeloid_activation": ["LYZ", "LST1", "TYROBP", "FCGR3A", "S100A8", "S100A9", "FCN1", "CD68", "MS4A7", "AIF1", "CST3", "CTSS"],
    "emt_stromal": ["VIM", "FN1", "COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "FAP", "ACTA2", "TAGLN", "MMP2", "SPARC"],
    "epithelial_identity": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT17", "KRT5", "KRT14", "CDH1", "MUC1", "MSLN", "CLDN4", "TACSTD2"],
    "g2m_checkpoint": ["MKI67", "TOP2A", "UBE2C", "BIRC5", "CCNB1", "CCNB2", "CDK1", "CDC20", "AURKB", "NUSAP1", "CENPF", "KIF11"],
    "e2f_targets": ["MCM2", "MCM4", "MCM5", "MCM6", "PCNA", "TYMS", "RRM2", "E2F1", "CDK2", "CDC6", "CCNE1", "TK1"],
    "hypoxia": ["VEGFA", "CA9", "SLC2A1", "LDHA", "ENO1", "PGK1", "ANGPTL4", "BNIP3", "NDRG1", "ADM", "P4HA1", "EGLN3"],
    "angiogenesis": ["VEGFA", "KDR", "FLT1", "PECAM1", "VWF", "CDH5", "ANGPT2", "ESAM", "ENG", "TEK", "MCAM", "ROBO4"],
    "oxidative_phosphorylation": ["NDUFA1", "NDUFB8", "SDHB", "UQCRC1", "COX4I1", "COX5A", "ATP5F1A", "ATP5F1B", "ATP5MC1", "ATP5PO", "NDUFS1", "UQCRQ"],
    "fatty_acid_metabolism": ["ACADM", "ACADVL", "CPT1A", "CPT2", "HADHA", "HADHB", "ACOX1", "EHHADH", "FABP1", "SCD", "FASN", "ACSL1"],
    "xenobiotic_metabolism": ["CYP3A4", "CYP2E1", "CYP2C9", "UGT1A1", "UGT2B7", "GSTA1", "GSTA2", "GSTP1", "ALDH1A1", "AKR1C1", "NQO1", "ABCC2"],
}


def parse_gene_value_file(path: Path) -> pd.Series:
    rows: List[Tuple[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            gene, value = line.split(",", 1)
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            gene, value = parts[0], parts[1]
        try:
            rows.append((gene.strip().upper(), float(value)))
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"no valid rows in {path}")
    return pd.DataFrame(rows, columns=["gene", "value"]).groupby("gene")["value"].sum().astype(np.float32)


def transform_external_values(values: pd.Series) -> pd.Series:
    arr = values.to_numpy(dtype=np.float32)
    frac_nonint = float(np.mean(np.abs(arr - np.round(arr)) > 1e-6))
    vmax = float(np.max(arr)) if len(arr) else 0.0
    if vmax > 50 or frac_nonint < 0.1:
        return np.log1p(values.clip(lower=0)).astype(np.float32)
    return values.astype(np.float32)


def load_pool_expression(pool: str, details_path: Path, input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    details = pd.read_csv(details_path)
    genes = sorted({g for gs in PATHWAYS.values() for g in gs})
    rows = []
    meta_rows = []
    for rec in details.itertuples(index=False):
        values = transform_external_values(parse_gene_value_file(input_dir / str(getattr(rec, "file"))))
        row = values.reindex(genes).astype(float)
        rows.append(row)
        meta_rows.append(
            {
                "pool": pool,
                "file": str(getattr(rec, "file")),
                "project": str(getattr(rec, "project")),
                "expected_site_family": str(getattr(rec, "expected_site_family")),
                "expected_disease_family": str(getattr(rec, "expected_disease_family")),
                "semantic_state_family": str(getattr(rec, "semantic_state_family")),
                "semantic_state_subprofile": str(getattr(rec, "semantic_state_subprofile")),
                "semantic_disease_semantic_status": str(getattr(rec, "semantic_disease_semantic_status")),
                "anchor_context_state": str(getattr(rec, "anchor_context_state")),
            }
        )
    expr = pd.DataFrame(rows).reset_index(drop=True)
    meta = pd.DataFrame(meta_rows)
    return expr, meta


def zscore_pool(expr: pd.DataFrame) -> pd.DataFrame:
    out = expr.copy()
    for col in out.columns:
        vals = pd.to_numeric(out[col], errors="coerce")
        mean = vals.mean(skipna=True)
        std = vals.std(skipna=True)
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        out[col] = (vals - mean) / std
    return out


def score_pathways(expr_z: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, genes in PATHWAYS.items():
        present = [g for g in genes if g in expr_z.columns]
        if present:
            rows[f"{name}_score"] = expr_z[present].mean(axis=1, skipna=True)
            rows[f"{name}_n_present"] = len(present)
        else:
            rows[f"{name}_score"] = np.nan
            rows[f"{name}_n_present"] = 0
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    score_cols = [f"{p}_score" for p in PATHWAYS]
    by_state = df.groupby(["pool", "semantic_state_family"], dropna=False)[score_cols].mean().reset_index()
    counts = df.groupby(["pool", "semantic_state_family"], dropna=False).size().reset_index(name="n")
    by_state = by_state.merge(counts, on=["pool", "semantic_state_family"], how="left")
    by_status = df.groupby(["pool", "semantic_disease_semantic_status"], dropna=False)[score_cols].mean().reset_index()
    status_counts = df.groupby(["pool", "semantic_disease_semantic_status"], dropna=False).size().reset_index(name="n")
    by_status = by_status.merge(status_counts, on=["pool", "semantic_disease_semantic_status"], how="left")
    return by_state, by_status


def color_for_value(v: float) -> str:
    if not np.isfinite(v):
        return "#f5f5f5"
    v = max(-1.5, min(1.5, float(v))) / 1.5
    if v >= 0:
        r = 255
        g = int(246 - 125 * v)
        b = int(230 - 165 * v)
    else:
        v = abs(v)
        r = int(225 - 145 * v)
        g = int(238 - 92 * v)
        b = 255
    return f"#{r:02x}{g:02x}{b:02x}"


def write_heatmap(summary: pd.DataFrame, pool: str, path: Path) -> None:
    sub = summary.loc[summary["pool"] == pool].copy().sort_values("n", ascending=False)
    states = sub["semantic_state_family"].tolist()
    pathways = list(PATHWAYS.keys())
    cell_w, cell_h = 84, 28
    left, top = 190, 84
    width = left + cell_w * len(pathways) + 36
    height = top + cell_h * len(states) + 76
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">{html.escape(pool)} pathway validation by semantic state</text>',
        '<text x="24" y="56" font-family="Arial" font-size="12" fill="#666">Pool gene-z-scored mean module scores; red is higher, blue is lower.</text>',
    ]
    for j, pathway in enumerate(pathways):
        x = left + j * cell_w + cell_w / 2
        parts.append(f'<text x="{x}" y="{top - 10}" font-family="Arial" font-size="10" text-anchor="end" transform="rotate(-45 {x} {top - 10})">{html.escape(pathway.replace("_", " "))}</text>')
    for i, row in enumerate(sub.itertuples(index=False)):
        y = top + i * cell_h
        state = str(row.semantic_state_family)
        n = int(row.n)
        parts.append(f'<text x="18" y="{y + 18}" font-family="Arial" font-size="11">{html.escape(state)} (n={n})</text>')
        for j, pathway in enumerate(pathways):
            val = float(getattr(row, f"{pathway}_score"))
            x = left + j * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{color_for_value(val)}" stroke="#fff"/>')
            parts.append(f'<text x="{x + cell_w / 2}" y="{y + 18}" font-family="Arial" font-size="10" text-anchor="middle" fill="#222">{val:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def state_values(by_state: pd.DataFrame, pool: str, state: str, pathways: List[str]) -> dict:
    row = by_state.loc[(by_state["pool"] == pool) & (by_state["semantic_state_family"] == state)]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {p: float(r[f"{p}_score"]) for p in pathways}


def write_summary(df: pd.DataFrame, by_state: pd.DataFrame) -> None:
    immune_paths = ["ifn_alpha", "ifn_gamma", "tnfa_nfkb", "inflammatory_response", "complement", "t_cell_cytotoxic", "myeloid_activation"]
    epithelial_paths = ["epithelial_identity", "emt_stromal", "g2m_checkpoint", "e2f_targets", "hypoxia"]
    lines = [
        "# Pathway-level validation of portrait states",
        "",
        "## Purpose",
        "",
        "This analysis validates semantic state families with compact Hallmark-style pathway modules computed directly from external expression profiles. The model is used only for grouping labels; pathway scores are independent post hoc expression summaries.",
        "",
        "## Data",
        "",
    ]
    for pool, sub in df.groupby("pool"):
        lines.append(f"- {pool}: `{len(sub)}` samples / `{sub['project'].nunique()}` projects")
    lines.extend(["", "## Directional checks", ""])
    for pool in sorted(df["pool"].unique()):
        lines.extend([f"### {pool}", ""])
        for state in ["hematologic_override", "epithelial_override", "generic_context_override", "clean_anchor_override", "stable_consensus", "unsupported_semantics"]:
            row = by_state.loc[(by_state["pool"] == pool) & (by_state["semantic_state_family"] == state)]
            if row.empty:
                continue
            n = int(row.iloc[0]["n"])
            immune_mean = float(np.mean(list(state_values(by_state, pool, state, immune_paths).values())))
            epithelial_mean = float(np.mean(list(state_values(by_state, pool, state, epithelial_paths).values())))
            top = (
                row[[f"{p}_score" for p in PATHWAYS]]
                .iloc[0]
                .sort_values(ascending=False)
                .head(4)
                .rename(index=lambda x: x.replace("_score", ""))
                .to_dict()
            )
            top = {str(k): float(v) for k, v in top.items()}
            lines.append(
                f"- `{state}` n={n}: immune_module_mean `{immune_mean:.3f}`, epithelial/stromal/proliferation_module_mean `{epithelial_mean:.3f}`, top pathways `{top}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "This pathway analysis aggregates multiple expression modules and complements the marker-program and cell-composition analyses. It is not a deconvolution analysis and should be interpreted as pathway-level grounding.",
            "",
            "## Output files",
            "",
            "- `t4b_pathway_scores_by_sample.csv`",
            "- `t4b_pathway_scores_by_state.csv`",
            "- `t4b_pathway_scores_by_disease_status.csv`",
            "- `external_180_state_pathway_heatmap.svg` and `multisource_450_state_pathway_heatmap.svg`",
            "",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for pool, paths in POOLS.items():
        expr, meta = load_pool_expression(pool, paths["details"], paths["inputs"])
        scores = score_pathways(zscore_pool(expr))
        frames.append(pd.concat([meta.reset_index(drop=True), scores.reset_index(drop=True)], axis=1))
    df = pd.concat(frames, ignore_index=True)
    by_state, by_status = summarize(df)
    df.to_csv(OUTDIR / "t4b_pathway_scores_by_sample.csv", index=False)
    by_state.to_csv(OUTDIR / "t4b_pathway_scores_by_state.csv", index=False)
    by_status.to_csv(OUTDIR / "t4b_pathway_scores_by_disease_status.csv", index=False)
    write_heatmap(by_state, "External-180", OUTDIR / "external_180_state_pathway_heatmap.svg")
    write_heatmap(by_state, "MultiSource-450", OUTDIR / "multisource_450_state_pathway_heatmap.svg")
    write_summary(df, by_state)


if __name__ == "__main__":
    main()
