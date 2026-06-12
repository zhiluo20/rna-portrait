#!/usr/bin/env python3
"""Cell-composition signature validation of semantic state families.

This is a transparent pre-deconvolution screen. It uses compact signatures
inspired by MCP-counter, xCell, EPIC, ESTIMATE, and LM22/CIBERSORT-style panels,
but it does not call the official deconvolution implementations.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


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
OUTDIR = SUPP_DIR / "T4c_cell_composition_signature_validation"

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

SIGNATURES: Dict[str, Dict[str, object]] = {
    "mcp_t_cells": {
        "panel": "MCP-counter-like",
        "group": "lymphoid",
        "genes": ["CD3D", "CD3E", "CD3G", "TRAC", "CD2", "CD247", "IL7R"],
    },
    "mcp_cd8_t_cells": {
        "panel": "MCP-counter-like",
        "group": "lymphoid",
        "genes": ["CD8A", "CD8B", "GZMK", "GZMA", "GZMB", "PRF1", "NKG7"],
    },
    "mcp_cytotoxic_lymphocytes": {
        "panel": "MCP-counter-like",
        "group": "lymphoid",
        "genes": ["NKG7", "GNLY", "GZMB", "PRF1", "CTSW", "KLRD1", "CCL5"],
    },
    "mcp_b_lineage": {
        "panel": "MCP-counter-like",
        "group": "lymphoid",
        "genes": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "CD22", "PAX5"],
    },
    "mcp_nk_cells": {
        "panel": "MCP-counter-like",
        "group": "lymphoid",
        "genes": ["KLRD1", "KLRF1", "NCR1", "NKG7", "GNLY", "PRF1"],
    },
    "mcp_monocytic_lineage": {
        "panel": "MCP-counter-like",
        "group": "myeloid",
        "genes": ["LYZ", "LST1", "FCN1", "S100A8", "S100A9", "VCAN", "CTSS"],
    },
    "mcp_myeloid_dendritic": {
        "panel": "MCP-counter-like",
        "group": "myeloid",
        "genes": ["CLEC9A", "BATF3", "IRF8", "ITGAX", "CD1C", "FCER1A", "LILRA4"],
    },
    "mcp_neutrophils": {
        "panel": "MCP-counter-like",
        "group": "myeloid",
        "genes": ["FCGR3B", "CSF3R", "CXCR2", "S100A8", "S100A9", "MNDA"],
    },
    "mcp_endothelial_cells": {
        "panel": "MCP-counter-like",
        "group": "stromal",
        "genes": ["PECAM1", "VWF", "CDH5", "KDR", "FLT1", "ENG", "ESAM", "TEK"],
    },
    "mcp_fibroblasts": {
        "panel": "MCP-counter-like",
        "group": "stromal",
        "genes": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "FAP", "PDGFRA", "ACTA2"],
    },
    "epic_b_cells": {
        "panel": "EPIC-like",
        "group": "lymphoid",
        "genes": ["CD19", "MS4A1", "CD79A", "CD79B", "BANK1"],
    },
    "epic_cd4_t_cells": {
        "panel": "EPIC-like",
        "group": "lymphoid",
        "genes": ["CD4", "IL7R", "CCR7", "LTB", "LEF1", "MAL"],
    },
    "epic_cd8_t_cells": {
        "panel": "EPIC-like",
        "group": "lymphoid",
        "genes": ["CD8A", "CD8B", "GZMK", "CCL5", "GZMB"],
    },
    "epic_nk_cells": {
        "panel": "EPIC-like",
        "group": "lymphoid",
        "genes": ["GNLY", "NKG7", "KLRD1", "PRF1"],
    },
    "epic_macrophages": {
        "panel": "EPIC-like",
        "group": "myeloid",
        "genes": ["CD68", "CD163", "CSF1R", "MRC1", "MSR1", "TYROBP"],
    },
    "epic_caf": {
        "panel": "EPIC-like",
        "group": "stromal",
        "genes": ["FAP", "COL1A1", "COL1A2", "PDGFRA", "DCN", "LUM"],
    },
    "epic_endothelial": {
        "panel": "EPIC-like",
        "group": "stromal",
        "genes": ["PECAM1", "VWF", "CDH5", "KDR"],
    },
    "epic_epithelial_tumor": {
        "panel": "EPIC-like",
        "group": "epithelial",
        "genes": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT17", "CDH1", "TACSTD2"],
    },
    "xcell_immune": {
        "panel": "xCell-like",
        "group": "immune",
        "genes": ["PTPRC", "HLA-DRA", "CD74", "B2M", "HLA-A", "HLA-B", "HLA-C"],
    },
    "xcell_lymphoid": {
        "panel": "xCell-like",
        "group": "lymphoid",
        "genes": ["CD3D", "CD3E", "MS4A1", "CD79A", "NKG7", "GNLY"],
    },
    "xcell_myeloid": {
        "panel": "xCell-like",
        "group": "myeloid",
        "genes": ["LYZ", "LST1", "TYROBP", "FCGR3A", "S100A8", "S100A9", "C1QA", "C1QB"],
    },
    "xcell_stromal": {
        "panel": "xCell-like",
        "group": "stromal",
        "genes": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "ACTA2", "TAGLN"],
    },
    "xcell_endothelial": {
        "panel": "xCell-like",
        "group": "stromal",
        "genes": ["PECAM1", "VWF", "CDH5", "ENG", "ESAM"],
    },
    "xcell_epithelial": {
        "panel": "xCell-like",
        "group": "epithelial",
        "genes": ["EPCAM", "KRT8", "KRT18", "KRT19", "CLDN4", "MUC1"],
    },
    "estimate_immune": {
        "panel": "ESTIMATE-like",
        "group": "immune",
        "genes": ["PTPRC", "CD53", "IL10RA", "HLA-DRA", "CD74", "LAPTM5", "LCP2"],
    },
    "estimate_stromal": {
        "panel": "ESTIMATE-like",
        "group": "stromal",
        "genes": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "FAP", "ACTA2", "TAGLN", "VIM", "FN1"],
    },
    "lm22_t_cells": {
        "panel": "LM22-like",
        "group": "lymphoid",
        "genes": ["CD3D", "CD3E", "TRAC", "CD2", "CD247"],
    },
    "lm22_b_cells": {
        "panel": "LM22-like",
        "group": "lymphoid",
        "genes": ["MS4A1", "CD79A", "CD79B", "CD19"],
    },
    "lm22_plasma_cells": {
        "panel": "LM22-like",
        "group": "lymphoid",
        "genes": ["MZB1", "JCHAIN", "XBP1", "IGKC", "IGHG1"],
    },
    "lm22_macrophages": {
        "panel": "LM22-like",
        "group": "myeloid",
        "genes": ["CD68", "CD163", "MSR1", "MRC1", "C1QA", "C1QB"],
    },
    "lm22_dendritic_cells": {
        "panel": "LM22-like",
        "group": "myeloid",
        "genes": ["ITGAX", "FCER1A", "CLEC10A", "CD1C", "LILRA4"],
    },
    "lm22_mast_cells": {
        "panel": "LM22-like",
        "group": "myeloid",
        "genes": ["KIT", "TPSAB1", "TPSB2", "CPA3", "MS4A2"],
    },
    "lm22_neutrophils": {
        "panel": "LM22-like",
        "group": "myeloid",
        "genes": ["S100A8", "S100A9", "FCGR3B", "CSF3R", "CXCR2"],
    },
    "proliferation": {
        "panel": "cell-cycle",
        "group": "proliferation",
        "genes": ["MKI67", "TOP2A", "PCNA", "UBE2C", "BIRC5", "CCNB1", "CDK1", "AURKB"],
    },
}

COMPOSITES: Dict[str, List[str]] = {
    "hematopoietic_composite": [
        "xcell_immune",
        "estimate_immune",
        "mcp_t_cells",
        "mcp_b_lineage",
        "mcp_monocytic_lineage",
        "lm22_t_cells",
        "lm22_b_cells",
        "lm22_macrophages",
    ],
    "lymphoid_composite": [
        "mcp_t_cells",
        "mcp_cd8_t_cells",
        "mcp_cytotoxic_lymphocytes",
        "mcp_b_lineage",
        "mcp_nk_cells",
        "epic_cd4_t_cells",
        "epic_cd8_t_cells",
        "epic_b_cells",
        "epic_nk_cells",
        "lm22_t_cells",
        "lm22_b_cells",
        "lm22_plasma_cells",
    ],
    "myeloid_composite": [
        "mcp_monocytic_lineage",
        "mcp_myeloid_dendritic",
        "mcp_neutrophils",
        "epic_macrophages",
        "xcell_myeloid",
        "lm22_macrophages",
        "lm22_dendritic_cells",
        "lm22_mast_cells",
        "lm22_neutrophils",
    ],
    "stromal_composite": ["mcp_fibroblasts", "epic_caf", "xcell_stromal", "estimate_stromal"],
    "endothelial_composite": ["mcp_endothelial_cells", "epic_endothelial", "xcell_endothelial"],
    "epithelial_composite": ["epic_epithelial_tumor", "xcell_epithelial"],
    "proliferation_composite": ["proliferation"],
    "estimate_context_composite": ["estimate_immune", "estimate_stromal"],
}

DISPLAY_COMPOSITES = [
    "hematopoietic_composite_z",
    "lymphoid_composite_z",
    "myeloid_composite_z",
    "stromal_composite_z",
    "endothelial_composite_z",
    "epithelial_composite_z",
    "proliferation_composite_z",
    "estimate_context_composite_z",
    "epithelial_minus_context_z",
]

PRIMARY_STATES = [
    "hematologic_override",
    "epithelial_override",
    "generic_context_override",
    "clean_anchor_override",
    "stable_consensus",
    "unsupported_semantics",
]


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


def all_signature_genes() -> List[str]:
    genes = set()
    for spec in SIGNATURES.values():
        genes.update(str(g).upper() for g in spec["genes"])
    return sorted(genes)


def load_pool_expression(pool: str, details_path: Path, input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    details = pd.read_csv(details_path)
    genes = all_signature_genes()
    expr_rows = []
    meta_rows = []
    for rec in details.itertuples(index=False):
        file_name = str(getattr(rec, "file"))
        values = transform_external_values(parse_gene_value_file(input_dir / file_name))
        expr_rows.append(values.reindex(genes).astype(float))
        meta_rows.append(
            {
                "pool": pool,
                "file": file_name,
                "project": str(getattr(rec, "project")),
                "expected_site_family": str(getattr(rec, "expected_site_family")),
                "expected_disease_family": str(getattr(rec, "expected_disease_family")),
                "semantic_state_family": str(getattr(rec, "semantic_state_family")),
                "semantic_state_subprofile": str(getattr(rec, "semantic_state_subprofile")),
                "semantic_disease_semantic_status": str(getattr(rec, "semantic_disease_semantic_status")),
                "anchor_context_state": str(getattr(rec, "anchor_context_state")),
            }
        )
    return pd.DataFrame(expr_rows).reset_index(drop=True), pd.DataFrame(meta_rows)


def zscore_columns(df: pd.DataFrame, cols: Iterable[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    target_cols = list(cols) if cols is not None else list(out.columns)
    for col in target_cols:
        vals = pd.to_numeric(out[col], errors="coerce")
        mean = vals.mean(skipna=True)
        std = vals.std(skipna=True)
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        out[col] = (vals - mean) / std
    return out


def score_signatures(expr: pd.DataFrame) -> pd.DataFrame:
    expr_z = zscore_columns(expr)
    rows = {}
    for name, spec in SIGNATURES.items():
        genes = [str(g).upper() for g in spec["genes"]]
        present = [g for g in genes if g in expr_z.columns]
        rows[f"{name}_score"] = expr_z[present].mean(axis=1, skipna=True) if present else np.nan
        rows[f"{name}_n_present"] = len(present)
    return pd.DataFrame(rows)


def add_score_zscores(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    score_cols = [f"{name}_score" for name in SIGNATURES]
    score_z = zscore_columns(out[score_cols])
    for col in score_cols:
        out[col.replace("_score", "_z")] = score_z[col]
    return out


def add_composites(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for comp, members in COMPOSITES.items():
        member_cols = [f"{m}_z" for m in members if f"{m}_z" in out.columns]
        out[f"{comp}_z"] = out[member_cols].mean(axis=1, skipna=True) if member_cols else np.nan
    out["epithelial_minus_context_z"] = out["epithelial_composite_z"] - out[
        ["hematopoietic_composite_z", "stromal_composite_z", "endothelial_composite_z"]
    ].mean(axis=1, skipna=True)
    return out


def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    score_cols = [c for c in df.columns if c.endswith("_z") and not c.endswith("_n_present")]
    by_state = df.groupby(["pool", "semantic_state_family"], dropna=False)[score_cols].mean().reset_index()
    counts = df.groupby(["pool", "semantic_state_family"], dropna=False).size().reset_index(name="n")
    by_state = by_state.merge(counts, on=["pool", "semantic_state_family"], how="left")
    by_status = df.groupby(["pool", "semantic_disease_semantic_status"], dropna=False)[score_cols].mean().reset_index()
    status_counts = df.groupby(["pool", "semantic_disease_semantic_status"], dropna=False).size().reset_index(name="n")
    by_status = by_status.merge(status_counts, on=["pool", "semantic_disease_semantic_status"], how="left")
    return by_state, by_status


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    p = pvals.astype(float).to_numpy()
    out = np.full(len(p), np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return pd.Series(out, index=pvals.index)
    valid_idx = np.where(valid)[0]
    order = valid_idx[np.argsort(p[valid])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.clip(ranked, 0, 1)
    return pd.Series(out, index=pvals.index)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    if not np.isfinite(pooled) or pooled < 1e-9:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled)


def state_vs_rest_effects(df: pd.DataFrame) -> pd.DataFrame:
    cols = DISPLAY_COMPOSITES + [f"{name}_z" for name in SIGNATURES]
    rows = []
    for (pool, state), sub in df.groupby(["pool", "semantic_state_family"], dropna=False):
        rest = df.loc[(df["pool"] == pool) & (df["semantic_state_family"] != state)]
        for col in cols:
            a = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(rest[col], errors="coerce").to_numpy(dtype=float)
            a_clean = a[np.isfinite(a)]
            b_clean = b[np.isfinite(b)]
            if len(a_clean) >= 3 and len(b_clean) >= 3:
                try:
                    pval = float(mannwhitneyu(a_clean, b_clean, alternative="two-sided").pvalue)
                except ValueError:
                    pval = np.nan
            else:
                pval = np.nan
            rows.append(
                {
                    "pool": pool,
                    "semantic_state_family": state,
                    "score": col,
                    "n_state": len(a_clean),
                    "n_rest": len(b_clean),
                    "state_mean": float(np.mean(a_clean)) if len(a_clean) else np.nan,
                    "rest_mean": float(np.mean(b_clean)) if len(b_clean) else np.nan,
                    "mean_diff": float(np.mean(a_clean) - np.mean(b_clean)) if len(a_clean) and len(b_clean) else np.nan,
                    "cohen_d": cohen_d(a_clean, b_clean),
                    "mannwhitney_p": pval,
                }
            )
    effects = pd.DataFrame(rows)
    effects["fdr_bh"] = np.nan
    for pool, idx in effects.groupby("pool").groups.items():
        effects.loc[idx, "fdr_bh"] = benjamini_hochberg(effects.loc[idx, "mannwhitney_p"])
    return effects


def color_for_value(v: float) -> str:
    if not np.isfinite(v):
        return "#f5f5f5"
    v = max(-1.6, min(1.6, float(v))) / 1.6
    if v >= 0:
        r = 255
        g = int(246 - 122 * v)
        b = int(230 - 164 * v)
    else:
        v = abs(v)
        r = int(225 - 146 * v)
        g = int(238 - 94 * v)
        b = 255
    return f"#{r:02x}{g:02x}{b:02x}"


def write_composite_heatmap(summary: pd.DataFrame, pool: str, path: Path) -> None:
    sub = summary.loc[summary["pool"] == pool].copy().sort_values("n", ascending=False)
    labels = [c.replace("_composite_z", "").replace("_z", "").replace("_", " ") for c in DISPLAY_COMPOSITES]
    cell_w, cell_h = 108, 28
    left, top = 195, 82
    width = left + cell_w * len(DISPLAY_COMPOSITES) + 36
    height = top + cell_h * len(sub) + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">{html.escape(pool)} cell-composition signature validation</text>',
        '<text x="24" y="56" font-family="Arial" font-size="12" fill="#666">Values are state means of pool-normalized signature composites; red is higher, blue is lower.</text>',
    ]
    for j, label in enumerate(labels):
        x = left + j * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x}" y="{top - 9}" font-family="Arial" font-size="10" text-anchor="end" '
            f'transform="rotate(-45 {x} {top - 9})">{html.escape(label)}</text>'
        )
    for i, row in enumerate(sub.itertuples(index=False)):
        y = top + i * cell_h
        state = str(row.semantic_state_family)
        parts.append(f'<text x="18" y="{y + 18}" font-family="Arial" font-size="11">{html.escape(state)} (n={int(row.n)})</text>')
        for j, col in enumerate(DISPLAY_COMPOSITES):
            val = float(getattr(row, col))
            x = left + j * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{color_for_value(val)}" stroke="#fff"/>')
            parts.append(f'<text x="{x + cell_w / 2}" y="{y + 18}" font-family="Arial" font-size="10" text-anchor="middle" fill="#222">{val:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_effect_heatmap(effects: pd.DataFrame, pool: str, path: Path) -> None:
    sub = effects.loc[
        (effects["pool"] == pool)
        & (effects["semantic_state_family"].isin(PRIMARY_STATES))
        & (effects["score"].isin(DISPLAY_COMPOSITES))
    ].copy()
    states = [s for s in PRIMARY_STATES if s in set(sub["semantic_state_family"])]
    labels = [c.replace("_composite_z", "").replace("_z", "").replace("_", " ") for c in DISPLAY_COMPOSITES]
    cell_w, cell_h = 108, 28
    left, top = 195, 82
    width = left + cell_w * len(DISPLAY_COMPOSITES) + 36
    height = top + cell_h * len(states) + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">{html.escape(pool)} state-vs-rest effect sizes</text>',
        '<text x="24" y="56" font-family="Arial" font-size="12" fill="#666">Cells show Cohen d for each semantic state against all other states in the same pool.</text>',
    ]
    for j, label in enumerate(labels):
        x = left + j * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x}" y="{top - 9}" font-family="Arial" font-size="10" text-anchor="end" '
            f'transform="rotate(-45 {x} {top - 9})">{html.escape(label)}</text>'
        )
    for i, state in enumerate(states):
        y = top + i * cell_h
        parts.append(f'<text x="18" y="{y + 18}" font-family="Arial" font-size="11">{html.escape(state)}</text>')
        for j, col in enumerate(DISPLAY_COMPOSITES):
            row = sub.loc[(sub["semantic_state_family"] == state) & (sub["score"] == col)]
            val = float(row.iloc[0]["cohen_d"]) if not row.empty else np.nan
            fdr = float(row.iloc[0]["fdr_bh"]) if not row.empty else np.nan
            x = left + j * cell_w
            star = "*" if np.isfinite(fdr) and fdr < 0.05 else ""
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{color_for_value(val)}" stroke="#fff"/>')
            parts.append(f'<text x="{x + cell_w / 2}" y="{y + 18}" font-family="Arial" font-size="10" text-anchor="middle" fill="#222">{val:.2f}{star}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def top_composites(row: pd.Series, n: int = 4) -> Dict[str, float]:
    vals = row[DISPLAY_COMPOSITES].sort_values(ascending=False).head(n)
    return {str(k).replace("_composite_z", "").replace("_z", ""): float(v) for k, v in vals.items()}


def write_summary(df: pd.DataFrame, by_state: pd.DataFrame, effects: pd.DataFrame) -> None:
    lines = [
        "# Cell-composition signature validation",
        "",
        "## Purpose",
        "",
        "This analysis adds a deconvolution-style biological check to the marker and pathway analyses. It computes compact cell-composition signatures directly from the external expression profiles and then asks whether semantic state families are enriched for expected immune, stromal, endothelial, epithelial, or proliferative context.",
        "",
        "Important limitation: these are MCP-counter-like, xCell-like, EPIC-like, ESTIMATE-like, and LM22-like transparent signature panels, not official software outputs from MCP-counter, xCell, EPIC, ESTIMATE, or CIBERSORTx. They should be interpreted as reproducible signature summaries rather than official deconvolution outputs.",
        "",
        "## Data",
        "",
    ]
    for pool, sub in df.groupby("pool"):
        lines.append(f"- {pool}: `{len(sub)}` samples / `{sub['project'].nunique()}` projects")
    lines.extend(["", "## Directional checks", ""])
    for pool in sorted(df["pool"].unique()):
        lines.extend([f"### {pool}", ""])
        for state in PRIMARY_STATES:
            row = by_state.loc[(by_state["pool"] == pool) & (by_state["semantic_state_family"] == state)]
            if row.empty:
                continue
            r = row.iloc[0]
            state_effects = effects.loc[
                (effects["pool"] == pool)
                & (effects["semantic_state_family"] == state)
                & (effects["score"].isin(DISPLAY_COMPOSITES))
            ]
            effect_map = {
                str(rec.score).replace("_composite_z", "").replace("_z", ""): float(rec.cohen_d)
                for rec in state_effects.sort_values("cohen_d", ascending=False).head(3).itertuples(index=False)
            }
            lines.append(
                f"- `{state}` n={int(r['n'])}: mean composites `{top_composites(r)}`, top state-vs-rest effects `{effect_map}`"
            )
        lines.append("")
    lines.extend(["## Primary interpretation", ""])
    for pool in sorted(df["pool"].unique()):
        lines.append(f"### {pool}")
        for state in ["hematologic_override", "generic_context_override", "clean_anchor_override", "epithelial_override"]:
            row = by_state.loc[(by_state["pool"] == pool) & (by_state["semantic_state_family"] == state)]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(
                "- "
                + f"`{state}`: hematopoietic `{float(r['hematopoietic_composite_z']):.3f}`, "
                + f"lymphoid `{float(r['lymphoid_composite_z']):.3f}`, myeloid `{float(r['myeloid_composite_z']):.3f}`, "
                + f"stromal `{float(r['stromal_composite_z']):.3f}`, endothelial `{float(r['endothelial_composite_z']):.3f}`, "
                + f"epithelial `{float(r['epithelial_composite_z']):.3f}`, proliferation `{float(r['proliferation_composite_z']):.3f}`"
            )
        lines.append("")
    def avg_state(state: str, metrics: List[str]) -> float:
        row = by_state.loc[by_state["semantic_state_family"] == state]
        if row.empty:
            return np.nan
        return float(np.nanmean(row[metrics].to_numpy(dtype=float)))

    hematologic_immune = avg_state(
        "hematologic_override",
        ["hematopoietic_composite_z", "lymphoid_composite_z", "myeloid_composite_z"],
    )
    hematologic_context = avg_state(
        "hematologic_override",
        ["stromal_composite_z", "endothelial_composite_z", "epithelial_composite_z"],
    )
    generic_context = avg_state(
        "generic_context_override",
        ["stromal_composite_z", "endothelial_composite_z", "epithelial_composite_z", "proliferation_composite_z"],
    )
    generic_context_immune_mean = avg_state(
        "generic_context_override",
        ["hematopoietic_composite_z", "lymphoid_composite_z", "myeloid_composite_z"],
    )
    clean_context = avg_state(
        "clean_anchor_override",
        ["stromal_composite_z", "endothelial_composite_z", "epithelial_composite_z", "proliferation_composite_z"],
    )
    epithelial_immune = avg_state(
        "epithelial_override",
        ["hematopoietic_composite_z", "lymphoid_composite_z", "myeloid_composite_z"],
    )
    epithelial_epithelial = avg_state("epithelial_override", ["epithelial_composite_z"])
    lines.extend(
        [
            "## Directional summary",
            "",
            f"- `hematologic_override` is supported as a blood/immune direction: cross-pool immune composite mean `{hematologic_immune:.3f}` versus context composite mean `{hematologic_context:.3f}`.",
            f"- `generic_context_override` is not supported as immune-only: cross-pool stromal/endothelial/epithelial/proliferation context mean `{generic_context:.3f}` versus immune mean `{generic_context_immune_mean:.3f}`.",
            f"- `clean_anchor_override` is the strongest clean non-immune context state in this pre-screen: cross-pool context mean `{clean_context:.3f}`.",
            f"- `epithelial_override` should not be described as purely epithelial from this evidence alone: cross-pool immune mean `{epithelial_immune:.3f}` and epithelial mean `{epithelial_epithelial:.3f}`. It may need to be split or renamed after larger validation.",
            "",
        ]
    )
    lines.extend(
        [
            "## Output files",
            "",
            "- `t4c_cell_composition_scores_by_sample.csv`",
            "- `t4c_cell_composition_scores_by_state.csv`",
            "- `t4c_cell_composition_scores_by_disease_status.csv`",
            "- `t4c_state_vs_rest_effects.csv`",
            "- `external_180_cell_composition_heatmap.svg` and `multisource_450_cell_composition_heatmap.svg`",
            "- `external_180_state_vs_rest_effect_heatmap.svg` and `multisource_450_state_vs_rest_effect_heatmap.svg`",
            "",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for pool, paths in POOLS.items():
        expr, meta = load_pool_expression(pool, paths["details"], paths["inputs"])
        scores = add_composites(add_score_zscores(score_signatures(expr)))
        frames.append(pd.concat([meta.reset_index(drop=True), scores.reset_index(drop=True)], axis=1))
    df = pd.concat(frames, ignore_index=True)
    by_state, by_status = summarize(df)
    effects = state_vs_rest_effects(df)

    df.to_csv(OUTDIR / "t4c_cell_composition_scores_by_sample.csv", index=False)
    by_state.to_csv(OUTDIR / "t4c_cell_composition_scores_by_state.csv", index=False)
    by_status.to_csv(OUTDIR / "t4c_cell_composition_scores_by_disease_status.csv", index=False)
    effects.to_csv(OUTDIR / "t4c_state_vs_rest_effects.csv", index=False)

    write_composite_heatmap(by_state, "External-180", OUTDIR / "external_180_cell_composition_heatmap.svg")
    write_composite_heatmap(by_state, "MultiSource-450", OUTDIR / "multisource_450_cell_composition_heatmap.svg")
    write_effect_heatmap(effects, "External-180", OUTDIR / "external_180_state_vs_rest_effect_heatmap.svg")
    write_effect_heatmap(effects, "MultiSource-450", OUTDIR / "multisource_450_state_vs_rest_effect_heatmap.svg")
    write_summary(df, by_state, effects)


if __name__ == "__main__":
    main()
