#!/usr/bin/env python3
"""First-pass biological validation of semantic state families with marker programs."""

from __future__ import annotations

import html
import json
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
OUTDIR = SUPP_DIR / "T4_marker_program_state_validation"

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

SIGNATURES: Dict[str, List[str]] = {
    "immune_core": ["PTPRC", "B2M", "HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "CD74", "CXCL10"],
    "t_cell_nk": ["CD3D", "CD3E", "TRAC", "IL7R", "NKG7", "GNLY", "GZMB", "PRF1"],
    "myeloid_inflammation": ["LYZ", "LST1", "TYROBP", "FCGR3A", "S100A8", "S100A9", "C1QA", "C1QB"],
    "hematologic_lineage": ["PTPRC", "MS4A1", "CD79A", "CD74", "HLA-DRA", "NKG7", "LST1", "TYROBP"],
    "epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT17", "KRT5", "KRT14", "MSLN"],
    "stromal_ecm": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "FAP", "ACTA2", "TAGLN"],
    "proliferation": ["MKI67", "TOP2A", "PCNA", "UBE2C", "BIRC5", "CCNB1", "CDK1", "AURKB"],
    "interferon": ["ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1", "OAS1", "OAS2", "STAT1"],
    "liver_metabolic": ["ALB", "APOA1", "APOB", "TTR", "FGB", "CYP3A4", "APOC3", "HP"],
    "neural": ["SNAP25", "RBFOX3", "SYT1", "MAP2", "SLC17A7", "GAD1", "GFAP", "MBP"],
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


def signature_score(values: pd.Series, genes: List[str]) -> Tuple[float, int]:
    present = [g for g in genes if g in values.index]
    if not present:
        return np.nan, 0
    return float(values.loc[present].mean()), len(present)


def compute_pool(pool: str, details_path: Path, input_dir: Path) -> pd.DataFrame:
    details = pd.read_csv(details_path)
    rows = []
    for rec in details.itertuples(index=False):
        values = transform_external_values(parse_gene_value_file(input_dir / str(getattr(rec, "file"))))
        row = {
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
        for sig, genes in SIGNATURES.items():
            score, n_present = signature_score(values, genes)
            row[f"{sig}_score"] = score
            row[f"{sig}_n_present"] = n_present
        rows.append(row)
    return pd.DataFrame(rows)


def add_pool_zscores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for sig in SIGNATURES:
        col = f"{sig}_score"
        zcol = f"{sig}_z"
        out[zcol] = np.nan
        for pool, idx in out.groupby("pool").groups.items():
            vals = out.loc[idx, col].astype(float)
            mean = vals.mean(skipna=True)
            std = vals.std(skipna=True)
            if not np.isfinite(std) or std < 1e-6:
                std = 1.0
            out.loc[idx, zcol] = (vals - mean) / std
    return out


def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    zcols = [f"{sig}_z" for sig in SIGNATURES]
    by_state = (
        df.groupby(["pool", "semantic_state_family"], dropna=False)[zcols]
        .mean()
        .reset_index()
    )
    counts = df.groupby(["pool", "semantic_state_family"], dropna=False).size().reset_index(name="n")
    by_state = by_state.merge(counts, on=["pool", "semantic_state_family"], how="left")

    by_status = (
        df.groupby(["pool", "semantic_disease_semantic_status"], dropna=False)[zcols]
        .mean()
        .reset_index()
    )
    status_counts = df.groupby(["pool", "semantic_disease_semantic_status"], dropna=False).size().reset_index(name="n")
    by_status = by_status.merge(status_counts, on=["pool", "semantic_disease_semantic_status"], how="left")
    return by_state, by_status


def color_for_value(v: float) -> str:
    if not np.isfinite(v):
        return "#f5f5f5"
    v = max(-1.5, min(1.5, float(v))) / 1.5
    if v >= 0:
        r = 255
        g = int(245 - 120 * v)
        b = int(230 - 170 * v)
    else:
        v = abs(v)
        r = int(225 - 145 * v)
        g = int(238 - 95 * v)
        b = 255
    return f"#{r:02x}{g:02x}{b:02x}"


def write_heatmap(summary: pd.DataFrame, pool: str, path: Path) -> None:
    sub = summary.loc[summary["pool"] == pool].copy()
    sub = sub.sort_values("n", ascending=False)
    states = sub["semantic_state_family"].tolist()
    sigs = list(SIGNATURES.keys())
    cell_w, cell_h = 86, 28
    left, top = 190, 72
    width = left + cell_w * len(sigs) + 40
    height = top + cell_h * len(states) + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">{html.escape(pool)} marker-program validation by semantic state</text>',
        '<text x="24" y="56" font-family="Arial" font-size="12" fill="#666">Values are pool-z-scored mean marker-program expression; red is higher, blue is lower.</text>',
    ]
    for j, sig in enumerate(sigs):
        x = left + j * cell_w + cell_w / 2
        label = sig.replace("_", " ")
        parts.append(f'<text x="{x}" y="{top - 8}" font-family="Arial" font-size="10" text-anchor="end" transform="rotate(-45 {x} {top - 8})">{html.escape(label)}</text>')
    for i, row in enumerate(sub.itertuples(index=False)):
        y = top + i * cell_h
        state = str(row.semantic_state_family)
        n = int(row.n)
        parts.append(f'<text x="18" y="{y + 18}" font-family="Arial" font-size="11">{html.escape(state)} (n={n})</text>')
        for j, sig in enumerate(sigs):
            val = float(getattr(row, f"{sig}_z"))
            x = left + j * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{color_for_value(val)}" stroke="#fff"/>')
            parts.append(f'<text x="{x + cell_w / 2}" y="{y + 18}" font-family="Arial" font-size="10" text-anchor="middle" fill="#222">{val:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary(df: pd.DataFrame, by_state: pd.DataFrame) -> None:
    lines = [
        "# Marker-program validation of portrait states",
        "",
        "## Purpose",
        "",
        "This first-pass biological validation uses expression marker programs computed directly from the external input profiles. The model outputs are used only as grouping labels; the marker scores are independent post hoc expression summaries.",
        "",
        "## Data",
        "",
    ]
    for pool, sub in df.groupby("pool"):
        lines.append(f"- {pool}: `{len(sub)}` samples / `{sub['project'].nunique()}` projects")
    lines.extend(["", "## Directional checks", ""])

    for pool, sub in by_state.groupby("pool"):
        lines.extend([f"### {pool}", ""])
        for state in ["hematologic_override", "epithelial_override", "generic_context_override", "clean_anchor_override", "stable_consensus", "unsupported_semantics"]:
            row = sub.loc[sub["semantic_state_family"] == state]
            if row.empty:
                continue
            r = row.iloc[0]
            vals = {
                "immune_core": float(r["immune_core_z"]),
                "hematologic_lineage": float(r["hematologic_lineage_z"]),
                "epithelial": float(r["epithelial_z"]),
                "stromal_ecm": float(r["stromal_ecm_z"]),
                "proliferation": float(r["proliferation_z"]),
                "interferon": float(r["interferon_z"]),
            }
            lines.append(f"- `{state}` n={int(r['n'])}: `{vals}`")
        lines.append("")

    lines.extend(
        [
            "## Output files",
            "",
            "- `t4_marker_scores_by_sample.csv`: per-sample marker scores and z-scores.",
            "- `t4_marker_scores_by_state.csv`: state-family mean marker z-scores.",
            "- `t4_marker_scores_by_disease_status.csv`: disease-status mean marker z-scores.",
            "- `external_180_state_marker_heatmap.svg` and `multisource_450_state_marker_heatmap.svg`.",
            "",
            "## Interpretation",
            "",
            "This marker-program analysis is an independent expression-based check of whether portrait families show immune, hematologic, epithelial, stromal or proliferative structure. It complements the pathway and cell-composition analyses generated by the later workflow steps.",
            "",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for pool, paths in POOLS.items():
        frames.append(compute_pool(pool, paths["details"], paths["inputs"]))
    df = add_pool_zscores(pd.concat(frames, ignore_index=True))
    by_state, by_status = summarize(df)
    df.to_csv(OUTDIR / "t4_marker_scores_by_sample.csv", index=False)
    by_state.to_csv(OUTDIR / "t4_marker_scores_by_state.csv", index=False)
    by_status.to_csv(OUTDIR / "t4_marker_scores_by_disease_status.csv", index=False)
    write_heatmap(by_state, "External-180", OUTDIR / "external_180_state_marker_heatmap.svg")
    write_heatmap(by_state, "MultiSource-450", OUTDIR / "multisource_450_state_marker_heatmap.svg")
    write_summary(df, by_state)


if __name__ == "__main__":
    main()
