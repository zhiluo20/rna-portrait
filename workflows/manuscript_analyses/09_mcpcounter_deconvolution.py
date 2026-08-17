#!/usr/bin/env python3
"""Official MCP-counter validation of semantic state families."""

from __future__ import annotations

import hashlib
import html
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

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
EPIC_DIR = SUPP_DIR / "T4d_official_EPIC_deconvolution"
OUTDIR = SUPP_DIR / "T4e_official_MCPcounter_deconvolution"
R_SCRIPT = OUTDIR / "run_mcpcounter_deconvolution.R"
MCP_COUNTER_REVISION = "b6eac73e91c246fcff0bb1a5c68a816cd588fc48"
SIGNATURE_FILES = {
    "mcpcounter_signatures_genes.txt": (
        f"https://raw.githubusercontent.com/ebecht/MCPcounter/{MCP_COUNTER_REVISION}/Signatures/genes.txt",
        "408f6c5d02c8f9bd2f1599c598367881853520934f8ce398a4886a8b296922bb",
    ),
    "mcpcounter_signatures_probesets.txt": (
        f"https://raw.githubusercontent.com/ebecht/MCPcounter/{MCP_COUNTER_REVISION}/Signatures/probesets.txt",
        "68778e6632fefac944127dba8d1ae1e8b8889ba36c71500776f8a7f00098c573",
    ),
}

POOLS = ["External-180", "MultiSource-450"]
POOL_SLUGS = {
    "External-180": "external_180",
    "MultiSource-450": "multisource_450",
}

PRIMARY_STATES = [
    "hematologic_override",
    "epithelial_override",
    "generic_context_override",
    "clean_anchor_override",
    "stable_consensus",
    "unsupported_semantics",
]

MCP_POPULATIONS = [
    "T cells",
    "Cytotoxic lymphocytes",
    "B lineage",
    "NK cells",
    "Monocytic lineage",
    "Myeloid dendritic cells",
    "Neutrophils",
    "Endothelial cells",
    "Fibroblasts",
]

POPULATION_COLUMNS = {
    "T cells": "mcp_t_cells",
    "Cytotoxic lymphocytes": "mcp_cytotoxic_lymphocytes",
    "B lineage": "mcp_b_lineage",
    "NK cells": "mcp_nk_cells",
    "Monocytic lineage": "mcp_monocytic_lineage",
    "Myeloid dendritic cells": "mcp_myeloid_dendritic_cells",
    "Neutrophils": "mcp_neutrophils",
    "Endothelial cells": "mcp_endothelial_cells",
    "Fibroblasts": "mcp_fibroblasts",
}

RAW_SCORE_COLS = list(POPULATION_COLUMNS.values())

COMPOSITE_COLS = [
    "mcp_immune_z_mean",
    "mcp_t_nk_cytotoxic_z_mean",
    "mcp_myeloid_z_mean",
    "mcp_stromal_z_mean",
    "mcp_endothelial_z",
    "mcp_fibroblast_z",
]

DISPLAY_METRICS = COMPOSITE_COLS
DISPLAY_LABELS = {
    "mcp_immune_z_mean": "immune",
    "mcp_t_nk_cytotoxic_z_mean": "T/NK/cytotoxic",
    "mcp_myeloid_z_mean": "myeloid",
    "mcp_stromal_z_mean": "stromal",
    "mcp_endothelial_z": "endothelial",
    "mcp_fibroblast_z": "fibroblast",
}


def ensure_signature_file(path: Path, url: str, expected_sha256: str) -> None:
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest == expected_sha256:
            return
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(
            f"Checksum mismatch for {url}: expected {expected_sha256}, got {digest}"
        )
    path.write_bytes(payload)


def write_r_script() -> None:
    genes_path = OUTDIR / "mcpcounter_signatures_genes.txt"
    probesets_path = OUTDIR / "mcpcounter_signatures_probesets.txt"
    for filename, (url, sha256) in SIGNATURE_FILES.items():
        ensure_signature_file(OUTDIR / filename, url, sha256)
    script = f"""
lib <- normalizePath("{R_LIB}", mustWork=TRUE)
.libPaths(c(lib, .libPaths()))
if (!requireNamespace("MCPcounter", quietly=TRUE)) {{
  stop("MCPcounter is not installed in project R library: ", lib)
}}
library(MCPcounter)
genes_path <- "{genes_path}"
probesets_path <- "{probesets_path}"
genes <- read.table(genes_path, sep="\\t", stringsAsFactors=FALSE, header=TRUE, colClasses="character", check.names=FALSE)
probesets <- read.table(probesets_path, sep="\\t", stringsAsFactors=FALSE, colClasses="character")
run_one <- function(pool_slug) {{
  matrix_path <- file.path("{EPIC_DIR}", paste0(pool_slug, "_cpm_matrix.tsv"))
  output_path <- file.path("{OUTDIR}", paste0(pool_slug, "_mcpcounter_scores.tsv"))
  mat <- as.matrix(read.table(matrix_path, header=TRUE, sep="\\t", row.names=1, check.names=FALSE))
  storage.mode(mat) <- "numeric"
  res <- MCPcounter::MCPcounter.estimate(
    mat,
    featuresType="HUGO_symbols",
    genes=genes,
    probesets=probesets
  )
  write.table(res, output_path, sep="\\t", quote=FALSE, col.names=NA)
}}
run_one("external_180")
run_one("multisource_450")
""".strip()
    R_SCRIPT.write_text(script + "\n", encoding="utf-8")


def run_mcpcounter() -> None:
    write_r_script()
    env = {
        **dict(),
    }
    log_path = OUTDIR / "mcpcounter_run.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            ["Rscript", str(R_SCRIPT)],
            cwd=str(RUN_CWD),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env=None if not env else env,
        )


def read_mcpcounter_scores(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", index_col=0)
    raw.index.name = "population"
    raw = raw.reindex(MCP_POPULATIONS)
    raw = raw.rename(index=POPULATION_COLUMNS)
    df = raw.T.reset_index().rename(columns={"index": "sample_id"})
    for col in RAW_SCORE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def zscore_by_pool(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[f"{col}_z"] = np.nan
        for _, idx in out.groupby("pool").groups.items():
            vals = pd.to_numeric(out.loc[idx, col], errors="coerce")
            mean = vals.mean(skipna=True)
            std = vals.std(skipna=True)
            if not np.isfinite(std) or std < 1e-9:
                std = 1.0
            out.loc[idx, f"{col}_z"] = (vals - mean) / std
    return out


def add_composites(df: pd.DataFrame) -> pd.DataFrame:
    out = zscore_by_pool(df, RAW_SCORE_COLS)
    z = {col: f"{col}_z" for col in RAW_SCORE_COLS}
    out["mcp_immune_z_mean"] = out[
        [
            z["mcp_t_cells"],
            z["mcp_cytotoxic_lymphocytes"],
            z["mcp_b_lineage"],
            z["mcp_nk_cells"],
            z["mcp_monocytic_lineage"],
            z["mcp_myeloid_dendritic_cells"],
            z["mcp_neutrophils"],
        ]
    ].mean(axis=1)
    out["mcp_t_nk_cytotoxic_z_mean"] = out[
        [z["mcp_t_cells"], z["mcp_cytotoxic_lymphocytes"], z["mcp_nk_cells"]]
    ].mean(axis=1)
    out["mcp_myeloid_z_mean"] = out[
        [z["mcp_monocytic_lineage"], z["mcp_myeloid_dendritic_cells"], z["mcp_neutrophils"]]
    ].mean(axis=1)
    out["mcp_stromal_z_mean"] = out[[z["mcp_endothelial_cells"], z["mcp_fibroblasts"]]].mean(axis=1)
    out["mcp_endothelial_z"] = out[z["mcp_endothelial_cells"]]
    out["mcp_fibroblast_z"] = out[z["mcp_fibroblasts"]]
    return out


def collect_results() -> pd.DataFrame:
    frames = []
    for pool in POOLS:
        slug = POOL_SLUGS[pool]
        meta = pd.read_csv(EPIC_DIR / f"{slug}_metadata.csv")
        scores = read_mcpcounter_scores(OUTDIR / f"{slug}_mcpcounter_scores.tsv")
        frames.append(meta.merge(scores, on="sample_id", how="left"))
    return add_composites(pd.concat(frames, ignore_index=True))


def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = RAW_SCORE_COLS + [f"{c}_z" for c in RAW_SCORE_COLS] + COMPOSITE_COLS
    by_state = df.groupby(["pool", "semantic_state_family"], dropna=False)[cols].mean().reset_index()
    counts = df.groupby(["pool", "semantic_state_family"], dropna=False).size().reset_index(name="n")
    by_state = by_state.merge(counts, on=["pool", "semantic_state_family"], how="left")
    by_status = df.groupby(["pool", "semantic_disease_semantic_status"], dropna=False)[cols].mean().reset_index()
    status_counts = df.groupby(["pool", "semantic_disease_semantic_status"], dropna=False).size().reset_index(name="n")
    by_status = by_status.merge(status_counts, on=["pool", "semantic_disease_semantic_status"], how="left")
    return by_state, by_status


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    if not np.isfinite(pooled) or pooled < 1e-12:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled)


def effects_state_vs_rest(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = [f"{c}_z" for c in RAW_SCORE_COLS] + DISPLAY_METRICS
    for (pool, state), sub in df.groupby(["pool", "semantic_state_family"], dropna=False):
        rest = df.loc[(df["pool"] == pool) & (df["semantic_state_family"] != state)]
        for col in cols:
            a = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(rest[col], errors="coerce").to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            pval = np.nan
            if len(a) >= 3 and len(b) >= 3:
                pval = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
            rows.append(
                {
                    "pool": pool,
                    "semantic_state_family": state,
                    "score": col,
                    "n_state": len(a),
                    "n_rest": len(b),
                    "state_mean": float(np.mean(a)) if len(a) else np.nan,
                    "rest_mean": float(np.mean(b)) if len(b) else np.nan,
                    "mean_diff": float(np.mean(a) - np.mean(b)) if len(a) and len(b) else np.nan,
                    "cohen_d": cohen_d(a, b),
                    "mannwhitney_p": pval,
                }
            )
    return pd.DataFrame(rows)


def color_for_value(v: float) -> str:
    if not np.isfinite(v):
        return "#f5f5f5"
    v = max(-1.8, min(1.8, float(v))) / 1.8
    if v >= 0:
        r = 255
        g = int(246 - 126 * v)
        b = int(230 - 168 * v)
    else:
        v = abs(v)
        r = int(225 - 146 * v)
        g = int(238 - 94 * v)
        b = 255
    return f"#{r:02x}{g:02x}{b:02x}"


def write_heatmap(summary: pd.DataFrame, pool: str, path: Path) -> None:
    sub = summary.loc[summary["pool"] == pool].copy().sort_values("n", ascending=False)
    labels = [DISPLAY_LABELS[c] for c in DISPLAY_METRICS]
    cell_w, cell_h = 112, 28
    left, top = 195, 82
    width = left + cell_w * len(DISPLAY_METRICS) + 36
    height = top + cell_h * len(sub) + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">{html.escape(pool)} official MCP-counter by semantic state</text>',
        '<text x="24" y="56" font-family="Arial" font-size="12" fill="#666">Values are pool-z-scored MCP-counter cell-population composites; red is higher, blue is lower.</text>',
    ]
    for j, label in enumerate(labels):
        x = left + j * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x}" y="{top - 9}" font-family="Arial" font-size="10" text-anchor="end" transform="rotate(-45 {x} {top - 9})">{html.escape(label)}</text>'
        )
    for i, row in enumerate(sub.itertuples(index=False)):
        y = top + i * cell_h
        state = str(row.semantic_state_family)
        parts.append(f'<text x="18" y="{y + 18}" font-family="Arial" font-size="11">{html.escape(state)} (n={int(row.n)})</text>')
        for j, col in enumerate(DISPLAY_METRICS):
            val = float(getattr(row, col))
            x = left + j * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{color_for_value(val)}" stroke="#fff"/>')
            parts.append(f'<text x="{x + cell_w / 2}" y="{y + 18}" font-family="Arial" font-size="10" text-anchor="middle" fill="#222">{val:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def mean_state(by_state: pd.DataFrame, state: str, metrics: List[str]) -> float:
    row = by_state.loc[by_state["semantic_state_family"] == state]
    if row.empty:
        return np.nan
    return float(np.nanmean(row[metrics].to_numpy(dtype=float)))


def top_metrics(row: pd.Series) -> Dict[str, float]:
    vals = row[DISPLAY_METRICS].sort_values(ascending=False).head(4)
    return {DISPLAY_LABELS[str(k)]: float(v) for k, v in vals.items()}


def write_summary(df: pd.DataFrame, by_state: pd.DataFrame) -> None:
    lines = [
        "# MCP-counter deconvolution validation",
        "",
        "## Purpose",
        "",
        "This analysis runs the MCP-counter R package on the same CPM-normalized external bulk RNA matrices used for EPIC deconvolution. MCP-counter returns abundance scores rather than cell fractions, so all population scores are z-scored within each validation pool before state-level comparisons.",
        "",
        "## Environment",
        "",
        "- R package: `MCPcounter 1.2.0` from `ebecht/MCPcounter`, `Source` subdirectory",
        f"- signature tables: upstream revision `{MCP_COUNTER_REVISION}`, verified by SHA-256",
        f"- local R library: `{R_LIB}`",
        "- input normalization: per-sample CPM from raw/count-like validation profiles",
        "- feature type: `HUGO_symbols`",
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
            lines.append(
                f"- `{state}` n={int(r['n'])}: top z-scored MCP-counter composites `{top_metrics(r)}`, "
                f"immune `{float(r['mcp_immune_z_mean']):.3f}`, stromal `{float(r['mcp_stromal_z_mean']):.3f}`"
            )
        lines.append("")
    hematologic_immune = mean_state(by_state, "hematologic_override", ["mcp_immune_z_mean", "mcp_t_nk_cytotoxic_z_mean"])
    generic_context = mean_state(by_state, "generic_context_override", ["mcp_stromal_z_mean", "mcp_endothelial_z", "mcp_fibroblast_z"])
    clean_context = mean_state(by_state, "clean_anchor_override", ["mcp_stromal_z_mean", "mcp_endothelial_z", "mcp_fibroblast_z"])
    epithelial_immune = mean_state(by_state, "epithelial_override", ["mcp_immune_z_mean", "mcp_t_nk_cytotoxic_z_mean"])
    epithelial_context = mean_state(by_state, "epithelial_override", ["mcp_stromal_z_mean", "mcp_endothelial_z", "mcp_fibroblast_z"])
    lines.extend(
        [
            "## Directional summary",
            "",
            f"- `hematologic_override`: official MCP-counter immune/cytotoxic mean z `{hematologic_immune:.3f}`.",
            f"- `generic_context_override`: official MCP-counter stromal/endothelial/fibroblast context mean z `{generic_context:.3f}`.",
            f"- `clean_anchor_override`: official MCP-counter stromal/endothelial/fibroblast context mean z `{clean_context:.3f}`.",
            f"- `epithelial_override`: MCP-counter immune mean z `{epithelial_immune:.3f}` and context mean z `{epithelial_context:.3f}`; keep cautious/mixed interpretation.",
            "",
            "## Output files",
            "",
            "- `t4e_mcpcounter_scores_by_sample.csv`",
            "- `t4e_mcpcounter_scores_by_state.csv`",
            "- `t4e_mcpcounter_scores_by_disease_status.csv`",
            "- `t4e_mcpcounter_state_vs_rest_effects.csv`",
            "- `external_180_mcpcounter_state_heatmap.svg` and `multisource_450_mcpcounter_state_heatmap.svg`",
            "- raw MCP-counter outputs: `*_mcpcounter_scores.tsv`",
            "",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    missing = [EPIC_DIR / f"{slug}_cpm_matrix.tsv" for slug in POOL_SLUGS.values() if not (EPIC_DIR / f"{slug}_cpm_matrix.tsv").exists()]
    if missing:
        raise FileNotFoundError(f"Missing CPM matrices from the EPIC deconvolution setup: {missing}")
    run_mcpcounter()
    df = collect_results()
    by_state, by_status = summarize(df)
    effects = effects_state_vs_rest(df)
    df.to_csv(OUTDIR / "t4e_mcpcounter_scores_by_sample.csv", index=False)
    by_state.to_csv(OUTDIR / "t4e_mcpcounter_scores_by_state.csv", index=False)
    by_status.to_csv(OUTDIR / "t4e_mcpcounter_scores_by_disease_status.csv", index=False)
    effects.to_csv(OUTDIR / "t4e_mcpcounter_state_vs_rest_effects.csv", index=False)
    write_heatmap(by_state, "External-180", OUTDIR / "external_180_mcpcounter_state_heatmap.svg")
    write_heatmap(by_state, "MultiSource-450", OUTDIR / "multisource_450_mcpcounter_state_heatmap.svg")
    write_summary(df, by_state)


if __name__ == "__main__":
    main()
