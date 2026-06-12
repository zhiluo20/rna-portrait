#!/usr/bin/env python3
"""Official EPIC deconvolution validation of semantic state families."""

from __future__ import annotations

import html
import subprocess
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
OUTDIR = SUPP_DIR / "T4d_official_EPIC_deconvolution"
R_SCRIPT = OUTDIR / "run_epic_deconvolution.R"

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

PRIMARY_STATES = [
    "hematologic_override",
    "epithelial_override",
    "generic_context_override",
    "clean_anchor_override",
    "stable_consensus",
    "unsupported_semantics",
]

EPIC_FRACTIONS = [
    "Bcells",
    "CD4_Tcells",
    "CD8_Tcells",
    "NKcells",
    "Macrophages",
    "CAFs",
    "Endothelial",
    "otherCells",
]

DISPLAY_METRICS = [
    "epic_immune_fraction",
    "epic_tcell_fraction",
    "epic_macrophage_fraction",
    "epic_caf_fraction",
    "epic_endothelial_fraction",
    "epic_stromal_fraction",
    "epic_other_fraction",
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
            val = float(value)
        except ValueError:
            continue
        if np.isfinite(val):
            rows.append((gene.strip().upper(), max(val, 0.0)))
    if not rows:
        raise ValueError(f"no gene-value rows parsed from {path}")
    return pd.DataFrame(rows, columns=["gene", "value"]).groupby("gene")["value"].sum().astype(np.float64)


def to_cpm(values: pd.Series) -> pd.Series:
    values = values.clip(lower=0.0)
    total = float(values.sum())
    if total <= 0 or not np.isfinite(total):
        return values * 0.0
    return values / total * 1_000_000.0


def sample_id_from_file(file_name: str) -> str:
    return Path(file_name).stem


def prepare_pool_matrix(pool: str, details_path: Path, input_dir: Path) -> pd.DataFrame:
    details = pd.read_csv(details_path).copy()
    rows = []
    series = []
    for rec in details.itertuples(index=False):
        file_name = str(getattr(rec, "file"))
        sample_id = sample_id_from_file(file_name)
        expr = to_cpm(parse_gene_value_file(input_dir / file_name)).rename(sample_id)
        series.append(expr)
        rows.append(
            {
                "pool": pool,
                "sample_id": sample_id,
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
    meta = pd.DataFrame(rows)
    matrix = pd.concat(series, axis=1).fillna(0.0)
    matrix.index.name = "gene"
    matrix_path = OUTDIR / f"{pool.lower().replace('-', '_')}_cpm_matrix.tsv"
    meta_path = OUTDIR / f"{pool.lower().replace('-', '_')}_metadata.csv"
    matrix.to_csv(matrix_path, sep="\t", float_format="%.8g")
    meta.to_csv(meta_path, index=False)
    return meta


def write_r_script() -> None:
    script = f"""
lib <- normalizePath("{R_LIB}", mustWork=TRUE)
.libPaths(c(lib, .libPaths()))
if (!requireNamespace("EPIC", quietly=TRUE)) {{
  stop("EPIC is not installed in project R library: ", lib)
}}
library(EPIC)
run_one <- function(pool_slug) {{
  matrix_path <- file.path("{OUTDIR}", paste0(pool_slug, "_cpm_matrix.tsv"))
  fractions_path <- file.path("{OUTDIR}", paste0(pool_slug, "_epic_cell_fractions.tsv"))
  mrna_path <- file.path("{OUTDIR}", paste0(pool_slug, "_epic_mrna_proportions.tsv"))
  gof_path <- file.path("{OUTDIR}", paste0(pool_slug, "_epic_fit_gof.tsv"))
  mat <- as.matrix(read.table(matrix_path, header=TRUE, sep="\\t", row.names=1, check.names=FALSE))
  res <- EPIC::EPIC(bulk=mat)
  write.table(res$cellFractions, fractions_path, sep="\\t", quote=FALSE, col.names=NA)
  write.table(res$mRNAProportions, mrna_path, sep="\\t", quote=FALSE, col.names=NA)
  write.table(res$fit.gof, gof_path, sep="\\t", quote=FALSE, col.names=NA)
}}
run_one("external_180")
run_one("multisource_450")
""".strip()
    R_SCRIPT.write_text(script + "\n", encoding="utf-8")


def run_epic() -> None:
    write_r_script()
    log_path = OUTDIR / "epic_run.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(["Rscript", str(R_SCRIPT)], cwd=str(RUN_CWD), stdout=log, stderr=subprocess.STDOUT, check=True)


def read_tsv_with_index(path: Path, index_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index.name = index_name
    return df.reset_index()


def zscore_by_pool(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[f"{col}_z"] = np.nan
        for pool, idx in out.groupby("pool").groups.items():
            vals = pd.to_numeric(out.loc[idx, col], errors="coerce")
            mean = vals.mean(skipna=True)
            std = vals.std(skipna=True)
            if not np.isfinite(std) or std < 1e-9:
                std = 1.0
            out.loc[idx, f"{col}_z"] = (vals - mean) / std
    return out


def add_composites(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in EPIC_FRACTIONS:
        if col not in out.columns:
            out[col] = 0.0
    out["epic_immune_fraction"] = out[["Bcells", "CD4_Tcells", "CD8_Tcells", "NKcells", "Macrophages"]].sum(axis=1)
    out["epic_tcell_fraction"] = out[["CD4_Tcells", "CD8_Tcells"]].sum(axis=1)
    out["epic_macrophage_fraction"] = out["Macrophages"]
    out["epic_caf_fraction"] = out["CAFs"]
    out["epic_endothelial_fraction"] = out["Endothelial"]
    out["epic_stromal_fraction"] = out[["CAFs", "Endothelial"]].sum(axis=1)
    out["epic_other_fraction"] = out["otherCells"]
    return zscore_by_pool(out, DISPLAY_METRICS)


def collect_results() -> pd.DataFrame:
    frames = []
    for pool in POOLS:
        slug = pool.lower().replace("-", "_")
        meta = pd.read_csv(OUTDIR / f"{slug}_metadata.csv")
        frac = read_tsv_with_index(OUTDIR / f"{slug}_epic_cell_fractions.tsv", "sample_id")
        gof = read_tsv_with_index(OUTDIR / f"{slug}_epic_fit_gof.tsv", "sample_id")
        merged = meta.merge(frac, on="sample_id", how="left").merge(gof, on="sample_id", how="left", suffixes=("", "_gof"))
        frames.append(merged)
    return add_composites(pd.concat(frames, ignore_index=True))


def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = EPIC_FRACTIONS + DISPLAY_METRICS + [f"{m}_z" for m in DISPLAY_METRICS]
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
    cols = [f"{m}_z" for m in DISPLAY_METRICS]
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


def write_heatmap(summary: pd.DataFrame, pool: str, path: Path) -> None:
    sub = summary.loc[summary["pool"] == pool].copy().sort_values("n", ascending=False)
    cols = [f"{m}_z" for m in DISPLAY_METRICS]
    labels = [c.replace("epic_", "").replace("_fraction_z", "").replace("_", " ") for c in cols]
    cell_w, cell_h = 104, 28
    left, top = 195, 82
    width = left + cell_w * len(cols) + 36
    height = top + cell_h * len(sub) + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">{html.escape(pool)} official EPIC deconvolution by semantic state</text>',
        '<text x="24" y="56" font-family="Arial" font-size="12" fill="#666">Values are pool-z-scored EPIC cell-fraction composites; red is higher, blue is lower.</text>',
    ]
    for j, label in enumerate(labels):
        x = left + j * cell_w + cell_w / 2
        parts.append(f'<text x="{x}" y="{top - 9}" font-family="Arial" font-size="10" text-anchor="end" transform="rotate(-45 {x} {top - 9})">{html.escape(label)}</text>')
    for i, row in enumerate(sub.itertuples(index=False)):
        y = top + i * cell_h
        state = str(row.semantic_state_family)
        parts.append(f'<text x="18" y="{y + 18}" font-family="Arial" font-size="11">{html.escape(state)} (n={int(row.n)})</text>')
        for j, col in enumerate(cols):
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
    vals = row[[f"{m}_z" for m in DISPLAY_METRICS]].sort_values(ascending=False).head(4)
    return {str(k).replace("epic_", "").replace("_fraction_z", ""): float(v) for k, v in vals.items()}


def write_summary(df: pd.DataFrame, by_state: pd.DataFrame, effects: pd.DataFrame) -> None:
    lines = [
        "# EPIC deconvolution validation",
        "",
        "## Purpose",
        "",
        "This analysis runs the official EPIC R package on CPM-normalized external bulk RNA profiles and groups the resulting cell fractions by the frozen semantic state families. Unlike T4c, this is an official deconvolution package output rather than a hand-curated signature pre-screen.",
        "",
        "## Environment",
        "",
        "- R package: `EPIC 1.1.7`",
        f"- local R library: `{R_LIB}`",
        "- input normalization: per-sample CPM from raw/count-like validation profiles",
        "",
        "## Data and convergence",
        "",
    ]
    for pool, sub in df.groupby("pool"):
        nonconv = int((pd.to_numeric(sub.get("convergeCode", 0), errors="coerce").fillna(0) != 0).sum())
        lines.append(f"- {pool}: `{len(sub)}` samples / `{sub['project'].nunique()}` projects; nonzero EPIC convergeCode `{nonconv}`")
    lines.extend(["", "## Directional checks", ""])
    for pool in sorted(df["pool"].unique()):
        lines.extend([f"### {pool}", ""])
        for state in PRIMARY_STATES:
            row = by_state.loc[(by_state["pool"] == pool) & (by_state["semantic_state_family"] == state)]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(
                f"- `{state}` n={int(r['n'])}: top z-scored EPIC composites `{top_metrics(r)}`, "
                f"raw immune `{float(r['epic_immune_fraction']):.3f}`, stromal `{float(r['epic_stromal_fraction']):.3f}`, other `{float(r['epic_other_fraction']):.3f}`"
            )
        lines.append("")
    hematologic_immune = mean_state(by_state, "hematologic_override", ["epic_immune_fraction_z", "epic_tcell_fraction_z", "epic_macrophage_fraction_z"])
    generic_stromal = mean_state(by_state, "generic_context_override", ["epic_caf_fraction_z", "epic_endothelial_fraction_z", "epic_stromal_fraction_z", "epic_other_fraction_z"])
    clean_stromal = mean_state(by_state, "clean_anchor_override", ["epic_caf_fraction_z", "epic_endothelial_fraction_z", "epic_stromal_fraction_z", "epic_other_fraction_z"])
    epithelial_immune = mean_state(by_state, "epithelial_override", ["epic_immune_fraction_z", "epic_tcell_fraction_z", "epic_macrophage_fraction_z"])
    epithelial_stromal = mean_state(by_state, "epithelial_override", ["epic_stromal_fraction_z", "epic_other_fraction_z"])
    lines.extend(
        [
            "## Directional summary",
            "",
            f"- `hematologic_override`: official EPIC immune-direction mean z `{hematologic_immune:.3f}`.",
            f"- `generic_context_override`: official EPIC stromal/other-context mean z `{generic_stromal:.3f}`.",
            f"- `clean_anchor_override`: official EPIC stromal/other-context mean z `{clean_stromal:.3f}`.",
            f"- `epithelial_override`: EPIC immune mean z `{epithelial_immune:.3f}` and stromal/other mean z `{epithelial_stromal:.3f}`; keep this label cautious until the state is split or independently validated.",
            "",
            "## Output files",
            "",
            "- `t4d_epic_scores_by_sample.csv`",
            "- `t4d_epic_scores_by_state.csv`",
            "- `t4d_epic_scores_by_disease_status.csv`",
            "- `t4d_epic_state_vs_rest_effects.csv`",
            "- `external_180_epic_state_heatmap.svg` and `multisource_450_epic_state_heatmap.svg`",
            "- raw EPIC outputs: `*_epic_cell_fractions.tsv`, `*_epic_mrna_proportions.tsv`, `*_epic_fit_gof.tsv`",
            "",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for pool, paths in POOLS.items():
        prepare_pool_matrix(pool, paths["details"], paths["inputs"])
    run_epic()
    df = collect_results()
    by_state, by_status = summarize(df)
    effects = effects_state_vs_rest(df)
    df.to_csv(OUTDIR / "t4d_epic_scores_by_sample.csv", index=False)
    by_state.to_csv(OUTDIR / "t4d_epic_scores_by_state.csv", index=False)
    by_status.to_csv(OUTDIR / "t4d_epic_scores_by_disease_status.csv", index=False)
    effects.to_csv(OUTDIR / "t4d_epic_state_vs_rest_effects.csv", index=False)
    write_heatmap(by_state, "External-180", OUTDIR / "external_180_epic_state_heatmap.svg")
    write_heatmap(by_state, "MultiSource-450", OUTDIR / "multisource_450_epic_state_heatmap.svg")
    write_summary(df, by_state, effects)


if __name__ == "__main__":
    main()
