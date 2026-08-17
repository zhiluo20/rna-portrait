#!/usr/bin/env python3
"""Build a merged GEO + TCGA training dataset in a sharded layout."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
GEO_ROOT = ROOT / "data" / "geo_processed_source"
TCGA_ROOT = ROOT / "data" / "tcga_processed_source"
OUT_ROOT = ROOT / "data" / "training_dataset_merged"
SHARD_SIZE = 512


def build_caption(row: pd.Series) -> str:
    free_value = row.get("metadata_text")
    free = "" if pd.isna(free_value) else str(free_value).strip()
    parts = []
    age = row.get("feat_age")
    if pd.notna(age):
        parts.append(f"age {int(round(float(age)))}")
    parts.append(f"sex {row.get('feat_sex', 'unknown')}")
    parts.append(f"biospecimen {row.get('feat_biospecimen_type', 'unknown')}")
    parts.append(f"anatomical site {row.get('feat_anatomical_site', 'unknown')}")
    parts.append(f"tumor status {row.get('feat_tumor_status', 'unknown')}")
    parts.append(f"tissue context {row.get('feat_tissue_context', 'unknown')}")
    parts.append(f"disease {row.get('feat_disease_label', 'unknown')}")
    parts.append(f"disease severity {row.get('feat_disease_severity', 'unknown')}")
    parts.append(f"sample role {row.get('feat_sample_role', 'unknown')}")
    parts.append(f"gse {row.get('gse', 'unknown')}")
    structured = "Structured metadata: " + "; ".join(parts) + "."
    return f"{free} {structured}".strip() if free else structured


def safe_dirname(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def load_geo(geo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    expr = pd.read_parquet(geo_root / "expr_log.parquet")
    meta = pd.read_parquet(geo_root / "meta.parquet").copy()
    sample_ids = np.load(geo_root / "sample_ids.npy", allow_pickle=True).astype(str)
    if len(meta) != len(sample_ids) or expr.shape[1] != len(sample_ids):
        raise ValueError("GEO expression, metadata and sample_ids.npy dimensions differ")
    meta["sample_id"] = sample_ids
    if "caption_text" not in meta.columns:
        meta["caption_text"] = meta.apply(build_caption, axis=1)
    meta["source_dataset"] = "geo"
    meta["expr_log_path"] = "inline_geo"
    expr.columns = sample_ids
    return expr, meta


def load_tcga_meta(tcga_root: Path) -> pd.DataFrame:
    meta = pd.read_parquet(tcga_root / "meta.parquet").copy()
    if "caption_text" not in meta.columns:
        meta["caption_text"] = meta.apply(build_caption, axis=1)
    meta["source_dataset"] = "tcga"
    return meta


def copy_tcga_shards(
    tcga_root: Path,
    out_root: Path,
    expected_genes: np.ndarray,
) -> int:
    out_projects = out_root / "projects"
    out_projects.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(tcga_root.glob("projects/*/shard_*")):
        rel = src.relative_to(tcga_root / "projects")
        dst = out_projects / rel
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        shard_genes = np.load(src / "genes.npy", allow_pickle=True).astype(str)
        if not np.array_equal(shard_genes, expected_genes.astype(str)):
            raise ValueError(f"TCGA shard gene order differs from GEO: {src}")
        shutil.copy2(src / "meta.parquet", dst / "meta.parquet")
        shutil.copy2(src / "sample_ids.npy", dst / "sample_ids.npy")
        shutil.copy2(src / "genes.npy", dst / "genes.npy")
        shutil.copy2(src / "expr_log1p_cpm.parquet", dst / "expr_log.parquet")
        count += 1
    if count == 0:
        raise RuntimeError(f"No TCGA expression shards found under {tcga_root}")
    return count


def write_geo_shards(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    out_root: Path,
    shard_size: int,
) -> int:
    out_projects = out_root / "projects"
    out_projects.mkdir(parents=True, exist_ok=True)
    count = 0
    for gse, project_meta in meta.groupby("gse", sort=True):
        project_meta = project_meta.reset_index(drop=True)
        n_shards = math.ceil(len(project_meta) / shard_size)
        project_dir = out_projects / safe_dirname(gse)
        if project_dir.exists():
            shutil.rmtree(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        for shard_idx in range(n_shards):
            start = shard_idx * shard_size
            end = min((shard_idx + 1) * shard_size, len(project_meta))
            shard_meta = project_meta.iloc[start:end].copy().reset_index(drop=True)
            sample_ids = shard_meta["sample_id"].tolist()
            shard_dir = project_dir / f"shard_{shard_idx:04d}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_meta.to_parquet(shard_dir / "meta.parquet", index=False)
            np.save(shard_dir / "sample_ids.npy", np.array(sample_ids, dtype=object), allow_pickle=True)
            np.save(shard_dir / "genes.npy", expr.index.to_numpy(), allow_pickle=True)
            expr.loc[:, sample_ids].to_parquet(shard_dir / "expr_log.parquet")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geo-root", type=Path, default=GEO_ROOT)
    parser.add_argument("--tcga-root", type=Path, default=TCGA_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = parser.parse_args()
    if args.shard_size < 1:
        parser.error("--shard-size must be positive")

    geo_root = args.geo_root.resolve()
    tcga_root = args.tcga_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    geo_expr, geo_meta = load_geo(geo_root)
    tcga_meta = load_tcga_meta(tcga_root)
    genes = np.load(geo_root / "genes.npy", allow_pickle=True).astype(str)
    if not np.array_equal(geo_expr.index.astype(str).to_numpy(), genes):
        raise ValueError("GEO genes.npy order does not match expr_log.parquet")

    merged_meta = pd.concat([geo_meta, tcga_meta], ignore_index=True, sort=False)
    merged_meta = merged_meta.sort_values(["source_dataset", "gse", "sample_id"]).reset_index(drop=True)
    if merged_meta["sample_id"].astype(str).duplicated().any():
        raise ValueError("GEO and TCGA sample identifiers are not globally unique")
    merged_sample_ids = merged_meta["sample_id"].to_numpy()

    np.save(out_root / "genes.npy", genes, allow_pickle=True)
    np.save(out_root / "sample_ids.npy", merged_sample_ids, allow_pickle=True)
    merged_meta.to_parquet(out_root / "meta.parquet", index=False)
    merged_meta.to_csv(out_root / "meta.csv", index=False)

    geo_shards = write_geo_shards(
        geo_expr,
        geo_meta,
        out_root,
        shard_size=args.shard_size,
    )
    tcga_shards = copy_tcga_shards(tcga_root, out_root, genes)

    summary = {
        "n_samples_total": int(len(merged_meta)),
        "n_samples_geo": int((merged_meta["source_dataset"] == "geo").sum()),
        "n_samples_tcga": int((merged_meta["source_dataset"] == "tcga").sum()),
        "n_genes": int(len(genes)),
        "n_projects_total": int(merged_meta["gse"].nunique()),
        "n_projects_geo": int(geo_meta["gse"].nunique()),
        "n_projects_tcga": int(tcga_meta["gse"].nunique()),
        "n_shards_geo": int(geo_shards),
        "n_shards_tcga": int(tcga_shards),
        "shard_size": int(args.shard_size),
        "layout": "projects/<gse>/shard_xxxx/{meta.parquet, sample_ids.npy, genes.npy, expr_log.parquet}",
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("[OK] wrote merged GEO+TCGA training dataset")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
