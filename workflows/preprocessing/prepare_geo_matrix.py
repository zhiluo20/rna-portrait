#!/usr/bin/env python3
"""Harmonize a GEO expression matrix and sample metadata for model training.

The input matrix must have genes on rows and samples on columns. Expression
values must already use the transformation reported for the source dataset;
this script does not infer or alter that transformation. It standardizes gene
symbols, averages duplicate gene rows, aligns metadata to matrix columns and
writes the file layout consumed by the training workflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: Path, *, index_col: int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, index_col=index_col)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", index_col=index_col)
    raise ValueError(f"Unsupported table format: {path}")


def normalize_gene_symbol(value: object) -> str:
    return str(value).strip().upper()


def harmonize_expression(expr: pd.DataFrame) -> pd.DataFrame:
    if expr.empty:
        raise ValueError("Expression matrix is empty")
    out = expr.copy()
    out.index = pd.Index(
        [normalize_gene_symbol(value) for value in out.index],
        name="gene_symbol",
    )
    out = out.loc[out.index != ""]
    out = out.apply(pd.to_numeric, errors="coerce")
    if out.isna().any().any():
        raise ValueError("Expression matrix contains missing or non-numeric values")
    out = out.groupby(level=0, sort=True).mean()
    out.columns = out.columns.astype(str)
    if out.columns.duplicated().any():
        duplicates = sorted(set(out.columns[out.columns.duplicated()].tolist()))
        raise ValueError(f"Duplicate sample columns: {duplicates[:5]}")
    return out.astype(np.float32)


def align_metadata(expr: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" not in meta.columns:
        raise ValueError("Metadata must contain a sample_id column")
    if "gse" not in meta.columns:
        raise ValueError("Metadata must contain a gse column")
    out = meta.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    if out["sample_id"].duplicated().any():
        duplicates = sorted(set(out.loc[out["sample_id"].duplicated(), "sample_id"]))
        raise ValueError(f"Duplicate metadata sample_id values: {duplicates[:5]}")

    matrix_ids = list(expr.columns)
    metadata_ids = set(out["sample_id"])
    missing = [sample_id for sample_id in matrix_ids if sample_id not in metadata_ids]
    extra = sorted(metadata_ids.difference(matrix_ids))
    if missing or extra:
        raise ValueError(
            "Matrix/metadata sample mismatch: "
            f"{len(missing)} matrix samples lack metadata and "
            f"{len(extra)} metadata rows lack expression columns"
        )
    return out.set_index("sample_id").loc[matrix_ids].reset_index()


def write_dataset(expr: pd.DataFrame, meta: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sample_ids = np.asarray(expr.columns, dtype=object)
    genes = np.asarray(expr.index, dtype=object)

    expr.to_parquet(outdir / "expr_log.parquet")
    meta.to_parquet(outdir / "meta.parquet", index=False)
    meta.to_csv(outdir / "meta.csv", index=False)
    np.save(outdir / "sample_ids.npy", sample_ids, allow_pickle=True)
    np.save(outdir / "genes.npy", genes, allow_pickle=True)

    summary = {
        "n_genes": int(len(genes)),
        "n_samples": int(len(sample_ids)),
        "n_projects": int(meta["gse"].astype(str).nunique()),
        "expression_transform": "preserved from input; see source metadata",
        "gene_harmonization": "strip whitespace, uppercase, mean duplicate rows",
        "sample_alignment": "metadata reordered to expression columns; exact set required",
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expression",
        type=Path,
        required=True,
        help="Parquet, CSV or TSV matrix with genes on rows and samples on columns.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Parquet, CSV or TSV metadata containing sample_id and gse.",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    expr = harmonize_expression(read_table(args.expression, index_col=0))
    meta = align_metadata(expr, read_table(args.metadata))
    write_dataset(expr, meta, args.outdir.resolve())
    print(
        f"[OK] wrote {args.outdir} with "
        f"{expr.shape[0]} genes and {expr.shape[1]} samples"
    )


if __name__ == "__main__":
    main()
