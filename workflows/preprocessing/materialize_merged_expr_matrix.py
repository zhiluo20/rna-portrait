#!/usr/bin/env python3
"""Materialize a monolithic expr_log.parquet for the merged training dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "training_dataset_merged"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()

    meta = pd.read_parquet(data_dir / "meta.parquet")
    sample_ids = meta["sample_id"].astype(str).tolist()
    genes = np.load(data_dir / "genes.npy", allow_pickle=True).astype(str).tolist()

    parts = []
    for shard_expr_path in sorted((data_dir / "projects").glob("*/shard_*/expr_log.parquet")):
        expr = pd.read_parquet(shard_expr_path)
        parts.append(expr)
    if not parts:
        raise RuntimeError(f"No expression shards found under {data_dir / 'projects'}")

    expr_all = pd.concat(parts, axis=1)
    if expr_all.columns.astype(str).duplicated().any():
        raise ValueError("Duplicate sample columns found across expression shards")
    expr_all.columns = expr_all.columns.astype(str)
    expr_all = expr_all.loc[genes, sample_ids]
    expr_all.index.name = "gene_symbol"
    output_path = data_dir / "expr_log.parquet"
    expr_all.to_parquet(output_path)
    print(f"[OK] wrote {output_path} with shape {expr_all.shape}")


if __name__ == "__main__":
    main()
