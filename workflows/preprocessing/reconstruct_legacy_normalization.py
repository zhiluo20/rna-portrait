#!/usr/bin/env python3
"""Reconstruct the historical whole-cohort log-median-ratio transformation.

This is an independently implemented numerical reconstruction, not an import
of OmicVerse and not a prospective single-sample preprocessing API. See the
recovery note for evidence, precision choices and remaining provenance limits.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalize_cohort(counts: pd.DataFrame, *, output_dtype: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return log1p normalized counts, size factors and fitted log references.

    Rows are already harmonized genes; columns are the complete source cohort.
    Only genes positive in every cohort sample contribute to the reference.
    No filtering, symbol mapping, duplicate collapse or silent imputation occurs.
    """
    if output_dtype not in {'float32', 'float64'}:
        raise ValueError('output_dtype must be explicitly float32 or float64')
    if counts.empty or not counts.index.is_unique or not counts.columns.is_unique:
        raise ValueError('Counts must be nonempty with unique gene and sample identifiers')
    if counts.index.isna().any() or counts.columns.isna().any():
        raise ValueError('Gene and sample identifiers cannot be missing')
    if any(not str(v).strip() for v in list(counts.index) + list(counts.columns)):
        raise ValueError('Gene and sample identifiers cannot be blank')
    values = counts.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Counts must be finite and nonnegative; missing is not zero')
    if not np.equal(values, np.floor(values)).all():
        raise ValueError('Expected integer-valued raw counts, not normalized expression')
    reference_rows = np.all(values > 0, axis=1)
    if not reference_rows.any():
        raise ValueError('No gene is positive in every sample; the historical estimator is undefined')
    logs = np.log(values[reference_rows])
    reference = np.mean(logs, axis=1)
    offsets = np.median(logs - reference[:, None], axis=0)
    factors = np.exp(offsets)
    if not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError('Size-factor estimation did not produce finite positive factors')
    transformed = np.log1p(values / factors[None, :]).astype(output_dtype)
    return (pd.DataFrame(transformed, index=counts.index.copy(), columns=counts.columns.copy()),
            pd.Series(factors, index=counts.columns.copy(), name='size_factor'),
            pd.Series(reference, index=counts.index[reference_rows].copy(), name='mean_log_count'))


def reconstruct(counts_path: Path, output_dir: Path, *, output_dtype: str, cohort_id: str) -> dict:
    """Write a new, checksum-bound cohort reconstruction; never overwrite a run."""
    counts_path = counts_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError('Output directory already exists; use a new versioned directory')
    if not cohort_id.strip():
        raise ValueError('A source cohort identifier is required')
    input_hash = sha256(counts_path)
    if counts_path.suffix == '.parquet':
        counts = pd.read_parquet(counts_path)
    elif counts_path.suffix == '.csv':
        counts = pd.read_csv(counts_path, index_col=0, keep_default_na=False, float_precision='round_trip')
    else:
        raise ValueError('Use a gene-by-sample .parquet or .csv count matrix')
    logged, factors, reference = normalize_cohort(counts, output_dtype=output_dtype)
    if sha256(counts_path) != input_hash:
        raise RuntimeError('Input file changed during reconstruction')
    output_dir.mkdir(parents=True, exist_ok=False)
    logged.to_parquet(output_dir / 'expr_log.parquet')
    factors.to_frame().to_parquet(output_dir / 'size_factors.parquet')
    reference.to_frame().to_parquet(output_dir / 'reference_mean_log_counts.parquet')
    manifest = {
        'schema_version': 1, 'cohort_id': cohort_id,
        'scope': 'Historical whole-cohort reconstruction, not training-only fitting or single-sample inference',
        'formula': 'r_g = mean_j(log(x_gj)); s_j = exp(median_g(log(x_gj)-r_g)); y_gj = log1p(x_gj/s_j)',
        'reference_gene_rule': 'Use genes positive in every supplied cohort sample',
        'gene_mapping_or_imputation_performed': False,
        'samples': counts.shape[1], 'genes': counts.shape[0], 'reference_genes': len(reference),
        'output_dtype': output_dtype, 'single_sample_warning': counts.shape[1] == 1,
        'input_file': counts_path.name, 'input_sha256': input_hash,
        'driver_sha256': sha256(Path(__file__)), 'python': platform.python_version(),
        'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'pyarrow']},
        'outputs_sha256': {p.name: sha256(p) for p in sorted(output_dir.glob('*.parquet'))},
        'historical_installed_omicverse_version_claimed': False,
    }
    (output_dir / 'normalization_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cohort-id', required=True)
    parser.add_argument('--output-dtype', choices=['float32', 'float64'], required=True)
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.counts, args.output_dir, output_dtype=args.output_dtype, cohort_id=args.cohort_id), indent=2))


if __name__ == '__main__':
    main()
