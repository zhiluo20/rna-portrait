from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from workflows.preprocessing.reconstruct_legacy_normalization import normalize_cohort, reconstruct, sha256


def counts():
    return pd.DataFrame({'S1': [1, 4, 0], 'S2': [4, 16, 2]}, index=['A', 'B', 'C'])


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_known_factors_and_precision(dtype):
    logged, factors, reference = normalize_cohort(counts(), output_dtype=dtype)
    np.testing.assert_allclose(factors, [.5, 2.], rtol=1e-14)
    np.testing.assert_allclose(reference, np.log([2, 8]), rtol=1e-14)
    np.testing.assert_allclose(logged, np.log1p([[2, 2], [8, 8], [0, 1]]), rtol=1e-7)
    assert logged.dtypes.eq(np.dtype(dtype)).all()


def test_permutations_preserve_identifiers_and_values():
    source = counts()
    expected, _, _ = normalize_cohort(source, output_dtype='float64')
    actual, _, _ = normalize_cohort(source.iloc[::-1, ::-1], output_dtype='float64')
    np.testing.assert_allclose(actual.loc[source.index, source.columns], expected, rtol=0, atol=1e-14)


def test_single_sample_is_identity_size_factor_not_cpm():
    source = counts().iloc[:, :1]
    actual, factors, _ = normalize_cohort(source, output_dtype='float64')
    np.testing.assert_array_equal(factors, [1.])
    np.testing.assert_array_equal(actual, np.log1p(source))


@pytest.mark.parametrize('bad', [-1., np.nan, np.inf, 1.5])
def test_invalid_count_values_rejected(bad):
    source = counts().astype(float)
    source.iloc[0, 0] = bad
    with pytest.raises(ValueError):
        normalize_cohort(source, output_dtype='float64')


def test_no_joint_positive_gene_rejected():
    with pytest.raises(ValueError, match='undefined'):
        normalize_cohort(pd.DataFrame([[1, 0], [0, 2]]), output_dtype='float64')


@pytest.mark.parametrize('axis', ['index', 'columns'])
def test_duplicate_identifiers_rejected(axis):
    source = counts()
    setattr(source, axis, ['same'] * len(getattr(source, axis)))
    with pytest.raises(ValueError, match='unique'):
        normalize_cohort(source, output_dtype='float64')


def test_empty_and_unspecified_precision_rejected():
    with pytest.raises(ValueError):
        normalize_cohort(pd.DataFrame(), output_dtype='float64')
    with pytest.raises(ValueError):
        normalize_cohort(counts(), output_dtype='float16')


def test_run_manifest_and_no_overwrite(tmp_path):
    source = tmp_path / 'counts.parquet'
    counts().to_parquet(source)
    original_hash = sha256(source)
    output = tmp_path / 'run'
    manifest = reconstruct(source, output, output_dtype='float64', cohort_id='synthetic_fixture')
    assert manifest['input_sha256'] == original_hash == sha256(source)
    assert manifest['samples'] == 2 and manifest['reference_genes'] == 2
    assert manifest == json.loads((output / 'normalization_manifest.json').read_text())
    assert all(sha256(output / name) == digest for name, digest in manifest['outputs_sha256'].items())
    with pytest.raises(FileExistsError):
        reconstruct(source, output, output_dtype='float64', cohort_id='synthetic_fixture')


def test_invalid_input_does_not_create_run_directory(tmp_path):
    source = tmp_path / 'bad.parquet'
    pd.DataFrame({'S1': [0], 'S2': [1]}).to_parquet(source)
    output = tmp_path / 'run'
    with pytest.raises(ValueError):
        reconstruct(source, output, output_dtype='float64', cohort_id='fixture')
    assert not output.exists()


def test_cli_runs_and_records_actual_files(tmp_path):
    source = tmp_path / 'counts.parquet'
    counts().to_parquet(source)
    script = Path(__file__).resolve().parents[1] / 'workflows/preprocessing/reconstruct_legacy_normalization.py'
    output = tmp_path / 'cli_run'
    result = subprocess.run([sys.executable, str(script), '--counts', str(source),
        '--output-dir', str(output), '--cohort-id', 'synthetic_fixture', '--output-dtype', 'float64'],
        check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report['cohort_id'] == 'synthetic_fixture' and report['samples'] == 2
    assert report['input_sha256'] == sha256(source)
    assert report['driver_sha256'] == sha256(script)
