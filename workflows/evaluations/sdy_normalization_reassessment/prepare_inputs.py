"""Freeze SDY whole-cohort input transforms without reading independent labels."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import shutil
import numpy as np
import pandas as pd


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(study_root, output_root):
    r, out = Path(study_root).resolve(), Path(output_root).resolve()
    protocol = json.loads((out / 'PROTOCOL.json').read_text())
    assert not (out / 'preparation.json').exists()
    assert shutil.disk_usage(out).free > 1024**3
    cohort = r / 'repair_v1/paired_candidates/SDY2823'
    pool_path = cohort / 'pairing_v1/primary_baseline_candidate_pool.csv'
    source = cohort / 'rna_primary/GSE279552_P462_genecounts.csv.gz'
    mapping_path = cohort / 'model_gene_mapping_v2.csv'
    assert sha(pool_path) == protocol['candidate_pool_sha256']
    assert sha(source) == protocol['count_matrix_sha256']
    old = json.loads((cohort / 'full_portrait_v1/protocol.json').read_text())
    assert sha(mapping_path) == old['mapping_sha256']
    assert sha(r / 'queries.json') == protocol['query_file_sha256']
    # Hash the label file without parsing or accessing label values.
    assert sha(cohort / 'composition_pilot_v1/qrels.csv') == protocol['qrels_sha256']
    normalizer = r / 'preprocessing_recovery_20261013/github_patch/workflows/preprocessing/reconstruct_legacy_normalization.py'
    assert sha(normalizer) == protocol['normalizer_implementation_sha256']
    specification = importlib.util.spec_from_file_location('frozen_legacy_normalizer', normalizer)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    pool = pd.read_csv(pool_path).set_index('library_id')
    assert len(pool) == 96 and pool.index.is_unique and pool.donor.is_unique
    ids = pool.index.to_numpy(dtype=str)
    mapping = pd.read_csv(mapping_path, keep_default_na=False)
    resolved = mapping.loc[mapping.status.eq('resolved')].set_index('model_gene').ensembl_id
    assert len(resolved) == 23323 and resolved.index.is_unique and resolved.is_unique
    first = pd.read_csv(source, nrows=0).columns[0]
    full = pd.read_csv(source, index_col=0, usecols=[first, *ids], dtype={x: np.int32 for x in ids})
    full = full.loc[:, ids]
    assert full.shape == (58302, 96) and full.index.is_unique
    assert np.isfinite(full.to_numpy()).all() and (full.to_numpy() >= 0).all()
    mapped = full.loc[resolved.to_list()].copy()
    mapped.index = resolved.index
    previous_counts = cohort / 'full_portrait_v1/RNA_only_mapped_counts.npz'
    saved = np.load(previous_counts)
    assert np.array_equal(saved['counts'], mapped.to_numpy(dtype=np.int32).T)
    assert np.array_equal(saved['library_id'], ids) and saved['model_gene'].tolist() == mapped.index.tolist()
    sources = [pool_path, source, mapping_path, previous_counts, r / 'queries.json', normalizer]
    old_files = [cohort / 'composition_pilot_v3' / n for n in ['rankings.csv.gz', 'per_query_paraphrase_results.csv', 'evaluation_summary.json']]
    old_files += [cohort / 'full_portrait_v1' / n for n in ['protocol.json', 'execution_contract.json', 'execution_summary.json', 'portrait_texts.csv', 'rankings_all_30_queries.csv.gz']]
    arrays = {'library_id': ids, 'model_gene': mapped.index.to_numpy(dtype=str), 'counts': mapped.to_numpy(dtype=np.int32).T}
    rows = []
    for arm, matrix in [('sf_reference', mapped), ('sf_full_measured', full)]:
        normalized, factors, reference = module.normalize_cohort(matrix, output_dtype='float64')
        # Separate scalar-factor reconstruction validates the full matrix path.
        expected = np.log1p(mapped.to_numpy(dtype=float) / factors.loc[ids].to_numpy()[None, :])
        if arm == 'sf_reference':
            actual = normalized.loc[mapped.index, ids].to_numpy()
        else:
            actual = normalized.loc[resolved.to_list(), ids].to_numpy()
        assert np.array_equal(actual, expected)
        arrays[arm] = actual.T
        arrays[arm + '_factors'] = factors.loc[ids].to_numpy()
        arrays[arm + '_reference_genes'] = reference.index.to_numpy(dtype=str)
        arrays[arm + '_mean_log_counts'] = reference.to_numpy()
        rows.append({'arm': arm, 'cohort_libraries': 96, 'normalization_input_genes': matrix.shape[0],
                     'positive_reference_genes': len(reference), 'factor_min': float(factors.min()),
                     'factor_median': float(factors.median()), 'factor_max': float(factors.max())})
    target = out / 'prepared_inputs.npz'
    assert not target.exists()
    np.savez_compressed(target, **arrays)
    result = {'recorded_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha(out / 'PROTOCOL.json'),
        'prepared_inputs_sha256': sha(target), 'driver_sha256': sha(Path(__file__)),
        'source_sha256': {str(p.relative_to(r)): sha(p) for p in sources},
        'immutable_old_files_sha256': {str(p.relative_to(r)): sha(p) for p in old_files},
        'arms': rows, 'mapped_gene_count': 23323, 'candidate_count': 96,
        'previous_mapped_counts_exactly_preserved': True, 'independent_labels_read': False,
        'new_model_inference_started': False, 'training_started': False,
        'normalization_fit_is_transductive': True,
        'scope': 'Input preparation only. The full-measured Ensembl arm is a sensitivity analysis, not an assertion of exact historical GEO preprocessing.'}
    (out / 'preparation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['arms', 'mapped_gene_count', 'candidate_count', 'independent_labels_read', 'training_started']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.study_root, args.output_root)
