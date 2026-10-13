"""Freeze a CPTAC normalization-reassessment contract and prepare RNA-only inputs.

The public PR5 normalizer is reused, not altered. No phenotype/qrel values are
read in this script. All historical inputs/outputs remain immutable.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main(r, out):
    r, out = r.resolve(), out.resolve()
    c = r / 'repair_v1/paired_candidates/CPTAC_PanCancer'
    normalizer = r / 'preprocessing_recovery_20261013/github_patch/workflows/preprocessing/reconstruct_legacy_normalization.py'
    assert sha(normalizer) == 'b1a9d38c2e3ade61d685d7124c07b46974d835209f64da8225862464ead5a5b5'
    pool_path = c / 'rna_inputs_v1/mechanically_valid_candidates.csv'
    counts_path = c / 'raw/RNA_Broad_v1/ALL.gene_reads.parquet'
    mapping_path = c / 'rna_inputs_v1/model_gene_mapping.csv'
    frozen_protein = json.loads((c / 'protein_concordance_v2/protocol.json').read_text())
    assert sha(pool_path) == frozen_protein['candidate_pool_sha256']
    assert sha(r / 'queries.json') == frozen_protein['question_file_sha256']
    old_qc = json.loads((c / 'rna_inputs_v1/QC_summary.json').read_text())
    assert sha(mapping_path) == old_qc['mapping_sha256']
    assert sha(c / 'rna_inputs_v1/model_input_arrays.npz') == old_qc['input_arrays_sha256']
    pool = pd.read_csv(pool_path, keep_default_na=False)
    assert len(pool) == 942 and pool.public_case_id.is_unique and pool.broad_sample_id.is_unique
    mapping = pd.read_csv(mapping_path, keep_default_na=False)
    resolved = mapping.loc[mapping.status.eq('resolved')].set_index('model_gene').measured_gene_id
    assert len(resolved) == 23628 and resolved.index.is_unique and resolved.is_unique
    sources = [pool_path, counts_path, mapping_path, c / 'broad_gene_annotation.json',
               c / 'rna_inputs_v1/sample_input_QC.csv', c / 'rna_inputs_v1/model_input_arrays.npz',
               c / 'rna_inputs_v1/protocol.json', c / 'protein_concordance_v2/protocol.json',
               c / 'protein_concordance_v2/qrels.csv', r / 'queries.json',
               r / 'repair_v1/pilot_rankings_v1/text_embeddings.npz', normalizer,
               r / 'rna-portrait/models/rna_language_alignment/rna_language_alignment.pt',
               r / 'lab_runtime_20261013_v2_offline/validation_v1/linux_runtime_contract.candidate.json']
    old_files = sorted(p for folder in ['rna_predictions_v1', 'cellwhisperer_v1', 'conventional_rankings_v1',
                                      'retrieval_evaluation_v1', 'protein_concordance_v2', 'full_portrait_v1']
                       for p in (c / folder).rglob('*') if p.is_file() and not p.name.startswith('._'))
    source_hashes = {str(p.relative_to(r)): sha(p) for p in sources}
    old_hashes = {str(p.relative_to(r)): sha(p) for p in old_files}
    out.mkdir(exist_ok=False)
    protocol = {
        'protocol_id': 'CPTAC-fixed-pool-size-factor-reassessment-v1-20261013',
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'stage': 'Exploratory post hoc normalization repair; old CPTAC and SDY outcomes already known. Not preregistration or confirmation.',
        'pool': 'Unchanged 942 Broad tumor-case columns. 443 prior alignment-test and 499 no-recorded-case-match; neither guarantees new patient independence.',
        'source_sha256': source_hashes,
        'immutable_old_files_sha256': old_hashes,
        'primary_normalization': 'sf_reference',
        'arms': {
            'sf_reference': 'Fit on all 23628 uniquely mapped reference genes, then apply log1p(count / factor). Same mapped universe as old reference CPM; change formula only.',
            'sf_full_measured': 'Gene-universe sensitivity: fit on all 59808 measured Broad Ensembl rows, then project onto the identical 23628 model symbols.'},
        'fit_population': 'Exactly all 942 locked candidate RNA libraries jointly in both arms. No cohort/disease label, protein measurement or model score is used to fit or select a normalization.',
        'fit_scope': 'Unsupervised transductive normalization, not a frozen training-only or prospective single-sample reference. This harmonized Broad pool is not asserted identical to any original GDC/GEO fitting cohort or annotation.',
        'formula': 'For genes positive in every fit sample: r_g=mean_j(log(count_gj)); s_j=exp(median_g(log(count_gj)-r_g)); output=log1p(count_gj/s_j).',
        'precision': 'Float64 fitted factors and mapped values; existing encoder float32 projection/clipping retained.',
        'input_scale': 'training_log pass-through; this API name alone does not establish distributional equivalence.',
        'mapping': 'Frozen unique mapping; 3502 measured selected genes, 594 unmeasured model features retain the old missing-to-zero convention. No new rescue/filtering.',
        'anchor': 'Same pretransformed mapped values as RNA branch; pretransformed=True and allow_alternate=False verified per sample.',
        'model_conditions_per_arm': ['rna_direct', 'rna_fused', 'full_portrait_released_calibration', 'full_portrait_no_calibration'],
        'primary_controls': {
            'rna_value_shuffled_sf_reference': 'Permute only the 3502 measured selected-feature log values within each sample, preserving missing-feature positions and observed multiset.',
            'rna_mismatched_global_sf_reference': 'Derange primary direct embeddings globally; exactly equivalent to reassigning already transformed profiles.',
            'rna_mismatched_within_cohort_sf_reference': 'Derange within the 10 original source strata; labels used only to define this control, never as RNA model input.',
            'rna_mask_10pct_sf_reference': 'Set 10 percent of measured selected log inputs to zero after the fixed transform, without refitting factors. Missing-input sensitivity, not reprocessed raw library loss.',
            'rna_mask_30pct_sf_reference': 'As above, 30 percent.',
            'rna_mask_50pct_sf_reference': 'As above, 50 percent.'},
        'randomization': 'Seed 20261010, old ordering: selected shuffle by case, then masks at 10/30/50 percent by case, then global and sorted-cohort derangements. Single draws are descriptive, not independent biological replicates.',
        'seed': 20261010,
        'text_fields': ['headline', 'biological_summary', 'what_to_trust', 'caution', 'recommended_reading', 'confidence_posture'],
        'text_rule': 'All six complete fields, unchanged prefixes. No query-specific selection or rewriting.',
        'evaluation': 'Unchanged 30 questions x 3 rewrites, 18 old methods retained plus 14 new conditions; original 13 contexts. Per-query identical judged pools, original protein panels and qrels, deterministic nDCG@10 plus exact-tie expectation, P/Recall/strong discovery, full-pool judged coverage, cohort representation and paraphrase stability.',
        'aggregation': 'Rewrites within query, then cohort blocks within query when applicable, then queries. Related panels remain correlated; no per-rewrite superiority P values.',
        'boundaries': ['Protein concordance is not cell counts, function, mechanism or independent statement correctness.', 'Known-case exclusions and original anchor training membership remain limited.', 'No new cases, queries, outcomes, weights, calibrations, priors or prototypes; no retraining.', 'No replacing primary by a more favorable sensitivity; all failures and weak results retained.', 'Old output files immutable; normalization correction does not prove Cell utility.'],
        'labels_read_during_preparation_or_inference': False,
        'driver_sha256': sha(__file__),
    }
    save_new(out / 'PROTOCOL.json', protocol)
    print('Protocol frozen before input transformation', sha(out / 'PROTOCOL.json'), flush=True)
    annotation = json.loads((c / 'broad_gene_annotation.json').read_text())
    frame = pq.read_table(counts_path, columns=['Name', *pool.broad_sample_id.tolist()]).to_pandas()
    assert frame.index.tolist() == [x['gene_id'] for x in annotation] and len(frame) == 59808
    counts = frame[pool.broad_sample_id.tolist()]
    counts.columns = pool.public_case_id.tolist()
    assert np.issubdtype(counts.to_numpy().dtype, np.integer) and (counts.to_numpy() >= 0).all()
    qc = pd.read_csv(c / 'rna_inputs_v1/sample_input_QC.csv').set_index('public_case_id').loc[counts.columns]
    for sample in counts.columns:
        actual = hashlib.sha256(np.ascontiguousarray(counts[sample].to_numpy(dtype=np.int32)).tobytes()).hexdigest()
        assert actual == qc.at[sample, 'counts_sha256_gene_axis_fixed']
    mapped = counts.loc[resolved.tolist()].copy()
    mapped.index = resolved.index.tolist()
    old = np.load(c / 'rna_inputs_v1/model_input_arrays.npz')
    assert old['public_case_id'].tolist() == counts.columns.tolist() and old['eligible'].all()
    selected = old['selected_genes'].astype(str)
    present = np.array([g in mapped.index for g in selected])
    assert np.array_equal(present, old['measured_feature_mask']) and present.sum() == 3502
    assert np.array_equal(mapped.loc[selected[present]].to_numpy().T, old['selected_counts'][:, present])
    assert np.array_equal(mapped.sum(axis=0).to_numpy(dtype=float), old['reference_library_sum'])
    old_recalc = np.log1p(old['selected_counts'].astype(float) / old['reference_library_sum'][:, None] * 1e6).astype(np.float32)
    assert np.array_equal(old_recalc, old['log1p_reference_cpm'])
    spec = importlib.util.spec_from_file_location('historical_normalizer', normalizer)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    arrays = {'sample_id': np.array(counts.columns, dtype=str), 'model_gene': np.array(mapped.index, dtype=str),
              'selected_genes': selected, 'measured_selected_mask': present}
    summaries, diagnostics = {}, []
    for arm, fitting in [('sf_reference', mapped), ('sf_full_measured', counts)]:
        logged, factors, reference = module.normalize_cohort(fitting, output_dtype='float64')
        # Independent factor check using sorted midpoint(s), not np.median.
        values = fitting.loc[reference.index].to_numpy(dtype=float)
        ratios = np.log(values) - np.log(values).sum(axis=1)[:, None] / values.shape[1]
        sorted_ratios = np.sort(ratios, axis=0)
        middle = len(sorted_ratios) // 2
        offsets = sorted_ratios[middle] if len(sorted_ratios) % 2 else (sorted_ratios[middle-1] + sorted_ratios[middle]) / 2
        factor_error = float(np.max(np.abs(np.exp(offsets) - factors.to_numpy())))
        assert factor_error <= 1e-10
        arrays[arm] = np.log1p(mapped.to_numpy(dtype=float) / factors.to_numpy()[None, :]).T
        assert np.isfinite(arrays[arm]).all() and (arrays[arm] >= 0).all()
        expected = logged if arm == 'sf_reference' else logged.loc[resolved.tolist()]
        assert np.array_equal(arrays[arm], expected.to_numpy().T)
        factors.to_csv(out / (arm + '_size_factors.csv'))
        reference.to_csv(out / (arm + '_reference_log_counts.csv'))
        summaries[arm] = {'fit_genes': len(fitting), 'positive_in_all_samples_genes': len(reference),
                          'factor_min': float(factors.min()), 'factor_median': float(factors.median()), 'factor_max': float(factors.max()),
                          'independent_factor_max_abs_error': factor_error,
                          'mapped_log_min': float(arrays[arm].min()), 'mapped_log_max': float(arrays[arm].max())}
        for i, sample in enumerate(counts.columns):
            diagnostics.append({'sample_id': sample, 'arm': arm, 'size_factor': float(factors.iloc[i]),
                                'mapped_log_median': float(np.median(arrays[arm][i])),
                                'mapped_zero_fraction': float(np.mean(arrays[arm][i] == 0))})
        print(arm, summaries[arm], flush=True)
        del logged, values, ratios, sorted_ratios
    np.savez_compressed(out / 'prepared_inputs.npz', **arrays)
    pd.DataFrame(diagnostics).to_csv(out / 'preparation_sample_diagnostics.csv', index=False)
    for path, expected in source_hashes.items():
        assert sha(r / path) == expected, path
    for path, expected in old_hashes.items():
        assert sha(r / path) == expected, path
    preparation = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha(out / 'PROTOCOL.json'),
                   'prepared_inputs_sha256': sha(out / 'prepared_inputs.npz'), 'source_sha256': source_hashes,
                   'immutable_old_files_sha256': old_hashes, 'samples': 942, 'mapped_reference_genes': 23628,
                   'measured_selected_genes': 3502, 'all_raw_columns_match_old_QC_hashes': True,
                   'old_CPM_reconstruction_exact': True, 'summaries': summaries, 'qrel_values_read': False,
                   'new_training': False, 'old_files_unchanged': True}
    save_new(out / 'preparation.json', preparation)
    print('Preparation complete; assay-label values not read', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    main(args.study_root, args.output_root)
