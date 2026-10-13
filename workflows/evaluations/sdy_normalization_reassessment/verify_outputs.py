"""Independent arithmetic and lineage checks without rerunning inference."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(study_root, input_root, output_root):
    r, inp, out = [Path(x).resolve() for x in [study_root, input_root, output_root]]
    prep = json.loads((inp / 'preparation.json').read_text())
    execution = json.loads((out / 'execution_summary.json').read_text())
    evaluation = json.loads((out / 'evaluation_summary.json').read_text())
    contract = json.loads((out / 'execution_contract.json').read_text())
    assert prep['protocol_sha256'] == sha(inp / 'PROTOCOL.json')
    assert prep['prepared_inputs_sha256'] == sha(inp / 'prepared_inputs.npz')
    assert execution['execution_contract_sha256'] == sha(out / 'execution_contract.json')
    assert evaluation['evaluation_contract_sha256'] == sha(out / 'evaluation_contract.json')
    assert contract['driver_sha256'] == sha(out / 'run_inference.py')
    for payload in [execution, evaluation]:
        for relative, expected in payload['output_sha256'].items():
            assert sha(out / relative) == expected, relative
    for relative, expected in prep['immutable_old_files_sha256'].items():
        assert sha(r / relative) == expected, relative
    for relative, expected in contract['source_sha256'].items():
        assert sha(r / 'rna-portrait' / relative) == expected, relative
    data = np.load(inp / 'prepared_inputs.npz')
    counts = data['counts'].astype(float)
    positive = np.all(counts > 0, axis=0)
    assert counts.shape == (96, 23323) and int(positive.sum()) == 9585
    logs = np.log(counts[:, positive])
    factors = np.exp(np.median(logs - logs.mean(axis=0), axis=1))
    factor_error = float(np.max(np.abs(factors-data['sf_reference_factors'])))
    assert factor_error < 1e-12
    assert np.allclose(np.log1p(counts/factors[:, None]), data['sf_reference'], rtol=0, atol=1e-12)
    for arm in ['sf_reference', 'sf_full_measured']:
        assert np.array_equal(np.log1p(counts/data[arm+'_factors'][:, None]), data[arm])
    ids = data['library_id'].astype(str)
    sample_index = {sample: i for i, sample in enumerate(ids)}
    rna = np.load(out / 'rna_embeddings.npz')
    text = np.load(out / 'portrait_embeddings.npz')
    assert np.array_equal(rna['library_id'], ids) and np.array_equal(text['library_id'], ids)
    mismatch = rna['mismatched_source_library_id'].astype(str)
    assert set(mismatch) == set(ids) and np.all(mismatch != ids)
    assert np.array_equal(rna['rna_direct_sf_reference'][[sample_index[s] for s in mismatch]], rna['rna_mismatched_sf_reference'])
    queries = np.load(r / 'repair_v1/pilot_rankings_v1/text_embeddings.npz')
    query_index = {(q, str(v)): i for i, (q, v) in enumerate(zip(queries['query_id'], queries['variant']))}
    ranks = pd.read_csv(out / 'rankings_all_30_queries.csv.gz', dtype={'variant': str}, float_precision='round_trip')
    score_error, score_groups = 0.0, 0
    for method in execution['added_methods_and_controls']:
        bank, query = (rna, queries['query_rna']) if method.startswith('rna_') else (text, queries['query_raw'])
        scores = query @ bank[method].T
        for (qid, variant), group in ranks.loc[ranks.method.eq(method)].groupby(['query_id', 'variant']):
            assert len(group) == 96 and group.sample_id.is_unique
            row = scores[query_index[(qid, variant)]]
            stored = group.set_index('sample_id').loc[ids].score.to_numpy()
            score_error = max(score_error, float(np.max(np.abs(stored-row))))
            assert group.sort_values('rank').sample_id.tolist() == ids[np.lexsort((ids, -row))].tolist()
            score_groups += 1
    assert score_groups == 900 and score_error == 0
    protocol = json.loads((inp / 'PROTOCOL.json').read_text())
    texts = pd.read_csv(out / 'portrait_texts.csv')
    record_count = 0
    for row in texts.itertuples():
        record = json.loads((out / 'portraits' / row.method / (row.library_id+'.json')).read_text())
        assert record['status'] == 'ok' and record['library_id'] == row.library_id
        assert record['execution_contract_sha256'] == execution['execution_contract_sha256']
        assert record['input_route_audit'] == {'same_values_both_branches': True, 'anchor_runner_declared_transform': True}
        arm = 'sf_full_measured' if row.method.endswith('sf_full_measured') else 'sf_reference'
        assert record['input_row_sha256'] == hashlib.sha256(data[arm][sample_index[row.library_id]].tobytes()).hexdigest()
        portrait = record['payload']['semantic_explanation']['semantic_portrait']
        assert row.portrait_text == '\n'.join(f.replace('_', ' ')+': '+portrait[f] for f in protocol['text_fields'])
        record_count += 1
    assert record_count == 384
    qrels = pd.read_csv(r / 'repair_v1/paired_candidates/SDY2823/composition_pilot_v1/qrels.csv')
    truth = qrels.set_index(['query_id', 'sample_id']).grade.to_dict()
    measured = pd.read_csv(out / 'rankings_measured_three_queries.csv.gz', dtype={'variant': str}, float_precision='round_trip')
    results = pd.read_csv(out / 'per_query_paraphrase_results.csv', dtype={'variant': str}, float_precision='round_trip').set_index(['method', 'query_id', 'variant'])
    metric_error, groups = 0.0, 0
    for key, group in measured.groupby(['method', 'query_id', 'variant']):
        method, qid, variant = key
        assert len(group) == 96 and group.sample_id.is_unique and set(group.sample_id) == set(ids)
        grades = np.array([truth[(qid, s)] for s in group.sample_id], dtype=int)
        gains = 2.0**grades-1
        tied = ndcg_score(gains[None, :], group.score.to_numpy()[None, :], k=10, ignore_ties=False)
        top = group.sort_values(['score', 'sample_id'], ascending=[False, True]).head(10)
        ordered = np.array([truth[(qid, s)] for s in top.sample_id])
        discounts = 1/np.log2(np.arange(2, 12))
        deterministic = np.dot(2.0**ordered-1, discounts)/np.dot(np.sort(gains)[::-1][:10], discounts)
        checks = {'ndcg_at_k': deterministic, 'tie_expected_ndcg_at_10': tied,
                  'precision_at_k': np.mean(ordered > 0),
                  'recall_at_10': np.count_nonzero(ordered > 0)/np.count_nonzero(grades > 0),
                  'upper_quartile_discovery_at_10': np.mean(ordered == 2)}
        for field, value in checks.items():
            metric_error = max(metric_error, abs(float(value)-results.loc[key, field]))
        groups += 1
    assert groups == 243 and metric_error < 1e-12
    means = pd.read_csv(out / 'three_question_descriptive_means.csv', float_precision='round_trip').set_index('method')
    recalculated = results[['ndcg_at_k', 'tie_expected_ndcg_at_10']].groupby(['method', 'query_id']).mean().groupby('method').mean()
    assert np.allclose(recalculated, means.loc[recalculated.index, recalculated.columns], rtol=0, atol=1e-12)
    return {'all_output_hashes_verified': True, 'old_outputs_unchanged': True,
            'reference_size_factor_max_abs_error': factor_error, 'score_groups_reconstructed': score_groups,
            'score_max_abs_error': score_error, 'full_portrait_records_verified': record_count,
            'independent_metric_groups_checked': groups, 'independent_metric_max_abs_error': metric_error,
            'both_RNA_branches_input_route_verified': True, 'no_self_mismatched_samples': True,
            'scope': 'Score reconstruction from persisted embeddings and arithmetic verification, not a second independent forward pass or biological validation'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root', type=Path, required=True)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    checks = verify(a.study_root, a.input_root, a.output_root)
    target = a.output_root / 'independent_validation.json'
    if target.exists():
        assert json.loads(target.read_text()) == checks
    else:
        with target.open('x') as stream:
            stream.write(json.dumps(checks, indent=2)+'\n')
    print(json.dumps(checks, indent=2))
