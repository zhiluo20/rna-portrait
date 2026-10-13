"""Evaluate all fixed conditions after inference; preserve the 17 old methods."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import itertools
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from retrieval_metrics import metrics_for_ranking


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_new(path, payload):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def evaluate_run(study_root, input_root, output_root):
    r, inp, out = [Path(x).resolve() for x in [study_root, input_root, output_root]]
    cohort = r / 'repair_v1/paired_candidates/SDY2823'
    protocol = json.loads((inp / 'PROTOCOL.json').read_text())
    prep = json.loads((inp / 'preparation.json').read_text())
    execution = json.loads((out / 'execution_summary.json').read_text())
    assert execution['successful_full_inferences'] == 384 and execution['failed_inferences'] == 0
    assert execution['assay_label_values_read'] is False and execution['training_started'] is False
    assert execution['execution_contract_sha256'] == sha(out / 'execution_contract.json')
    for relative, expected in execution['output_sha256'].items():
        assert sha(out / relative) == expected
    for relative, expected in prep['immutable_old_files_sha256'].items():
        assert sha(r / relative) == expected
    qrel_path = cohort / 'composition_pilot_v1/qrels.csv'
    pool_path = cohort / 'pairing_v1/primary_baseline_candidate_pool.csv'
    old_rank_path = cohort / 'composition_pilot_v3/rankings.csv.gz'
    old_metric_path = cohort / 'composition_pilot_v3/per_query_paraphrase_results.csv'
    assert sha(qrel_path) == protocol['qrels_sha256']
    assert sha(pool_path) == protocol['candidate_pool_sha256']
    contract = {
        'driver_sha256': sha(__file__), 'helper_sha256': sha(Path(__file__).with_name('retrieval_metrics.py')),
        'protocol_sha256': sha(inp / 'PROTOCOL.json'), 'inference_summary_sha256': sha(out / 'execution_summary.json'),
        'qrels_sha256': sha(qrel_path), 'old_rankings_sha256': sha(old_rank_path),
        'old_metrics_sha256': sha(old_metric_path), 'candidate_pool_sha256': sha(pool_path),
        'query_ids': ['Q01', 'Q02', 'Q19'], 'k': 10,
        'scope': 'Exploratory reassessment after prior outcomes were seen; no tuning or superiority hypothesis test',
    }
    contract_path = out / 'evaluation_contract.json'
    if contract_path.exists():
        assert json.loads(contract_path.read_text()) == contract, 'Evaluation contract changed'
    else:
        save_new(contract_path, contract)
    if (out / 'evaluation_summary.json').exists():
        summary = json.loads((out / 'evaluation_summary.json').read_text())
        assert summary['evaluation_contract_sha256'] == sha(contract_path)
        for relative, expected in summary['output_sha256'].items():
            assert sha(out / relative) == expected
        print('Already evaluated and verified; no files rewritten')
        return
    # Assay-derived label values are first read here, after inference completion.
    pool, qrels = pd.read_csv(pool_path), pd.read_csv(qrel_path)
    assert len(pool) == 96 and pool.library_id.is_unique and pool.donor.is_unique
    ids = set(pool.library_id)
    assert len(qrels) == 288 and not qrels.duplicated(['query_id', 'sample_id']).any()
    assert qrels.grade.isin([0, 1, 2]).all()
    assert not qrels.evidence_status.isin(['insufficient evidence', 'insufficient_evidence', 'unjudged']).any()
    truth = {(row.query_id, row.sample_id): int(row.grade) for row in qrels.itertuples()}
    for qid in contract['query_ids']:
        assert set(qrels.loc[qrels.query_id.eq(qid), 'sample_id']) == ids
    old = pd.read_csv(old_rank_path, dtype={'variant': str}, float_precision='round_trip')
    new = pd.read_csv(out / 'rankings_all_30_queries.csv.gz', dtype={'variant': str}, float_precision='round_trip')
    assert len(new) == 86400 and new.method.nunique() == 10 and new.query_id.nunique() == 30
    assert old.method.nunique() == 17 and set(old.method).isdisjoint(new.method)
    assert not new.duplicated(['method', 'query_id', 'variant', 'sample_id']).any()
    for _, group in new.groupby(['method', 'query_id', 'variant']):
        assert len(group) == 96 and set(group.sample_id) == ids and np.isfinite(group.score).all()
    ranks = pd.concat([old, new.loc[new.query_id.isin(contract['query_ids'])]], ignore_index=True)
    assert len(ranks) == 23328 and ranks.method.nunique() == 27
    rows, errors, top_sets = [], [], {}
    for key, group in ranks.groupby(['method', 'query_id', 'variant']):
        method, qid, variant = key
        assert len(group) == 96 and group.sample_id.is_unique and set(group.sample_id) == ids
        records = [(row.sample_id, row.score, truth[(qid, row.sample_id)]) for row in group.itertuples()]
        values = metrics_for_ranking(records)
        top = group.sort_values(['score', 'sample_id'], ascending=[False, True]).head(10)
        grades = np.array([truth[(qid, sample)] for sample in group.sample_id])
        gains = 2.0**grades-1
        tied = float(ndcg_score(gains[None, :], group.score.to_numpy()[None, :], k=10, ignore_ties=False))
        idcg = sum((2**g-1)/math.log2(i+2) for i, g in enumerate(sorted(grades, reverse=True)[:10]))
        deterministic = sum((2**truth[(qid, sample)]-1)/math.log2(i+2) for i, sample in enumerate(top.sample_id)) / idcg
        errors += [abs(tied-values['tie_expected_ndcg_at_10']), abs(deterministic-values['ndcg_at_k'])]
        rows.append({'method': method, 'query_id': qid, 'variant': variant, **values})
        top_sets[key] = set(top.sample_id)
    result = pd.DataFrame(rows).set_index(['method', 'query_id', 'variant'])
    assert len(result) == 243 and max(errors) < 1e-12 and not result.isna().any().any()
    metrics = ['ndcg_at_k', 'precision_at_k', 'recall_at_10', 'upper_quartile_discovery_at_10',
               'tie_expected_ndcg_at_10', 'tie_expected_precision_at_10', 'tie_expected_recall_at_10',
               'tie_expected_strong_discovery_at_10']
    prior = pd.read_csv(old_metric_path, dtype={'variant': str}, float_precision='round_trip').set_index(['method', 'query_id', 'variant'])
    assert len(prior) == 153
    checked_columns = metrics + ['judged_at_k', 'k', 'unique_scores', 'maximum_tie_size', 'cutoff_tie_size']
    old_error = float(np.max(np.abs(result.loc[prior.index, checked_columns].to_numpy(dtype=float) - prior[checked_columns].to_numpy(dtype=float))))
    assert old_error < 1e-12
    questions = result[metrics].groupby(['method', 'query_id']).mean()
    means = questions.groupby('method').mean()
    stability_rows = []
    for (method, qid), group in result.groupby(['method', 'query_id']):
        variants = list(group.index.get_level_values('variant'))
        assert len(variants) == 3
        pairs = [(top_sets[(method, qid, a)], top_sets[(method, qid, b)]) for a, b in itertools.combinations(variants, 2)]
        stability_rows.append({'method': method, 'query_id': qid,
                               'paraphrase_top10_jaccard': float(np.mean([len(a & b)/len(a | b) for a, b in pairs]))})
    stability = pd.DataFrame(stability_rows)
    means['paraphrase_top10_jaccard'] = stability.groupby('method').paraphrase_top10_jaccard.mean()
    means['strong_hits_per_10'] = means.upper_quartile_discovery_at_10 * 10
    means['grade0_candidates_per_10'] = (1 - means.precision_at_k) * 10
    texts = pd.read_csv(out / 'portrait_texts.csv')
    calibration_equal = {}
    for arm in protocol['arms']:
        left = texts.loc[texts.method.eq('full_portrait_released_calibration_' + arm)].set_index('library_id').portrait_text
        right = texts.loc[texts.method.eq('full_portrait_no_calibration_' + arm)].set_index('library_id').portrait_text
        calibration_equal[arm] = int(left.eq(right.loc[left.index]).sum())
    diagnostics = pd.read_csv(out / 'input_diagnostics.csv')
    assert len(diagnostics) == 192 and diagnostics.matched_genes.eq(3524).all()
    assert diagnostics.normalization_library_sum.isna().all()
    clipping = diagnostics.groupby('arm')[['matched_feature_upper_clip_fraction', 'matched_feature_lower_clip_fraction']].median().to_dict('index')
    outputs = {'rankings_measured_three_queries.csv.gz': ranks, 'per_query_paraphrase_results.csv': result,
               'per_question_results.csv': questions, 'three_question_descriptive_means.csv': means,
               'paraphrase_stability.csv': stability}
    for name, frame in outputs.items():
        assert not (out / name).exists(), 'Inspect partial finalization before retrying'
        frame.to_csv(out / name, index=name not in ['rankings_measured_three_queries.csv.gz', 'paraphrase_stability.csv'],
                     compression='gzip' if name.endswith('.gz') else None)
    summary = {
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'evaluation_contract_sha256': sha(contract_path),
        'candidate_donors': 96, 'questions': contract['query_ids'], 'query_variants': 9,
        'methods_and_controls': 27, 'independently_checked_metric_groups': 243,
        'maximum_independent_ndcg_difference': max(errors), 'previous_153_metric_groups_maximum_difference': old_error,
        'previous_results_unchanged': True, 'join_loss_or_duplicate_rows': 0,
        'full_portrait_unique_texts_by_condition': texts.groupby('method').portrait_text.nunique().to_dict(),
        'same_text_with_and_without_calibration': calibration_equal, 'median_clipping': clipping,
        'means': means.reset_index().to_dict('records'), 'unvalidated_original_questions': 27, 'new_training': False,
        'boundaries': [
            'Three correlated T/B composition queries in one cohort; not cross-cohort generalization.',
            'Whole-candidate-cohort unsupervised normalization is transductive, not a prospective single-sample contract.',
            'Flow-based percentile relevance grades are not clinical thresholds or human statement ratings.',
            'All prior comparator inputs and method-specific preprocessing are retained, not replaced by size factors.',
            'Prior CPM outcomes were observed: this is exploratory, not confirmatory.',
            'New gene shuffle covers all 23323 mapped values; the older shuffle covered selected measured genes only.',
            'Sample discovery counts are measured; human review time and downstream experimental benefit are not.',
        ],
        'output_sha256': {name: sha(out / name) for name in outputs},
    }
    save_new(out / 'evaluation_summary.json', summary)
    print(means[['ndcg_at_k', 'tie_expected_ndcg_at_10', 'strong_hits_per_10', 'paraphrase_top10_jaccard']].to_string())
    print('Maximum independent metric discrepancy:', max(errors))
    print('Maximum discrepancy versus prior 153 groups:', old_error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    evaluate_run(args.study_root, args.input_root, args.output_root)
