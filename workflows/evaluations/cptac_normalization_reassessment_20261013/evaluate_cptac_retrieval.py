"""Evaluate all completed methods on frozen protein-concordance references.

Requires the complete portrait run; no silent partial-method comparison. The
main metric is conditional on common measurable protein evidence, with separate
full-pool coverage. These are not functional, diagnostic, or statement ratings.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import argparse
import itertools
import sys
import numpy as np
import pandas as pd


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(study_root, reassessment_root):
    STUDY=study_root.resolve(); ROOT=STUDY/'repair_v1'
    COHORT=ROOT/'paired_candidates/CPTAC_PanCancer'
    NEW=reassessment_root.resolve(); INFERENCE=NEW/'inference_v1'
    OUT=NEW/'evaluation_v1'
    sys.path.insert(0,str(ROOT))
    from cptac_retrieval_metrics import measure
    pool_path=COHORT/'rna_inputs_v1/mechanically_valid_candidates.csv'
    pool=pd.read_csv(pool_path).set_index('public_case_id')
    ids=pool.index.to_numpy(dtype=str);assert len(ids)==942 and pool.index.is_unique
    sources=[COHORT/f'{name}/rankings.csv.gz' for name in ['rna_predictions_v1','cellwhisperer_v1','conventional_rankings_v1','full_portrait_v1']]
    full=json.loads((COHORT/'full_portrait_v1/execution_summary.json').read_text())
    assert full['successful_inferences']==1884 and all(r=={'written':942,'ok':942,'failed':0} for r in full['conditions'].values())
    fresh=json.loads((INFERENCE/'execution_summary.json').read_text())
    assert fresh['successful_full_inferences']==3768 and fresh['failed_inferences']==0
    assert len(fresh['full_conditions'])==4 and set(fresh['full_conditions'].values())=={942}
    for relative,expected in fresh['output_sha256'].items():
        assert sha(INFERENCE/relative)==expected,relative
    sources += [INFERENCE/'rankings_all_30_queries.csv.gz']
    ranking_sources=sources.copy()
    sources += [COHORT/'protein_concordance_v2/qrels.csv',COHORT/'protein_concordance_v2/protocol.json',pool_path]
    sources += [NEW/'PROTOCOL.json',NEW/'preparation.json',INFERENCE/'execution_contract.json',INFERENCE/'execution_summary.json']
    contract={'created_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':{str(p.relative_to(STUDY)):sha(p) for p in sources},
              'driver_sha256':sha(__file__),'metric_helper_sha256':sha(ROOT/'cptac_retrieval_metrics.py'),
              'scope':'Exploratory fixed-panel protein concordance. Complete-evidence conditional pools shared by all methods; report full-pool coverage, do not treat unjudged as negative.',
              'strata':'Pooled all, pooled prior alignment-test, pooled no-recorded-case-match; and within each cancer cohort with cohort-specific protein references. Rankings are frozen and merely restricted, not refit to strata.',
              'macro_average':'Average paraphrases within each query; then eligible cohort blocks within each query; then queries. Report evaluable query/block counts. No per-paraphrase p-values.'}
    OUT.mkdir(exist_ok=False)
    path=OUT/'contract.json'
    if path.exists():
        old=json.loads(path.read_text());contract['created_utc']=old['created_utc'];assert old==contract
    else:path.write_text(json.dumps(contract,indent=2)+'\n')
    scores={}
    expected_query_ids={q['query_id'] for q in json.loads((STUDY/'queries.json').read_text())['queries']}
    for source in ranking_sources:
        frame=pd.read_csv(source,usecols=['method','query_id','variant','sample_id','score'],float_precision='round_trip')
        assert not frame.duplicated(['method','query_id','variant','sample_id']).any()
        for method,part in frame.groupby('method',sort=True):
            assert method not in scores and set(part.query_id)==expected_query_ids
            matrix=part.pivot(index=['query_id','variant'],columns='sample_id',values='score').reindex(columns=ids)
            assert matrix.shape==(90,942) and np.isfinite(matrix.to_numpy()).all()
            assert set(matrix.index.get_level_values('variant'))=={1,2,3}
            scores[method]=matrix
    assert len(scores)==32
    qrels=pd.read_csv(COHORT/'protein_concordance_v2/qrels.csv')
    assert not qrels.duplicated(['scope','cohort','query_id','sample_id']).any()
    labels={(scope,cancer):part.pivot(index='query_id',columns='sample_id',values='grade') for (scope,cancer),part in qrels.groupby(['scope','cohort'])}
    conditions=[('pooled','all','all',ids),
                ('pooled','all','prior_alignment_test',pool.index[pool.exposure_status.eq('known_alignment_test_case')].to_numpy(dtype=str)),
                ('pooled','all','no_recorded_case_match',pool.index[pool.exposure_status.eq('unresolved_no_case_match_in_this_crosswalk')].to_numpy(dtype=str))]
    conditions += [('within_cohort',cancer,'all',group.index.to_numpy(dtype=str)) for cancer,group in pool.groupby('cohort',sort=True)]
    results=[]; stability=[]
    for scope,cancer,stratum,subset in conditions:
        truth=labels[(scope,cancer)].reindex(columns=subset)
        assert set(truth.index)==expected_query_ids
        cancer_labels=pool.loc[subset,'cohort'].to_numpy(dtype=str)
        for method,matrix in scores.items():
            for (qid,variant),row in matrix[subset].iterrows():
                metric=measure(subset,row.to_numpy(),truth.loc[qid].to_numpy(),cancer_labels)
                results.append({'scope':scope,'cohort':cancer,'stratum':stratum,'method':method,'query_id':qid,'variant':int(variant),**metric})
            for qid in sorted(expected_query_ids):
                known=np.isfinite(truth.loc[qid].to_numpy())
                judged_ids=np.asarray(subset)[known]
                if len(judged_ids)<10:
                    stability.append({'scope':scope,'cohort':cancer,'stratum':stratum,'method':method,'query_id':qid,
                                      'judged_cases':len(judged_ids),'conditional_top10_jaccard':np.nan})
                    continue
                sets=[]
                for variant in [1,2,3]:
                    values=matrix.loc[(qid,variant),judged_ids].to_numpy()
                    sets.append(set(judged_ids[np.lexsort((judged_ids,-values))[:10]]))
                value=float(np.mean([len(a&b)/len(a|b) for a,b in itertools.combinations(sets,2)]))
                stability.append({'scope':scope,'cohort':cancer,'stratum':stratum,'method':method,'query_id':qid,
                                  'judged_cases':len(judged_ids),'conditional_top10_jaccard':value})
        print('Evaluated',scope,cancer,stratum,len(subset),'cases',flush=True)
    result=pd.DataFrame(results)
    assert len(result)==37440
    keys=['scope','cohort','stratum','method','query_id','variant']
    old=pd.read_csv(COHORT/'retrieval_evaluation_v1/per_query_paraphrase_results.csv').set_index(keys).sort_index()
    rebuilt_old=result.set_index(keys).loc[old.index].sort_index()
    old_numeric=old.select_dtypes(include=[np.number]).columns.tolist()
    assert len(old)==21060 and np.allclose(old[old_numeric],rebuilt_old[old_numeric],atol=1e-12,rtol=0,equal_nan=True)
    old_error=float(np.nanmax(np.abs(old[old_numeric].to_numpy()-rebuilt_old[old_numeric].to_numpy())))
    for column in old.columns.difference(old_numeric):
        assert old[column].fillna('').equals(rebuilt_old[column].fillna('')),column
    result.to_csv(OUT/'per_query_paraphrase_results.csv',index=False)
    pd.DataFrame(stability).to_csv(OUT/'paraphrase_stability.csv',index=False)
    keys=['scope','cohort','stratum','method','query_id']
    metrics=['conditional_ndcg_at_10','conditional_precision_at_10','conditional_recall_at_10','conditional_strong_discovery_at_10',
             'conditional_tie_expected_ndcg_at_10','conditional_tie_expected_precision_at_10','conditional_cohorts_in_top10',
             'full_pool_top10_judged_fraction','full_pool_precision_at_10','full_pool_ndcg_at_10']
    questions=result.groupby(keys,dropna=False)[metrics].mean().reset_index()
    questions.to_csv(OUT/'per_question_results.csv',index=False)
    # Primary descriptive macro is only the specified common measurable pool.
    # Full-pool precision/nDCG remain per-query because method-specific missing
    # judgments make averaging just the nonmissing groups unfair.
    macro_metrics=[m for m in metrics if m not in ['full_pool_precision_at_10','full_pool_ndcg_at_10']]
    query_means=questions.groupby(['scope','stratum','method','query_id'])[macro_metrics].mean()
    means=query_means.groupby(['scope','stratum','method']).mean()
    means['evaluable_queries']=query_means.conditional_ndcg_at_10.groupby(['scope','stratum','method']).count()
    means.to_csv(OUT/'descriptive_macro_results.csv')
    coverage=result.groupby(['scope','cohort','stratum','query_id']).agg(candidate_cases=('candidate_cases','first'),judged_cases=('judged_cases','first'),conditional_groups=('conditional_ndcg_at_10','count')).reset_index()
    coverage.to_csv(OUT/'query_candidate_coverage.csv',index=False)
    report={'candidate_cases':942,'methods':sorted(scores),'questions':30,'query_variants':90,'metric_groups':len(result),
            'candidate_strata':{stratum:len(subset) for scope,cancer,stratum,subset in conditions if scope=='pooled'},
            'full_portrait_complete':True,'primary_scope':'pooled/all conditional common measurable pools',
            'old_metric_groups_reproduced':len(old),'old_metric_max_abs_error':old_error,
            'corrected_scale_full_inferences':3768,'new_methods_and_controls':14,
            'macro_results':json.loads(means.reset_index().to_json(orient='records')),
            'limitations':['Protein panels are molecular proxies, not independent functional truth or statement ratings.',
                           'Protein detection makes conditional-pool selection nonrandom; full-pool coverage is mandatory.',
                           'Prior-test and no-recorded-match strata are not a guarantee of strict new external independence.',
                           'Queries share biological panels; paraphrases are repeated measures. No journal-suitability or mechanism claim.']}
    for relative,expected in contract['source_sha256'].items():assert sha(STUDY/relative)==expected,relative
    (OUT/'evaluation_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(means.loc[('pooled','all'),['conditional_ndcg_at_10','conditional_tie_expected_ndcg_at_10','conditional_precision_at_10','full_pool_top10_judged_fraction']].to_string())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,required=True)
    parser.add_argument('--reassessment-root',type=Path,required=True)
    args=parser.parse_args()
    main(args.study_root,args.reassessment_root)
